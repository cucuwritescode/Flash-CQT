//the big fft stages on tensor cores at near fp32 accuracy. every stage is
//a matmul against the fixed 256 point dft matrix, run as tf32 wmma with the
//three way split done in registers, high and low halves of both operands,
//three mma per real product, fp32 accumulate. the torch level version of
//this split lost to its own memory traffic (bench history), here the split
//lives in shared and registers and touches hbm once per tensor.
//
//four stage flavours from one template
//  s1 fwd  real input, times F1, times twiddle, transposed store
//  s1 inv  complex input (mirrored spectrum), same shape
//  s2 fwd  complex input, times F1, direct store (the matrix layout)
//  s2 inv  complex input, times F1, real part only, strided store, scaled
//
//tiles, 64 by 64 output per block of 4 warps, each warp owns a 2 by 2 grid
//of 16 by 16 fragments, k in steps of 16 (two 8 wide mma slices). needs
//sm80 or newer for tf32 wmma.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
using namespace nvcuda;

#define TM 64
#define TN 64
#define TK 16

//OUT_MODE 0 transposed two plane store (stage one feeding stage two)
//OUT_MODE 1 direct two plane store (the spectrum matrix layout)
//OUT_MODE 2 real part only, sample index row + 256 col, scaled
template <bool A_COL, bool REAL_IN, bool TWIDDLE, int OUT_MODE>
__global__ void stage_kernel(
    const float* __restrict__ ar_g, const float* __restrict__ ai_g,
    const float* __restrict__ brh, const float* __restrict__ brl,
    const float* __restrict__ bih, const float* __restrict__ bil,
    const float* __restrict__ nbih, const float* __restrict__ nbil,
    const float* __restrict__ twr, const float* __restrict__ twi,
    float* outr, float* outi, float scale)
{
    //dynamic shared, the a tile splits while the k loop runs, the whole
    //area is reused as epilogue staging afterwards
    extern __shared__ float sh[];
    float* sah = sh;                    //[TM][TK]
    float* sal = sh + TM * TK;
    float* sih = sh + 2 * TM * TK;
    float* sil = sh + 3 * TM * TK;
    float* ep = sh;                     //[2][TM][TN] after the loop

    const int b = blockIdx.z;
    const int row0 = blockIdx.x * TM;
    const int col0 = blockIdx.y * TN;
    const long abase = (long)b * 65536;

    const int warp = threadIdx.x >> 5;
    const int wr = warp >> 1;
    const int wc = warp & 1;

    //hh terms and the scaled cross terms cannot share an accumulator,
    //two scales, two sets
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> accr[2][2], acci[2][2];
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> accr2[2][2], acci2[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(accr[i][j], 0.f);
            wmma::fill_fragment(acci[i][j], 0.f);
            wmma::fill_fragment(accr2[i][j], 0.f);
            wmma::fill_fragment(acci2[i][j], 0.f);
        }

    for (int k0 = 0; k0 < 256; k0 += TK) {
        //load and split the a tile, s1 reads the strided sample layout,
        //s2 reads the row major planes its stage one stored
        for (int i = threadIdx.x; i < TM * TK; i += blockDim.x) {
            const int r = i / TK;
            const int c = i % TK;
            const long src = A_COL
                ? abase + 256L * (k0 + c) + (row0 + r)
                : abase + 256L * (row0 + r) + (k0 + c);
            const float v = ar_g[src];
            const float h = wmma::__float_to_tf32(v);
            sah[r * TK + c] = h;
            //residual scaled up so its tf32 rounding costs nothing, the
            //cross accumulator is descaled once in the epilogue
            sal[r * TK + c] = (v - h) * 2048.f;
            if (!REAL_IN) {
                const float u = ai_g[src];
                const float g = wmma::__float_to_tf32(u);
                sih[r * TK + c] = g;
                sil[r * TK + c] = (u - g) * 2048.f;
            }
        }
        __syncthreads();

        for (int s = 0; s < 2; s++) {
            //a fragments once per slice, shared with both column subtiles.
            //the high halves were rounded when the tile was staged, only
            //the residual fragments still need the formal conversion
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                           wmma::row_major> Ah[2], Al[2], Ch[2], Cl[2];
            for (int i = 0; i < 2; i++) {
                const int ra = (wr * 32 + i * 16) * TK + s * 8;
                wmma::load_matrix_sync(Ah[i], sah + ra, TK);
                wmma::load_matrix_sync(Al[i], sal + ra, TK);
                for (int t = 0; t < Al[i].num_elements; t++)
                    Al[i].x[t] = wmma::__float_to_tf32(Al[i].x[t]);
                if (!REAL_IN) {
                    wmma::load_matrix_sync(Ch[i], sih + ra, TK);
                    wmma::load_matrix_sync(Cl[i], sil + ra, TK);
                    for (int t = 0; t < Cl[i].num_elements; t++)
                        Cl[i].x[t] = wmma::__float_to_tf32(Cl[i].x[t]);
                }
            }
            for (int j = 0; j < 2; j++) {
                const long bo = (long)(k0 + s * 8) * 256 + col0 + wc * 32 + j * 16;
                //table fragments, pre rounded on the host, loaded once per
                //column subtile and used by both row subtiles. four way
                //split, high by high, high by low, low by high, low by low
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                               wmma::row_major> Brh, Brl, Bqh, Bql;
                wmma::load_matrix_sync(Brh, brh + bo, 256);
                wmma::load_matrix_sync(Brl, brl + bo, 256);
                wmma::load_matrix_sync(Bqh, bih + bo, 256);
                wmma::load_matrix_sync(Bql, bil + bo, 256);
                for (int i = 0; i < 2; i++) {
                    wmma::mma_sync(accr[i][j], Ah[i], Brh, accr[i][j]);
                    wmma::mma_sync(accr2[i][j], Ah[i], Brl, accr2[i][j]);
                    wmma::mma_sync(accr2[i][j], Al[i], Brh, accr2[i][j]);
                    wmma::mma_sync(acci[i][j], Ah[i], Bqh, acci[i][j]);
                    wmma::mma_sync(acci2[i][j], Ah[i], Bql, acci2[i][j]);
                    wmma::mma_sync(acci2[i][j], Al[i], Bqh, acci2[i][j]);
                    if (!REAL_IN) {
                        wmma::mma_sync(acci[i][j], Ch[i], Brh, acci[i][j]);
                        wmma::mma_sync(acci2[i][j], Ch[i], Brl, acci2[i][j]);
                        wmma::mma_sync(acci2[i][j], Cl[i], Brh, acci2[i][j]);
                    }
                }
                if (!REAL_IN) {
                    //minus ai times bi through the negated tables
                    wmma::load_matrix_sync(Brh, nbih + bo, 256);
                    wmma::load_matrix_sync(Brl, nbil + bo, 256);
                    for (int i = 0; i < 2; i++) {
                        wmma::mma_sync(accr[i][j], Ch[i], Brh, accr[i][j]);
                        wmma::mma_sync(accr2[i][j], Ch[i], Brl, accr2[i][j]);
                        wmma::mma_sync(accr2[i][j], Cl[i], Brh, accr2[i][j]);
                    }
                }
            }
        }
        __syncthreads();
    }

    //stage all four accumulator planes through shared, then combine with
    //the descale, out = hh + cross / 2048
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            float* dst = ep + (wr * 32 + i * 16) * TN + wc * 32 + j * 16;
            wmma::store_matrix_sync(dst, accr[i][j], TN, wmma::mem_row_major);
            wmma::store_matrix_sync(dst + TM * TN, acci[i][j], TN,
                                    wmma::mem_row_major);
            wmma::store_matrix_sync(dst + 2 * TM * TN, accr2[i][j], TN,
                                    wmma::mem_row_major);
            wmma::store_matrix_sync(dst + 3 * TM * TN, acci2[i][j], TN,
                                    wmma::mem_row_major);
        }
    __syncthreads();

    for (int i = threadIdx.x; i < TM * TN; i += blockDim.x) {
        const int r = i / TN;
        const int c = i % TN;
        float cr = ep[r * TN + c] + ep[2 * TM * TN + r * TN + c] * (1.f / 2048.f);
        float ci = ep[TM * TN + r * TN + c] + ep[3 * TM * TN + r * TN + c] * (1.f / 2048.f);
        if (TWIDDLE) {
            const float tr = twr[(long)(row0 + r) * 256 + col0 + c];
            const float ti = twi[(long)(row0 + r) * 256 + col0 + c];
            const float xr = cr * tr - ci * ti;
            const float xi = cr * ti + ci * tr;
            cr = xr;
            ci = xi;
        }
        if (OUT_MODE == 0) {
            //transposed store, stage two reads its rows contiguously
            outr[abase + 256L * (col0 + c) + row0 + r] = cr;
            outi[abase + 256L * (col0 + c) + row0 + r] = ci;
        } else if (OUT_MODE == 1) {
            outr[abase + 256L * (row0 + r) + col0 + c] = cr;
            outi[abase + 256L * (row0 + r) + col0 + c] = ci;
        } else {
            //real signal out, sample index row + 256 col
            outr[abase + (row0 + r) + 256L * (col0 + c)] = cr * scale;
        }
    }
}

//launch through a function pointer so template commas survive the macro
#define LAUNCH(...)                                                         \
    auto kp = __VA_ARGS__;                                                  \
    dim3 grid(4, 4, (unsigned)ar.size(0));                                  \
    const int shmem = 4 * TM * TN * (int)sizeof(float);                     \
    static bool cfg = false;                                                \
    if (!cfg) {                                                             \
        cudaFuncSetAttribute(kp,                                            \
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);            \
        cfg = true;                                                         \
    }                                                                       \
    kp<<<grid, 128, shmem, at::cuda::getCurrentCUDAStream()>>>(                                               \
        ar.data_ptr<float>(), ai.data_ptr<float>(),                         \
        brh.data_ptr<float>(), brl.data_ptr<float>(),                       \
        bih.data_ptr<float>(), bil.data_ptr<float>(),                       \
        nbih.data_ptr<float>(), nbil.data_ptr<float>(),                     \
        twr.data_ptr<float>(), twi.data_ptr<float>(),                       \
        outr.data_ptr<float>(), outi.data_ptr<float>(), (float)scale)

void s1_fwd(torch::Tensor ar, torch::Tensor ai,
            torch::Tensor outr, torch::Tensor outi,
            torch::Tensor brh, torch::Tensor brl,
            torch::Tensor bih, torch::Tensor bil,
            torch::Tensor nbih, torch::Tensor nbil,
            torch::Tensor twr, torch::Tensor twi, double scale)
{
    LAUNCH(stage_kernel<true, true, true, 0>);
}

void s1_inv(torch::Tensor ar, torch::Tensor ai,
            torch::Tensor outr, torch::Tensor outi,
            torch::Tensor brh, torch::Tensor brl,
            torch::Tensor bih, torch::Tensor bil,
            torch::Tensor nbih, torch::Tensor nbil,
            torch::Tensor twr, torch::Tensor twi, double scale)
{
    LAUNCH(stage_kernel<true, false, true, 0>);
}

void s2_fwd(torch::Tensor ar, torch::Tensor ai,
            torch::Tensor outr, torch::Tensor outi,
            torch::Tensor brh, torch::Tensor brl,
            torch::Tensor bih, torch::Tensor bil,
            torch::Tensor nbih, torch::Tensor nbil,
            torch::Tensor twr, torch::Tensor twi, double scale)
{
    LAUNCH(stage_kernel<false, false, false, 1>);
}

void s2_inv(torch::Tensor ar, torch::Tensor ai,
            torch::Tensor outr, torch::Tensor outi,
            torch::Tensor brh, torch::Tensor brl,
            torch::Tensor bih, torch::Tensor bil,
            torch::Tensor nbih, torch::Tensor nbil,
            torch::Tensor twr, torch::Tensor twi, double scale)
{
    LAUNCH(stage_kernel<false, false, false, 2>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("s1_fwd", &s1_fwd, "stage one forward, real in, twiddle, transposed out");
    m.def("s1_inv", &s1_inv, "stage one inverse, complex in, twiddle, transposed out");
    m.def("s2_fwd", &s2_fwd, "stage two forward, complex in, direct out");
    m.def("s2_inv", &s2_inv, "stage two inverse, complex in, real strided out");
}
