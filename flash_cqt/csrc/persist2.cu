//the factored resident pair. the whole factored transform in one
//cooperative launch per direction, every phase body transcribed from a
//kernel already gated on its own, the packing folded into the first pass
//load, grid wide syncs where the launch boundaries were.
//
//forward   pass one (reads the interleaved signal directly), sync, pass
//          two, sync, pass three, sync, radix band analysis
//inverse   zero the half spectrum, sync, radix band synthesis (atomic
//          scatter, measured faster than the ell edges on ampere), sync,
//          retangle, sync, three inverse passes with syncs, unpack

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;

#include "radix_impl.cuh"

#define LL 32768
#define NN 65536
#define TPB 128
#define SH_FLOATS 8256

struct P2 {
    const float* x;
    float* kr; float* ki;
    float* frre; float* frim;
    float* xo;
    float* war; float* wai;   //pass ping pong planes, [B, 32768]
    float* wbr; float* wbi;
    const float* w32r; const float* w32i;
    const float* wlpr; const float* wlpi;
    const float* wkpr; const float* wkpi;
    const float* wnr; const float* wni;
    const int* xroute;                //radix band tables, access order
    const float* xgdv; const int* xpos;
    const float* xrr; const float* xri;
    const int* xdesc;
    long n_k;
    int nbands, batch;
};

//one 32 transform group of one pass, mode 0 reads the interleaved real
//signal, mode 1 complex planes first pass layout, mode 2 second pass,
//mode 3 third pass (permuted store, no twiddle)
__device__ void pass_body(const P2& p, int task, int mode,
                          const float* inr, const float* ini,
                          float* outr, float* outi,
                          const float* twr_t, const float* twi_t, float* sh)
{
    float* sr = sh;
    float* si = sh + 32 * 33;
    float* w32a = sh + 2 * 32 * 33;
    float* w32b = w32a + 32;
    const int g = task % 32;
    const int b = task / 32;
    const long base = (long)b * LL;
    if (threadIdx.x < 32) {
        w32a[threadIdx.x] = p.w32r[threadIdx.x];
        w32b[threadIdx.x] = p.w32i[threadIdx.x];
    }
    for (int i = threadIdx.x; i < 32 * 32; i += TPB) {
        const int row = i >> 5;
        const int col = i & 31;
        const long src = (mode == 3)
            ? base + (long)g * 1024 + row * 32 + col
            : base + (long)row * 1024 + g * 32 + col;
        if (mode == 0) {
            const long n = (long)row * 1024 + g * 32 + col;
            sr[row * 33 + col] = p.x[(long)b * NN + 2 * n];
            si[row * 33 + col] = p.x[(long)b * NN + 2 * n + 1];
        } else {
            sr[row * 33 + col] = inr[src];
            si[row * 33 + col] = ini[src];
        }
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rb = (int)(__brev((unsigned)lane) >> 27);
    const unsigned full = 0xffffffffu;
    for (int c = warp; c < 32; c += TPB / 32) {
        float ar = sr[rb * 33 + c];
        float ai = si[rb * 33 + c];
        for (int s = 0; s < 5; s++) {
            const int h = 1 << s;
            const float pr = __shfl_xor_sync(full, ar, h);
            const float pi = __shfl_xor_sync(full, ai, h);
            const int isv = lane & h;
            const float vvr = isv ? ar : pr;
            const float vvi = isv ? ai : pi;
            const float uur = isv ? pr : ar;
            const float uui = isv ? pi : ai;
            const int tix = (lane & (h - 1)) << (4 - s);
            const float tr = vvr * w32a[tix] - vvi * w32b[tix];
            const float ti = vvr * w32b[tix] + vvi * w32a[tix];
            ar = isv ? uur - tr : uur + tr;
            ai = isv ? uui - ti : uui + ti;
        }
        long dst;
        if (mode <= 1) {
            const int bt = g * 32 + c;
            const int tix = bt * 32 + lane;
            const float xr = ar * twr_t[tix] - ai * twi_t[tix];
            ai = ar * twi_t[tix] + ai * twr_t[tix];
            ar = xr;
            dst = base + (long)bt * 32 + lane;
        } else if (mode == 2) {
            const int tix = g * 32 + lane;
            const float xr = ar * twr_t[tix] - ai * twi_t[tix];
            ai = ar * twi_t[tix] + ai * twr_t[tix];
            ar = xr;
            dst = base + (long)c * 1024 + g * 32 + lane;
        } else {
            dst = base + (long)g * 1024 + c * 32 + lane;
        }
        outr[dst] = ar;
        outi[dst] = ai;
    }
    __syncthreads();
}

//the band phases run the radix engine, same bodies as radix.cu via the
//shared header, the resident shell only adapts the parameter block
__device__ inline RP p2rp(const P2& p, const float* zr, const float* zi)
{
    RP r{};
    r.zr = zr; r.zi = zi;
    r.kr = p.kr; r.ki = p.ki;
    r.route = (const uint4*)p.xroute;
    r.fr = p.frre; r.fi = p.frim;
    r.gdv = p.xgdv; r.pos = p.xpos;
    r.rr = p.xrr; r.ri = p.xri;
    r.desc = p.xdesc;
    r.n_k = p.n_k;
    return r;
}

__global__ void fwd2_kernel(P2 p)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sh[];
    const int np = 32 * p.batch;
    for (int t = blockIdx.x; t < np; t += gridDim.x)
        pass_body(p, t, 0, p.x, p.x, p.war, p.wai, p.wlpr, p.wlpi, sh);
    grid.sync();
    for (int t = blockIdx.x; t < np; t += gridDim.x)
        pass_body(p, t, 2, p.war, p.wai, p.wbr, p.wbi, p.wkpr, p.wkpi, sh);
    grid.sync();
    for (int t = blockIdx.x; t < np; t += gridDim.x)
        pass_body(p, t, 3, p.wbr, p.wbi, p.war, p.wai, p.wkpr, p.wkpi, sh);
    grid.sync();
    const RP rp = p2rp(p, p.war, p.wai);
    const int nbt = p.nbands * p.batch;
    for (int t = blockIdx.x; t < nbt; t += gridDim.x) {
        radix_band<0>(rp, t % p.nbands, t / p.nbands, sh);
        //terminal barrier, the next task reuses the shared planes
        __syncthreads();
    }
}

__global__ void inv2_kernel(P2 p)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sh[];
    const long nfr = (long)p.batch * (LL + 1);
    for (long i = (long)blockIdx.x * TPB + threadIdx.x; i < nfr;
         i += (long)gridDim.x * TPB) {
        p.frre[i] = 0.f;
        p.frim[i] = 0.f;
    }
    grid.sync();
    const RP rp = p2rp(p, p.war, p.wai);
    const int nbt = p.nbands * p.batch;
    for (int t = blockIdx.x; t < nbt; t += gridDim.x) {
        radix_band<1>(rp, t % p.nbands, t / p.nbands, sh);
        __syncthreads();
    }
    grid.sync();
    //retangle into wa
    const long nl = (long)p.batch * LL;
    for (long i = (long)blockIdx.x * TPB + threadIdx.x; i < nl;
         i += (long)gridDim.x * TPB) {
        const long b = i / LL;
        const long q = i % LL;
        const long nf = LL + 1;
        const float x0r = p.frre[b * nf + q];
        const float x0i = p.frim[b * nf + q];
        const float x1r = p.frre[b * nf + (LL - q)];
        const float x1i = -p.frim[b * nf + (LL - q)];
        const float er = 0.5f * (x0r + x1r);
        const float ei = 0.5f * (x0i + x1i);
        const float dr = 0.5f * (x0r - x1r);
        const float di = 0.5f * (x0i - x1i);
        const float orr = dr * p.wnr[q] + di * p.wni[q];
        const float oii = di * p.wnr[q] - dr * p.wni[q];
        p.war[b * LL + q] = er - oii;
        p.wai[b * LL + q] = -(ei + orr);
    }
    grid.sync();
    const int np = 32 * p.batch;
    for (int t = blockIdx.x; t < np; t += gridDim.x)
        pass_body(p, t, 1, p.war, p.wai, p.wbr, p.wbi, p.wlpr, p.wlpi, sh);
    grid.sync();
    for (int t = blockIdx.x; t < np; t += gridDim.x)
        pass_body(p, t, 2, p.wbr, p.wbi, p.war, p.wai, p.wkpr, p.wkpi, sh);
    grid.sync();
    for (int t = blockIdx.x; t < np; t += gridDim.x)
        pass_body(p, t, 3, p.war, p.wai, p.wbr, p.wbi, p.wkpr, p.wkpi, sh);
    grid.sync();
    //unpack, permuted read, interleaved write, conj and scale
    for (long i = (long)blockIdx.x * TPB + threadIdx.x; i < nl;
         i += (long)gridDim.x * TPB) {
        const long b = i / LL;
        const long n = i % LL;
        const long idx = ((n & 31) * 1024) + (((n >> 5) & 31) * 32) + (n >> 10);
        const float sc = 1.f / (float)LL;
        p.xo[b * (long)NN + 2 * n] = p.wbr[b * LL + idx] * sc;
        p.xo[b * (long)NN + 2 * n + 1] = -p.wbi[b * LL + idx] * sc;
    }
}

static void launch(const void* kp, P2& p, cudaStream_t stream)
{
    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int per = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per, kp, TPB, SH_FLOATS * sizeof(float));
    if (per < 1) per = 1;
    dim3 grid(sms * per);
    dim3 block(TPB);
    void* args[] = {&p};
    cudaError_t err = cudaLaunchCooperativeKernel(
        kp, grid, block, args, SH_FLOATS * sizeof(float), stream);
    TORCH_CHECK(err == cudaSuccess, "cooperative launch failed, ",
                cudaGetErrorString(err));
}

#define FILL_ARGS                                                            \
    torch::Tensor x, torch::Tensor kr, torch::Tensor ki,                     \
    torch::Tensor frre, torch::Tensor frim, torch::Tensor xo,                \
    torch::Tensor war, torch::Tensor wai,                                    \
    torch::Tensor wbr, torch::Tensor wbi,                                    \
    torch::Tensor w32r, torch::Tensor w32i,                                  \
    torch::Tensor wlpr, torch::Tensor wlpi,                                  \
    torch::Tensor wkpr, torch::Tensor wkpi,                                  \
    torch::Tensor wnr, torch::Tensor wni,                                    \
    torch::Tensor xdesc, torch::Tensor xroute,                               \
    torch::Tensor xgdv, torch::Tensor xpos,                                  \
    torch::Tensor xrr, torch::Tensor xri,                                    \
    int64_t n_k, int64_t batch

static P2 fill(FILL_ARGS)
{
    P2 p;
    p.x = x.data_ptr<float>();
    p.kr = kr.data_ptr<float>(); p.ki = ki.data_ptr<float>();
    p.frre = frre.data_ptr<float>(); p.frim = frim.data_ptr<float>();
    p.xo = xo.data_ptr<float>();
    p.war = war.data_ptr<float>(); p.wai = wai.data_ptr<float>();
    p.wbr = wbr.data_ptr<float>(); p.wbi = wbi.data_ptr<float>();
    p.w32r = w32r.data_ptr<float>(); p.w32i = w32i.data_ptr<float>();
    p.wlpr = wlpr.data_ptr<float>(); p.wlpi = wlpi.data_ptr<float>();
    p.wkpr = wkpr.data_ptr<float>(); p.wkpi = wkpi.data_ptr<float>();
    p.wnr = wnr.data_ptr<float>(); p.wni = wni.data_ptr<float>();
    p.xdesc = xdesc.data_ptr<int>();
    p.xroute = xroute.data_ptr<int>();
    p.xgdv = xgdv.data_ptr<float>();
    p.xpos = xpos.data_ptr<int>();
    p.xrr = xrr.data_ptr<float>(); p.xri = xri.data_ptr<float>();
    p.n_k = n_k;
    p.nbands = xdesc.size(0);
    p.batch = (int)batch;
    return p;
}

void fwd2(FILL_ARGS)
{
    P2 p = fill(x, kr, ki, frre, frim, xo, war, wai, wbr, wbi, w32r, w32i,
                wlpr, wlpi, wkpr, wkpi, wnr, wni, xdesc, xroute,
                xgdv, xpos, xrr, xri, n_k, batch);
    launch((const void*)fwd2_kernel, p, at::cuda::getCurrentCUDAStream());
}

void inv2(FILL_ARGS)
{
    P2 p = fill(x, kr, ki, frre, frim, xo, war, wai, wbr, wbi, w32r, w32i,
                wlpr, wlpi, wkpr, wkpi, wnr, wni, xdesc, xroute,
                xgdv, xpos, xrr, xri, n_k, batch);
    launch((const void*)inv2_kernel, p, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fwd2", &fwd2, "factored resident forward");
    m.def("inv2", &inv2, "factored resident inverse");
}
