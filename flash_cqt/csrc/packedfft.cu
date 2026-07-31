//the packed big fft, F_32768 as three radix 32 passes, the first layer of
//the factored transform. one warp owns one 32 point transform, the input
//element sits in a lane, the dft is 32 lane broadcasts against the 32 by 32
//matrix staged in shared, the inter pass twiddle multiplies in registers.
//index scheme validated against torch.fft in bench/packedfft.py before this
//file was written, the kernel is a transcription.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define LL 32768
#define RD 32
#define TPB 128

//each block owns 32 transforms of one pass, stages their 32 by 32 input
//tile through shared with coalesced loads (the naive per lane strided load
//cost 3x at batch), then each warp runs eight transforms through the five
//shuffle butterfly stages. pass three stores in the permuted order
//Zp[k1*1024 + k2*32 + k3], coalesced, the router tables absorb the
//permutation at build time for free.
//PASS 1  group g owns b in [32g, 32g+32), in z[a*1024 + b], twiddle
//        WL[b*lane], out c1[b*32 + lane]
//PASS 2  group g is b2, transforms k1, in c1[a2*1024 + b2*32 + k1],
//        twiddle W1k[b2*lane], out c2[k1*1024 + b2*32 + lane]
//PASS 3  group g is k1, transforms k2, in c2[k1*1024 + b2*32 + k2],
//        no twiddle, out Zp[k1*1024 + k2*32 + lane]
template <int PASS>
__global__ void pass_kernel(
    const float* __restrict__ inr, const float* __restrict__ ini,
    float* __restrict__ outr, float* __restrict__ outi,
    const float* __restrict__ fr_g, const float* __restrict__ fi_g,
    const float* __restrict__ twr, const float* __restrict__ twi)
{
    //tile rows padded to 33 so the butterfly column reads spread banks
    __shared__ float sr[RD * 33], si[RD * 33];
    __shared__ float wr32[RD], wi32[RD];
    if (threadIdx.x < RD) {
        wr32[threadIdx.x] = fr_g[RD + threadIdx.x];
        wi32[threadIdx.x] = fi_g[RD + threadIdx.x];
    }
    const int g = blockIdx.x;
    const long base = (long)blockIdx.y * LL;
    for (int i = threadIdx.x; i < RD * RD; i += TPB) {
        const int row = i >> 5;
        const int col = i & 31;
        const long src = (PASS == 3)
            ? base + (long)g * 1024 + row * 32 + col
            : base + (long)row * 1024 + g * 32 + col;
        sr[row * 33 + col] = inr[src];
        si[row * 33 + col] = ini[src];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rb = (int)(__brev((unsigned)lane) >> 27);
    const unsigned full = 0xffffffffu;
    for (int c = warp; c < RD; c += TPB / 32) {
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
            const float tr = vvr * wr32[tix] - vvi * wi32[tix];
            const float ti = vvr * wi32[tix] + vvi * wr32[tix];
            ar = isv ? uur - tr : uur + tr;
            ai = isv ? uui - ti : uui + ti;
        }
        long dst;
        if (PASS == 1) {
            const int b = g * 32 + c;
            //twiddle tables are pre reordered to [transform, lane]
            const int tix = b * 32 + lane;
            const float xr = ar * twr[tix] - ai * twi[tix];
            ai = ar * twi[tix] + ai * twr[tix];
            ar = xr;
            dst = base + (long)b * 32 + lane;
        } else if (PASS == 2) {
            const int tix = g * 32 + lane;
            const float xr = ar * twr[tix] - ai * twi[tix];
            ai = ar * twi[tix] + ai * twr[tix];
            ar = xr;
            dst = base + (long)c * 1024 + g * 32 + lane;
        } else {
            dst = base + (long)g * 1024 + c * 32 + lane;
        }
        outr[dst] = ar;
        outi[dst] = ai;
    }
}

void fft3(torch::Tensor zr, torch::Tensor zi,
          torch::Tensor t1r, torch::Tensor t1i,
          torch::Tensor t2r, torch::Tensor t2i,
          torch::Tensor outr, torch::Tensor outi,
          torch::Tensor FR, torch::Tensor FI,
          torch::Tensor WLR, torch::Tensor WLI,
          torch::Tensor WKR, torch::Tensor WKI, int64_t batch)
{
    dim3 grid(32, (unsigned)batch);
    dim3 block(TPB);
    auto s = at::cuda::getCurrentCUDAStream();
    pass_kernel<1><<<grid, block, 0, s>>>(
        zr.data_ptr<float>(), zi.data_ptr<float>(),
        t1r.data_ptr<float>(), t1i.data_ptr<float>(),
        FR.data_ptr<float>(), FI.data_ptr<float>(),
        WLR.data_ptr<float>(), WLI.data_ptr<float>());
    pass_kernel<2><<<grid, block, 0, s>>>(
        t1r.data_ptr<float>(), t1i.data_ptr<float>(),
        t2r.data_ptr<float>(), t2i.data_ptr<float>(),
        FR.data_ptr<float>(), FI.data_ptr<float>(),
        WKR.data_ptr<float>(), WKI.data_ptr<float>());
    pass_kernel<3><<<grid, block, 0, s>>>(
        t2r.data_ptr<float>(), t2i.data_ptr<float>(),
        outr.data_ptr<float>(), outi.data_ptr<float>(),
        FR.data_ptr<float>(), FI.data_ptr<float>(),
        WKR.data_ptr<float>(), WKI.data_ptr<float>());
}

//the inverse tail. retangle folds the half spectrum back into the packed
//complex form with the ifft conjugation baked into the store, the three
//pass kernel then runs as the inverse transform, and unpack absorbs the
//permuted layout and the interleave in one gather. maths validated against
//irfft in bench/router.py before transcription, 131.5 db.
__global__ void retangle_kernel(
    const float* __restrict__ frr, const float* __restrict__ fri,
    float* __restrict__ wr, float* __restrict__ wi,
    const float* __restrict__ wnr, const float* __restrict__ wni)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long nfr = LL + 1;
    const long b = blockIdx.y;
    if (i >= LL) return;
    const float x0r = frr[b * nfr + i];
    const float x0i = fri[b * nfr + i];
    const float x1r = frr[b * nfr + (LL - i)];
    const float x1i = -fri[b * nfr + (LL - i)];
    const float er = 0.5f * (x0r + x1r);
    const float ei = 0.5f * (x0i + x1i);
    const float dr = 0.5f * (x0r - x1r);
    const float di = 0.5f * (x0i - x1i);
    //odd = d times conj(wn)
    const float orr = dr * wnr[i] + di * wni[i];
    const float oii = di * wnr[i] - dr * wni[i];
    //z = even + i odd, stored conjugated for the ifft trick
    wr[b * LL + i] = er - oii;
    wi[b * LL + i] = -(ei + orr);
}

__global__ void unpack_kernel(
    const float* __restrict__ zpr, const float* __restrict__ zpi,
    float* __restrict__ xo)
{
    const long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long b = blockIdx.y;
    if (n >= LL) return;
    const long idx = ((n & 31) * 1024) + (((n >> 5) & 31) * 32) + (n >> 10);
    const float sc = 1.f / (float)LL;
    xo[b * (2L * LL) + 2 * n] = zpr[b * LL + idx] * sc;
    xo[b * (2L * LL) + 2 * n + 1] = -zpi[b * LL + idx] * sc;
}

//retangle over the ell edges, the reduction of at most three slots per
//bin fused into the load, predicated selects so unwritten slots are never
//touched arithmetically
__global__ void retangle_ell_kernel(
    const float* __restrict__ er, const float* __restrict__ ei,
    const int* __restrict__ deg,
    float* __restrict__ wr, float* __restrict__ wi,
    const float* __restrict__ wnr, const float* __restrict__ wni)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long ne = 3L * (LL + 1);
    const long b = blockIdx.y;
    if (i >= LL) return;
    const long q0 = i;
    const long q1 = LL - i;
    float x0r = 0.f, x0i = 0.f, x1r = 0.f, x1i = 0.f;
    const int d0 = deg[q0];
    const int d1 = deg[q1];
    for (int s = 0; s < 3; s++) {
        if (s < d0) {
            x0r += er[b * ne + 3 * q0 + s];
            x0i += ei[b * ne + 3 * q0 + s];
        }
        if (s < d1) {
            x1r += er[b * ne + 3 * q1 + s];
            x1i += ei[b * ne + 3 * q1 + s];
        }
    }
    x1i = -x1i;
    const float er_ = 0.5f * (x0r + x1r);
    const float ei_ = 0.5f * (x0i + x1i);
    const float dr = 0.5f * (x0r - x1r);
    const float di = 0.5f * (x0i - x1i);
    const float orr = dr * wnr[i] + di * wni[i];
    const float oii = di * wnr[i] - dr * wni[i];
    wr[b * LL + i] = er_ - oii;
    wi[b * LL + i] = -(ei_ + orr);
}

void inv_tail_ell(torch::Tensor er, torch::Tensor ei, torch::Tensor deg,
                  torch::Tensor wr, torch::Tensor wi,
                  torch::Tensor t1r, torch::Tensor t1i,
                  torch::Tensor t2r, torch::Tensor t2i,
                  torch::Tensor zpr, torch::Tensor zpi, torch::Tensor xo,
                  torch::Tensor FR, torch::Tensor FI,
                  torch::Tensor WLR, torch::Tensor WLI,
                  torch::Tensor WKR, torch::Tensor WKI,
                  torch::Tensor WNR, torch::Tensor WNI, int64_t batch)
{
    auto s = at::cuda::getCurrentCUDAStream();
    dim3 eb(256);
    dim3 eg((LL + 255) / 256, (unsigned)batch);
    retangle_ell_kernel<<<eg, eb, 0, s>>>(
        er.data_ptr<float>(), ei.data_ptr<float>(), deg.data_ptr<int>(),
        wr.data_ptr<float>(), wi.data_ptr<float>(),
        WNR.data_ptr<float>(), WNI.data_ptr<float>());
    fft3(wr, wi, t1r, t1i, t2r, t2i, zpr, zpi,
         FR, FI, WLR, WLI, WKR, WKI, batch);
    unpack_kernel<<<eg, eb, 0, s>>>(
        zpr.data_ptr<float>(), zpi.data_ptr<float>(), xo.data_ptr<float>());
}

void inv_tail(torch::Tensor frr, torch::Tensor fri,
              torch::Tensor wr, torch::Tensor wi,
              torch::Tensor t1r, torch::Tensor t1i,
              torch::Tensor t2r, torch::Tensor t2i,
              torch::Tensor zpr, torch::Tensor zpi, torch::Tensor xo,
              torch::Tensor FR, torch::Tensor FI,
              torch::Tensor WLR, torch::Tensor WLI,
              torch::Tensor WKR, torch::Tensor WKI,
              torch::Tensor WNR, torch::Tensor WNI, int64_t batch)
{
    auto s = at::cuda::getCurrentCUDAStream();
    dim3 eb(256);
    dim3 eg((LL + 255) / 256, (unsigned)batch);
    retangle_kernel<<<eg, eb, 0, s>>>(
        frr.data_ptr<float>(), fri.data_ptr<float>(),
        wr.data_ptr<float>(), wi.data_ptr<float>(),
        WNR.data_ptr<float>(), WNI.data_ptr<float>());
    fft3(wr, wi, t1r, t1i, t2r, t2i, zpr, zpi,
         FR, FI, WLR, WLI, WKR, WKI, batch);
    unpack_kernel<<<eg, eb, 0, s>>>(
        zpr.data_ptr<float>(), zpi.data_ptr<float>(), xo.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fft3", &fft3, "packed 32768 point fft, three radix 32 passes");
    m.def("inv_tail", &inv_tail, "retangle, packed inverse fft, unpack");
    m.def("inv_tail_ell", &inv_tail_ell,
          "ell reduce and retangle, packed inverse fft, unpack");
}
