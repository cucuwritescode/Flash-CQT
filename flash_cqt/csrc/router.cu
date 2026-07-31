//the router layer of the factored transform. band prologues read the
//packed half length spectrum through two taps, tap = A Z[p1] + B conj
//Z[p2], with the real fft untangle and the analysis window folded into the
//complex coefficients at table build time, then the band inverse dfts run
//exactly as the proven block kernels do. tables validated against the
//reference taps in bench/router.py before this file was written.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define LL 32768

//two stage bands, one thread block per band and batch element
__global__ void router_two_kernel(
    const float* __restrict__ zpr, const float* __restrict__ zpi,
    float* kr, float* ki,
    const int* __restrict__ p1, const int* __restrict__ p2,
    const float* __restrict__ tar, const float* __restrict__ tai,
    const float* __restrict__ tbr, const float* __restrict__ tbi,
    const float* __restrict__ f1r, const float* __restrict__ f1i,
    const float* __restrict__ twr, const float* __restrict__ twi,
    const float* __restrict__ f2r, const float* __restrict__ f2i,
    const int* __restrict__ task,
    long n_k, int n1, int n2, float scale)
{
    extern __shared__ float sh[];
    const int m = n1 * n2;
    float* car = sh;
    float* cai = sh + m;
    float* sr = sh + 2 * m;
    float* si = sh + 3 * m;

    const int band = blockIdx.x;
    const int b = blockIdx.y;
    const int ab = task[band * 2 + 0];
    const int ob = task[band * 2 + 1];

    //router prologue, two taps per element, conj in for the ifft trick
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int n1i = i / n2;
        const int n2i = i % n2;
        const int j = ab + n2i + n2 * n1i;
        const int q1 = p1[j];
        const int q2 = p2[j];
        const float z1r = zpr[(long)b * LL + q1];
        const float z1i = zpi[(long)b * LL + q1];
        const float z2r = zpr[(long)b * LL + q2];
        const float z2i = zpi[(long)b * LL + q2];
        const float vr = tar[j] * z1r - tai[j] * z1i
                       + tbr[j] * z2r + tbi[j] * z2i;
        const float vi = tar[j] * z1i + tai[j] * z1r
                       + tbi[j] * z2r - tbr[j] * z2i;
        car[n2i * n1 + n1i] = vr;
        cai[n2i * n1 + n1i] = -vi;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int n2i = i / n1;
        const int k1 = i % n1;
        float ar = 0.f, ai = 0.f;
        for (int n1i = 0; n1i < n1; n1i++) {
            const float xr = car[n2i * n1 + n1i];
            const float xi = cai[n2i * n1 + n1i];
            const float wr = f1r[n1i * n1 + k1];
            const float wi = f1i[n1i * n1 + k1];
            ar = fmaf(xr, wr, fmaf(-xi, wi, ar));
            ai = fmaf(xr, wi, fmaf(xi, wr, ai));
        }
        const float tr = twr[n2i * n1 + k1];
        const float ti = twi[n2i * n1 + k1];
        sr[n2i * n1 + k1] = ar * tr - ai * ti;
        si[n2i * n1 + k1] = ar * ti + ai * tr;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int k2 = i / n1;
        const int k1 = i % n1;
        float dr = 0.f, di = 0.f;
        for (int n2i = 0; n2i < n2; n2i++) {
            const float xr = sr[n2i * n1 + k1];
            const float xi = si[n2i * n1 + k1];
            const float wr = f2r[n2i * n2 + k2];
            const float wi = f2i[n2i * n2 + k2];
            dr = fmaf(xr, wr, fmaf(-xi, wi, dr));
            di = fmaf(xr, wi, fmaf(xi, wr, di));
        }
        const int oo = k1 + n1 * k2;
        kr[(long)b * n_k + ob + oo] = dr * scale;
        ki[(long)b * n_k + ob + oo] = -di * scale;
    }
}

//direct family, matrix already holds the inverse dft, no conj trick
__global__ void router_direct_kernel(
    const float* __restrict__ zpr, const float* __restrict__ zpi,
    float* kr, float* ki,
    const int* __restrict__ p1, const int* __restrict__ p2,
    const float* __restrict__ tar, const float* __restrict__ tai,
    const float* __restrict__ tbr, const float* __restrict__ tbi,
    const float* __restrict__ mr, const float* __restrict__ mi,
    const int* __restrict__ task, long n_k)
{
    extern __shared__ float sh[];
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    const int* t = task + row * 6;
    const int pb = t[0], mb = t[1], ob = t[2];
    const int m = t[3], nb = t[4], n0 = t[5];
    const int nn = min(64, m - n0);
    float* car = sh;
    float* cai = sh + nb * m;
    for (int i = threadIdx.x; i < nb * m; i += blockDim.x) {
        const int r = i / m;
        const int c = i % m;
        const int j = pb + r * m + c;
        const int q1 = p1[j];
        const int q2 = p2[j];
        const float z1r = zpr[(long)b * LL + q1];
        const float z1i = zpi[(long)b * LL + q1];
        const float z2r = zpr[(long)b * LL + q2];
        const float z2i = zpi[(long)b * LL + q2];
        car[i] = tar[j] * z1r - tai[j] * z1i + tbr[j] * z2r + tbi[j] * z2i;
        cai[i] = tar[j] * z1i + tai[j] * z1r + tbi[j] * z2r - tbr[j] * z2i;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nb * nn; i += blockDim.x) {
        const int r = i / nn;
        const int c = n0 + i % nn;
        float vr = 0.f, vi = 0.f;
        for (int k = 0; k < m; k++) {
            const float xr = car[r * m + k];
            const float xi = cai[r * m + k];
            const float wr = mr[mb + k * m + c];
            const float wi = mi[mb + k * m + c];
            vr = fmaf(xr, wr, fmaf(-xi, wi, vr));
            vi = fmaf(xr, wi, fmaf(xi, wr, vi));
        }
        kr[(long)b * n_k + ob + r * m + c] = vr;
        ki[(long)b * n_k + ob + r * m + c] = vi;
    }
}

//the atomic free inverse writers. every live tap owns a unique edge in a
//degree 3 ell buffer, plain stores, nothing contended, nothing zeroed,
//the reduction happens in the retangle kernel of the inverse tail.
//slot tables built and degree asserted in fused.py, reduce equivalence
//proven against index_add in the local gate before this was written
__global__ void ell_two_kernel(
    const float* __restrict__ kr, const float* __restrict__ ki,
    float* __restrict__ er, float* __restrict__ ei,
    const float* __restrict__ f1r, const float* __restrict__ f1i,
    const float* __restrict__ twr, const float* __restrict__ twi,
    const float* __restrict__ f2r, const float* __restrict__ f2i,
    const float* __restrict__ gdv, const int* __restrict__ eidx,
    const int* __restrict__ task,
    long n_k, long ne, int n1, int n2)
{
    extern __shared__ float sh[];
    const int m = n1 * n2;
    float* car = sh;
    float* cai = sh + m;
    float* sr = sh + 2 * m;
    float* si = sh + 3 * m;
    const int band = blockIdx.x;
    const int b = blockIdx.y;
    const int ab = task[band * 2 + 0];
    const int ob = task[band * 2 + 1];
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int n1i = i / n2;
        const int n2i = i % n2;
        car[n2i * n1 + n1i] = kr[(long)b * n_k + ob + n2i + n2 * n1i];
        cai[n2i * n1 + n1i] = ki[(long)b * n_k + ob + n2i + n2 * n1i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int n2i = i / n1;
        const int k1 = i % n1;
        float ar = 0.f, ai = 0.f;
        for (int n1i = 0; n1i < n1; n1i++) {
            const float xr = car[n2i * n1 + n1i];
            const float xi = cai[n2i * n1 + n1i];
            const float wr = f1r[n1i * n1 + k1];
            const float wi = f1i[n1i * n1 + k1];
            ar = fmaf(xr, wr, fmaf(-xi, wi, ar));
            ai = fmaf(xr, wi, fmaf(xi, wr, ai));
        }
        const float tr = twr[n2i * n1 + k1];
        const float ti = twi[n2i * n1 + k1];
        sr[n2i * n1 + k1] = ar * tr - ai * ti;
        si[n2i * n1 + k1] = ar * ti + ai * tr;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int k2 = i / n1;
        const int k1 = i % n1;
        float dr = 0.f, di = 0.f;
        for (int n2i = 0; n2i < n2; n2i++) {
            const float xr = sr[n2i * n1 + k1];
            const float xi = si[n2i * n1 + k1];
            const float wr = f2r[n2i * n2 + k2];
            const float wi = f2i[n2i * n2 + k2];
            dr = fmaf(xr, wr, fmaf(-xi, wi, dr));
            di = fmaf(xr, wi, fmaf(xi, wr, di));
        }
        const int oo = k1 + n1 * k2;
        const float g = gdv[ab + oo];
        if (g != 0.f) {
            const int e = eidx[ab + oo];
            er[b * ne + e] = dr * g;
            ei[b * ne + e] = di * g;
        }
    }
}

__global__ void ell_direct_kernel(
    const float* __restrict__ kr, const float* __restrict__ ki,
    float* __restrict__ er, float* __restrict__ ei,
    const float* __restrict__ mr, const float* __restrict__ mi,
    const float* __restrict__ gdv, const int* __restrict__ eidx,
    const int* __restrict__ task, long n_k, long ne)
{
    extern __shared__ float sh[];
    const int row = blockIdx.x;
    const int b = blockIdx.y;
    const int* t = task + row * 6;
    const int pb = t[0], mb = t[1], ob = t[2];
    const int m = t[3], nb = t[4], n0 = t[5];
    const int nn = min(64, m - n0);
    float* car = sh;
    float* cai = sh + nb * m;
    for (int i = threadIdx.x; i < nb * m; i += blockDim.x) {
        const int r = i / m;
        const int c = i % m;
        car[i] = kr[(long)b * n_k + ob + r * m + c];
        cai[i] = ki[(long)b * n_k + ob + r * m + c];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nb * nn; i += blockDim.x) {
        const int r = i / nn;
        const int c = n0 + i % nn;
        float vr = 0.f, vi = 0.f;
        for (int k = 0; k < m; k++) {
            const float xr = car[r * m + k];
            const float xi = cai[r * m + k];
            const float wr = mr[mb + k * m + c];
            const float wi = mi[mb + k * m + c];
            vr = fmaf(xr, wr, fmaf(-xi, wi, vr));
            vi = fmaf(xr, wi, fmaf(xi, wr, vi));
        }
        const int e0 = pb + r * m + c;
        const float g = gdv[e0];
        if (g != 0.f) {
            const int e = eidx[e0];
            er[b * ne + e] = vr * g;
            ei[b * ne + e] = vi * g;
        }
    }
}

void ell_two(torch::Tensor kr, torch::Tensor ki,
             torch::Tensor er, torch::Tensor ei,
             torch::Tensor f1r, torch::Tensor f1i,
             torch::Tensor twr, torch::Tensor twi,
             torch::Tensor f2r, torch::Tensor f2i,
             torch::Tensor gdv, torch::Tensor eidx, torch::Tensor task,
             int64_t n_k, int64_t ne, int64_t n1, int64_t n2, int64_t batch)
{
    const int rows = task.size(0);
    const int m = (int)(n1 * n2);
    dim3 grid(rows, (unsigned)batch);
    const int shmem = 4 * m * (int)sizeof(float);
    ell_two_kernel<<<grid, 128, shmem, at::cuda::getCurrentCUDAStream()>>>(
        kr.data_ptr<float>(), ki.data_ptr<float>(),
        er.data_ptr<float>(), ei.data_ptr<float>(),
        f1r.data_ptr<float>(), f1i.data_ptr<float>(),
        twr.data_ptr<float>(), twi.data_ptr<float>(),
        f2r.data_ptr<float>(), f2i.data_ptr<float>(),
        gdv.data_ptr<float>(), eidx.data_ptr<int>(), task.data_ptr<int>(),
        n_k, ne, (int)n1, (int)n2);
}

void ell_direct(torch::Tensor kr, torch::Tensor ki,
                torch::Tensor er, torch::Tensor ei,
                torch::Tensor mr, torch::Tensor mi,
                torch::Tensor gdv, torch::Tensor eidx, torch::Tensor task,
                int64_t n_k, int64_t ne, int64_t batch)
{
    const int rows = task.size(0);
    dim3 grid(rows, (unsigned)batch);
    const int shmem = 2 * 16 * 256 * (int)sizeof(float);
    ell_direct_kernel<<<grid, 128, shmem, at::cuda::getCurrentCUDAStream()>>>(
        kr.data_ptr<float>(), ki.data_ptr<float>(),
        er.data_ptr<float>(), ei.data_ptr<float>(),
        mr.data_ptr<float>(), mi.data_ptr<float>(),
        gdv.data_ptr<float>(), eidx.data_ptr<int>(), task.data_ptr<int>(),
        n_k, ne);
}

void router_two(torch::Tensor zpr, torch::Tensor zpi,
                torch::Tensor kr, torch::Tensor ki,
                torch::Tensor p1, torch::Tensor p2,
                torch::Tensor tar, torch::Tensor tai,
                torch::Tensor tbr, torch::Tensor tbi,
                torch::Tensor f1r, torch::Tensor f1i,
                torch::Tensor twr, torch::Tensor twi,
                torch::Tensor f2r, torch::Tensor f2i, torch::Tensor task,
                int64_t n_k, int64_t n1, int64_t n2, double scale,
                int64_t batch)
{
    const int rows = task.size(0);
    const int m = (int)(n1 * n2);
    dim3 grid(rows, (unsigned)batch);
    const int shmem = 4 * m * (int)sizeof(float);
    router_two_kernel<<<grid, 128, shmem, at::cuda::getCurrentCUDAStream()>>>(
        zpr.data_ptr<float>(), zpi.data_ptr<float>(),
        kr.data_ptr<float>(), ki.data_ptr<float>(),
        p1.data_ptr<int>(), p2.data_ptr<int>(),
        tar.data_ptr<float>(), tai.data_ptr<float>(),
        tbr.data_ptr<float>(), tbi.data_ptr<float>(),
        f1r.data_ptr<float>(), f1i.data_ptr<float>(),
        twr.data_ptr<float>(), twi.data_ptr<float>(),
        f2r.data_ptr<float>(), f2i.data_ptr<float>(), task.data_ptr<int>(),
        n_k, (int)n1, (int)n2, (float)scale);
}

void router_direct(torch::Tensor zpr, torch::Tensor zpi,
                   torch::Tensor kr, torch::Tensor ki,
                   torch::Tensor p1, torch::Tensor p2,
                   torch::Tensor tar, torch::Tensor tai,
                   torch::Tensor tbr, torch::Tensor tbi,
                   torch::Tensor mr, torch::Tensor mi, torch::Tensor task,
                   int64_t n_k, int64_t batch)
{
    const int rows = task.size(0);
    dim3 grid(rows, (unsigned)batch);
    //largest direct tile is 16 bins by 256 taps, two planes
    const int shmem = 2 * 16 * 256 * (int)sizeof(float);
    router_direct_kernel<<<grid, 128, shmem, at::cuda::getCurrentCUDAStream()>>>(
        zpr.data_ptr<float>(), zpi.data_ptr<float>(),
        kr.data_ptr<float>(), ki.data_ptr<float>(),
        p1.data_ptr<int>(), p2.data_ptr<int>(),
        tar.data_ptr<float>(), tai.data_ptr<float>(),
        tbr.data_ptr<float>(), tbi.data_ptr<float>(),
        mr.data_ptr<float>(), mi.data_ptr<float>(), task.data_ptr<int>(),
        n_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("router_two", &router_two, "router plus two stage band inverse dft");
    m.def("router_direct", &router_direct, "router plus direct band inverse dft");
    m.def("ell_two", &ell_two, "two stage band inverse into the ell edges");
    m.def("ell_direct", &ell_direct, "direct band inverse into the ell edges");
}
