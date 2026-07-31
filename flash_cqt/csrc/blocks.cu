//inverse two stage block kernel in raw cuda. the triton compiler past 3.1
//generates code 2 to 6x slower than its own 3.1 output for this kernel, so
//it moves off triton entirely. one thread block per (band, batch element),
//coefficients and the stage one intermediate live in shared memory, the
//dft and twiddle tables stream from global (they are shared by every block
//and sit in l2). plain fp32 cores in this first version, tensor cores can
//come later if this is not already fast enough.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void inv_two_kernel(
    const float* __restrict__ kr, const float* __restrict__ ki,
    float* frre, float* frim,
    const float* __restrict__ f1r, const float* __restrict__ f1i,
    const float* __restrict__ twr, const float* __restrict__ twi,
    const float* __restrict__ f2r, const float* __restrict__ f2i,
    const float* __restrict__ gdv, const int* __restrict__ pos,
    const int* __restrict__ task,
    int n_fr, long n_k, int n1, int n2)
{
    //shared layout, c then the stage one result, both as split planes
    extern __shared__ float sh[];
    const int m = n1 * n2;
    float* cr = sh;
    float* ci = sh + m;
    float* sr = sh + 2 * m;
    float* si = sh + 3 * m;

    const int band = blockIdx.x;
    const int b = blockIdx.y;
    const int ab = task[band * 2 + 0];
    const int ob = task[band * 2 + 1];

    //load the band's coefficients, element (n2i, n1i) sits at ob + n2i + n2*n1i
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int n1i = i / n2;
        const int n2i = i % n2;
        cr[n2i * n1 + n1i] = kr[(long)b * n_k + ob + n2i + n2 * n1i];
        ci[n2i * n1 + n1i] = ki[(long)b * n_k + ob + n2i + n2 * n1i];
    }
    __syncthreads();

    //stage one, s[n2i, k1] = (sum over n1i of c F1) times twiddle
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int n2i = i / n1;
        const int k1 = i % n1;
        float ar = 0.f, ai = 0.f;
        for (int n1i = 0; n1i < n1; n1i++) {
            const float xr = cr[n2i * n1 + n1i];
            const float xi = ci[n2i * n1 + n1i];
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

    //stage two plus the scatter, d[k1, k2] = sum over n2i of s F2,
    //output index k1 + n1*k2, dual window then atomic add into the half
    //spectrum, lanes with zero window skipped (they carry pos 0)
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
            const int p = pos[ab + oo];
            atomicAdd(frre + (long)b * n_fr + p, dr * g);
            atomicAdd(frim + (long)b * n_fr + p, di * g);
        }
    }
}

//forward twin, gather from the matrix layout spectrum with the analysis
//window (the conjugate for the side blocks is a sign baked into the
//imaginary window plane), inverse dft through the conjugation trick, store
//the coefficient block. same shared staging as the inverse
__global__ void fwd_two_kernel(
    const float* __restrict__ dre, const float* __restrict__ dim,
    float* kr, float* ki,
    const int* __restrict__ pidx,
    const float* __restrict__ wre, const float* __restrict__ wim,
    const float* __restrict__ f1r, const float* __restrict__ f1i,
    const float* __restrict__ twr, const float* __restrict__ twi,
    const float* __restrict__ f2r, const float* __restrict__ f2i,
    const int* __restrict__ task,
    long n_d, long n_k, int n1, int n2, float scale)
{
    extern __shared__ float sh[];
    const int m = n1 * n2;
    float* ar_ = sh;
    float* ai_ = sh + m;
    float* sr = sh + 2 * m;
    float* si = sh + 3 * m;

    const int band = blockIdx.x;
    const int b = blockIdx.y;
    const int ab = task[band * 2 + 0];
    const int ob = task[band * 2 + 1];

    //gather and window, conj in for the ifft trick
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int n1i = i / n2;
        const int n2i = i % n2;
        const int j = ab + n2i + n2 * n1i;
        const int p = pidx[j];
        ar_[n2i * n1 + n1i] = dre[(long)b * n_d + p] * wre[j];
        ai_[n2i * n1 + n1i] = -(dim[(long)b * n_d + p] * wim[j]);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        const int n2i = i / n1;
        const int k1 = i % n1;
        float ar = 0.f, ai = 0.f;
        for (int n1i = 0; n1i < n1; n1i++) {
            const float xr = ar_[n2i * n1 + n1i];
            const float xi = ai_[n2i * n1 + n1i];
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

    //stage two, conj out and scale, store the block
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

void fwd_two(torch::Tensor dre, torch::Tensor dim,
             torch::Tensor kr, torch::Tensor ki,
             torch::Tensor pidx, torch::Tensor wre, torch::Tensor wim,
             torch::Tensor f1r, torch::Tensor f1i,
             torch::Tensor twr, torch::Tensor twi,
             torch::Tensor f2r, torch::Tensor f2i, torch::Tensor task,
             int64_t n_d, int64_t n_k, int64_t n1, int64_t n2,
             double scale, int64_t batch)
{
    const int rows = task.size(0);
    const int m = (int)(n1 * n2);
    dim3 grid(rows, (unsigned)batch);
    const int shmem = 4 * m * (int)sizeof(float);
    fwd_two_kernel<<<grid, 128, shmem, at::cuda::getCurrentCUDAStream()>>>(
        dre.data_ptr<float>(), dim.data_ptr<float>(),
        kr.data_ptr<float>(), ki.data_ptr<float>(),
        pidx.data_ptr<int>(), wre.data_ptr<float>(), wim.data_ptr<float>(),
        f1r.data_ptr<float>(), f1i.data_ptr<float>(),
        twr.data_ptr<float>(), twi.data_ptr<float>(),
        f2r.data_ptr<float>(), f2i.data_ptr<float>(), task.data_ptr<int>(),
        n_d, n_k, (int)n1, (int)n2, (float)scale);
}

void inv_two(torch::Tensor kr, torch::Tensor ki,
             torch::Tensor frre, torch::Tensor frim,
             torch::Tensor f1r, torch::Tensor f1i,
             torch::Tensor twr, torch::Tensor twi,
             torch::Tensor f2r, torch::Tensor f2i,
             torch::Tensor gdv, torch::Tensor pos, torch::Tensor task,
             int64_t n_fr, int64_t n_k, int64_t n1, int64_t n2, int64_t batch)
{
    const int rows = task.size(0);
    const int m = (int)(n1 * n2);
    dim3 grid(rows, (unsigned)batch);
    const int shmem = 4 * m * (int)sizeof(float);
    inv_two_kernel<<<grid, 128, shmem, at::cuda::getCurrentCUDAStream()>>>(
        kr.data_ptr<float>(), ki.data_ptr<float>(),
        frre.data_ptr<float>(), frim.data_ptr<float>(),
        f1r.data_ptr<float>(), f1i.data_ptr<float>(),
        twr.data_ptr<float>(), twi.data_ptr<float>(),
        f2r.data_ptr<float>(), f2i.data_ptr<float>(),
        gdv.data_ptr<float>(), pos.data_ptr<int>(), task.data_ptr<int>(),
        (int)n_fr, n_k, (int)n1, (int)n2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("inv_two", &inv_two, "inverse two stage block transform");
    m.def("fwd_two", &fwd_two, "forward two stage block transform");
}
