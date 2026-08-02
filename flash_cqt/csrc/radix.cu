//the radix band engine. every band transform is a mixed radix fft built
//from warp shuffle butterflies, radices 8 16 32, at most three levels,
//replacing the dense small dft loops of the router and scatter kernels.
//schedules, swizzles and twiddle exponents match the level by level
//reference in bench/radix_reference.py, gated on cpu before any cuda.
//
//forward   two tap router prologue, mixed radix inverse dft, scaled store
//          into the coefficient heap in natural order
//inverse   natural order heap load, mixed radix forward dft, dual window
//          gain and atomic scatter into the half spectrum
//
//one cta owns one band and batch task from gather to store, 128 threads,
//dynamic shared is two float planes of the largest band plus the staged
//32 point butterfly roots. the per length root tables stay in global on
//the l1 path for the few inter level twiddles, stored with the synthesis
//sign, the analysis side negates the imaginary part on load

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "radix_impl.cuh"

#define LL 32768
#define TPB 128

//mode 0 analysis, sign plus one with a final 1 over m. mode 1 synthesis,
//sign minus one, no scale, one cta per band and batch task
template<int MODE>
__global__ void radix_kernel(RP p)
{
    extern __shared__ float sh[];
    radix_band<MODE>(p, blockIdx.x, blockIdx.y, sh);
}

static RP pack(torch::Tensor desc, torch::Tensor rr, torch::Tensor ri,
               int64_t n_k)
{
    RP p{};
    p.rr = rr.data_ptr<float>();
    p.ri = ri.data_ptr<float>();
    p.desc = desc.data_ptr<int>();
    p.n_k = n_k;
    return p;
}

void fwd_bands(torch::Tensor zpr, torch::Tensor zpi,
               torch::Tensor kr, torch::Tensor ki,
               torch::Tensor route,
               torch::Tensor desc, torch::Tensor rr, torch::Tensor ri,
               int64_t n_k, int64_t batch)
{
    RP p = pack(desc, rr, ri, n_k);
    p.zr = zpr.data_ptr<float>();
    p.zi = zpi.data_ptr<float>();
    p.kr = kr.data_ptr<float>();
    p.ki = ki.data_ptr<float>();
    p.route = (const uint4*)route.data_ptr<int>();
    dim3 grid(desc.size(0), (unsigned)batch);
    const int shmem = (2 * 2048 + 64) * (int)sizeof(float);
    radix_kernel<0><<<grid, TPB, shmem, at::cuda::getCurrentCUDAStream()>>>(p);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fwd_bands launch failed");
}

void inv_bands(torch::Tensor kr, torch::Tensor ki,
               torch::Tensor fr, torch::Tensor fi,
               torch::Tensor gdv, torch::Tensor pos,
               torch::Tensor desc, torch::Tensor rr, torch::Tensor ri,
               int64_t n_k, int64_t batch, int64_t variant)
{
    RP p = pack(desc, rr, ri, n_k);
    p.kr = kr.data_ptr<float>();
    p.ki = ki.data_ptr<float>();
    p.fr = fr.data_ptr<float>();
    p.fi = fi.data_ptr<float>();
    p.gdv = gdv.data_ptr<float>();
    p.pos = pos.data_ptr<int>();
    dim3 grid(desc.size(0), (unsigned)batch);
    if (variant == 1) {
        //second complex buffer past the roots, higher shared, no stack
        const int shmem = (4 * 2048 + 64) * (int)sizeof(float);
        radix_kernel<1><<<grid, TPB, shmem,
                          at::cuda::getCurrentCUDAStream()>>>(p);
    } else {
        //single buffer with a register pull, the operand arrays sit on
        //the stack at higher occupancy, measured fastest at batch
        const int shmem = (2 * 2048 + 64) * (int)sizeof(float);
        radix_kernel<2><<<grid, TPB, shmem,
                          at::cuda::getCurrentCUDAStream()>>>(p);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "inv_bands launch failed");
}

void adj_bands(torch::Tensor kr, torch::Tensor ki,
               torch::Tensor fr, torch::Tensor fi,
               torch::Tensor agr, torch::Tensor agi, torch::Tensor aidx,
               torch::Tensor desc, torch::Tensor rr, torch::Tensor ri,
               int64_t n_k, int64_t batch)
{
    //adjoint of the analysis, the band dft of the incoming coefficient
    //gradients scattered at the analysis gather indices, two gain
    //planes carry the conj sign, constants baked at table build
    RP p = pack(desc, rr, ri, n_k);
    p.kr = kr.data_ptr<float>();
    p.ki = ki.data_ptr<float>();
    p.fr = fr.data_ptr<float>();
    p.fi = fi.data_ptr<float>();
    p.gdv = agr.data_ptr<float>();
    p.gdi = agi.data_ptr<float>();
    p.pos = aidx.data_ptr<int>();
    dim3 grid(desc.size(0), (unsigned)batch);
    const int shmem = (2 * 2048 + 64) * (int)sizeof(float);
    radix_kernel<3><<<grid, TPB, shmem, at::cuda::getCurrentCUDAStream()>>>(p);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "adj_bands launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fwd_bands", &fwd_bands, "router prologue plus mixed radix band idft");
    m.def("inv_bands", &inv_bands, "mixed radix band dft plus atomic scatter");
    m.def("adj_bands", &adj_bands, "analysis adjoint, band dft plus signed scatter");
}
