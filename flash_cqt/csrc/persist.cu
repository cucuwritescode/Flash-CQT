//the two fat kernels. one resident forward, one resident inverse, the
//whole per slice transform inside a single cooperative launch with
//grid wide syncs where the kernel boundaries used to be. plain exact fp32
//in this first version, the win at small batch is erasing the boundaries,
//not the arithmetic. block count is sized to what fits co resident on the
//card, work is taken grid stride from per step task lists.
//
//forward   stage one tiles, sync, stage two tiles, sync, blocks
//inverse   zero the half spectrum, sync, blocks scatter, sync, mirror,
//          sync, stage one tiles, sync, stage two tiles

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;

#define NN 65536
#define RR 256
#define HALFN 32768
#define TPB 128
#define SH_FLOATS 8192  //32 kb dynamic shared, the largest any step needs

struct Params {
    //signal planes
    const float* x;        //forward input [B, 65536]
    float* cr; float* ci;  //stage intermediate planes, transposed layout
    float* dr; float* di;  //spectrum matrix layout planes
    float* kr; float* ki;  //coefficient planes
    float* frre; float* frim;  //half spectrum planes [B, 32769]
    float* xo;             //inverse output [B, 65536]
    //big stage tables
    const float* f1r; const float* f1i;
    const float* twr; const float* twi;
    //two stage block family, packed descriptors and one table heap
    const int* two_desc;   //[rows2, 8] n1 n2 ab ob f1o two f2o pad
    const float* two_tabs; //concatenated per bucket tables
    const int* two_pidx;
    const float* two_wre; const float* two_wim;
    const float* two_gdv; const int* two_pos;
    //direct block family, the flat tables the python side already holds
    const int* d_task;     //[rowsd, 6] pb mb ob m nb n0
    const int* d_pidx;
    const float* d_wre; const float* d_wim;
    const float* d_mr; const float* d_mi;  //fwd inverse dft, inv forward dft
    const float* d_gdv; const int* d_pos;
    long n_k;
    int rows2, rowsd, batch;
    float inv_scale;       //two stage forward output scale is per bucket,
                           //this is the big inverse fft 1/N
};

//one 64 by 64 tile of a 256 point stage matmul, plain fp32, a chunk of the
//a operand staged through shared, tables streamed from l2
__device__ void stage_tile(const Params& p, int task, bool s1, bool inv,
                           float* sh)
{
    const int b = task / 16;
    const int t16 = task % 16;
    const int row0 = (t16 / 4) * 64;
    const int col0 = (t16 % 4) * 64;
    const long abase = (long)b * NN;
    float* ar = sh;               //[64][16]
    float* ai = sh + 1024;
    float* fr = sh + 2048;        //[16][64] table chunk, reused by all rows
    float* fi = sh + 3072;
    float accr[32], acci[32];
    for (int u = 0; u < 32; u++) { accr[u] = 0.f; acci[u] = 0.f; }
    //thread u owns outputs (r = 2*(tid/64) + 4*(u/16)? keep it simple,
    //thread tid owns rows tid/2 stepped by 64, cols (tid%2)*32 + 0..31
    const int tr0 = threadIdx.x / 2;          //0..63 row within tile
    const int tc0 = (threadIdx.x % 2) * 32;   //column half
    for (int k0 = 0; k0 < 256; k0 += 16) {
        for (int i = threadIdx.x; i < 1024; i += TPB) {
            const int r = i / 16;
            const int c = i % 16;
            long src;
            if (s1)  //sample layout, a[n2][n1] at 256 n1 + n2
                src = abase + 256L * (k0 + c) + (row0 + r);
            else     //transposed planes from stage one, row major
                src = abase + 256L * (row0 + r) + (k0 + c);
            if (s1 && !inv) {
                ar[i] = p.x[src];
                ai[i] = 0.f;
            } else if (s1) {
                //mirrored conjugated planes prepared by the mirror step
                ar[i] = p.dr[src];
                ai[i] = p.di[src];
            } else {
                ar[i] = p.cr[src];
                ai[i] = p.ci[src];
            }
        }
        //stage the table chunk as well, v1 streamed it from l2 thousands
        //of times per thread and paid 3x for it, the flashfftconv lesson
        for (int i = threadIdx.x; i < 1024; i += TPB) {
            const int kk = i / 64;
            const int cc = i % 64;
            fr[i] = p.f1r[(k0 + kk) * 256 + col0 + cc];
            fi[i] = p.f1i[(k0 + kk) * 256 + col0 + cc];
        }
        __syncthreads();
        for (int kk = 0; kk < 16; kk++) {
            const float xr = ar[tr0 * 16 + kk];
            const float xi = ai[tr0 * 16 + kk];
            for (int u = 0; u < 32; u++) {
                const float wr = fr[kk * 64 + tc0 + u];
                const float wi = fi[kk * 64 + tc0 + u];
                accr[u] = fmaf(xr, wr, fmaf(-xi, wi, accr[u]));
                acci[u] = fmaf(xr, wi, fmaf(xi, wr, acci[u]));
            }
        }
        __syncthreads();
    }
    //epilogue
    const int gr = row0 + tr0;
    for (int u = 0; u < 32; u++) {
        const int gc = col0 + tc0 + u;
        float vr = accr[u];
        float vi = acci[u];
        if (s1) {
            const float tr_ = p.twr[gr * 256 + gc];
            const float ti_ = p.twi[gr * 256 + gc];
            const float wr = vr * tr_ - vi * ti_;
            vi = vr * ti_ + vi * tr_;
            vr = wr;
            //transposed store for stage two
            p.cr[abase + 256L * gc + gr] = vr;
            p.ci[abase + 256L * gc + gr] = vi;
        } else if (!inv) {
            p.dr[abase + 256L * gr + gc] = vr;
            p.di[abase + 256L * gr + gc] = vi;
        } else {
            //real signal out, sample gr + 256 gc
            p.xo[abase + gr + 256L * gc] = vr * p.inv_scale;
        }
    }
}

//one two stage band, the proven block kernel body as a device function
__device__ void two_band(const Params& p, int task, bool fwd, float* sh)
{
    const int band = task % p.rows2;
    const int b = task / p.rows2;
    const int* d = p.two_desc + band * 8;
    const int n1 = d[0], n2 = d[1], ab = d[2], ob = d[3];
    const float* f1 = p.two_tabs + d[4];
    const float* tw = p.two_tabs + d[5];
    const float* f2 = p.two_tabs + d[6];
    const int m = n1 * n2;
    float* car = sh;
    float* cai = sh + m;
    float* sr = sh + 2 * m;
    float* si = sh + 3 * m;
    //tables hold F1R then F1I (each n1 n1), TWR TWI (n2 n1), F2R F2I (n2 n2)
    const float* f1r = f1; const float* f1i = f1 + n1 * n1;
    const float* twr = tw; const float* twi = tw + n2 * n1;
    const float* f2r = f2; const float* f2i = f2 + n2 * n2;

    for (int i = threadIdx.x; i < m; i += TPB) {
        const int n1i = i / n2;
        const int n2i = i % n2;
        const int j = ab + n2i + n2 * n1i;
        if (fwd) {
            const int pp = p.two_pidx[j];
            car[n2i * n1 + n1i] = p.dr[(long)b * NN + pp] * p.two_wre[j];
            cai[n2i * n1 + n1i] = -(p.di[(long)b * NN + pp] * p.two_wim[j]);
        } else {
            car[n2i * n1 + n1i] = p.kr[(long)b * p.n_k + ob + n2i + n2 * n1i];
            cai[n2i * n1 + n1i] = p.ki[(long)b * p.n_k + ob + n2i + n2 * n1i];
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < m; i += TPB) {
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
    const float bscale = 1.f / (float)m;
    for (int i = threadIdx.x; i < m; i += TPB) {
        const int k2 = i / n1;
        const int k1 = i % n1;
        float dr_ = 0.f, di_ = 0.f;
        for (int n2i = 0; n2i < n2; n2i++) {
            const float xr = sr[n2i * n1 + k1];
            const float xi = si[n2i * n1 + k1];
            const float wr = f2r[n2i * n2 + k2];
            const float wi = f2i[n2i * n2 + k2];
            dr_ = fmaf(xr, wr, fmaf(-xi, wi, dr_));
            di_ = fmaf(xr, wi, fmaf(xi, wr, di_));
        }
        const int oo = k1 + n1 * k2;
        if (fwd) {
            p.kr[(long)b * p.n_k + ob + oo] = dr_ * bscale;
            p.ki[(long)b * p.n_k + ob + oo] = -di_ * bscale;
        } else {
            const float g = p.two_gdv[ab + oo];
            if (g != 0.f) {
                const int pp = p.two_pos[ab + oo];
                atomicAdd(p.frre + (long)b * (HALFN + 1) + pp, dr_ * g);
                atomicAdd(p.frim + (long)b * (HALFN + 1) + pp, di_ * g);
            }
        }
    }
    __syncthreads();
}

//one direct family row, gather, one dft matrix, store or scatter
__device__ void direct_row(const Params& p, int task, bool fwd, float* sh)
{
    const int row = task % p.rowsd;
    const int b = task / p.rowsd;
    const int* t = p.d_task + row * 6;
    const int pb = t[0], mb = t[1], ob = t[2];
    const int m = t[3], nb = t[4], n0 = t[5];
    const int nn = min(64, m - n0);
    float* car = sh;          //[nb][m] gathered or loaded operand
    float* cai = sh + nb * m;
    for (int i = threadIdx.x; i < nb * m; i += TPB) {
        const int r = i / m;
        const int c = i % m;
        if (fwd) {
            const int j = pb + r * m + c;
            const int pp = p.d_pidx[j];
            car[i] = p.dr[(long)b * NN + pp] * p.d_wre[j];
            cai[i] = p.di[(long)b * NN + pp] * p.d_wim[j];
        } else {
            car[i] = p.kr[(long)b * p.n_k + ob + r * m + c];
            cai[i] = p.ki[(long)b * p.n_k + ob + r * m + c];
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nb * nn; i += TPB) {
        const int r = i / nn;
        const int c = n0 + i % nn;
        float vr = 0.f, vi = 0.f;
        for (int k = 0; k < m; k++) {
            const float xr = car[r * m + k];
            const float xi = cai[r * m + k];
            const float wr = p.d_mr[mb + k * m + c];
            const float wi = p.d_mi[mb + k * m + c];
            vr = fmaf(xr, wr, fmaf(-xi, wi, vr));
            vi = fmaf(xr, wi, fmaf(xi, wr, vi));
        }
        const int e = pb + r * m + c;
        if (fwd) {
            p.kr[(long)b * p.n_k + ob + r * m + c] = vr;
            p.ki[(long)b * p.n_k + ob + r * m + c] = vi;
        } else {
            const float g = p.d_gdv[e];
            if (g != 0.f) {
                const int pp = p.d_pos[e];
                atomicAdd(p.frre + (long)b * (HALFN + 1) + pp, vr * g);
                atomicAdd(p.frim + (long)b * (HALFN + 1) + pp, vi * g);
            }
        }
    }
    __syncthreads();
}

__global__ void fwd_kernel(Params p)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sh[];
    const int nstage = 16 * p.batch;
    for (int t = blockIdx.x; t < nstage; t += gridDim.x)
        stage_tile(p, t, true, false, sh);
    grid.sync();
    for (int t = blockIdx.x; t < nstage; t += gridDim.x)
        stage_tile(p, t, false, false, sh);
    grid.sync();
    const int n2t = p.rows2 * p.batch;
    const int ndt = p.rowsd * p.batch;
    for (int t = blockIdx.x; t < n2t + ndt; t += gridDim.x) {
        if (t < n2t) two_band(p, t, true, sh);
        else direct_row(p, t - n2t, true, sh);
    }
}

__global__ void inv_kernel(Params p)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float sh[];
    //zero the half spectrum accumulators
    const long nfr = (long)p.batch * (HALFN + 1);
    for (long i = (long)blockIdx.x * TPB + threadIdx.x; i < nfr;
         i += (long)gridDim.x * TPB) {
        p.frre[i] = 0.f;
        p.frim[i] = 0.f;
    }
    grid.sync();
    const int n2t = p.rows2 * p.batch;
    const int ndt = p.rowsd * p.batch;
    for (int t = blockIdx.x; t < n2t + ndt; t += gridDim.x) {
        if (t < n2t) two_band(p, t, false, sh);
        else direct_row(p, t - n2t, false, sh);
    }
    grid.sync();
    //mirror into the full conjugated planes
    const long nfull = (long)p.batch * NN;
    for (long i = (long)blockIdx.x * TPB + threadIdx.x; i < nfull;
         i += (long)gridDim.x * TPB) {
        const long b = i / NN;
        const long j = i % NN;
        const long idx = j <= HALFN ? j : NN - j;
        const float s = j <= HALFN ? -1.f : 1.f;
        p.dr[i] = p.frre[b * (HALFN + 1) + idx];
        p.di[i] = p.frim[b * (HALFN + 1) + idx] * s;
    }
    grid.sync();
    const int nstage = 16 * p.batch;
    for (int t = blockIdx.x; t < nstage; t += gridDim.x)
        stage_tile(p, t, true, true, sh);
    grid.sync();
    for (int t = blockIdx.x; t < nstage; t += gridDim.x)
        stage_tile(p, t, false, true, sh);
}

static void launch(const void* kp, Params& p, cudaStream_t stream)
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

static Params fill(torch::Tensor x, torch::Tensor cr, torch::Tensor ci,
                   torch::Tensor dr, torch::Tensor di,
                   torch::Tensor kr, torch::Tensor ki,
                   torch::Tensor frre, torch::Tensor frim, torch::Tensor xo,
                   torch::Tensor f1r, torch::Tensor f1i,
                   torch::Tensor twr, torch::Tensor twi,
                   torch::Tensor two_desc, torch::Tensor two_tabs,
                   torch::Tensor two_pidx, torch::Tensor two_wre,
                   torch::Tensor two_wim, torch::Tensor two_gdv,
                   torch::Tensor two_pos,
                   torch::Tensor d_task, torch::Tensor d_pidx,
                   torch::Tensor d_wre, torch::Tensor d_wim,
                   torch::Tensor d_mr, torch::Tensor d_mi,
                   torch::Tensor d_gdv, torch::Tensor d_pos,
                   int64_t n_k, int64_t batch)
{
    Params p;
    p.x = x.data_ptr<float>();
    p.cr = cr.data_ptr<float>(); p.ci = ci.data_ptr<float>();
    p.dr = dr.data_ptr<float>(); p.di = di.data_ptr<float>();
    p.kr = kr.data_ptr<float>(); p.ki = ki.data_ptr<float>();
    p.frre = frre.data_ptr<float>(); p.frim = frim.data_ptr<float>();
    p.xo = xo.data_ptr<float>();
    p.f1r = f1r.data_ptr<float>(); p.f1i = f1i.data_ptr<float>();
    p.twr = twr.data_ptr<float>(); p.twi = twi.data_ptr<float>();
    p.two_desc = two_desc.data_ptr<int>();
    p.two_tabs = two_tabs.data_ptr<float>();
    p.two_pidx = two_pidx.data_ptr<int>();
    p.two_wre = two_wre.data_ptr<float>();
    p.two_wim = two_wim.data_ptr<float>();
    p.two_gdv = two_gdv.data_ptr<float>();
    p.two_pos = two_pos.data_ptr<int>();
    p.d_task = d_task.data_ptr<int>();
    p.d_pidx = d_pidx.data_ptr<int>();
    p.d_wre = d_wre.data_ptr<float>();
    p.d_wim = d_wim.data_ptr<float>();
    p.d_mr = d_mr.data_ptr<float>();
    p.d_mi = d_mi.data_ptr<float>();
    p.d_gdv = d_gdv.data_ptr<float>();
    p.d_pos = d_pos.data_ptr<int>();
    p.n_k = n_k;
    p.rows2 = two_desc.size(0);
    p.rowsd = d_task.size(0);
    p.batch = (int)batch;
    p.inv_scale = 1.f / (float)NN;
    return p;
}

void persist_fwd(torch::Tensor x, torch::Tensor cr, torch::Tensor ci,
                 torch::Tensor dr, torch::Tensor di,
                 torch::Tensor kr, torch::Tensor ki,
                 torch::Tensor frre, torch::Tensor frim, torch::Tensor xo,
                 torch::Tensor f1r, torch::Tensor f1i,
                 torch::Tensor twr, torch::Tensor twi,
                 torch::Tensor two_desc, torch::Tensor two_tabs,
                 torch::Tensor two_pidx, torch::Tensor two_wre,
                 torch::Tensor two_wim, torch::Tensor two_gdv,
                 torch::Tensor two_pos,
                 torch::Tensor d_task, torch::Tensor d_pidx,
                 torch::Tensor d_wre, torch::Tensor d_wim,
                 torch::Tensor d_mr, torch::Tensor d_mi,
                 torch::Tensor d_gdv, torch::Tensor d_pos,
                 int64_t n_k, int64_t batch)
{
    Params p = fill(x, cr, ci, dr, di, kr, ki, frre, frim, xo, f1r, f1i,
                    twr, twi, two_desc, two_tabs, two_pidx, two_wre, two_wim,
                    two_gdv, two_pos, d_task, d_pidx, d_wre, d_wim,
                    d_mr, d_mi, d_gdv, d_pos, n_k, batch);
    launch((const void*)fwd_kernel, p, at::cuda::getCurrentCUDAStream());
}

void persist_inv(torch::Tensor x, torch::Tensor cr, torch::Tensor ci,
                 torch::Tensor dr, torch::Tensor di,
                 torch::Tensor kr, torch::Tensor ki,
                 torch::Tensor frre, torch::Tensor frim, torch::Tensor xo,
                 torch::Tensor f1r, torch::Tensor f1i,
                 torch::Tensor twr, torch::Tensor twi,
                 torch::Tensor two_desc, torch::Tensor two_tabs,
                 torch::Tensor two_pidx, torch::Tensor two_wre,
                 torch::Tensor two_wim, torch::Tensor two_gdv,
                 torch::Tensor two_pos,
                 torch::Tensor d_task, torch::Tensor d_pidx,
                 torch::Tensor d_wre, torch::Tensor d_wim,
                 torch::Tensor d_mr, torch::Tensor d_mi,
                 torch::Tensor d_gdv, torch::Tensor d_pos,
                 int64_t n_k, int64_t batch)
{
    Params p = fill(x, cr, ci, dr, di, kr, ki, frre, frim, xo, f1r, f1i,
                    twr, twi, two_desc, two_tabs, two_pidx, two_wre, two_wim,
                    two_gdv, two_pos, d_task, d_pidx, d_wre, d_wim,
                    d_mr, d_mi, d_gdv, d_pos, n_k, batch);
    launch((const void*)inv_kernel, p, at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("persist_fwd", &persist_fwd, "resident forward transform");
    m.def("persist_inv", &persist_inv, "resident inverse transform");
}
