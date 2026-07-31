"""prove the raw cuda build loop works end to end on the rented gpu.

compiles a trivial kernel through torch's inline extension machinery and
runs it. if this prints ok, the flagship cuda kernels can be developed in
the same loop, nvcc, headers, ninja and the runtime all line up.

run   modal run bench/cloud.py --cuda --cmd "python bench/toolchain.py"
"""
import sys
import subprocess
import torch


def main():
    r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        print("nvcc missing, this image cannot build cuda extensions")
        sys.exit(1)
    print(r.stdout.strip().splitlines()[-1])
    if not torch.cuda.is_available():
        print("no cuda device")
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    from torch.utils.cpp_extension import load_inline
    src = r"""
    __global__ void axpy_kernel(const float* x, float* y, float a, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a * x[i] + y[i];
    }
    torch::Tensor axpy(torch::Tensor x, torch::Tensor y, double a) {
        int n = x.numel();
        axpy_kernel<<<(n + 255) / 256, 256>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), (float)a, n);
        return y;
    }
    """
    mod = load_inline(name="axpy_probe", cpp_sources="",
                      cuda_sources=src, functions=["axpy"], verbose=True)
    x = torch.randn(1 << 20, device="cuda")
    y = torch.randn(1 << 20, device="cuda")
    ref = 2.5 * x + y
    out = mod.axpy(x, y.clone(), 2.5)
    err = (out - ref).abs().max().item()
    print(f"axpy max err {err:.2e}  ->", "ok, the cuda loop is live" if err == 0 else "CHECK")


if __name__ == "__main__":
    main()
