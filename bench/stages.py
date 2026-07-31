"""should the big fft stages move to cublas.

the stage kernels are complex matmuls against a fixed 256 by 256 dft matrix,
which is exactly what cublas is best at. this races the triton stage pair
against the same maths as torch complex matmuls (cublas underneath), forward
and inverse, fp32 exact and with tf32 allowed, before any hand written cuda
is spent on them.

run   modal run bench/cloud.py --cuda --cmd "python bench/stages.py"
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, _stage1, _stage2, _mirror, N, R, HALF

FS = 44100
SL = 65536


def snr_db(x, xh):
    d = ((x - xh).abs() ** 2).sum()
    e = (x.abs() ** 2).sum()
    return (10 * torch.log10(e / d)).item() if d > 0 else float("inf")


def med(fn, reps=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    if not torch.cuda.is_available():
        print("no cuda device found")
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")
    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=SL, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    F1 = torch.complex(f.F1R, f.F1I)
    TW = torch.complex(f.TWR, f.TWI)

    for B in (1, 32):
        x = torch.randn(B, SL, device="cuda")
        w = f._workspace(B)
        with torch.no_grad():
            #triton forward stages, the shipped pair
            c1, c2 = f.s1f["ieee"], f.s2f["ieee"]

            def tri_fwd():
                _stage1[(B, R // c1["BM"], R // c1["BN"])](
                    x, w["DR"], w["DI"], f.F1R, f.F1I, f.TWR, f.TWI,
                    w["CR"], w["CI"], IS_INV=False, PREC="ieee", **c1)
                _stage2[(B, R // c2["BM"], R // c2["BN"])](
                    w["CR"], w["CI"], f.F1R, f.F1I, w["DR"], w["DI"], w["XO"],
                    1.0 / N, IS_INV=False, PREC="ieee", **c2)

            #the same maths through cublas
            def blas_fwd():
                A = x.view(B, R, R).transpose(1, 2).to(torch.complex64)
                C = (A @ F1) * TW
                D = C.transpose(1, 2) @ F1
                return D

            tri_fwd()
            ref = torch.complex(w["DR"], w["DI"]).clone()
            D = blas_fwd().reshape(B, N)
            v = snr_db(ref, D)
            print(f"B {B:>3}  blas vs triton fwd stages  {v:8.2f} db  ->",
                  "MATCH" if v > 120 else "MISMATCH")

            for allow, lab in ((False, "fp32"), (True, "tf32")):
                torch.backends.cuda.matmul.allow_tf32 = allow
                tb = med(blas_fwd)
                print(f"B {B:>3}  fwd stages  triton {med(tri_fwd):8.3f} ms   "
                      f"blas {lab} {tb:8.3f} ms")
            torch.backends.cuda.matmul.allow_tf32 = False

            #inverse stages, mirror plus two matmuls plus real out
            i1, i2 = f.s1i["ieee"], f.s2i["ieee"]
            w["FR"].normal_()
            w["FI"].normal_()

            def tri_inv():
                _mirror[(B, N // 1024)](
                    w["FR"], w["FI"], w["DR"], w["DI"], HALF, BLOCK=1024)
                _stage1[(B, R // i1["BM"], R // i1["BN"])](
                    w["XO"], w["DR"], w["DI"], f.F1R, f.F1I, f.TWR, f.TWI,
                    w["CR"], w["CI"], IS_INV=True, PREC="ieee", **i1)
                _stage2[(B, R // i2["BM"], R // i2["BN"])](
                    w["CR"], w["CI"], f.F1R, f.F1I, w["DR"], w["DI"], w["XO"],
                    1.0 / N, IS_INV=True, PREC="ieee", **i2)

            j = torch.arange(N, device="cuda")
            sel = j <= HALF
            idx = torch.where(sel, j, N - j)
            sgn = torch.where(sel, -1.0, 1.0).float()

            def blas_inv():
                XR = w["FR"][:, idx]
                XI = w["FI"][:, idx] * sgn
                A = torch.complex(XR, XI).view(B, R, R).transpose(1, 2)
                C = (A @ F1) * TW
                D = C.transpose(1, 2) @ F1
                return D.real.transpose(1, 2).reshape(B, N) / N

            tri_inv()
            ref = w["XO"].clone()
            xo = blas_inv()
            v = snr_db(ref, xo)
            print(f"B {B:>3}  blas vs triton inv stages  {v:8.2f} db  ->",
                  "MATCH" if v > 120 else "MISMATCH")
            for allow, lab in ((False, "fp32"), (True, "tf32")):
                torch.backends.cuda.matmul.allow_tf32 = allow
                tb = med(blas_inv)
                print(f"B {B:>3}  inv stages  triton {med(tri_inv):8.3f} ms   "
                      f"blas {lab} {tb:8.3f} ms")
            torch.backends.cuda.matmul.allow_tf32 = False
            print()


if __name__ == "__main__":
    main()
