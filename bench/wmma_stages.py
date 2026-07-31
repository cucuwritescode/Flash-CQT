"""gate and race the wmma stage kernels against the triton stages.

run   modal run bench/cloud.py --cuda --cmd "python bench/wmma_stages.py"

tf32x3 mode at batch, the wmma path against the triton stage path through
the full transform, coefficients and roundtrip must agree and hold above
110 db against the input before the timings mean anything.
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, _stages_ext

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
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    if not _stages_ext():
        print("stage extension failed to build")
        sys.exit(1)
    print("stage extension built\n")

    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=SL, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    f.prec = "tf32x3"

    for B in (32, 128):
        x = torch.randn(B, SL, device="cuda")
        f._workspace(B)
        with torch.no_grad():
            f.use_cuda_stages = False
            KR0, KI0 = (u.clone() for u in f.fwd(x))
            xo0 = f.bwd(KR0.clone(), KI0.clone()).clone()
            f.use_cuda_stages = True
            KR1, KI1 = f.fwd(x)
            vc = min(snr_db(KR0, KR1), snr_db(KI0, KI1))
            xo1 = f.bwd(KR1.clone(), KI1.clone())
            vx = snr_db(x, xo1)
            ok = vc > 100 and vx > 110
            print(f"B {B:>3}  coefs wmma vs triton {vc:8.2f} db   "
                  f"roundtrip vs input {vx:8.2f} db  ->",
                  "MATCH" if ok else "MISMATCH, do not trust the timings")

            def rt_triton():
                f.use_cuda_stages = False
                f.bwd(*f.fwd(x))
                f.use_cuda_stages = True

            tt = med(rt_triton)
            tc = med(lambda: f.bwd(*f.fwd(x)))
            print(f"B {B:>3}  roundtrip  triton stages {tt:8.3f} ms   "
                  f"wmma stages {tc:8.3f} ms   {tt / tc:5.2f}x\n")


if __name__ == "__main__":
    main()
