"""gate and race the raw cuda inverse block kernel against the triton one.

run   modal run bench/cloud.py --cuda --cmd "python bench/cuda_blocks.py"

builds flash_cqt/csrc/blocks.cu on the remote gpu, checks its scatter output
against the triton path bucket by bucket (they must agree to fp32 atomics
order, snr above 120 db), then times both at B 1 and B 32. the triton
numbers here are whatever the installed triton produces, the cuda kernel is
the same source everywhere, that is the point of it.
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, HALF

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


def build_ext():
    from torch.utils.cpp_extension import load
    src = pathlib.Path(__file__).resolve().parents[1] / "flash_cqt" / "csrc" / "blocks.cu"
    return load(name="flash_cqt_blocks", sources=[str(src)], verbose=False)


def main():
    if not torch.cuda.is_available():
        print("no cuda device found")
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    ext = build_ext()
    print("extension built\n")

    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=SL, device="cuda")
    f = FusedOctCQT(cqt, "cuda")

    for B in (1, 32):
        x = torch.randn(B, SL, device="cuda")
        w = f._workspace(B)
        with torch.no_grad():
            #forward gate, cuda fwd blocks against triton fwd blocks
            f.use_cuda_blocks = False
            KRt, KIt = (u.clone() for u in f.fwd(x))
            f.use_cuda_blocks = True
            KR, KI = f.fwd(x)
            v = min(snr_db(KRt, KR), snr_db(KIt, KI))
            print(f"B {B:>3}  cuda vs triton forward  {v:8.2f} db  ->",
                  "MATCH" if v > 120 else "MISMATCH, do not trust the timings")

            def tri_fwd():
                f.use_cuda_blocks = False
                f.fwd(x)
                f.use_cuda_blocks = True

            tf = med(tri_fwd)
            cf = med(lambda: f.fwd(x))
            print(f"B {B:>3}  full forward  triton {tf:8.3f} ms   "
                  f"cuda blocks {cf:8.3f} ms   {tf / cf:5.2f}x")

            def triton_all():
                f.use_cuda_blocks = False
                w["FR"].zero_()
                w["FI"].zero_()
                for t in f.two:
                    f._inv_two(t, B, w, KR, KI, "ieee")
                f.use_cuda_blocks = True

            def cuda_all():
                f.use_cuda_blocks = True
                w["FR"].zero_()
                w["FI"].zero_()
                for t in f.two:
                    f._inv_two(t, B, w, KR, KI, "ieee")

            triton_all()
            rfr, rfi = w["FR"].clone(), w["FI"].clone()
            cuda_all()
            v = min(snr_db(rfr, w["FR"]), snr_db(rfi, w["FI"]))
            ok = v > 120
            print(f"B {B:>3}  cuda vs triton scatter  {v:8.2f} db  ->",
                  "MATCH" if ok else "MISMATCH, do not trust the timings")

            tt = med(triton_all)
            tc = med(cuda_all)
            print(f"B {B:>3}  all inverse buckets  triton {tt:8.3f} ms   "
                  f"cuda {tc:8.3f} ms   {tt / tc:5.2f}x\n")


if __name__ == "__main__":
    main()
