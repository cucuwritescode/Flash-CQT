"""gate and race the resident kernels against the shipped graphed pipeline.

run   modal run bench/cloud.py --cuda --cmd "python bench/persist.py"

the resident pair replaces about 13 graphed kernels with 2 cooperative
launches. correctness first, forward coefficients against the shipped
forward and the roundtrip against the input must both hold above 120 db,
then the race, 2 eager launches against the replayed graph.
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, _persist_ext

FS = 44100
SL = 65536


def snr_db(x, xh):
    d = ((x - xh).abs() ** 2).sum()
    e = (x.abs() ** 2).sum()
    return (10 * torch.log10(e / d)).item() if d > 0 else float("inf")


def med(fn, reps=50):
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


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def main():
    if not torch.cuda.is_available():
        print("no cuda device found")
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    if not _persist_ext():
        print("persist extension failed to build")
        sys.exit(1)
    print("persist extension built\n")

    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=SL, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    f.prec = "ieee"

    for B in (1, 8, 32):
        x = torch.randn(B, SL, device="cuda")
        f._workspace(B)
        with torch.no_grad():
            KR0, KI0 = (u.clone() for u in f.fwd(x))
            KR1, KI1 = f.persist_fwd(x)
            vf = min(snr_db(KR0, KR1), snr_db(KI0, KI1))
            xo = f.persist_inv(KR1.clone(), KI1.clone())
            vr = snr_db(x, xo)
            ok = vf > 120 and vr > 120
            print(f"B {B:>3}  fwd coefs vs shipped {vf:8.2f} db   "
                  f"roundtrip vs input {vr:8.2f} db  ->",
                  "MATCH" if ok else "MISMATCH, do not trust the timings")

            g = capture(lambda: f.bwd(*f.fwd(x)))
            tg = med(g.replay)
            tp = med(lambda: f.persist_inv(*f.persist_fwd(x)))
            print(f"B {B:>3}  roundtrip  shipped graph {tg:8.3f} ms   "
                  f"resident pair {tp:8.3f} ms   {tg / tp:5.2f}x\n")


if __name__ == "__main__":
    main()
