"""the factored resident pair, two cooperative launches for the roundtrip.

run   modal run bench/cloud.py --cuda --cmd "python bench/resident2.py"

gates first, the resident forward against the shipped forward and the
resident roundtrip against the input, then the race, two launches against
the graphed cufft pipeline, the graphed shipped pipeline, and the graphed
multi launch factored path.
"""
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, N, _persist2_ext
import packedfft as pf

FS = 44100
L = N // 2


def snr_db(x, xh):
    e = x.abs().double().square().sum()
    d = (x - xh).abs().double().square().sum()
    return float(10 * torch.log10(e / d)) if d > 0 else float("inf")


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
    ext = _persist2_ext()
    if not ext:
        print("persist2 extension failed to build")
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")

    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=N, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    f.prec = "ieee"

    F, WL, W1k = pf.tables("cuda")
    w32 = F[1].contiguous()
    b1024 = torch.arange(1024, device="cuda")
    lane = torch.arange(32, device="cuda")
    WLp = WL[(b1024[:, None] * lane[None, :]) % L].reshape(-1)
    g32 = torch.arange(32, device="cuda")
    WKp = W1k[(g32[:, None] * lane[None, :]) % 1024].reshape(-1)
    qq = torch.arange(L, device="cuda", dtype=torch.float64)
    WN = torch.exp(-2j * math.pi * qq / N).to(torch.complex64)

    #the band phases run the radix engine, same tables as radix.cu
    rx = f.radix

    print(f"{'B':>4} | {'cufft':>7} | {'shipped':>8} | {'factlnch':>9} | "
          f"{'resident':>9} {'snr':>6} {'vs cufft':>8}")
    print("-" * 72)
    for B in (1, 8, 32, 128):
        x = torch.randn(B, N, device="cuda")
        w = f._workspace(B)
        war = torch.empty(B, L, device="cuda")
        wai = torch.empty(B, L, device="cuda")
        wbr = torch.empty(B, L, device="cuda")
        wbi = torch.empty(B, L, device="cuda")

        def resident_rt():
            args = (x, w["KR"], w["KI"], w["FR"], w["FI"], w["XO"],
                    war, wai, wbr, wbi,
                    w32.real.contiguous(), w32.imag.contiguous(),
                    WLp.real.contiguous(), WLp.imag.contiguous(),
                    WKp.real.contiguous(), WKp.imag.contiguous(),
                    WN.real.contiguous(), WN.imag.contiguous(),
                    rx["desc"], rx["route"],
                    rx["gdv"], rx["pos"], rx["rr"], rx["ri"],
                    f.ncoef, B)
            ext.fwd2(*args)
            ext.inv2(*args)

        with torch.no_grad():
            #correctness gates
            KR0, KI0 = (u.clone() for u in f.fwd(x))
            resident_rt()
            vf = min(snr_db(KR0, w["KR"]), snr_db(KI0, w["KI"]))
            vr = snr_db(x, w["XO"])
            ok = vf > 118 and vr > 118
            print(f"gate B {B}  fwd vs shipped {vf:.2f} db  roundtrip {vr:.2f} db ->",
                  "PASS" if ok else "FAIL")
            if not ok:
                continue

            def cu_rt():
                bl, sd = cqt.fwd(x)
                return cqt.bwd(bl, sd)

            a = med(capture(cu_rt).replay)
            s_ = med(capture(lambda: f.bwd(*f.fwd(x))).replay)
            r_ = med(resident_rt)
        print(f"{B:>4} | {a:>7.3f} | {s_:>8.3f} | {'-':>9} | "
              f"{r_:>9.3f} {vr:>6.1f} {a / r_:>7.2f}x")


if __name__ == "__main__":
    main()
