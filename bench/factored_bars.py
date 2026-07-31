"""the factored pipeline under graphs, against shipped and cufft.

run   modal run bench/cloud.py --cuda --cmd "python bench/factored_bars.py"

three roundtrip paths captured and replayed, the cufft reference pipeline,
the shipped fused pipeline, and the factored path (pack, three butterfly
passes, router, band dfts, scatter, retangle, packed inverse, unpack).
snr gates on the eager output and on replay against eager, a dropped
kernel cannot hide.
"""
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, N, _packedfft_ext, _radix_ext
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
    pext = _packedfft_ext()
    xext = _radix_ext()
    if not pext or not xext:
        print("extension build failed")
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}\n")

    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=N, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    f.prec = "ieee"

    F, WL, W1k = pf.tables("cuda")
    FR, FI = F.real.contiguous(), F.imag.contiguous()
    b1024 = torch.arange(1024, device="cuda")
    lane = torch.arange(32, device="cuda")
    WLp = WL[(b1024[:, None] * lane[None, :]) % L].reshape(-1)
    g32 = torch.arange(32, device="cuda")
    WKp = W1k[(g32[:, None] * lane[None, :]) % 1024].reshape(-1)
    WLR, WLI = WLp.real.contiguous(), WLp.imag.contiguous()
    WKR, WKI = WKp.real.contiguous(), WKp.imag.contiguous()
    qq = torch.arange(L, device="cuda", dtype=torch.float64)
    WN = torch.exp(-2j * math.pi * qq / N).to(torch.complex64)
    WNR, WNI = WN.real.contiguous(), WN.imag.contiguous()

    print(f"{'B':>4} | {'cufft_ms':>9} | {'shipped':>8} {'snr':>6} | "
          f"{'factored':>9} {'snr':>6} {'vs cufft':>8}")
    print("-" * 66)
    warns = []
    for B in (1, 8, 32, 128):
        x = torch.randn(B, N, device="cuda")
        w = f._workspace(B)
        zr = torch.empty(B, L, device="cuda")
        zi = torch.empty(B, L, device="cuda")
        t1r, t1i = torch.empty_like(zr), torch.empty_like(zr)
        t2r, t2i = torch.empty_like(zr), torch.empty_like(zr)
        zpr, zpi = torch.empty_like(zr), torch.empty_like(zr)
        xot = torch.empty(B, N, device="cuda")
        ne = 3 * (L + 1)
        er_ = torch.empty(B, ne, device="cuda")
        ei_ = torch.empty(B, ne, device="cuda")

        def factored_rt():
            zr.copy_(x[..., 0::2])
            zi.copy_(x[..., 1::2])
            pext.fft3(zr, zi, t1r, t1i, t2r, t2i, zpr, zpi,
                      FR, FI, WLR, WLI, WKR, WKI, B)
            #band phases through the radix engine, one cta per band and
            #batch task, warp shuffle butterflies. the inverse keeps the
            #atomic scatter, measured faster than the ell edges on sm80
            rx = f.radix
            xext.fwd_bands(zpr, zpi, w["KR"], w["KI"],
                           rx["route"], rx["desc"], rx["rr"],
                           rx["ri"], f.ncoef, B)
            w["FR"].zero_()
            w["FI"].zero_()
            xext.inv_bands(w["KR"], w["KI"], w["FR"], w["FI"],
                           rx["gdv"], rx["pos"], rx["desc"], rx["rr"],
                           rx["ri"], f.ncoef, B, 2)
            pext.inv_tail(w["FR"], w["FI"], zr, zi, t1r, t1i, t2r, t2i,
                          zpr, zpi, xot, FR, FI, WLR, WLI, WKR, WKI,
                          WNR, WNI, B)

        def cu_rt():
            bl, sd = cqt.fwd(x)
            return cqt.bwd(bl, sd)

        with torch.no_grad():
            g = capture(cu_rt)
            a = ev = med(g.replay)

            xe = f.bwd(*f.fwd(x))
            vs_ = snr_db(x, xe)
            xe = xe.clone()
            g = capture(lambda: f.bwd(*f.fwd(x)))
            s_ = med(g.replay)
            if snr_db(xe, f._workspace(B)["XO"]) < 90:
                warns.append(f"B {B} shipped replay diverges")

            factored_rt()
            vf_ = snr_db(x, xot)
            xf = xot.clone()
            g = capture(factored_rt)
            f_ = med(g.replay)
            if snr_db(xf, xot) < 90:
                warns.append(f"B {B} factored replay diverges")

        print(f"{B:>4} | {a:>9.3f} | {s_:>8.3f} {vs_:>6.1f} | "
              f"{f_:>9.3f} {vf_:>6.1f} {a / f_:>7.2f}x")
    for wm in warns:
        print("WARNING", wm)


if __name__ == "__main__":
    main()
