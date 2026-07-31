"""per phase graphed timings of the factored path at batch.

run   modal run bench/cloud.py --cuda --cmd "python bench/factored_parts.py"

four phases timed as captured graphs, the packed fft, the router bands,
the scatter, and the inverse tail, so the batch gap to cufft names its
owner before anything gets optimised.
"""
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, N, _packedfft_ext, _router_ext
import packedfft as pf

FS = 44100
L = N // 2


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
    pext = _packedfft_ext()
    rext = _router_ext()
    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=N, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    f.prec = "ieee"
    print(f"gpu {torch.cuda.get_device_name(0)}\n")

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

    print(f"{'phase':<16} | {'B32_ms':>8} {'us/slice':>9} | {'B128_ms':>8} {'us/slice':>9}")
    print("-" * 60)
    rows = {}
    for B in (32, 128):
        x = torch.randn(B, N, device="cuda")
        w = f._workspace(B)
        zr = torch.empty(B, L, device="cuda")
        zi = torch.empty(B, L, device="cuda")
        t1r, t1i = torch.empty_like(zr), torch.empty_like(zr)
        t2r, t2i = torch.empty_like(zr), torch.empty_like(zr)
        zpr, zpi = torch.empty_like(zr), torch.empty_like(zr)
        xot = torch.empty(B, N, device="cuda")

        def ph_fft():
            zr.copy_(x[..., 0::2])
            zi.copy_(x[..., 1::2])
            pext.fft3(zr, zi, t1r, t1i, t2r, t2i, zpr, zpi,
                      FR, FI, WLR, WLI, WKR, WKI, B)

        def ph_router():
            for t in f.two:
                rt = t["router"]
                rext.router_two(zpr, zpi, w["KR"], w["KI"],
                                rt["p1"], rt["p2"], rt["ar"], rt["ai"],
                                rt["br"], rt["bi"],
                                t["F1R"], t["F1I"], t["TWR"], t["TWI"],
                                t["F2R"], t["F2I"], t["task"],
                                f.ncoef, t["N1"], t["N2"], t["scale"], B)
            dr = f.d_router
            rext.router_direct(zpr, zpi, w["KR"], w["KI"],
                               dr["p1"], dr["p2"], dr["ar"], dr["ai"],
                               dr["br"], dr["bi"],
                               f.d_wir, f.d_wii, f.d_task, f.ncoef, B)

        def ph_scatter():
            f._scatter(w["KR"], w["KI"], w, B, "ieee")

        def ph_tail():
            pext.inv_tail(w["FR"], w["FI"], zr, zi, t1r, t1i, t2r, t2i,
                          zpr, zpi, xot, FR, FI, WLR, WLI, WKR, WKI,
                          WNR, WNI, B)

        with torch.no_grad():
            ph_fft(); ph_router(); ph_scatter(); ph_tail()
            for name, fn in (("pack+fft3", ph_fft), ("router bands", ph_router),
                             ("scatter bands", ph_scatter), ("inverse tail", ph_tail)):
                g = capture(fn)
                v = med(g.replay)
                rows.setdefault(name, {})[B] = v
    for name, d in rows.items():
        print(f"{name:<16} | {d[32]:>8.3f} {1000 * d[32] / 32:>9.1f} | "
              f"{d[128]:>8.3f} {1000 * d[128] / 128:>9.1f}")
    tot32 = sum(d[32] for d in rows.values())
    tot128 = sum(d[128] for d in rows.values())
    print(f"{'total':<16} | {tot32:>8.3f} {1000 * tot32 / 32:>9.1f} | "
          f"{tot128:>8.3f} {1000 * tot128 / 128:>9.1f}")
    print(f"\ncost model marginal 11.4 us/slice, cufft ~30/26 us/slice at B 32/128")


if __name__ == "__main__":
    main()
