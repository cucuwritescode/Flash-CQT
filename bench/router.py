"""the router layer of the factored transform, sim and kernel gates.

every band tap becomes tap = A Z[p1] + B conj(Z[p2]) on the packed half
length spectrum, with the real fft untangle and the analysis window folded
into the complex coefficients A and B (side band conjugation swaps the
roles). the tables are built in fused.py where the raw band data lives,
this file proves them, first as a torch sim against the reference taps,
then the cuda kernels against the shipped forward.

run   python bench/router.py             local sim gates, cpu
run   modal run bench/cloud.py --cuda --cmd "python bench/router.py --kernel"
"""
import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, N

FS = 44100
L = N // 2


def snr_db(x, xh):
    e = x.abs().double().square().sum()
    d = (x - xh).abs().double().square().sum()
    return float(10 * torch.log10(e / d)) if d > 0 else float("inf")


def build(device):
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=N, device=device)
    return FusedOctCQT(cqt, device)


def zp_planes(x):
    #packed spectrum in the packed fft kernel's permuted storage order
    z = torch.complex(x[..., 0::2], x[..., 1::2])
    Z = torch.fft.fft(z)
    q = torch.arange(L, device=x.device)
    perm = (q & 31) * 1024 + ((q >> 5) & 31) * 32 + (q >> 10)
    Zp = torch.empty_like(Z)
    Zp[..., perm] = Z
    return Zp


def taps(Zp, rt):
    A = torch.complex(rt["ar"], rt["ai"])
    B = torch.complex(rt["br"], rt["bi"])
    return A * Zp[..., rt["p1"].long()] + B * Zp[..., rt["p2"].long()].conj()


def ref_taps(ft, rt, w):
    a = ft[..., rt["k"].long()]
    a = torch.where(rt["cf"], a.conj(), a)
    return a * w


def sim_gate(device="cpu"):
    torch.manual_seed(0)
    f = build(device)
    x = torch.randn(2, N, device=device)
    Zp = zp_planes(x)
    ft = torch.fft.rfft(x)
    ok = True
    v = snr_db(ref_taps(ft, f.d_router, f.d_wre), taps(Zp, f.d_router))
    print(f"router taps direct family  {v:8.2f} db  ->",
          "PASS" if v > 120 else "FAIL")
    ok &= v > 120
    for t in f.two:
        v = snr_db(ref_taps(ft, t["router"], t["wre"]),
                   taps(Zp, t["router"]))
        print(f"router taps bucket {t['N1']}x{t['N2']:<4}{v:8.2f} db  ->",
              "PASS" if v > 120 else "FAIL")
        ok &= v > 120
    return ok


def kernel_gate():
    from flash_cqt.fused import _packedfft_ext, _router_ext
    import packedfft as pf
    pext = _packedfft_ext()
    rext = _router_ext()
    if not pext or not rext:
        print("extension build failed", bool(pext), bool(rext))
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    torch.manual_seed(0)
    f = build("cuda")
    F, WL, W1k = pf.tables("cuda")
    FR, FI = F.real.contiguous(), F.imag.contiguous()
    b1024 = torch.arange(1024, device="cuda")
    lane = torch.arange(32, device="cuda")
    WLp = WL[(b1024[:, None] * lane[None, :]) % L].reshape(-1)
    g32 = torch.arange(32, device="cuda")
    WKp = W1k[(g32[:, None] * lane[None, :]) % 1024].reshape(-1)
    WLR, WLI = WLp.real.contiguous(), WLp.imag.contiguous()
    WKR, WKI = WKp.real.contiguous(), WKp.imag.contiguous()

    for B in (1, 32):
        x = torch.randn(B, N, device="cuda")
        w = f._workspace(B)
        zr = torch.empty(B, L, device="cuda")
        zi = torch.empty(B, L, device="cuda")
        t1r, t1i = torch.empty_like(zr), torch.empty_like(zr)
        t2r, t2i = torch.empty_like(zr), torch.empty_like(zr)
        zpr, zpi = torch.empty_like(zr), torch.empty_like(zr)

        def factored_fwd():
            zr.copy_(x[..., 0::2])
            zi.copy_(x[..., 1::2])
            pext.fft3(zr, zi, t1r, t1i, t2r, t2i, zpr, zpi,
                      FR, FI, WLR, WLI, WKR, WKI, B)
            for t in f.two:
                rt = t["router"]
                rext.router_two(zpr, zpi, w["KR"], w["KI"],
                                rt["p1"], rt["p2"], rt["ar"], rt["ai"],
                                rt["br"], rt["bi"],
                                t["F1R"], t["F1I"], t["TWR"], t["TWI"],
                                t["F2R"], t["F2I"], t["task"],
                                f.ncoef, t["N1"], t["N2"],
                                t["scale"], B)
            dr = f.d_router
            rext.router_direct(zpr, zpi, w["KR"], w["KI"],
                               dr["p1"], dr["p2"], dr["ar"], dr["ai"],
                               dr["br"], dr["bi"],
                               f.d_wir, f.d_wii, f.d_task, f.ncoef, B)
            return w["KR"], w["KI"]

        with torch.no_grad():
            f.use_cuda_blocks = True
            KR0, KI0 = (u.clone() for u in f.fwd(x))
            KR1, KI1 = factored_fwd()
            v = min(snr_db(KR0, KR1), snr_db(KI0, KI1))
            print(f"B {B:>3}  factored fwd vs shipped {v:8.2f} db  ->",
                  "PASS" if v > 118 else "FAIL")

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

            tf_ = med(factored_fwd)
            ts_ = med(lambda: f.fwd(x))
            print(f"B {B:>3}  forward eager  factored {tf_:7.3f} ms   "
                  f"shipped {ts_:7.3f} ms   {ts_ / tf_:5.2f}x")

            #the inverse tail, scatter stays on the proven path, the big
            #inverse fft goes packed
            import math as _m
            qq = torch.arange(L, device="cuda", dtype=torch.float64)
            WN = torch.exp(-2j * _m.pi * qq / N).to(torch.complex64)
            WNR, WNI = WN.real.contiguous(), WN.imag.contiguous()
            xo_ref = f.bwd(KR1.clone(), KI1.clone()).clone()
            xot = torch.empty(B, N, device="cuda")
            pext.inv_tail(w["FR"], w["FI"], zr, zi, t1r, t1i, t2r, t2i,
                          zpr, zpi, xot, FR, FI, WLR, WLI, WKR, WKI,
                          WNR, WNI, B)
            vt = snr_db(x, xot)
            vs = snr_db(xo_ref, xot)
            print(f"B {B:>3}  factored roundtrip vs input {vt:8.2f} db  "
                  f"vs shipped {vs:8.2f} db  ->",
                  "PASS" if vt > 118 else "FAIL")

            def factored_rt():
                KRa, KIa = factored_fwd()
                f._scatter(KRa, KIa, w, B, "ieee")
                pext.inv_tail(w["FR"], w["FI"], zr, zi, t1r, t1i, t2r, t2i,
                              zpr, zpi, xot, FR, FI, WLR, WLI, WKR, WKI,
                              WNR, WNI, B)

            tr_ = med(factored_rt)
            tsr = med(lambda: f.bwd(*f.fwd(x)))
            print(f"B {B:>3}  roundtrip eager  factored {tr_:7.3f} ms   "
                  f"shipped {tsr:7.3f} ms   {tsr / tr_:5.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="store_true")
    args = ap.parse_args()
    if args.kernel:
        if not sim_gate("cuda"):
            sys.exit(1)
        kernel_gate()
    else:
        sys.exit(0 if sim_gate() else 1)
