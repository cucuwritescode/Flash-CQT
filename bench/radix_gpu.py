"""gates and timings for the radix band engine.

  python bench/radix_gpu.py --check                  cpu table and schedule gates
  modal run bench/cloud.py --cuda --cmd "python bench/radix_gpu.py"

the cpu check drives the exact tables the kernel reads, with the
level by level reference standing in for the warp fft. a pass here means any
gpu failure is a transcription bug in radix.cu, not a table bug. the gpu
run gates the kernel against the shipped eager path, checks graph replay
integrity, then times the band phases against the dense kernels.
"""
import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, N, HALF, _packedfft_ext, _radix_ext

FS = 44100
L = N // 2


def snr(a, b):
    p = a.double().square().sum()
    e = (a.double() - b.double()).square().sum()
    return float("inf") if float(e) == 0 else float(10 * torch.log10(p / e))


def packed_spectrum(x):
    #the permuted packed half spectrum the router taps index into
    z = torch.complex(x[..., 0::2].contiguous(), x[..., 1::2].contiguous())
    Z = torch.fft.fft(z)
    q = torch.arange(L, device=x.device)
    zperm = (q & 31) * 1024 + ((q >> 5) & 31) * 32 + (q >> 10)
    zp = torch.empty_like(Z)
    zp[..., zperm] = Z
    return zp


def heap(cqt, x):
    blocks, side = cqt.fwd(x)
    B = x.shape[0]
    return torch.cat([c.reshape(B, -1) for c in list(blocks) + list(side)],
                     dim=-1)


def check_cpu():
    import radix_reference as p4
    torch.manual_seed(7)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=N, device="cpu")
    f = FusedOctCQT(cqt, "cpu")
    rx = f.radix
    rows = rx["desc"].tolist()
    buckets = {}
    for m, _, _, _ in rows:
        buckets[m] = buckets.get(m, 0) + 1
    print("bands", len(rows), "buckets", dict(sorted(buckets.items())))

    x = torch.randn(2, N)
    zp = packed_spectrum(x)
    ref = heap(cqt, x)

    from flash_cqt.fused import radix_load_perm, radix_store_perm
    with torch.no_grad():
        #analysis, route then the scheduled inverse dft per band. the
        #shipped tables are load ordered, invert the permutation here so
        #the reference runs on natural order, this cross checks the
        #kernel's index arithmetic against the table builder
        cand = torch.zeros_like(ref)
        for m, ab, ob, _ in rows:
            lp = radix_load_perm(m)
            A = torch.empty(m, dtype=torch.complex64)
            Bc = torch.empty(m, dtype=torch.complex64)
            i1 = torch.empty(m, dtype=torch.long)
            i2 = torch.empty(m, dtype=torch.long)
            A[lp] = torch.complex(rx["ar"][ab:ab + m], rx["ai"][ab:ab + m])
            Bc[lp] = torch.complex(rx["br"][ab:ab + m], rx["bi"][ab:ab + m])
            i1[lp] = rx["p1"][ab:ab + m].long()
            i2[lp] = rx["p2"][ab:ab + m].long()
            v = A * zp[:, i1] + Bc * zp[:, i2].conj()
            cand[:, ob:ob + m] = p4.scheduled_ifft(v)
        fwd_db = snr(torch.view_as_real(ref), torch.view_as_real(cand))

        #synthesis, scheduled dft, dual gain, scatter, irfft tail
        frr = torch.zeros(2, HALF + 1)
        fri = torch.zeros(2, HALF + 1)
        for m, ab, ob, _ in rows:
            sp = radix_store_perm(m)
            d = p4.scheduled_fft(ref[:, ob:ob + m])
            g = torch.empty(m)
            pos = torch.empty(m, dtype=torch.long)
            g[sp] = rx["gdv"][ab:ab + m]
            pos[sp] = rx["pos"][ab:ab + m].long()
            live = g != 0
            frr.index_add_(1, pos[live], d.real[:, live] * g[live])
            fri.index_add_(1, pos[live], d.imag[:, live] * g[live])
        sig = torch.fft.irfft(torch.complex(frr, fri), N)
        rt_db = snr(x, sig)

    #the compact record must reconstruct the six arrays it replaced
    rec = rx["route"].reshape(-1, 4)
    q = rec[:, 0].to(torch.int64) & 0xffffffff
    p1r = (q & 0xffff).to(torch.int32)
    p2r = (q >> 16).to(torch.int32)
    brr = rec[:, 3].view(torch.float32) - rec[:, 1].view(torch.float32)
    bir = -rec[:, 2].view(torch.float32)
    idx_ok = torch.equal(p1r, rx["p1"]) and torch.equal(p2r, rx["p2"])
    rec_db = min(snr(rx["br"], brr), snr(rx["bi"], bir))
    print(f"compact record vs arrays {rec_db:.2f} db, indices "
          + ("exact" if idx_ok else "WRONG"))

    print(f"analysis vs shipped heap {fwd_db:.2f} db")
    print(f"synthesis roundtrip      {rt_db:.2f} db")
    ok = fwd_db >= 120 and rt_db >= 120 and rec_db >= 120 and idx_ok
    print("TABLE-GATES-" + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


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


def main_gpu():
    import packedfft as pf
    from flash_cqt.fused import _router_ext
    xext = _radix_ext()
    assert xext, "radix extension failed to build, rerun with verbose load"
    pext = _packedfft_ext()
    rext = _router_ext()
    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=N, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    f.prec = "ieee"
    rx = f.radix
    print("gpu", torch.cuda.get_device_name(0), " torch", torch.__version__)

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

    hdr = (f"{'B':>4} | {'dense_fwd':>9} {'radix_fwd':>9} | "
           f"{'dense_inv':>9} {'inv_2buf':>9} {'inv_pull':>9} | "
           f"us/slice d->r best")
    rowsout = []
    for B in (1, 8, 32, 128):
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

        def radix_fwd():
            xext.fwd_bands(zpr, zpi, w["KR"], w["KI"],
                           rx["route"], rx["desc"], rx["rr"],
                           rx["ri"], f.ncoef, B)

        def dense_fwd():
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

        def radix_inv(variant=2):
            w["FR"].zero_()
            w["FI"].zero_()
            xext.inv_bands(w["KR"], w["KI"], w["FR"], w["FI"],
                           rx["gdv"], rx["pos"], rx["desc"], rx["rr"],
                           rx["ri"], f.ncoef, B, variant)

        def radix_inv1():
            radix_inv(1)

        def dense_inv():
            f._scatter(w["KR"], w["KI"], w, B, "ieee")

        def ph_tail():
            pext.inv_tail(w["FR"], w["FI"], zr, zi, t1r, t1i, t2r, t2i,
                          zpr, zpi, xot, FR, FI, WLR, WLI, WKR, WKI,
                          WNR, WNI, B)

        with torch.no_grad():
            #gates once per batch, radix vs the shipped eager path
            kr_ref, ki_ref = f.fwd(x)
            kr_ref, ki_ref = kr_ref.clone(), ki_ref.clone()
            ph_fft()
            radix_fwd()
            fdb = min(snr(kr_ref, w["KR"]), snr(ki_ref, w["KI"]))

            dense_inv()
            fr_ref, fi_ref = w["FR"].clone(), w["FI"].clone()
            radix_inv(1)
            i1 = min(snr(fr_ref, w["FR"]), snr(fi_ref, w["FI"]))
            radix_inv(2)
            idb = min(snr(fr_ref, w["FR"]), snr(fi_ref, w["FI"]), i1)
            ph_tail()
            rdb = snr(x, xot)
            print(f"gate B {B:<3} fwd {fdb:7.2f} db  scatter {idb:7.2f} db  "
                  f"roundtrip {rdb:7.2f} db -> "
                  + ("PASS" if min(fdb, idb, rdb) >= 118 else "FAIL"))
            if min(fdb, idb, rdb) < 118:
                continue

            #replay integrity. the scatter is atomic so summation order
            #differs between runs and bitwise equality is unreachable,
            #instead poison every buffer the graph must write, then the
            #replay has to rebuild the eager result from zpr alone. a
            #silently dropped kernel leaves poison behind and fails loud
            radix_fwd()
            radix_inv()
            er, ei = w["FR"].clone(), w["FI"].clone()
            g = capture(lambda: (radix_fwd(), radix_inv()))
            w["KR"].fill_(0.0)
            w["KI"].fill_(0.0)
            w["FR"].fill_(1e9)
            w["FI"].fill_(1e9)
            g.replay()
            torch.cuda.synchronize()
            rint = min(snr(er, w["FR"]), snr(ei, w["FI"]))
            if rint < 120:
                print(f"replay vs eager {rint:.2f} db, timings void")
                continue
            print(f"replay vs eager {rint:.2f} db (atomic order noise)")

            gd_f = capture(dense_fwd)
            gx_f = capture(radix_fwd)
            gd_i = capture(dense_inv)
            g1_i = capture(radix_inv1)
            g2_i = capture(radix_inv)
            df, xf = med(gd_f.replay), med(gx_f.replay)
            di = med(gd_i.replay)
            x1 = med(g1_i.replay)
            x2 = med(g2_i.replay)
            rowsout.append((B, df, xf, di, x1, x2))

    print("\n" + hdr)
    print("-" * len(hdr))
    for B, df, xf, di, x1, x2 in rowsout:
        xb = min(x1, x2)
        print(f"{B:>4} | {df:>9.3f} {xf:>9.3f} | {di:>9.3f} {x1:>9.3f} "
              f"{x2:>9.3f} | {1000 * (df + di) / B:>6.1f} -> "
              f"{1000 * (xf + xb) / B:.1f}")

    #per length breakdown at batch, one bucket at a time through a sliced
    #descriptor, so the batch gap names its owner length
    desc = rx["desc"]
    ms = desc[:, 0]
    print(f"\nper length, B=32, graphed ms and us/slice")
    print(f"{'M':>6} {'bands':>6} | {'fwd_ms':>8} {'us/sl':>7} | "
          f"{'inv_ms':>8} {'us/sl':>7}")
    B = 32
    x = torch.randn(B, N, device="cuda")
    w = f._workspace(B)
    zr = torch.empty(B, L, device="cuda")
    zi = torch.empty(B, L, device="cuda")
    t1r, t1i = torch.empty_like(zr), torch.empty_like(zr)
    t2r, t2i = torch.empty_like(zr), torch.empty_like(zr)
    zpr, zpi = torch.empty_like(zr), torch.empty_like(zr)
    zr.copy_(x[..., 0::2])
    zi.copy_(x[..., 1::2])
    pext.fft3(zr, zi, t1r, t1i, t2r, t2i, zpr, zpi,
              FR, FI, WLR, WLI, WKR, WKI, B)
    with torch.no_grad():
        for m in (32, 64, 128, 256, 512, 1024, 2048):
            sub = desc[ms == m].contiguous()
            nb = sub.shape[0]

            def bf():
                xext.fwd_bands(zpr, zpi, w["KR"], w["KI"],
                               rx["route"], sub, rx["rr"],
                               rx["ri"], f.ncoef, B)

            def bi_():
                xext.inv_bands(w["KR"], w["KI"], w["FR"], w["FI"],
                               rx["gdv"], rx["pos"], sub, rx["rr"],
                               rx["ri"], f.ncoef, B, 2)

            vf = med(capture(bf).replay)
            vi = med(capture(bi_).replay)
            print(f"{m:>6} {nb:>6} | {vf:>8.3f} {1000 * vf / B:>7.1f} | "
                  f"{vi:>8.3f} {1000 * vi / B:>7.1f}")

    print("\nRADIX-GATES-DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check_cpu())
    main_gpu()
