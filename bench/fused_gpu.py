"""gates and speed for the fused triton per slice cqt.

run   python bench/fused_gpu.py           on a cuda machine, paste back the output
run   python bench/fused_gpu.py --check   anywhere, no triton needed

the check mode validates the kernel tables and dataflow through the torch
simulator in flash_cqt/fused.py against the eager transform, per block family
and roundtrip. the cuda run then gates each kernel against that simulator
(any mismatch names its kernel), sweeps the precision knob, and times the
fused path against the two measured bars, 0.542 ms per slice at B 1 and 28 us
per slice at B 128 (graphed cufft on the a100).
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, HAVE_TRITON

FS = 44100
SL = 65536
NUM_OCTS = [3, 4, 2]
BINS = [8, 16, 32]
REPS = 100


def snr_db(x, xh):
    #works for real and complex alike
    d = ((x - xh).abs().float() ** 2).sum()
    e = (x.abs().float() ** 2).sum()
    return (10 * torch.log10(e / d)).item() if d > 0 else float("inf")


def build(device):
    cqt = OctCQT(NUM_OCTS, BINS, fs=FS, audio_len=SL, device=device)
    return cqt, FusedOctCQT(cqt, device)


def check_sim(device):
    #simulator against the eager transform, this validates every table
    print(f"== simulator gates on {device}, fp32 ==")
    torch.manual_seed(0)
    cqt, f = build(device)
    x = torch.randn(2, SL, device=device)
    ok = True
    with torch.no_grad():
        blocks, side = cqt.fwd(x)
        eager = list(blocks) + list(side)
        KR, KI = f.sim_fwd(x)
        ours = f.blocks_view(KR, KI)
        worst = min(snr_db(a, b) for a, b in zip(eager, ours))
        print(f"forward blocks vs eager, worst family  {worst:8.2f} db")
        ok &= worst > 100
        xo = f.sim_bwd(KR, KI)
        v = snr_db(x, xo)
        print(f"simulator roundtrip                    {v:8.2f} db")
        ok &= v > 100
        xe = cqt.bwd(blocks, side)
        v = snr_db(xe, xo)
        print(f"simulator inverse vs eager inverse     {v:8.2f} db")
        ok &= v > 100
    print("gates", "PASSED" if ok else "FAILED")
    return ok


def ev_ms(fn, reps=REPS):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
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
        print("no cuda device found, run with --check for the cpu gates")
        sys.exit(1)
    if not HAVE_TRITON:
        print("triton is not importable in this environment")
        sys.exit(1)
    dev = "cuda"
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"cuda {torch.version.cuda}")
    print(f"config fs {FS} sl_len {SL} num_octs {NUM_OCTS} bins {BINS}\n")
    if not check_sim(dev):
        print("aborting, the simulator must match eager before kernels mean anything")
        sys.exit(1)

    torch.manual_seed(0)
    cqt, f = build(dev)
    x = torch.randn(1, SL, device=dev)
    ok = True

    #kernel against simulator, stage by stage, fp32
    print("\n== kernel gates against the simulator, fp32 ==")
    f.prec = "ieee"
    with torch.no_grad():
        sKR, sKI = f.sim_fwd(x)
        KR, KI = f.fwd(x)
        v = snr_db(torch.complex(sKR, sKI), torch.complex(KR, KI))
        print(f"forward kernels vs simulator   {v:8.2f} db")
        ok &= v > 100
        sxo = f.sim_bwd(sKR, sKI)
        xo = f.bwd(sKR.clone(), sKI.clone())
        v = snr_db(sxo, xo)
        print(f"inverse kernels vs simulator   {v:8.2f} db")
        ok &= v > 100
        xo = f.bwd(*f.fwd(x))
        v = snr_db(x, xo)
        print(f"fused roundtrip vs input       {v:8.2f} db")
        ok &= v > 100
    if not ok:
        print("kernel gates FAILED, timings below are for debugging only")

    #precision knob
    print("\n== precision knob, per slice roundtrip B 1 ==")
    print(f"{'prec':<8} | {'snr_db':>8} | {'eager_ms':>9} {'graph_ms':>9}")
    print("-" * 44)
    results = {}
    with torch.no_grad():
        for prec in ("ieee", "tf32", "tf32x3"):
            f.prec = prec
            try:
                rt = lambda: f.bwd(*f.fwd(x))
                v = snr_db(x, rt())
                e = ev_ms(rt)
                g = capture(rt)
                gm = ev_ms(g.replay)
                results[prec] = (v, gm)
                print(f"{prec:<8} | {v:>8.2f} | {e:>9.3f} {gm:>9.3f}")
            except Exception as exc:
                print(f"{prec:<8} | SKIPPED {type(exc).__name__}: {str(exc)[:70]}")
    f.prec = "ieee"

    #which kernel eats the time, medians per launch
    print("\n== per kernel breakdown, fp32 ==")
    with torch.no_grad():
        p1 = f.bench_parts(x)
        x32 = torch.randn(32, SL, device=dev)
        f._workspace(32)
        p32 = f.bench_parts(x32)
    print(f"{'kernel':<16} | {'B1_ms':>8} | {'B32_ms':>8}")
    print("-" * 40)
    for (n1, v1), (_, v32) in zip(p1, p32):
        print(f"{n1:<16} | {v1:>8.3f} | {v32:>8.3f}")

    #config sweep, brute measurement instead of one hypothesis per cluster
    #round. tries tile shapes for the two worst stage kernels and split
    #factors for the widest block bucket, applies the winners to everything
    #below, so the bars table already runs with the best measured configs
    print("\n== config sweep, fp32, median ms per launch ==")
    CFGS = [
        dict(BM=32, BN=128, BK=64, num_warps=8, num_stages=2),
        dict(BM=32, BN=64, BK=64, num_warps=4, num_stages=2),
        dict(BM=64, BN=64, BK=64, num_warps=8, num_stages=2),
        dict(BM=32, BN=128, BK=64, num_warps=8, num_stages=1),
        dict(BM=64, BN=128, BK=32, num_warps=8, num_stages=2),
        dict(BM=32, BN=32, BK=64, num_warps=4, num_stages=2),
        dict(BM=32, BN=128, BK=64, num_warps=16, num_stages=2),
    ]
    xs = {1: x, 32: torch.randn(32, SL, device=dev)}
    f._workspace(32)
    print(f"{'stage cfg':<24} | {'s1i_B1':>7} {'s2f_B1':>7} | {'s1i_B32':>8} {'s2f_B32':>8}")
    print("-" * 64)
    best1 = best2 = None
    with torch.no_grad():
        for c in CFGS:
            label = f"BM{c['BM']} BN{c['BN']} BK{c['BK']} w{c['num_warps']} s{c['num_stages']}"
            try:
                f.s1i = dict(c)
                f.s2f = dict(c)
                p1 = dict(f.bench_parts(xs[1], reps=20))
                p32 = dict(f.bench_parts(xs[32], reps=20))
                v = (p1["inv stage1"], p1["fwd stage2"],
                     p32["inv stage1"], p32["fwd stage2"])
                print(f"{label:<24} | {v[0]:>7.3f} {v[1]:>7.3f} | {v[2]:>8.3f} {v[3]:>8.3f}")
                s1 = v[0] + v[2] / 32
                s2 = v[1] + v[3] / 32
                if best1 is None or s1 < best1[0]:
                    best1 = (s1, dict(c))
                if best2 is None or s2 < best2[0]:
                    best2 = (s2, dict(c))
            except Exception as exc:
                print(f"{label:<24} | SKIPPED {type(exc).__name__}: {str(exc)[:40]}")
        f.s1i = best1[1]
        f.s2f = best2[1]
        c1, c2 = f.s1i, f.s2f
        print(f"winners  s1i BM{c1['BM']} BN{c1['BN']} BK{c1['BK']} w{c1['num_warps']} "
              f"s{c1['num_stages']}, s2f BM{c2['BM']} BN{c2['BN']} BK{c2['BK']} "
              f"w{c2['num_warps']} s{c2['num_stages']}")

        #split factors for the widest block bucket, fwd and inv apart
        tb = [t for t in f.two if t["N2"] >= 64][0]
        name = f"two {tb['N1']}x{tb['N2']}"
        keep_bmax = f.split_bmax
        f.split_bmax = 1 << 30
        print(f"\n{'split':>5} | {'fwd_B1':>7} {'inv_B1':>7} | {'fwd_B32':>8} {'inv_B32':>8}")
        print("-" * 46)
        bestf = besti1 = besti32 = None
        for s in (1, 2, 4, 8):
            tb["split"] = s
            tb["spliti"] = s
            p1 = dict(f.bench_parts(xs[1], reps=20))
            p32 = dict(f.bench_parts(xs[32], reps=20))
            v = (p1["fwd " + name], p1["inv " + name],
                 p32["fwd " + name], p32["inv " + name])
            print(f"{s:>5} | {v[0]:>7.3f} {v[1]:>7.3f} | {v[2]:>8.3f} {v[3]:>8.3f}")
            if bestf is None or v[0] + v[2] / 32 < bestf[0]:
                bestf = (v[0] + v[2] / 32, s)
            if besti1 is None or v[1] < besti1[0]:
                besti1 = (v[1], s)
            if besti32 is None or v[3] < besti32[0]:
                besti32 = (v[3], s)
        tb["split"] = bestf[1]
        tb["spliti"] = besti1[1]
        f.split_bmax = keep_bmax if besti32[1] == 1 else (1 << 30)
        print(f"winners  fwd split {tb['split']}, inv split {tb['spliti']} "
              f"(unsplit past batch {f.split_bmax if f.split_bmax < 1 << 30 else 'never'})")

    #peak memory before the batch loop fills the workspace cache, so the
    #number means one B 1 roundtrip and not the sum of every batch size
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        f.bwd(*f.fwd(x))
    torch.cuda.synchronize()
    print(f"\npeak cuda memory at B 1 (plus the B 32 workspace above)  "
          f"{torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    #the bars, fp32 and the tensor core mode that held 117 db side by side
    print("\n== against the cufft baseline, graphed, per slice ==")
    print(f"{'B':>4} | {'cufft_ms':>9} | {'fp32_ms':>9} {'speedup':>7} | "
          f"{'tf32x3_ms':>9} {'speedup':>7}")
    print("-" * 62)
    with torch.no_grad():
        for B in (1, 8, 32, 128):
            xb = torch.randn(B, SL, device=dev)
            def cu_rt():
                bl, sd = cqt.fwd(xb)
                return cqt.bwd(bl, sd)
            g = capture(cu_rt)
            a = ev_ms(g.replay)
            f._workspace(B)
            f.prec = "ieee"
            g = capture(lambda: f.bwd(*f.fwd(xb)))
            b_ = ev_ms(g.replay)
            try:
                f.prec = "tf32x3"
                g = capture(lambda: f.bwd(*f.fwd(xb)))
                c_ = ev_ms(g.replay)
                tail = f"{c_:>9.3f} {a / c_:>6.2f}x"
            except Exception as exc:
                tail = f"SKIPPED {type(exc).__name__}"
            f.prec = "ieee"
            print(f"{B:>4} | {a:>9.3f} | {b_:>9.3f} {a / b_:>6.2f}x | {tail}")

    if "ieee" in results:
        v, gm = results["ieee"]
        print(f"\nsummary  fused fp32 graphed {gm:.3f} ms per slice at {v:.1f} db, "
              f"bars were 0.542 ms (B 1) and 0.028 ms (B 128, per slice)")


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check_sim("cpu") else 1)
    main()
