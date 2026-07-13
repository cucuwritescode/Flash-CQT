"""every cqt implementation we know of, one gpu, one table.

run   python bench/compare_gpu.py           on a cuda machine, paste back the output
run   python bench/compare_gpu.py --check   anywhere, correctness only, no timing

rows that need code this checkout does not have (xumx sliCQ, mr cqtdiff) or
packages that are not installed (cqt-pytorch, nnAudio) are skipped with a note,
nothing kills the run. pip install cqt-pytorch nnAudio for the full table.

whole buffer rows use 262144 samples at 44100. ours and multi_CQT run the
mr cqtdiff config [3,4,2] octaves at [8,16,32] bins, the single resolution
engines run 9 octaves at 16 bins from the same fmin. slicq gets sl_len 65536
to match ours, same latency, fair fight. nnaudio has no inverse so it only
gets a forward time. xrt is seconds of audio processed per second of compute
on the roundtrip (forward for nnaudio).

the last section times the per slice streaming unit eagerly and through a
recorded cuda graph (flash_cqt.Graphed), and checks the graph replay output
is identical to eager before trusting its time.
"""
import sys
import types
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "xumx-sliCQ"))
sys.path.insert(0, str(ROOT / "repo-refs" / "CQT_pytorch"))

FS = 44100
LS = 262144
SL = 65536
NUM_OCTS = [3, 4, 2]
BINS = [8, 16, 32]
FMIN = FS / 2 / 2 ** 9
FLO, FHI = 2 * FMIN, 0.95 * FS / 2
REPS = 50
AUDIO_S = LS / FS


def snr_db(x, xh):
    x, xh = x.float(), xh.float()
    return (10 * torch.log10((x ** 2).sum() / ((x - xh) ** 2).sum())).item()


def snr_inband_db(x, xh):
    X = torch.fft.rfft(x.float(), dim=-1)
    Xh = torch.fft.rfft(xh.float(), dim=-1)
    n = x.shape[-1]
    i0, i1 = int(FLO * n / FS), int(FHI * n / FS)
    return (10 * torch.log10((X[..., i0:i1].abs() ** 2).sum()
                             / ((X - Xh)[..., i0:i1].abs() ** 2).sum())).item()


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


def peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


#each engine returns (snr_raw, snr_inband, fwd_fn, inv_fn, rt_fn, note),
#timing happens outside so the check mode can skip it

def eng_flash(dev):
    from flash_cqt import SlicedCQT
    c = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device=dev)
    x = torch.randn(1, 1, LS, device=dev)
    b, sd = c.fwd(x)
    xh = c.bwd(b, sd, length=LS)
    return (snr_db(x, xh), snr_inband_db(x, xh),
            lambda: c.fwd(x), lambda: c.bwd(b, sd, length=LS),
            lambda: c.bwd(*c.fwd(x), length=LS), "ours, streamable")


def eng_flashoct(dev):
    from flash_cqt import OctCQT
    c = OctCQT(NUM_OCTS, BINS, fs=FS, audio_len=LS, device=dev)
    x = torch.randn(1, 1, LS, device=dev)
    b, sd = c.fwd(x)
    xh = c.bwd(b, sd)
    return (snr_db(x, xh), snr_inband_db(x, xh),
            lambda: c.fwd(x), lambda: c.bwd(b, sd),
            lambda: c.bwd(*c.fwd(x)), "ours, offline")


def eng_multicqt(dev):
    #what mr cqtdiff ships today, one cqt and one full fft per resolution group
    pkg = types.ModuleType("utils")
    pkg.__path__ = [str(ROOT / "MR-CQTdiff" / "src" / "utils")]
    sys.modules["utils"] = pkg
    from utils.cqt_nsgt_pytorch.multiCQT import multi_CQT
    fmax = [FS * 2 ** (-1 - sum(NUM_OCTS[1 + i:])) for i in range(3)]
    c = multi_CQT(numocts=NUM_OCTS, binsoct=BINS, fmax=fmax, mode="oct",
                  window="hann", fs=FS, audio_len=LS, dtype=torch.float32, device=dev)
    x = torch.randn(1, 1, LS, device=dev)
    b = c.fwd(x)
    xh = c.bwd(b)[..., :LS]
    return (snr_db(x, xh), snr_inband_db(x, xh),
            lambda: c.fwd(x), lambda: c.bwd(b),
            lambda: c.bwd(c.fwd(x)), "mr cqtdiff today")


def eng_refoct(dev):
    #the repo-refs oct cqt we started from, uniform 9x16
    from cqt_nsgt_pytorch import CQT_nsgt
    c = CQT_nsgt(9, 16, mode="oct", fs=FS, audio_len=LS, device=dev,
                 dtype=torch.float32)
    x = torch.randn(1, 1, LS, device=dev)
    b = c.fwd(x)
    xh = c.bwd(b)
    return (snr_db(x, xh), snr_inband_db(x, xh),
            lambda: c.fwd(x), lambda: c.bwd(b),
            lambda: c.bwd(c.fwd(x)), "eloimoliner base, 9x16")


def eng_slicq(dev):
    from xumx_slicq_v2.nsgt import NSGT_sliced, LogScale
    scl = LogScale(FMIN, FS / 2, 9 * 16)
    n = NSGT_sliced(scl, SL, SL // 4, FS, real=True, multichannel=True, device=dev)
    x = torch.randn(1, LS, device=dev)
    b = n.forward((x,))
    #backward mutates its input blocks, hand it clones
    xh = n.backward([t.clone() for t in b], LS)
    return (snr_db(x, xh), snr_inband_db(x, xh),
            lambda: n.forward((x,)),
            lambda: n.backward([t.clone() for t in b], LS),
            lambda: n.backward(n.forward((x,)), LS), "sevagh sliCQ, 9x16")


def eng_archinetai(dev):
    from cqt_pytorch import CQT
    c = CQT(num_octaves=9, num_bins_per_octave=16, sample_rate=FS,
            block_length=LS).to(dev)
    x = torch.randn(1, 1, LS, device=dev)
    z = c.encode(x)
    xh = c.decode(z)[..., :LS]
    return (snr_db(x, xh), snr_inband_db(x, xh),
            lambda: c.encode(x), lambda: c.decode(z),
            lambda: c.decode(c.encode(x)), "archinetai, 9x16, dense")


def eng_nnaudio(dev):
    from nnAudio.features.cqt import CQT1992v2
    c = CQT1992v2(sr=FS, hop_length=512, fmin=FMIN, n_bins=9 * 16,
                  bins_per_octave=16, verbose=False).to(dev)
    x = torch.randn(1, LS, device=dev)
    c(x)
    return (None, None, lambda: c(x), None, None,
            "conv based, forward only, not invertible")


def eng_stft(dev):
    win = torch.hann_window(2048, device=dev)
    x = torch.randn(1, LS, device=dev)
    fwd = lambda: torch.stft(x, 2048, hop_length=512, window=win,
                             center=True, return_complex=True)
    z = fwd()
    inv = lambda: torch.istft(z, 2048, hop_length=512, window=win,
                              center=True, length=LS)
    xh = inv()
    return (snr_db(x, xh), snr_inband_db(x, xh), fwd, inv,
            lambda: torch.istft(fwd(), 2048, hop_length=512, window=win,
                                center=True, length=LS),
            "single resolution reference")


ENGINES = [
    ("flash", eng_flash),
    ("flashoct", eng_flashoct),
    ("multicqt", eng_multicqt),
    ("refoct", eng_refoct),
    ("slicq", eng_slicq),
    ("archinetai", eng_archinetai),
    ("nnaudio", eng_nnaudio),
    ("stft", eng_stft),
]


def run_table(dev, timing):
    print(f"{'engine':<11} | {'snr_raw':>8} {'inband':>8} | "
          f"{'fwd_ms':>8} {'inv_ms':>8} {'rt_ms':>8} {'xRT':>7} | {'peak_MB':>8} | note")
    print("-" * 110)
    for name, build in ENGINES:
        try:
            torch.manual_seed(0)
            with torch.no_grad():
                sr, si, fwd, inv, rt, note = build(dev)
                if timing:
                    fm = ev_ms(fwd)
                    im = ev_ms(inv) if inv else None
                    rm = ev_ms(rt) if rt else None
                    pk = peak_mb(rt if rt else fwd)
                else:
                    fm = im = rm = pk = None
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"{name:<11} | SKIPPED {msg[:85]}")
            continue
        f = lambda v, w=8, p=2: " " * (w - 1) + "-" if v is None else f"{v:>{w}.{p}f}"
        xrt = " " * 6 + "-" if not rm else f"{1000 * AUDIO_S / rm:>7.0f}"
        if not rm and fm:
            xrt = f"{1000 * AUDIO_S / fm:>7.0f}"
        print(f"{name:<11} | {f(sr)} {f(si)} | {f(fm)} {f(im)} {f(rm)} {xrt} | "
              f"{f(pk, 8, 1)} | {note}")


def run_streaming(dev):
    #the per slice unit, eager against a recorded cuda graph
    from flash_cqt import SlicedCQT, Graphed
    print("\n== per slice streaming unit, B 1, eager vs cuda graph ==")
    torch.manual_seed(0)
    s = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device=dev)
    sl = torch.randn(1, 1, SL, device=dev) * s.tw

    def rt(t):
        b, sd = s.cqt.fwd(t)
        return s.cqt.bwd(b, sd)

    with torch.no_grad():
        eager = ev_ms(lambda: rt(sl), REPS * 2)
        g = Graphed(rt, sl)
        md = (g(sl) - rt(sl)).abs().max().item()
        graph = ev_ms(lambda: g(sl), REPS * 2)
    budget = 1000.0 * s.hop / FS
    print(f"graph output vs eager  max|diff| = {md:.2e}  "
          f"({'OK' if md == 0 else 'CHECK, expected exactly 0'})")
    print(f"eager    {eager:8.3f} ms/slice  {1000 / eager:7.0f} slices/s  "
          f"margin {budget / eager:5.0f}x realtime")
    print(f"graphed  {graph:8.3f} ms/slice  {1000 / graph:7.0f} slices/s  "
          f"margin {budget / graph:5.0f}x realtime")
    print(f"graph replay is {eager / graph:.2f}x eager, this is the baseline any "
          "fused kernel has to beat")


def main():
    check = "--check" in sys.argv
    if check:
        print("correctness only on cpu, timings need the cuda run\n")
        run_table("cpu", timing=False)
        return
    if not torch.cuda.is_available():
        print("no cuda device found, run with --check for the cpu gates")
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"cuda {torch.version.cuda}")
    print(f"config fs {FS} ls {LS} fmin {FMIN:.1f} hz, fp32, B 1\n")
    run_table("cuda", timing=True)
    run_streaming("cuda")


if __name__ == "__main__":
    main()
