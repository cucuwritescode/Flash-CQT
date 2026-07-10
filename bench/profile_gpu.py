"""cuda profile of the sliced oct cqt at the mr cqtdiff config.

self contained, needs only torch and this repo's flash_cqt package. run it on
the cluster from anywhere and paste back the output

  python bench/profile_gpu.py

it prints roundtrip snr, whole buffer timings and peak cuda memory, a per
slice breakdown (cufft floor, glue, launch and dispatch overhead measured by
cuda graph replay), a torch.stft reference, and a one line verdict on whether
a fused kernel is worth writing on this gpu.
"""
import sys
import time
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import SlicedCQT

FS = 44100
LS = 262144
SL = 65536
NUM_OCTS = [3, 4, 2]
BINS = [8, 16, 32]
REPS_BUF = 20
REPS_SLICE = 200


def ev_ms(fn, reps):
    #median wall time via cuda events
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


def snr_db(x, xh):
    x, xh = x.float(), xh.float()
    return (10 * torch.log10((x ** 2).sum() / ((x - xh) ** 2).sum())).item()


def main():
    if not torch.cuda.is_available():
        print("no cuda device found, this script must run on the gpu cluster")
        sys.exit(1)
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    print(f"gpu {name}  torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"config fs {FS} ls {LS} sl_len {SL} num_octs {NUM_OCTS} bins {BINS} fp32\n")

    torch.manual_seed(0)
    s = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device=dev)
    H, hop, tw, cqt = s.H, s.hop, s.tw, s.cqt

    #whole buffer numbers
    print("whole buffer (262144 samples)")
    print(f"{'batch':>5} | {'snr_db':>8} | {'fwd_ms':>8} {'inv_ms':>8} {'rt_ms':>8} | {'peak_MB':>8}")
    for B in (1, 4):
        x = torch.randn(B, 1, LS, device=dev)
        with torch.no_grad():
            c, side = s.fwd(x)
            xh = s.bwd(c, side, length=LS)
            fwd = ev_ms(lambda: s.fwd(x), REPS_BUF)
            inv = ev_ms(lambda: s.bwd(c, side, length=LS), REPS_BUF)
            rt = ev_ms(lambda: s.bwd(*s.fwd(x), length=LS), REPS_BUF)
            mem = peak_mb(lambda: s.bwd(*s.fwd(x), length=LS))
        print(f"{B:>5} | {snr_db(x, xh):>8.2f} | {fwd:>8.2f} {inv:>8.2f} {rt:>8.2f} | {mem:>8.1f}")

    #per slice, the real streaming unit, one hop of fresh input per slice
    print("\nper slice (streaming unit, B 1)")
    prev = torch.randn(1, 1, hop, device=dev)
    chunk = torch.randn(1, 1, hop, device=dev)

    def slice_fwd():
        sl = torch.cat([prev, chunk], dim=-1) * tw
        sl = torch.roll(sl, -H, -1)
        blocks, side = cqt.fwd(sl.unsqueeze(2))
        blocks = [torch.roll(b, r, -1) for b, r in zip(blocks, s._roll)]
        return blocks, side

    def slice_inv(blocks, side):
        blocks = [torch.roll(b, -r, -1) for b, r in zip(blocks, s._roll)]
        y = cqt.bwd(blocks, side)[:, :, 0]
        return torch.roll(y, +H, -1)

    with torch.no_grad():
        blocks, side = slice_fwd()
        #precompute every intermediate so the floor timers run only the ffts
        sl = torch.roll(torch.cat([prev, chunk], dim=-1) * tw, -H, -1).unsqueeze(2)
        ft = torch.fft.rfft(sl)
        gathered = [ft[..., idx] * gv for idx, gv, _, _, _ in cqt.groups]
        gathered += [torch.where(cj, ft[..., idx].conj(), ft[..., idx]) * gv
                     for idx, gv, _, _, cj in cqt.side_groups]
        allblocks = list(blocks) + list(side)
        fr = torch.zeros(1, cqt.nn // 2 + 1, dtype=torch.complex64, device=dev)

        fwd_floor = ev_ms(lambda: (torch.fft.rfft(sl),
                                   [torch.fft.ifft(a) for a in gathered]), REPS_SLICE)
        inv_floor = ev_ms(lambda: ([torch.fft.fft(b) for b in allblocks],
                                   torch.fft.irfft(fr, n=cqt.nn)), REPS_SLICE)
        fwd_full = ev_ms(slice_fwd, REPS_SLICE)
        inv_full = ev_ms(lambda: slice_inv(blocks, side), REPS_SLICE)
        rt_full = ev_ms(lambda: slice_inv(*slice_fwd()), REPS_SLICE)

        #cuda graph replay removes launch and python dispatch cost, what is
        #left is the gpu work itself, the gap is the overhead a fused kernel
        #or graph capture would recover
        graph_ms = None
        try:
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    slice_inv(*slice_fwd())
            torch.cuda.current_stream().wait_stream(side_stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = slice_inv(*slice_fwd())
            graph_ms = ev_ms(g.replay, REPS_SLICE)
        except Exception as e:
            print(f"cuda graph capture failed ({type(e).__name__}: {e}), "
                  "overhead line below is unavailable")

    floor = fwd_floor + inv_floor
    glue = max(rt_full - floor, 0.0)
    budget = 1000.0 * hop / FS
    print(f"fwd {fwd_full:.3f} ms  inv {inv_full:.3f} ms  roundtrip {rt_full:.3f} ms")
    print(f"real time budget {budget:.0f} ms per slice, margin {budget / rt_full:.0f}x")
    print(f"cufft floor        {floor:.3f} ms  ({100 * floor / rt_full:.0f}% of roundtrip)")
    print(f"glue (window, gather, scatter, rolls)  {glue:.3f} ms  ({100 * glue / rt_full:.0f}%)")
    if graph_ms is not None:
        overhead = max(rt_full - graph_ms, 0.0)
        print(f"graph replay       {graph_ms:.3f} ms, so launch and dispatch overhead "
              f"{overhead:.3f} ms  ({100 * overhead / rt_full:.0f}%)")

    #stft reference, lighter single resolution transform, context not a target
    win = torch.hann_window(2048, device=dev)
    x1 = torch.randn(1, LS, device=dev)
    with torch.no_grad():
        z = torch.stft(x1, 2048, hop_length=512, window=win, center=True, return_complex=True)
        stft_rt = ev_ms(lambda: torch.istft(
            torch.stft(x1, 2048, hop_length=512, window=win, center=True, return_complex=True),
            2048, hop_length=512, window=win, center=True, length=LS), REPS_BUF)
    print(f"\ntorch.stft reference roundtrip (nfft 2048 hop 512, whole buffer) {stft_rt:.2f} ms")

    #verdict
    launch_share = (rt_full - graph_ms) / rt_full if graph_ms is not None else None
    glue_share = glue / rt_full
    if launch_share is not None and launch_share >= 0.3:
        v = (f"verdict launch bound ({100 * launch_share:.0f}% of the per slice roundtrip is "
             "launch and dispatch), a fused kernel (or cuda graphs) would help")
    elif glue_share >= 0.3:
        v = (f"verdict bandwidth bound ({100 * glue_share:.0f}% of the per slice roundtrip is "
             "glue between ffts), a fused kernel keeping intermediates on chip would help")
    else:
        v = (f"verdict cufft bound ({100 * floor / rt_full:.0f}% of the per slice roundtrip is "
             "the ffts themselves), a fused kernel would not help")
    print("\n" + v)


if __name__ == "__main__":
    main()
