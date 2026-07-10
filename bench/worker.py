"""benchmark worker, one config per process.

runs ONE (engine, device) config in its own process so peak rss from
resource.getrusage is attributable to this config alone. prints one json line.

engines
  oct       repo-refs/CQT_pytorch offline oct mode CQT
  slicq     xumx-sliCQ sliced nsgt (streaming reference)
  stft      torch.stft/istft reference at a fixed setting, lighter single
            resolution transform, a reference point not a target
  flash     our sliced flex oct cqt (flash_cqt.SlicedCQT)
  flashoct  our offline flex oct cqt (flash_cqt.OctCQT, one shared fft)
  multicqt  mr cqtdiff multi_CQT, offline flex oct with one cqt and one full
            fft per resolution group, the structure reference

metrics
  snr_raw_db     roundtrip snr vs the raw input
  snr_hpf_db     roundtrip snr vs the dc and nyquist filtered input (oct only,
                 oct mode discards the dc and nyquist bands by design)
  snr_inband_db  spectral snr away from the dc and nyquist seams, this is where
                 the transform is supposed to be exact
  fwd_ms inv_ms rt_ms   median wall times over reps after warmup
  peak_rss_mb    process high water mark resident memory
"""
import sys, json, time, resource, argparse, pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo-refs" / "CQT_pytorch"))
sys.path.insert(0, str(ROOT / "xumx-sliCQ"))


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def snr_db(x, xh):
    x = x.detach().cpu().float()
    xh = xh.detach().cpu().float()
    den = ((x - xh) ** 2).sum().item()
    if den <= 0:
        return float("inf")
    return 10.0 * torch.log10((x ** 2).sum() / den).item()


def snr_inband_db(x, xh, fs, flo, fhi):
    #spectral snr restricted to [flo, fhi], skips the dc and nyquist seams
    x = x.detach().cpu().float()
    xh = xh.detach().cpu().float()
    X = torch.fft.rfft(x, dim=-1)
    Xh = torch.fft.rfft(xh, dim=-1)
    n = x.shape[-1]
    i0, i1 = int(flo * n / fs), int(fhi * n / fs)
    den = ((X - Xh)[..., i0:i1].abs() ** 2).sum().item()
    if den <= 0:
        return float("inf")
    return 10.0 * torch.log10((X[..., i0:i1].abs() ** 2).sum() / den).item()


def timed_ms(fn, dev, reps):
    fn()  #warmup
    sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return round(ts[len(ts) // 2], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True,
                    choices=["oct", "slicq", "stft", "flash", "flashoct", "multicqt"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fs", type=int, default=44100)
    ap.add_argument("--ls", type=int, default=262144)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--numocts", type=int, default=9)
    ap.add_argument("--binsoct", type=int, default=16)
    ap.add_argument("--sllen", type=int, default=0)  #0 means use the suggested value
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = args.device
    out = {"engine": args.engine, "device": dev, "fs": args.fs, "ls": args.ls,
           "batch": args.batch, "dtype": "float32"}

    #in band interval, clear of the dc seam (below fmin) and the nyquist seam
    fmin = args.fs / 2 / 2 ** args.numocts
    flo, fhi = 2 * fmin, 0.95 * args.fs / 2

    try:
        if args.engine == "oct":
            from cqt_nsgt_pytorch import CQT_nsgt
            out["config"] = f"numocts={args.numocts} binsoct={args.binsoct}"
            cqt = CQT_nsgt(args.numocts, args.binsoct, mode="oct", fs=args.fs,
                           audio_len=args.ls, device=dev, dtype=torch.float32)
            x = torch.randn(args.batch, 1, args.ls, device=dev)
            with torch.no_grad():
                c = cqt.fwd(x)
                xh = cqt.bwd(c)[..., :args.ls]
                #oct drops dc and nyquist so also measure vs the filtered input
                xf = cqt.apply_hpf_DC(x)
                xfh = cqt.bwd(cqt.fwd(xf))[..., :args.ls]
            out["snr_raw_db"] = round(snr_db(x, xh), 2)
            out["snr_hpf_db"] = round(snr_db(xf, xfh), 2)
            out["snr_inband_db"] = round(snr_inband_db(x, xh, args.fs, flo, fhi), 2)
            out["coefs"] = sum(t.numel() for t in c) // (args.batch)
            with torch.no_grad():
                out["fwd_ms"] = timed_ms(lambda: cqt.fwd(x), dev, args.reps)
                out["inv_ms"] = timed_ms(lambda: cqt.bwd(c), dev, args.reps)
                out["rt_ms"] = timed_ms(lambda: cqt.bwd(cqt.fwd(x)), dev, args.reps)

        elif args.engine == "slicq":
            from xumx_slicq_v2.nsgt import NSGT_sliced, LogScale
            fmax = args.fs / 2
            fbins = args.numocts * args.binsoct
            scl = LogScale(fmin, fmax, fbins)
            sllen, trlen = scl.suggested_sllen_trlen(args.fs)
            if args.sllen:
                sllen = args.sllen
                trlen = sllen // 4
            out["config"] = f"fbins={fbins} fmin={fmin:.1f} sllen={sllen} trlen={trlen}"
            nsgt = NSGT_sliced(scl, sllen, trlen, args.fs, real=True,
                               multichannel=True, device=dev)
            #slicq has no batch dim, the leading dim of its blocks is the slice index
            x = torch.randn(1, args.ls, device=dev)
            with torch.no_grad():
                c = nsgt.forward((x,))
                #backward mutates its input blocks in place, hand it a deep copy
                xh = nsgt.backward([t.clone() for t in c], args.ls)
            out["snr_raw_db"] = round(snr_db(x, xh), 2)
            out["snr_inband_db"] = round(snr_inband_db(x, xh, args.fs, flo, fhi), 2)
            out["coefs"] = sum(t.numel() for t in c)
            with torch.no_grad():
                out["fwd_ms"] = timed_ms(lambda: nsgt.forward((x,)), dev, args.reps)
                out["inv_ms"] = timed_ms(
                    lambda: nsgt.backward([t.clone() for t in c], args.ls), dev, args.reps)
                out["rt_ms"] = timed_ms(
                    lambda: nsgt.backward(nsgt.forward((x,)), args.ls), dev, args.reps)

        elif args.engine == "flash":
            sys.path.insert(0, str(ROOT))
            from flash_cqt import SlicedCQT
            sl = args.sllen or 65536
            out["config"] = f"num_octs=[3,4,2] bins=[8,16,32] sl_len={sl}"
            cqt = SlicedCQT([3, 4, 2], [8, 16, 32], fs=args.fs, sl_len=sl, device=dev)
            x = torch.randn(args.batch, 1, args.ls, device=dev)
            with torch.no_grad():
                c, side = cqt.fwd(x)
                xh = cqt.bwd(c, side, length=args.ls)
            out["snr_raw_db"] = round(snr_db(x, xh), 2)
            out["snr_inband_db"] = round(snr_inband_db(x, xh, args.fs, flo, fhi), 2)
            out["coefs"] = (sum(t.numel() for t in c) + sum(t.numel() for t in side)) // args.batch
            with torch.no_grad():
                out["fwd_ms"] = timed_ms(lambda: cqt.fwd(x), dev, args.reps)
                out["inv_ms"] = timed_ms(lambda: cqt.bwd(c, side, length=args.ls), dev, args.reps)
                out["rt_ms"] = timed_ms(lambda: cqt.bwd(*cqt.fwd(x), length=args.ls), dev, args.reps)

        elif args.engine == "flashoct":
            sys.path.insert(0, str(ROOT))
            from flash_cqt import OctCQT
            out["config"] = "num_octs=[3,4,2] bins=[8,16,32] offline"
            cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=args.fs, audio_len=args.ls, device=dev)
            x = torch.randn(args.batch, 1, args.ls, device=dev)
            with torch.no_grad():
                c, side = cqt.fwd(x)
                xh = cqt.bwd(c, side)
            out["snr_raw_db"] = round(snr_db(x, xh), 2)
            out["snr_inband_db"] = round(snr_inband_db(x, xh, args.fs, flo, fhi), 2)
            out["coefs"] = (sum(t.numel() for t in c) + sum(t.numel() for t in side)) // args.batch
            with torch.no_grad():
                out["fwd_ms"] = timed_ms(lambda: cqt.fwd(x), dev, args.reps)
                out["inv_ms"] = timed_ms(lambda: cqt.bwd(c, side), dev, args.reps)
                out["rt_ms"] = timed_ms(lambda: cqt.bwd(*cqt.fwd(x)), dev, args.reps)

        elif args.engine == "multicqt":
            #mr cqtdiff resolves utils against other repos on this machine, pin it
            import types
            pkg = types.ModuleType("utils")
            pkg.__path__ = [str(ROOT / "MR-CQTdiff" / "src" / "utils")]
            sys.modules["utils"] = pkg
            from utils.cqt_nsgt_pytorch.multiCQT import multi_CQT
            num_octs = [3, 4, 2]
            fmax = [args.fs * 2 ** (-1 - sum(num_octs[1 + i:])) for i in range(3)]
            out["config"] = "num_octs=[3,4,2] bins=[8,16,32] one cqt per group"
            cqt = multi_CQT(numocts=num_octs, binsoct=[8, 16, 32], fmax=fmax, mode="oct",
                            window="hann", fs=args.fs, audio_len=args.ls,
                            dtype=torch.float32, device=dev)
            x = torch.randn(args.batch, 1, args.ls, device=dev)
            with torch.no_grad():
                c = cqt.fwd(x)
                xh = cqt.bwd(c)[..., :args.ls]
            out["snr_raw_db"] = round(snr_db(x, xh), 2)
            out["snr_inband_db"] = round(snr_inband_db(x, xh, args.fs, flo, fhi), 2)
            out["coefs"] = sum(t.numel() for t in c) // args.batch
            with torch.no_grad():
                out["fwd_ms"] = timed_ms(lambda: cqt.fwd(x), dev, args.reps)
                out["inv_ms"] = timed_ms(lambda: cqt.bwd(c), dev, args.reps)
                out["rt_ms"] = timed_ms(lambda: cqt.bwd(cqt.fwd(x)), dev, args.reps)

        elif args.engine == "stft":
            out["config"] = f"nfft={args.nfft} hop={args.hop}"
            win = torch.hann_window(args.nfft, device=dev)
            x = torch.randn(args.batch, args.ls, device=dev)
            fwd = lambda t: torch.stft(t, args.nfft, hop_length=args.hop,
                                       window=win, center=True, return_complex=True)
            inv = lambda z: torch.istft(z, args.nfft, hop_length=args.hop,
                                        window=win, center=True, length=args.ls)
            with torch.no_grad():
                z = fwd(x)
                xh = inv(z)
            out["snr_raw_db"] = round(snr_db(x, xh), 2)
            out["snr_inband_db"] = round(snr_inband_db(x, xh, args.fs, flo, fhi), 2)
            out["coefs"] = z.numel() // args.batch
            with torch.no_grad():
                out["fwd_ms"] = timed_ms(lambda: fwd(x), dev, args.reps)
                out["inv_ms"] = timed_ms(lambda: inv(z), dev, args.reps)
                out["rt_ms"] = timed_ms(lambda: inv(fwd(x)), dev, args.reps)

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
