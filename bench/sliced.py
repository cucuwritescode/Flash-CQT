"""verification and benchmark for the streamable sliced oct cqt.

run   python bench/sliced.py

checks, in order
  1 structure, offline flex oct blocks against mr cqtdiff multi_CQT at 262144
  2 correctness, octave blocks bit identical to repo-refs oct in the uniform
    config, round trip snr offline and sliced, with and without the dc and
    nyquist side blocks
  3 streaming, chunk by chunk output equal to the whole buffer path, state size
  4 latency against the lowest usable fmin, sl_len is the knob
  5 speed and memory per engine via bench/worker.py subprocesses
"""
import sys, json, subprocess, pathlib, types
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repo-refs" / "CQT_pytorch"))
WORKER = str(HERE / "worker.py")

FS, LS, SL = 44100, 262144, 65536
NUM_OCTS, BINS = [3, 4, 2], [8, 16, 32]


def snr_db(x, xh):
    x, xh = x.cpu().float(), xh.cpu().float()
    return (10 * torch.log10((x ** 2).sum() / ((x - xh) ** 2).sum())).item()


def snr_inband_db(x, xh, flo, fhi):
    X = torch.fft.rfft(x.cpu().float(), dim=-1)
    Xh = torch.fft.rfft(xh.cpu().float(), dim=-1)
    n = x.shape[-1]
    i0, i1 = int(flo * n / FS), int(fhi * n / FS)
    return (10 * torch.log10((X[..., i0:i1].abs() ** 2).sum()
                             / ((X - Xh)[..., i0:i1].abs() ** 2).sum())).item()


def check_structure():
    print("== 1 structure against mr cqtdiff multi_CQT ==")
    from flash_cqt import OctCQT
    pkg = types.ModuleType("utils")
    pkg.__path__ = [str(ROOT / "MR-CQTdiff" / "src" / "utils")]
    sys.modules["utils"] = pkg
    from utils.cqt_nsgt_pytorch.multiCQT import multi_CQT
    fmax = [FS * 2 ** (-1 - sum(NUM_OCTS[1 + i:])) for i in range(3)]
    ref = multi_CQT(numocts=NUM_OCTS, binsoct=BINS, fmax=fmax, mode="oct",
                    window="hann", fs=FS, audio_len=LS, dtype=torch.float32, device="cpu")
    ours = OctCQT(NUM_OCTS, BINS, fs=FS, audio_len=LS, device="cpu")
    x = torch.randn(1, 1, LS)
    with torch.no_grad():
        cr = ref.fwd(x)
        co, _ = ours.fwd(x)
    ok = len(cr) == len(co) and all(a.shape == b.shape and a.dtype == b.dtype
                                    for a, b in zip(cr, co))
    print("multi_CQT blocks", [tuple(t.shape) for t in cr])
    print("ours      blocks", [tuple(t.shape) for t in co])
    print("MATCH" if ok else "MISMATCH")
    return ok


def check_correctness():
    print("\n== 2 correctness ==")
    from flash_cqt import OctCQT, SlicedCQT
    from cqt_nsgt_pytorch import CQT_nsgt
    torch.manual_seed(0)
    x = torch.randn(1, 1, LS)
    fmin = FS / 2 / 2 ** 9
    ok = True

    #uniform config, octave blocks must equal the reference oct forward bitwise
    ref = CQT_nsgt(9, 16, mode="oct", fs=FS, audio_len=LS, device="cpu", dtype=torch.float32)
    ours = OctCQT(9, 16, fs=FS, audio_len=LS, device="cpu")
    with torch.no_grad():
        cr = ref.fwd(x)
        co, side = ours.fwd(x)
    md = max((a - b).abs().max().item() for a, b in zip(cr, co))
    print(f"uniform 9x16 max|ours-ref| = {md}  ->", "BIT-IDENTICAL" if md == 0 else "CHECK")
    ok &= md == 0

    #offline flex round trip
    o = OctCQT(NUM_OCTS, BINS, fs=FS, audio_len=LS, device="cpu")
    with torch.no_grad():
        c, side = o.fwd(x)
        s_with = snr_db(x, o.bwd(c, side))
        s_wo = snr_db(x, o.bwd(c))
    print(f"offline flex roundtrip: {s_with:.2f} db with side blocks, {s_wo:.2f} db without")
    ok &= s_with > 120

    #sliced flex round trip
    s = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device="cpu")
    with torch.no_grad():
        blocks, side = s.fwd(x)
        xh = s.bwd(blocks, side, length=LS)
        xh2 = s.bwd(blocks, None, length=LS)
    print(f"sliced roundtrip sl_len={SL}: {snr_db(x, xh):.2f} db raw with side "
          f"({snr_inband_db(x, xh, 2 * fmin, .95 * FS / 2):.2f} db inband), "
          f"{snr_db(x, xh2):.2f} db raw without ({snr_inband_db(x, xh2, 2 * fmin, .95 * FS / 2):.2f} db inband)")
    ok &= snr_db(x, xh) > 120

    #gradients reach the input
    xg = torch.randn(1, 1, LS, requires_grad=True)
    loss = sum((t.abs() ** 2).sum() for t in s.fwd(xg)[0])
    loss.backward()
    gok = xg.grad is not None and torch.isfinite(xg.grad).all().item()
    print("grad_ok:", gok)
    return ok and gok


def check_streaming():
    print("\n== 3 streaming equals whole buffer ==")
    from flash_cqt import SlicedCQT
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    ok = True
    for dev in devs:
        torch.manual_seed(0)
        s = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device=dev)
        x = torch.randn(1, 1, LS, device=dev)
        with torch.no_grad():
            blocks, side = s.fwd(x)
            chunks = [x[..., i * s.hop:(i + 1) * s.hop] for i in range(LS // s.hop)]
            out = list(s.stream_fwd(iter(chunks)))
            xs = torch.cat(list(s.stream_bwd(iter(out))), dim=-1)[..., :LS]
            xh = s.bwd(blocks, side, length=LS)
        m = s.size_per_oct
        md = max((c - blocks[i][..., k * m[i]:(k + 1) * m[i]]).abs().max().item()
                 for k, (bl, sd) in enumerate(out) for i, c in enumerate(bl))
        rd = (xs - xh).abs().max().item()
        state = 2 * s.hop * 4
        print(f"{dev}: slices={len(out)} max|stream-batch| coefs={md:.2e} recon={rd:.2e} "
              f"stream snr={snr_db(x, xs):.2f} db  state={state} bytes (B=C=1 fp32)")
        ok &= md < 1e-5 and rd < 1e-5
    return ok


def show_latency():
    print("\n== 4 latency against lowest fmin ==")
    from flash_cqt import SlicedCQT
    s = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device="cpu")
    print(s.latency_table((4096, 8192, 16384, 32768, 65536, 131072)))
    print(f"chosen default sl_len={SL} -> latency {s.latency_ms():.1f} ms, "
          f"covers the full {sum(NUM_OCTS)} octaves down to {s.frqs_hz[0]:.1f} hz")


def run_bench():
    print("\n== 5 speed and memory (subprocess per config) ==")
    print(f"{'engine':<9} {'device':<6} | {'snr_raw':>8} {'inband':>8} | "
          f"{'fwd_ms':>8} {'inv_ms':>8} {'rt_ms':>8} | {'peak_MB':>8}")
    print("-" * 84)
    mps = torch.backends.mps.is_available()
    for dev in ["cpu"] + (["mps"] if mps else []):
        for eng in ["flash", "flashoct", "multicqt", "slicq"]:
            cfg = ["--engine", eng, "--device", dev]
            out = subprocess.run([sys.executable, WORKER] + cfg, capture_output=True, text=True)
            r = None
            for line in reversed(out.stdout.strip().splitlines()):
                try:
                    r = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
            if r is None or "error" in (r or {}):
                msg = (r or {}).get("error", (out.stderr or "no output").strip()[-160:])
                print(f"{eng:<9} {dev:<6} | ERROR {msg}")
                continue
            print(f"{r['engine']:<9} {r['device']:<6} | {r.get('snr_raw_db', 0):>8.2f} "
                  f"{r.get('snr_inband_db', 0):>8.2f} | {r.get('fwd_ms', 0):>8.2f} "
                  f"{r.get('inv_ms', 0):>8.2f} {r.get('rt_ms', 0):>8.2f} | "
                  f"{r.get('peak_rss_mb', 0):>8.1f}")


def main():
    print(f"torch={torch.__version__} fs={FS} ls={LS} num_octs={NUM_OCTS} bins={BINS}\n")
    ok = check_structure()
    ok &= check_correctness()
    ok &= check_streaming()
    show_latency()
    run_bench()
    print("\nresult:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
