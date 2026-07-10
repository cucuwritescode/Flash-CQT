"""baseline table. spawns worker.py once per config and prints a table.

run   python bench/baselines.py
"""
import sys, json, subprocess, pathlib
import torch

HERE = pathlib.Path(__file__).resolve().parent
WORKER = str(HERE / "worker.py")

#one row per (engine, device), fs 44100 and ls 262144 to match the mr cqtdiff target
CONFIGS = []
for dev in ["cpu", "mps"]:
    CONFIGS.append(["--engine", "oct", "--device", dev])
    CONFIGS.append(["--engine", "slicq", "--device", dev])
    CONFIGS.append(["--engine", "stft", "--device", dev])


def run_one(cfg):
    out = subprocess.run([sys.executable, WORKER] + cfg, capture_output=True, text=True)
    for line in reversed(out.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"engine": cfg[1], "device": cfg[3], "error": (out.stderr or "no output").strip()[-200:]}


def fmt(v, w):
    if v is None:
        return " " * (w - 1) + "-"
    if isinstance(v, float):
        return f"{v:>{w}.2f}"
    return f"{v:>{w}}"


def main():
    mps = torch.backends.mps.is_available()
    print(f"torch={torch.__version__}  mps_available={mps}  fs=44100 ls=262144 batch=1 dtype=float32")
    print(f"{'engine':<7} {'device':<6} | {'snr_raw':>8} {'snr_hpf':>8} {'inband':>8} | "
          f"{'fwd_ms':>8} {'inv_ms':>8} {'rt_ms':>8} | {'peak_MB':>8} | config")
    print("-" * 110)
    for cfg in CONFIGS:
        if cfg[3] == "mps" and not mps:
            continue
        r = run_one(cfg)
        if "error" in r:
            print(f"{r.get('engine','?'):<7} {r.get('device','?'):<6} | ERROR {r['error']}")
            continue
        print(f"{r['engine']:<7} {r['device']:<6} | "
              f"{fmt(r.get('snr_raw_db'), 8)} {fmt(r.get('snr_hpf_db'), 8)} {fmt(r.get('snr_inband_db'), 8)} | "
              f"{fmt(r.get('fwd_ms'), 8)} {fmt(r.get('inv_ms'), 8)} {fmt(r.get('rt_ms'), 8)} | "
              f"{fmt(r.get('peak_rss_mb'), 8)} | {r.get('config','')}")
    print()
    print("snr_raw is vs the raw input. oct discards dc and nyquist bands by design, so its")
    print("raw snr is dominated by the seams below fmin and near nyquist. snr_hpf is vs the")
    print("dc and nyquist filtered input, inband is the spectral snr on [2*fmin, 0.95*nyq]")
    print("where the transform is meant to be exact. stft is a lighter single resolution")
    print("reference, not a target. all timings are this machine (cpu or mps), cuda pending.")


if __name__ == "__main__":
    main()
