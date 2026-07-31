"""single kernel probe for the block scatter kernels.

run   python bench/probe.py            per kernel medians at batch 32
run   python bench/probe.py --B 8      other batch sizes
run   python bench/probe.py --once     one inverse pass only, the ncu target

the interesting rows are fwd two 32x64 and inv two 32x64, the same matmul
shape in both directions, forward healthy, inverse sick on the a100 under
the newer triton and healthy on the v100 under the older one. profile the
inverse with

  ncu --kernel-name regex:_block_two --launch-count 6 \
      --section MemoryWorkloadAnalysis --section SchedulerStats \
      python bench/probe.py --once

if ncu reports ERR_NVGPUCTRPERM the cluster locks the counters, skip it.
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT

FS = 44100
SL = 65536


def main():
    if not torch.cuda.is_available():
        print("no cuda device found")
        sys.exit(1)
    B = 32
    if "--B" in sys.argv:
        B = int(sys.argv[sys.argv.index("--B") + 1])
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"cuda {torch.version.cuda}  B {B}")
    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=SL, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    x = torch.randn(B, SL, device="cuda")
    w = f._workspace(B)
    with torch.no_grad():
        KR, KI = f.fwd(x)
        f.bwd(KR, KI)
        if "--once" in sys.argv:
            #one pass of the widest inverse bucket only, for the profiler
            t = [t for t in f.two if t["N2"] >= 64][0]
            f._inv_two(t, B, w, KR, KI, "ieee")
            torch.cuda.synchronize()
            print("one inverse pass of the 32x64 bucket done")
            return
        parts = f.bench_parts(x, reps=30)
    for name, ms in parts:
        mark = "  <--" if name.endswith("two 32x64") else ""
        print(f"{name:<16} {ms:>8.3f} ms{mark}")


if __name__ == "__main__":
    main()
