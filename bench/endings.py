"""which memory operation the newer triton miscompiles in the inverse scatter.

the inverse two stage kernel runs 6.5x slower than its forward twin under
triton 3.7 and equal to it under 3.1, same source, same card. this probe
times the identical matmul body with six different endings, the ending that
carries the slowdown names the miscompiled pattern.

run   modal run bench/cloud.py --cmd "python bench/endings.py"
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT

import triton
import triton.language as tl

FS = 44100
SL = 65536
B = 32

EPIS = [
    "masked read add write (shipped colour path)",
    "masked atomic add",
    "indexed masked store, no read back",
    "indexed unmasked store",
    "contiguous store, forward style",
    "no store, dots kept alive by a scalar",
]


@triton.jit
def _probe(FRRE, FRIM, KR, KI, DUMR, DUMI,
           F1R, F1I, TWR, TWI, F2R, F2I, GDV, POS, TASK,
           n_fr, n_k,
           N1: tl.constexpr, N2: tl.constexpr, SPLIT: tl.constexpr,
           EPI: tl.constexpr):
    #the inverse two stage body of flash_cqt/fused.py verbatim, only the
    #ending differs by EPI
    pid = tl.program_id(0)
    b = tl.program_id(1)
    row = pid // SPLIT
    sp = pid % SPLIT
    t = TASK + row * 2
    ab = tl.load(t + 0)
    ob = tl.load(t + 1)
    r2 = tl.arange(0, N2)
    r1 = tl.arange(0, N1)
    coff = ob + r2[:, None] + N2 * r1[None, :]
    ar = tl.load(KR + b * n_k + coff)
    ai = tl.load(KI + b * n_k + coff)
    f1r = tl.load(F1R + r1[:, None] * N1 + r1[None, :])
    f1i = tl.load(F1I + r1[:, None] * N1 + r1[None, :])
    br = tl.dot(ar, f1r, input_precision="ieee") - tl.dot(ai, f1i, input_precision="ieee")
    bi = tl.dot(ar, f1i, input_precision="ieee") + tl.dot(ai, f1r, input_precision="ieee")
    twr = tl.load(TWR + r2[:, None] * N1 + r1[None, :])
    twi = tl.load(TWI + r2[:, None] * N1 + r1[None, :])
    cr = br * twr - bi * twi
    ci = br * twi + bi * twr
    ar2 = tl.trans(cr)
    ai2 = tl.trans(ci)
    rk2 = sp * (N2 // SPLIT) + tl.arange(0, N2 // SPLIT)
    f2r = tl.load(F2R + r2[:, None] * N2 + rk2[None, :])
    f2i = tl.load(F2I + r2[:, None] * N2 + rk2[None, :])
    dr = tl.dot(ar2, f2r, input_precision="ieee") - tl.dot(ai2, f2i, input_precision="ieee")
    di = tl.dot(ar2, f2i, input_precision="ieee") + tl.dot(ai2, f2r, input_precision="ieee")
    oo = r1[:, None] + N1 * rk2[None, :]
    if EPI == 0:
        g = tl.load(GDV + ab + oo)
        p = tl.load(POS + ab + oo)
        gm = g != 0.0
        vr = tl.load(FRRE + b * n_fr + p, mask=gm, other=0.0)
        vi = tl.load(FRIM + b * n_fr + p, mask=gm, other=0.0)
        tl.store(FRRE + b * n_fr + p, vr + dr * g, mask=gm)
        tl.store(FRIM + b * n_fr + p, vi + di * g, mask=gm)
    elif EPI == 1:
        g = tl.load(GDV + ab + oo)
        p = tl.load(POS + ab + oo)
        gm = g != 0.0
        tl.atomic_add(FRRE + b * n_fr + p, dr * g, mask=gm)
        tl.atomic_add(FRIM + b * n_fr + p, di * g, mask=gm)
    elif EPI == 2:
        g = tl.load(GDV + ab + oo)
        p = tl.load(POS + ab + oo)
        gm = g != 0.0
        tl.store(FRRE + b * n_fr + p, dr * g, mask=gm)
        tl.store(FRIM + b * n_fr + p, di * g, mask=gm)
    elif EPI == 3:
        g = tl.load(GDV + ab + oo)
        p = tl.load(POS + ab + oo)
        tl.store(FRRE + b * n_fr + p, dr * g)
        tl.store(FRIM + b * n_fr + p, di * g)
    elif EPI == 4:
        g = tl.load(GDV + ab + oo)
        tl.store(DUMR + b * n_k + ob + oo, dr * g)
        tl.store(DUMI + b * n_k + ob + oo, di * g)
    else:
        s = tl.sum(dr) + tl.sum(di)
        tl.store(FRRE + b * n_fr + row, s)


def med(fn, reps=30):
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


def main():
    if not torch.cuda.is_available():
        print("no cuda device found")
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"cuda {torch.version.cuda}  B {B}")
    import triton as tr
    print(f"triton {tr.__version__}\n")
    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=SL, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    x = torch.randn(B, SL, device="cuda")
    w = f._workspace(B)
    with torch.no_grad():
        KR, KI = f.fwd(x)
    t = [t for t in f.two if t["N2"] >= 64][0]
    dumr = torch.zeros(B, f.ncoef, device="cuda")
    dumi = torch.zeros(B, f.ncoef, device="cuda")
    SPLIT = 2
    grid = (t["task"].shape[0] * SPLIT, B)
    from flash_cqt.fused import HALF
    for epi, name in enumerate(EPIS):
        fn = lambda: _probe[grid](
            w["FR"], w["FI"], KR, KI, dumr, dumi,
            t["F1R"], t["F1I"], t["TWR"], t["TWI"], t["F2R"], t["F2I"],
            t["gdv"], t["pos"], t["task"], HALF + 1, f.ncoef,
            N1=t["N1"], N2=t["N2"], SPLIT=SPLIT, EPI=epi,
            num_warps=4, num_stages=2)
        print(f"epi {epi}  {med(fn):>8.3f} ms  {name}")

    #the probe body at split 2 ran 3x faster than the shipped kernel at its
    #baked split 4 on the same card, so either the split (16 wide dots at 4)
    #or the shipped kernel's constexpr branches carry the slowdown, race
    #them at every split
    from flash_cqt.fused import _block_two, N as NN
    print("\nsplit sweep, probe body against the shipped kernel, ms")
    print(f"{'split':>5} | {'probe_epi0':>10} | {'shipped':>8}")
    for sp in (1, 2, 4):
        g2 = (t["task"].shape[0] * sp, B)
        pf = lambda: _probe[g2](
            w["FR"], w["FI"], KR, KI, dumr, dumi,
            t["F1R"], t["F1I"], t["TWR"], t["TWI"], t["F2R"], t["F2I"],
            t["gdv"], t["pos"], t["task"], HALF + 1, f.ncoef,
            N1=t["N1"], N2=t["N2"], SPLIT=sp, EPI=0,
            num_warps=4, num_stages=2)
        rf = lambda: _block_two[g2](
            w["DR"], w["DI"], w["FR"], w["FI"], KR, KI,
            t["pidx"], t["wre"], t["wim"],
            t["F1R"], t["F1I"], t["TWR"], t["TWI"], t["F2R"], t["F2I"],
            t["gdv"], t["pos"], t["task"], NN, HALF + 1, f.ncoef,
            t["scale"], N1=t["N1"], N2=t["N2"], SPLIT=sp,
            ATOMIC=True, IS_FWD=False, PREC="ieee",
            num_warps=4, num_stages=2)
        print(f"{sp:>5} | {med(pf):>10.3f} | {med(rf):>8.3f}")


if __name__ == "__main__":
    main()
