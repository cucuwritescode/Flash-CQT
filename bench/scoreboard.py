"""the streamable head to head, one card, one run.

  modal run bench/cloud.py --cuda --cmd "python bench/scoreboard.py"

whole buffer 262144 at 44100, B 1, the mr cqtdiff config. slicq gets
sl_len 65536 to match ours, same latency. our sliced path runs twice,
once on the shipped per slice engine and once with the factored radix
engine dropped in as the per slice transform, so the multiplier over
slicq is a same run measurement, not arithmetic across containers.
slicq has no graph mode so every row is timed eagerly, ours also
reports the graphed replay it actually ships with, replay checked
against eager before it counts.
"""
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "xumx-sliCQ"))

from flash_cqt import SlicedCQT
from flash_cqt.fused import FusedOctCQT, N, HALF, _packedfft_ext, _radix_ext
import packedfft as pf

FS = 44100
LS = 262144
SL = 65536
L = N // 2
NUM_OCTS = [3, 4, 2]
BINS = [8, 16, 32]
FMIN = FS / 2 / 2 ** 9


def snr_db(x, xh):
    x, xh = x.float(), xh.float()
    return (10 * torch.log10((x ** 2).sum() / ((x - xh) ** 2).sum())).item()


def ev_ms(fn, reps=50):
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


class FactoredEngine:
    """octcqt shaped fwd and bwd on the factored radix kernels, so the
    sliced wrapper can use the new engine as its per slice transform"""

    def __init__(self, f):
        self.f = f
        self.cqt = f.cqt
        pext = _packedfft_ext()
        self.pext = pext
        self.xext = _radix_ext()
        F, WL, W1k = pf.tables("cuda")
        self.FR = F.real.contiguous()
        self.FI = F.imag.contiguous()
        b1024 = torch.arange(1024, device="cuda")
        lane = torch.arange(32, device="cuda")
        WLp = WL[(b1024[:, None] * lane[None, :]) % L].reshape(-1)
        g32 = torch.arange(32, device="cuda")
        WKp = W1k[(g32[:, None] * lane[None, :]) % 1024].reshape(-1)
        self.WLR, self.WLI = WLp.real.contiguous(), WLp.imag.contiguous()
        self.WKR, self.WKI = WKp.real.contiguous(), WKp.imag.contiguous()
        qq = torch.arange(L, device="cuda", dtype=torch.float64)
        WN = torch.exp(-2j * math.pi * qq / N).to(torch.complex64)
        self.WNR, self.WNI = WN.real.contiguous(), WN.imag.contiguous()
        self.ws = {}

    def _buf(self, B):
        if B not in self.ws:
            z = lambda *s: torch.empty(*s, device="cuda")
            self.ws[B] = dict(
                zr=z(B, L), zi=z(B, L), t1r=z(B, L), t1i=z(B, L),
                t2r=z(B, L), t2i=z(B, L), zpr=z(B, L), zpi=z(B, L),
                kr=z(B, self.f.ncoef), ki=z(B, self.f.ncoef),
                fr=z(B, HALF + 1), fi=z(B, HALF + 1), xo=z(B, N))
        return self.ws[B]

    def fwd(self, x):
        lead = x.shape[:-1]
        B = int(torch.tensor(lead).prod()) if lead else 1
        xx = x.reshape(B, N)
        w = self._buf(B)
        rx = self.f.radix
        w["zr"].copy_(xx[..., 0::2])
        w["zi"].copy_(xx[..., 1::2])
        self.pext.fft3(w["zr"], w["zi"], w["t1r"], w["t1i"], w["t2r"],
                       w["t2i"], w["zpr"], w["zpi"], self.FR, self.FI,
                       self.WLR, self.WLI, self.WKR, self.WKI, B)
        self.xext.fwd_bands(w["zpr"], w["zpi"], w["kr"], w["ki"],
                            rx["route"], rx["desc"], rx["rr"], rx["ri"],
                            self.f.ncoef, B)
        vs = self.f.blocks_view(w["kr"], w["ki"])
        ng = len(self.cqt.groups)
        blocks = [v.reshape(*lead, *v.shape[1:]) for v in vs[:ng]]
        side = [v.reshape(*lead, *v.shape[1:]) for v in vs[ng:]]
        return blocks, side

    def bwd(self, blocks, side=None):
        lead = blocks[0].shape[:-2]
        B = int(torch.tensor(lead).prod()) if lead else 1
        w = self._buf(B)
        rx = self.f.radix
        #pack the block structure back into the flat heap
        allb = list(blocks) + list(side or [])
        for v, (b, M, off) in zip(allb, self.f.gspec):
            c = v.reshape(B, b * M)
            w["kr"][:, off:off + b * M] = c.real
            w["ki"][:, off:off + b * M] = c.imag
        w["fr"].zero_()
        w["fi"].zero_()
        self.xext.inv_bands(w["kr"], w["ki"], w["fr"], w["fi"],
                            rx["gdv"], rx["pos"], rx["desc"], rx["rr"],
                            rx["ri"], self.f.ncoef, B, 2)
        self.pext.inv_tail(w["fr"], w["fi"], w["zr"], w["zi"], w["t1r"],
                           w["t1i"], w["t2r"], w["t2i"], w["zpr"],
                           w["zpi"], w["xo"], self.FR, self.FI,
                           self.WLR, self.WLI, self.WKR, self.WKI,
                           self.WNR, self.WNI, B)
        return w["xo"].reshape(*lead, N)


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
    print("gpu", torch.cuda.get_device_name(0))
    torch.manual_seed(0)
    x3 = torch.randn(1, 1, LS, device="cuda")
    rows = []

    with torch.no_grad():
        #ours on the shipped per slice engine
        c = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device="cuda")
        xh = c.bwd(*c.fwd(x3), length=LS)
        rows.append(("flash cqt, shipped engine", snr_db(x3, xh),
                     ev_ms(lambda: c.bwd(*c.fwd(x3), length=LS)), None))

        #ours with the factored radix engine per slice
        f = FusedOctCQT(c.cqt, "cuda")
        f.prec = "ieee"
        c.cqt = FactoredEngine(f)
        xh = c.bwd(*c.fwd(x3), length=LS)
        vr = snr_db(x3, xh)
        e = ev_ms(lambda: c.bwd(*c.fwd(x3), length=LS))
        xh0 = xh.clone()
        g = capture(lambda: c.bwd(*c.fwd(x3), length=LS))
        gm = ev_ms(g.replay)
        #replay integrity before the graphed number counts
        rdb = snr_db(xh0, c.bwd(*c.fwd(x3), length=LS))
        gdb = "ok" if rdb > 118 else f"REPLAY {rdb:.1f} db"
        rows.append(("flash cqt, radix engine", vr, e, (gm, gdb)))

        #slicq, the only other streamable invertible cqt
        try:
            from xumx_slicq_v2.nsgt import NSGT_sliced, LogScale
            scl = LogScale(FMIN, FS / 2, 9 * 16)
            n = NSGT_sliced(scl, SL, SL // 4, FS, real=True,
                            multichannel=True, device="cuda")
            x2 = x3[:, 0]
            b = n.forward((x2,))
            xh = n.backward([t.clone() for t in b], LS)
            rows.append(("slicq 9x16", snr_db(x2, xh),
                         ev_ms(lambda: n.backward(n.forward((x2,)), LS)),
                         None))
        except Exception as ex:
            print("slicq skipped,", ex)

    print(f"\n{'engine':<28} | {'snr_db':>7} | {'eager_ms':>9} | graphed")
    print("-" * 64)
    base = None
    for nm, s_, e, g in rows:
        gtxt = f"{g[0]:.3f} ({g[1]})" if g else "-"
        print(f"{nm:<28} | {s_:>7.1f} | {e:>9.3f} | {gtxt}")
        if "radix" in nm:
            base = (e, g[0] if g else e)
    for nm, s_, e, g in rows:
        if "slicq" in nm and base:
            print(f"\nslicq vs radix engine, eager {e / base[0]:.0f}x, "
                  f"against our graphed {e / base[1]:.0f}x")
    print("SCOREBOARD-DONE")


if __name__ == "__main__":
    main()
