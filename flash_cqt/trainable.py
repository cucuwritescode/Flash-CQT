"""the factored transform as a trainable module.

the transform is linear, so each backward pass is the adjoint run on
the same engines with different tables. backward of the analysis is
the scatter engine with the analysis window and the packed inverse
tail, backward of the synthesis is the router engine with the dual
window record. both adjoints and every baked constant are proven
against torch autograd in bench/autograd_check.py, and the wrapper is
gated on gpu by bench/autograd_gpu.py. nothing is saved for backward,
linear maps need no activations.

  cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=44100, audio_len=65536,
               device="cuda")
  t = TrainableCQT(cqt)
  blocks, side = t.fwd(x)      #x [B, 65536] real, requires_grad ok
  xh = t.bwd(blocks, side)     #gradients flow through both ways
"""
import math

import torch

from .fused import FusedOctCQT, N, HALF, _packedfft_ext, _radix_ext

L = N // 2
R = 256


class TrainableCQT:
    """octcqt shaped fwd and bwd on the factored kernels with gradients"""

    def __init__(self, cqt, device="cuda"):
        dev = torch.device(device)
        self.f = FusedOctCQT(cqt, dev)
        self.cqt = cqt
        self.pext = _packedfft_ext()
        self.xext = _radix_ext()
        assert self.pext and self.xext, "cuda extensions failed to build"

        #packed fft tables, access ordered twiddles per pass
        j = torch.arange(32, dtype=torch.float64)
        F = torch.exp(-2j * math.pi * j[:, None] * j[None, :] / 32)
        F = F.to(torch.complex64).to(dev)
        self.FR = F.real.contiguous()
        self.FI = F.imag.contiguous()
        kL = torch.arange(L, dtype=torch.float64)
        WL = torch.exp(-2j * math.pi * kL / L).to(torch.complex64).to(dev)
        k1 = torch.arange(1024, dtype=torch.float64)
        W1 = torch.exp(-2j * math.pi * k1 / 1024).to(torch.complex64).to(dev)
        b1024 = torch.arange(1024, device=dev)
        lane = torch.arange(32, device=dev)
        WLp = WL[(b1024[:, None] * lane[None, :]) % L].reshape(-1)
        g32 = torch.arange(32, device=dev)
        WKp = W1[(g32[:, None] * lane[None, :]) % 1024].reshape(-1)
        self.WLR = WLp.real.contiguous()
        self.WLI = WLp.imag.contiguous()
        self.WKR = WKp.real.contiguous()
        self.WKI = WKp.imag.contiguous()
        qq = torch.arange(L, device=dev, dtype=torch.float64)
        WN = torch.exp(-2j * math.pi * qq / N).to(torch.complex64)
        self.WNR = WN.real.contiguous()
        self.WNI = WN.imag.contiguous()
        self.dev = dev
        self.scr = {}

    def _scratch(self, B):
        #scratch planes reused between calls, outputs are always fresh
        if B not in self.scr:
            z = lambda *s: torch.empty(*s, device=self.dev)
            self.scr[B] = dict(zr=z(B, L), zi=z(B, L), t1r=z(B, L),
                               t1i=z(B, L), t2r=z(B, L), t2i=z(B, L),
                               zpr=z(B, L), zpi=z(B, L),
                               fr=z(B, HALF + 1), fi=z(B, HALF + 1))
        return self.scr[B]

    def _analysis(self, x):
        B = x.shape[0]
        w = self._scratch(B)
        rx = self.f.radix
        kr = torch.empty(B, self.f.ncoef, device=self.dev)
        ki = torch.empty(B, self.f.ncoef, device=self.dev)
        w["zr"].copy_(x[..., 0::2])
        w["zi"].copy_(x[..., 1::2])
        self.pext.fft3(w["zr"], w["zi"], w["t1r"], w["t1i"], w["t2r"],
                       w["t2i"], w["zpr"], w["zpi"], self.FR, self.FI,
                       self.WLR, self.WLI, self.WKR, self.WKI, B)
        self.xext.fwd_bands(w["zpr"], w["zpi"], kr, ki, rx["route"],
                            rx["desc"], rx["rr"], rx["ri"],
                            self.f.ncoef, B)
        return kr, ki

    def _tail(self, B, w):
        xo = torch.empty(B, N, device=self.dev)
        self.pext.inv_tail(w["fr"], w["fi"], w["zr"], w["zi"], w["t1r"],
                           w["t1i"], w["t2r"], w["t2i"], w["zpr"],
                           w["zpi"], xo, self.FR, self.FI, self.WLR,
                           self.WLI, self.WKR, self.WKI, self.WNR,
                           self.WNI, B)
        return xo

    def _synthesis(self, kr, ki):
        B = kr.shape[0]
        w = self._scratch(B)
        rx = self.f.radix
        w["fr"].zero_()
        w["fi"].zero_()
        self.xext.inv_bands(kr, ki, w["fr"], w["fi"], rx["gdv"],
                            rx["pos"], rx["desc"], rx["rr"], rx["ri"],
                            self.f.ncoef, B, 2)
        return self._tail(B, w)

    def _adj_analysis(self, gkr, gki):
        #gradient of the analysis, signed scatter then the inverse tail
        B = gkr.shape[0]
        w = self._scratch(B)
        rx = self.f.radix
        w["fr"].zero_()
        w["fi"].zero_()
        self.xext.adj_bands(gkr.contiguous(), gki.contiguous(), w["fr"],
                            w["fi"], rx["agr"], rx["agi"], rx["aidx"],
                            rx["desc"], rx["rr"], rx["ri"],
                            self.f.ncoef, B)
        return self._tail(B, w)

    def _adj_synthesis(self, gx):
        #gradient of the synthesis, the router with the dual record
        B = gx.shape[0]
        w = self._scratch(B)
        rx = self.f.radix
        gkr = torch.empty(B, self.f.ncoef, device=self.dev)
        gki = torch.empty(B, self.f.ncoef, device=self.dev)
        w["zr"].copy_(gx[..., 0::2])
        w["zi"].copy_(gx[..., 1::2])
        self.pext.fft3(w["zr"], w["zi"], w["t1r"], w["t1i"], w["t2r"],
                       w["t2i"], w["zpr"], w["zpi"], self.FR, self.FI,
                       self.WLR, self.WLI, self.WKR, self.WKI, B)
        self.xext.fwd_bands(w["zpr"], w["zpi"], gkr, gki, rx["route_b"],
                            rx["desc"], rx["rr"], rx["ri"],
                            self.f.ncoef, B)
        return gkr, gki

    def fwd(self, x):
        """x [..., N] real to (blocks, side), complex, gradients flow,
        any leading shape is preserved on the blocks"""
        lead = x.shape[:-1]
        kr, ki = _Analysis.apply(x.reshape(-1, N).contiguous(), self)
        vs = []
        for b, M, off in self.f.gspec:
            vs.append(torch.complex(kr[:, off:off + b * M],
                                    ki[:, off:off + b * M])
                      .view(*lead, b, M))
        ng = len(self.cqt.groups)
        return vs[:ng], vs[ng:]

    def bwd(self, blocks, side=None):
        """inverse of fwd, gradients flow"""
        allb = list(blocks) + list(side or [])
        lead = allb[0].shape[:-2]
        B = 1
        for d in lead:
            B *= d
        flat = torch.cat([c.reshape(B, -1) for c in allb], dim=-1)
        return _Synthesis.apply(flat.real.contiguous(),
                                flat.imag.contiguous(),
                                self).reshape(*lead, N)


class _Analysis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eng):
        ctx.eng = eng
        with torch.no_grad():
            return eng._analysis(x)

    @staticmethod
    def backward(ctx, gkr, gki):
        with torch.no_grad():
            return ctx.eng._adj_analysis(gkr, gki), None


class _Synthesis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kr, ki, eng):
        ctx.eng = eng
        with torch.no_grad():
            return eng._synthesis(kr, ki)

    @staticmethod
    def backward(ctx, gx):
        with torch.no_grad():
            gkr, gki = ctx.eng._adj_synthesis(gx.contiguous())
            return gkr, gki, None
