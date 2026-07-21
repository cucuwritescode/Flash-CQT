#facundo franchino, july 2026
"""fused triton per slice oct cqt.

the eager per slice transform is roughly 22 small kernels, each paying launch
latency and a full HBM round trip. here it is reduced to 12 fat ones (6 per direction),
graph capturable to one replay. every fft is cooley tukey matmul stages, the
math validated in bench/monarch.py, the fusion layout inspiration is borrowed from
flashfftconv (one kernel does load, matmul stages, pointwise and store, and
the ragged gather indices are precomputed into the mid decomposition layout
so the kernel reads them contiguously).
forward, per slice of 65536 samples
  k1  stage one of the big fft, real input, 256 point dft matmul plus twiddle
  k2  stage two, another 256 point dft matmul, spectrum kept in matrix layout
  k3  one launch per block family, gather from the spectrum matrix, multiply
      the analysis window, inverse dft (direct matrix up to 256, two matmul
      stages above), write the coefficient blocks
inverse mirrors it, dft of each block, dual window, atomic scatter into the
half spectrum, then the big inverse fft with hermitian completion done as
index arithmetic on load, nothing is materialised twice basically..

precision is a knob on the dots, "ieee" is fp32, "tf32" and "tf32x3" run the
tensor cores (tf32x3 held 116.6 db in the matmul form measurement).

everything heavy is precomputed in torch at init, and sim_fwd and sim_bwd
mirror the kernel dataflow op for op with the same tables, so the tables and
the math can be verified on any device without triton. triton itself is
imported lazily, this module works on cpu for table building and simulation.

flashfftconv fusion layout inspo is courtesy of the people at Hazy Research lab at Stanford
"""
import math
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

N = 65536
R = 256  #big fft factor, 65536 = 256 * 256
HALF = N // 2


def dft_planes(M):
    #forward dft matrix as float32 re and im planes, angles from integer
    #residues in float64 so the tables are exact to fp32
    n = torch.arange(M, dtype=torch.int64)
    e = (n[:, None] * n[None, :]) % M
    ang = -2.0 * math.pi * e.to(torch.float64) / M
    return torch.cos(ang).float(), torch.sin(ang).float()


def twiddle_planes(N1, N2):
    #T[n2, k1] = W_{N1 N2}^{n2 k1}
    n2 = torch.arange(N2, dtype=torch.int64)[:, None]
    k1 = torch.arange(N1, dtype=torch.int64)[None, :]
    e = (n2 * k1) % (N1 * N2)
    ang = -2.0 * math.pi * e.to(torch.float64) / (N1 * N2)
    return torch.cos(ang).float(), torch.sin(ang).float()


def split(M):
    N1 = 1 << (int(math.log2(M)) // 2)
    return N1, M // N1


class FusedOctCQT:
    """kernel tables and launchers for one OctCQT at audio_len 65536"""

    def __init__(self, cqt, device=None):
        assert cqt.nn == N, "fused path is built for sl_len 65536"
        dev = torch.device(device) if device is not None else cqt.device
        self.dev = dev
        self.cqt = cqt

        #group list in coefficient buffer order, octaves then dc then nyquist
        self.gspec = []  #(bins, M, coef_off)
        off = 0
        groups = list(cqt.groups) + list(cqt.side_groups)
        for idx, gv, pos, gdv, conj in groups:
            b, M = idx.shape
            self.gspec.append((b, M, off))
            off += b * M
        self.ncoef = off

        #big fft tables
        f1r, f1i = dft_planes(R)
        twr, twi = twiddle_planes(R, R)
        self.F1R, self.F1I = f1r.to(dev), f1i.to(dev)
        self.TWR, self.TWI = twr.to(dev), twi.to(dev)

        #spectrum index k maps to the matrix layout D[k1, k2] flat at k1*R + k2
        def dmap(k):
            return ((k % R) * R + k // R).to(torch.int32)

        #direct family, M up to 256, one dft matrix each
        #two stage family, M in 512 1024 2048, bucketed by factor pair
        d_pidx, d_wre, d_wim, d_gdv, d_pos, d_wir, d_wii, d_fmr, d_fmi = ([] for _ in range(9))
        d_rows = []
        two = {}
        p_off = 0
        m_off = 0
        BN = 64
        for gi, (idx, gv, pos, gdv, conj) in enumerate(groups):
            b, M = idx.shape
            pidx = dmap(idx.reshape(-1).cpu())
            gvf = gv.reshape(-1).float().cpu()
            sgn = torch.where(conj.reshape(-1).cpu(), -1.0, 1.0)
            gdvf = gdv.reshape(-1).float().cpu()
            posf = pos.reshape(-1).to(torch.int32).cpu()
            cbase = self.gspec[gi][2]
            if M <= 256:
                fr, fi = dft_planes(M)
                #inverse dft folded into the matrix, conj(F)/M
                d_wir.append((fr / M).reshape(-1))
                d_wii.append((-fi / M).reshape(-1))
                d_fmr.append(fr.reshape(-1))
                d_fmi.append(fi.reshape(-1))
                d_pidx.append(pidx)
                d_wre.append(gvf)
                d_wim.append(gvf * sgn)
                d_gdv.append(gdvf)
                d_pos.append(posf)
                for b0 in range(0, b, 16):
                    nb = min(16, b - b0)
                    for n0 in range(0, M, BN):
                        d_rows.append([p_off + b0 * M, m_off,
                                       cbase + b0 * M, M, nb, n0])
                p_off += b * M
                m_off += M * M
            else:
                N1, N2 = split(M)
                key = (N1, N2)
                if key not in two:
                    g1 = dft_planes(N1)
                    g2 = dft_planes(N2)
                    tw = twiddle_planes(N1, N2)
                    two[key] = dict(F1R=g1[0].to(dev), F1I=g1[1].to(dev),
                                    F2R=g2[0].to(dev), F2I=g2[1].to(dev),
                                    TWR=tw[0].to(dev), TWI=tw[1].to(dev),
                                    scale=1.0 / M,
                                    pidx=[], wre=[], wim=[], gdv=[], pos=[],
                                    rows=[], t_off=0)
                t = two[key]
                t["pidx"].append(pidx)
                t["wre"].append(gvf)
                t["wim"].append(gvf * sgn)
                t["gdv"].append(gdvf)
                t["pos"].append(posf)
                for bi in range(b):
                    t["rows"].append([t["t_off"] + bi * M, cbase + bi * M])
                t["t_off"] += b * M

        def cat(ts, dt):
            return torch.cat(ts).to(dt).to(dev) if ts else torch.zeros(0, dtype=dt, device=dev)

        self.d_pidx = cat(d_pidx, torch.int32)
        self.d_wre = cat(d_wre, torch.float32)
        self.d_wim = cat(d_wim, torch.float32)
        self.d_gdv = cat(d_gdv, torch.float32)
        self.d_pos = cat(d_pos, torch.int32)
        self.d_wir = cat(d_wir, torch.float32)
        self.d_wii = cat(d_wii, torch.float32)
        self.d_fmr = cat(d_fmr, torch.float32)
        self.d_fmi = cat(d_fmi, torch.float32)
        self.d_task = torch.tensor(d_rows, dtype=torch.int32, device=dev)

        self.two = []
        for (N1, N2), t in sorted(two.items()):
            #wide buckets split each bin's second stage across programs so
            #the launch fills more of the card, 33 programs was not enough.
            #split is the forward setting, spliti the inverse one, the sweep
            #tunes them apart because the inverse scatters through atomics
            self.two.append(dict(
                N1=N1, N2=N2, scale=t["scale"],
                split=4 if N2 >= 64 else 1, spliti=4 if N2 >= 64 else 1,
                spliti_big=1, inv_atomic=False, inv_atomic_big=False,
                F1R=t["F1R"], F1I=t["F1I"], F2R=t["F2R"], F2I=t["F2I"],
                TWR=t["TWR"], TWI=t["TWI"],
                pidx=cat(t["pidx"], torch.int32),
                wre=cat(t["wre"], torch.float32),
                wim=cat(t["wim"], torch.float32),
                gdv=cat(t["gdv"], torch.float32),
                pos=cat(t["pos"], torch.int32),
                task=torch.tensor(t["rows"], dtype=torch.int32, device=dev)))

        #colour the bands of each two stage bucket so no two bands in one
        #class share a scatter bin, then the inverse runs one plain read add
        #write launch per class instead of atomics. the inverse 2048 bucket
        #measured 10x its forward twin under atomic contention. supports are
        #contiguous and only neighbours overlap, so two classes suffice,
        #the assert below proves it on every build
        for t in self.two:
            rows = t["task"].tolist()
            M = t["N1"] * t["N2"]
            spans = []
            for ab, ob in rows:
                g = t["gdv"][ab:ab + M]
                pv = t["pos"][ab:ab + M][g != 0]
                spans.append((int(pv.min()), int(pv.max())) if pv.numel() else (0, -1))
            order = sorted(range(len(rows)), key=lambda i: spans[i][0])
            ends = []
            colour = [0] * len(rows)
            for i in order:
                lo, hi = spans[i]
                for c in range(len(ends) + 1):
                    if c == len(ends):
                        ends.append(hi)
                        colour[i] = c
                    elif ends[c] < lo:
                        ends[c] = hi
                        colour[i] = c
                    else:
                        continue
                    break
            t["tasks_c"] = []
            for c in range(len(ends)):
                sel = [rows[i] for i in range(len(rows)) if colour[i] == c]
                pu = torch.cat([t["pos"][ab:ab + M][t["gdv"][ab:ab + M] != 0]
                                for ab, ob in sel])
                assert pu.numel() == torch.unique(pu).numel(), "scatter collision in colour class"
                t["tasks_c"].append(torch.tensor(sel, dtype=torch.int32, device=dev))

        self._ws = {}  #workspaces per batch size

        #stage kernel launch configs, per direction and per precision, the
        #sweep in bench/fused_gpu.py overwrites them with measured winners.
        #fp32 wants tiny tiles (register budget), tensor core modes want
        #bigger ones, so they are tuned apart
        cfg = dict(BM=32, BN=128, BK=64, num_warps=8, num_stages=2)
        precs = ("ieee", "tf32", "tf32x3")
        self.s1f = {p: dict(cfg) for p in precs}
        self.s1i = {p: dict(cfg) for p in precs}
        self.s2f = {p: dict(cfg) for p in precs}
        self.s2i = {p: dict(cfg) for p in precs}
        #inverse block splits contend at batch (measured), below this batch
        #size the inverse uses spliti, above it spliti_big
        self.split_bmax = 8

        #measured winners from the cluster sweeps baked in per architecture
        #so the kernel is fast out of the box, the bench sweep still
        #overrides whatever it measures better on the day
        if self.dev.type == "cuda" and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(self.dev)
            small = dict(BM=32, BN=32, BK=64, num_warps=4, num_stages=2)
            if cap >= (8, 0):
                #a100 sweep winners, fp32 wants tiny tiles, tf32x3 fat ones
                self.s1i["ieee"] = dict(small)
                self.s2f["ieee"] = dict(small)
                self.s1i["tf32x3"] = dict(BM=32, BN=128, BK=64,
                                          num_warps=8, num_stages=1)
                self.s2f["tf32x3"] = dict(BM=32, BN=64, BK=64,
                                          num_warps=4, num_stages=2)
                self.s1i["tf32"] = dict(self.s1i["tf32x3"])
                self.s2f["tf32"] = dict(self.s2f["tf32x3"])
                for t in self.two:
                    if t["N2"] >= 64:
                        #the round five graphed bars (the best measured end
                        #to end, 0.294 ms tf32x3 at B 1) ran coloured
                        #classes, split 4 small batch and 2 large. eager
                        #rankings preferred atomics but the graphed
                        #roundtrip did not, bake what actually won
                        t["spliti"], t["inv_atomic"] = 4, False
                        t["spliti_big"], t["inv_atomic_big"] = 2, False
                    else:
                        t["inv_atomic"] = t["inv_atomic_big"] = False
            else:
                #v100 sweep winners, fp32 atomics are slow on sm70 so the
                #coloured classes win everywhere on the wide bucket
                self.s1i["ieee"] = dict(small)
                self.s2f["ieee"] = dict(BM=64, BN=128, BK=32,
                                        num_warps=8, num_stages=2)
                #no tensor cores here, every precision runs the same machine
                #code, share the winners rather than leave the generic tiles
                for p in ("tf32", "tf32x3"):
                    self.s1i[p] = dict(self.s1i["ieee"])
                    self.s2f[p] = dict(self.s2f["ieee"])
                for t in self.two:
                    if t["N2"] >= 64:
                        t["spliti"] = t["spliti_big"] = 4
                        t["inv_atomic"] = t["inv_atomic_big"] = False
                    else:
                        t["inv_atomic"] = t["inv_atomic_big"] = True

    def _workspace(self, B):
        if B not in self._ws:
            z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=self.dev)
            self._ws[B] = dict(CR=z(B, N), CI=z(B, N), DR=z(B, N), DI=z(B, N),
                               KR=z(B, self.ncoef), KI=z(B, self.ncoef),
                               FR=z(B, HALF + 1), FI=z(B, HALF + 1), XO=z(B, N))
        return self._ws[B]

    def blocks_view(self, KR, KI):
        #complex per group views of the flat coefficient planes
        out = []
        for b, M, off in self.gspec:
            out.append(torch.complex(KR[:, off:off + b * M],
                                     KI[:, off:off + b * M]).view(-1, b, M))
        return out

    #torch simulator, the same tables and the same dataflow as the kernels,
    #runs anywhere, this is what the kernels are validated against

    def sim_fwd(self, x):
        B = x.shape[0]
        F1 = torch.complex(self.F1R, self.F1I)
        TW = torch.complex(self.TWR, self.TWI)
        A = x.view(B, R, R).transpose(1, 2).to(torch.complex64)
        C = (A @ F1) * TW
        D = C.transpose(1, 2) @ F1
        DR, DI = D.real.reshape(B, N), D.imag.reshape(B, N)
        KR = torch.zeros(B, self.ncoef, device=x.device)
        KI = torch.zeros(B, self.ncoef, device=x.device)
        #direct family
        for row in self.d_task.tolist():
            pb, mb, ob, M, nb, n0 = row
            nn = min(64, M - n0)
            i = torch.arange(nb, device=x.device)[:, None] * M \
                + torch.arange(M, device=x.device)[None, :]
            pi = self.d_pidx[pb + i].long()
            ar = DR[:, :].gather(1, pi.reshape(1, -1).expand(B, -1)).view(B, nb, M) \
                * self.d_wre[pb + i]
            ai = DI[:, :].gather(1, pi.reshape(1, -1).expand(B, -1)).view(B, nb, M) \
                * self.d_wim[pb + i]
            a = torch.complex(ar, ai)
            W = torch.complex(self.d_wir[mb:mb + M * M],
                              self.d_wii[mb:mb + M * M]).view(M, M)
            blk = a @ W[:, n0:n0 + nn]
            j = torch.arange(nb, device=x.device)[:, None] * M \
                + n0 + torch.arange(nn, device=x.device)[None, :]
            KR[:, ob + j.reshape(-1)] = blk.real.reshape(B, -1)
            KI[:, ob + j.reshape(-1)] = blk.imag.reshape(B, -1)
        #two stage family
        for t in self.two:
            N1, N2 = t["N1"], t["N2"]
            F1b = torch.complex(t["F1R"], t["F1I"])
            F2b = torch.complex(t["F2R"], t["F2I"])
            TWb = torch.complex(t["TWR"], t["TWI"])
            for ab, ob in t["task"].tolist():
                j = torch.arange(N2, device=x.device)[:, None] \
                    + N2 * torch.arange(N1, device=x.device)[None, :]
                pi = t["pidx"][ab + j].long()
                ar = DR.gather(1, pi.reshape(1, -1).expand(B, -1)).view(B, N2, N1) \
                    * t["wre"][ab + j]
                ai = DI.gather(1, pi.reshape(1, -1).expand(B, -1)).view(B, N2, N1) \
                    * t["wim"][ab + j]
                a = torch.complex(ar, -ai)  #conj for the ifft trick
                d = ((a @ F1b) * TWb).transpose(1, 2) @ F2b  #[B, N1, N2]
                o = torch.arange(N1, device=x.device)[:, None] \
                    + N1 * torch.arange(N2, device=x.device)[None, :]
                KR[:, ob + o.reshape(-1)] = d.real.reshape(B, -1) * t["scale"]
                KI[:, ob + o.reshape(-1)] = -d.imag.reshape(B, -1) * t["scale"]
        return KR, KI

    def sim_bwd(self, KR, KI):
        B = KR.shape[0]
        dev = KR.device
        FR = torch.zeros(B, HALF + 1, device=dev)
        FI = torch.zeros(B, HALF + 1, device=dev)
        #direct family
        for row in self.d_task.tolist():
            pb, mb, ob, M, nb, n0 = row
            nn = min(64, M - n0)
            i = torch.arange(nb, device=dev)[:, None] * M \
                + torch.arange(M, device=dev)[None, :]
            c = torch.complex(KR[:, ob + i.reshape(-1)],
                              KI[:, ob + i.reshape(-1)]).view(B, nb, M)
            Fm = torch.complex(self.d_fmr[mb:mb + M * M],
                               self.d_fmi[mb:mb + M * M]).view(M, M)
            v = c @ Fm[:, n0:n0 + nn]
            j = torch.arange(nb, device=dev)[:, None] * M \
                + n0 + torch.arange(nn, device=dev)[None, :]
            g = self.d_gdv[pb + j]
            p = self.d_pos[pb + j].long().reshape(-1)
            FR.index_add_(1, p, (v.real * g).reshape(B, -1))
            FI.index_add_(1, p, (v.imag * g).reshape(B, -1))
        #two stage family, forward dft this time, no conj trick
        for t in self.two:
            N1, N2 = t["N1"], t["N2"]
            F1b = torch.complex(t["F1R"], t["F1I"])
            F2b = torch.complex(t["F2R"], t["F2I"])
            TWb = torch.complex(t["TWR"], t["TWI"])
            for ab, ob in t["task"].tolist():
                j = torch.arange(N2, device=dev)[:, None] \
                    + N2 * torch.arange(N1, device=dev)[None, :]
                c = torch.complex(KR[:, ob + j.reshape(-1)],
                                  KI[:, ob + j.reshape(-1)]).view(B, N2, N1)
                d = ((c @ F1b) * TWb).transpose(1, 2) @ F2b
                o = torch.arange(N1, device=dev)[:, None] \
                    + N1 * torch.arange(N2, device=dev)[None, :]
                g = t["gdv"][ab + o.reshape(-1)]
                p = t["pos"][ab + o.reshape(-1)].long()
                FR.index_add_(1, p, d.real.reshape(B, -1) * g)
                FI.index_add_(1, p, d.imag.reshape(B, -1) * g)
        #big inverse fft with hermitian completion on load, conj trick
        j = torch.arange(N, device=dev)
        sel = j <= HALF
        idx = torch.where(sel, j, N - j)
        s = torch.where(sel, -1.0, 1.0).float()
        XR = FR[:, idx]
        XI = FI[:, idx] * s
        F1 = torch.complex(self.F1R, self.F1I)
        TW = torch.complex(self.TWR, self.TWI)
        A = torch.complex(XR, XI).view(B, R, R).transpose(1, 2)
        C = (A @ F1) * TW
        D = C.transpose(1, 2) @ F1  #[B, k1, k2]
        #x[k1 + R k2] = real(conj(D)) / N
        xo = D.real.transpose(1, 2).reshape(B, N) / N
        return xo

    #triton launchers. num_stages caps triton's load pipelining, the inverse
    #stage1 has four dot operands and at the default depth its shared memory
    #blows past the a100 limit (measured, 172032 required vs 166912
    #available), two stages fit everywhere

    def fwd(self, x):
        B = x.shape[0]
        w = self._workspace(B)
        prec = getattr(self, "prec", "ieee")
        c1, c2 = self.s1f[prec], self.s2f[prec]
        #DR DI are passed to stage1 as the inverse input planes, unused in
        #the forward branch, any tensor of the right size does
        _stage1[(B, R // c1["BM"], R // c1["BN"])](
            x, w["DR"], w["DI"], self.F1R, self.F1I, self.TWR, self.TWI,
            w["CR"], w["CI"], IS_INV=False, PREC=prec, **c1)
        _stage2[(B, R // c2["BM"], R // c2["BN"])](
            w["CR"], w["CI"], self.F1R, self.F1I, w["DR"], w["DI"], w["XO"],
            1.0 / N, IS_INV=False, PREC=prec, **c2)
        _block_direct[(self.d_task.shape[0], B)](
            w["DR"], w["DI"], w["FR"], w["FI"], w["KR"], w["KI"],
            self.d_pidx, self.d_wre, self.d_wim, self.d_wir, self.d_wii,
            self.d_gdv, self.d_pos, self.d_task, N, HALF + 1, self.ncoef,
            IS_FWD=True, PREC=prec, num_warps=4, num_stages=2)
        for t in self.two:
            #dot widths must stay 16 or more, older triton enforces it
            sp = max(1, min(t["split"], t["N2"] // 16))
            _block_two[(t["task"].shape[0] * sp, B)](
                w["DR"], w["DI"], w["FR"], w["FI"], w["KR"], w["KI"],
                t["pidx"], t["wre"], t["wim"],
                t["F1R"], t["F1I"], t["TWR"], t["TWI"], t["F2R"], t["F2I"],
                t["gdv"], t["pos"], t["task"], N, HALF + 1, self.ncoef,
                t["scale"], N1=t["N1"], N2=t["N2"], SPLIT=sp,
                ATOMIC=True, IS_FWD=True, PREC=prec, num_warps=4, num_stages=2)
        return w["KR"], w["KI"]

    def _inv_two(self, t, B, w, KR, KI, prec):
        #inverse scatter for one bucket. atomic mode is one launch over all
        #bands, coloured mode is one launch per collision free class with
        #plain read add write. which wins depends on the card (sm70 fp32
        #atomics are slow, sm80 ones are fine), so it is a measured flag
        big = B >= self.split_bmax
        sp = t["spliti_big"] if big else t["spliti"]
        sp = max(1, min(sp, t["N2"] // 16))
        atomic = t["inv_atomic_big"] if big else t["inv_atomic"]
        tasks = [t["task"]] if atomic else t["tasks_c"]
        for tk in tasks:
            _block_two[(tk.shape[0] * sp, B)](
                w["DR"], w["DI"], w["FR"], w["FI"], KR, KI,
                t["pidx"], t["wre"], t["wim"],
                t["F1R"], t["F1I"], t["TWR"], t["TWI"], t["F2R"], t["F2I"],
                t["gdv"], t["pos"], tk, N, HALF + 1, self.ncoef,
                t["scale"], N1=t["N1"], N2=t["N2"], SPLIT=sp,
                ATOMIC=atomic, IS_FWD=False, PREC=prec,
                num_warps=4, num_stages=2)

    def bwd(self, KR, KI):
        B = KR.shape[0]
        w = self._workspace(B)
        prec = getattr(self, "prec", "ieee")
        w["FR"].zero_()
        w["FI"].zero_()
        _block_direct[(self.d_task.shape[0], B)](
            w["DR"], w["DI"], w["FR"], w["FI"], KR, KI,
            self.d_pidx, self.d_wre, self.d_wim, self.d_fmr, self.d_fmi,
            self.d_gdv, self.d_pos, self.d_task, N, HALF + 1, self.ncoef,
            IS_FWD=False, PREC=prec, num_warps=4, num_stages=2)
        for t in self.two:
            self._inv_two(t, B, w, KR, KI, prec)
        #expand the half spectrum into full conjugated planes so stage1
        #loads contiguously, DR DI are free in this direction
        _mirror[(B, N // 1024)](
            w["FR"], w["FI"], w["DR"], w["DI"], HALF, BLOCK=1024)
        c1, c2 = self.s1i[prec], self.s2i[prec]
        _stage1[(B, R // c1["BM"], R // c1["BN"])](
            w["XO"], w["DR"], w["DI"], self.F1R, self.F1I, self.TWR, self.TWI,
            w["CR"], w["CI"], IS_INV=True, PREC=prec, **c1)
        _stage2[(B, R // c2["BM"], R // c2["BN"])](
            w["CR"], w["CI"], self.F1R, self.F1I, w["DR"], w["DI"], w["XO"],
            1.0 / N, IS_INV=True, PREC=prec, **c2)
        return w["XO"]

    def bench_parts(self, x, reps=50):
        #median ms per kernel launch, diagnosis only, names the kernel that
        #eats the time instead of guessing. mirrors the launch lines above
        B = x.shape[0]
        w = self._workspace(B)
        prec = getattr(self, "prec", "ieee")
        KR, KI = self.fwd(x)
        self.bwd(KR, KI)

        def med(fn):
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

        c1f, c2f = self.s1f[prec], self.s2f[prec]
        c1i, c2i = self.s1i[prec], self.s2i[prec]
        parts = [("fwd stage1", med(lambda: _stage1[(B, R // c1f["BM"], R // c1f["BN"])](
            x, w["DR"], w["DI"], self.F1R, self.F1I, self.TWR, self.TWI,
            w["CR"], w["CI"], IS_INV=False, PREC=prec, **c1f)))]
        parts.append(("fwd stage2", med(lambda: _stage2[(B, R // c2f["BM"], R // c2f["BN"])](
            w["CR"], w["CI"], self.F1R, self.F1I, w["DR"], w["DI"], w["XO"],
            1.0 / N, IS_INV=False, PREC=prec, **c2f))))
        parts.append(("fwd direct", med(lambda: _block_direct[(self.d_task.shape[0], B)](
            w["DR"], w["DI"], w["FR"], w["FI"], w["KR"], w["KI"],
            self.d_pidx, self.d_wre, self.d_wim, self.d_wir, self.d_wii,
            self.d_gdv, self.d_pos, self.d_task, N, HALF + 1, self.ncoef,
            IS_FWD=True, PREC=prec, num_warps=4, num_stages=2))))
        for t in self.two:
            sp = max(1, min(t["split"], t["N2"] // 16))
            parts.append((f"fwd two {t['N1']}x{t['N2']}",
                          med(lambda t=t, sp=sp: _block_two[(t["task"].shape[0] * sp, B)](
                w["DR"], w["DI"], w["FR"], w["FI"], w["KR"], w["KI"],
                t["pidx"], t["wre"], t["wim"],
                t["F1R"], t["F1I"], t["TWR"], t["TWI"], t["F2R"], t["F2I"],
                t["gdv"], t["pos"], t["task"], N, HALF + 1, self.ncoef,
                t["scale"], N1=t["N1"], N2=t["N2"], SPLIT=sp,
                ATOMIC=True, IS_FWD=True, PREC=prec, num_warps=4, num_stages=2))))
        parts.append(("inv direct", med(lambda: _block_direct[(self.d_task.shape[0], B)](
            w["DR"], w["DI"], w["FR"], w["FI"], KR, KI,
            self.d_pidx, self.d_wre, self.d_wim, self.d_fmr, self.d_fmi,
            self.d_gdv, self.d_pos, self.d_task, N, HALF + 1, self.ncoef,
            IS_FWD=False, PREC=prec, num_warps=4, num_stages=2))))
        for t in self.two:
            parts.append((f"inv two {t['N1']}x{t['N2']}",
                          med(lambda t=t: self._inv_two(t, B, w, KR, KI, prec))))
        parts.append(("inv mirror", med(lambda: _mirror[(B, N // 1024)](
            w["FR"], w["FI"], w["DR"], w["DI"], HALF, BLOCK=1024))))
        parts.append(("inv stage1", med(lambda: _stage1[(B, R // c1i["BM"], R // c1i["BN"])](
            w["XO"], w["DR"], w["DI"], self.F1R, self.F1I, self.TWR, self.TWI,
            w["CR"], w["CI"], IS_INV=True, PREC=prec, **c1i))))
        parts.append(("inv stage2", med(lambda: _stage2[(B, R // c2i["BM"], R // c2i["BN"])](
            w["CR"], w["CI"], self.F1R, self.F1I, w["DR"], w["DI"], w["XO"],
            1.0 / N, IS_INV=True, PREC=prec, **c2i))))
        return parts


if HAVE_TRITON:

    @triton.jit
    def _mirror(FRRE, FRIM, XRE, XIM, half, BLOCK: tl.constexpr):
        #expand the half spectrum into full mirrored planes, conjugated for
        #the inverse fft trick. contiguous writes, so the inverse stage1 can
        #load plainly instead of gathering through computed indices, which
        #measured 15x slower than the plain load path
        b = tl.program_id(0)
        j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        sel = j <= half
        idx = tl.where(sel, j, 65536 - j)
        s = tl.where(sel, -1.0, 1.0)
        ar = tl.load(FRRE + b * (half + 1) + idx)
        ai = tl.load(FRIM + b * (half + 1) + idx) * s
        tl.store(XRE + b * 65536 + j, ar)
        tl.store(XIM + b * 65536 + j, ai)

    @triton.jit
    def _stage1(X, XRE, XIM, F1R, F1I, TWR, TWI, CRE, CIM,
                IS_INV: tl.constexpr, PREC: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        #C[n2, k1] = (sum_n1 A[n2, n1] F[n1, k1]) * T[n2, k1]
        #forward A is the real input, inverse A is the mirrored conjugated
        #spectrum the _mirror kernel prepared. C is stored transposed, k1
        #major, so stage2 can load it without a register transpose.
        #tile shape is a parameter, the config sweep in bench/fused_gpu.py
        #picks it from measurement, the inverse variant of this kernel was
        #the pipeline's worst at the fixed 32 by 128 shape
        b = tl.program_id(0)
        r2 = tl.program_id(1) * BM + tl.arange(0, BM)
        rk = tl.program_id(2) * BN + tl.arange(0, BN)
        accr = tl.zeros((BM, BN), tl.float32)
        acci = tl.zeros((BM, BN), tl.float32)
        for n0 in range(0, 256, BK):
            rn = n0 + tl.arange(0, BK)
            j = 256 * rn[None, :] + r2[:, None]
            fr = tl.load(F1R + rn[:, None] * 256 + rk[None, :])
            fi = tl.load(F1I + rn[:, None] * 256 + rk[None, :])
            if IS_INV:
                ar = tl.load(XRE + b * 65536 + j)
                ai = tl.load(XIM + b * 65536 + j)
                accr += tl.dot(ar, fr, input_precision=PREC) - tl.dot(ai, fi, input_precision=PREC)
                acci += tl.dot(ar, fi, input_precision=PREC) + tl.dot(ai, fr, input_precision=PREC)
            else:
                ar = tl.load(X + b * 65536 + j)
                accr += tl.dot(ar, fr, input_precision=PREC)
                acci += tl.dot(ar, fi, input_precision=PREC)
        twr = tl.load(TWR + r2[:, None] * 256 + rk[None, :])
        twi = tl.load(TWI + r2[:, None] * 256 + rk[None, :])
        #transposed store, element (n2, k1) lands at k1 * 256 + n2
        off = b * 65536 + rk[None, :] * 256 + r2[:, None]
        tl.store(CRE + off, accr * twr - acci * twi)
        tl.store(CIM + off, accr * twi + acci * twr)

    @triton.jit
    def _stage2(CRE, CIM, F2R, F2I, DRE, DIM, XO, inv_scale,
                IS_INV: tl.constexpr, PREC: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        #D[k1, k2] = sum_n2 C[n2, k1] F[n2, k2], forward keeps the spectrum
        #in matrix layout, inverse stores the real signal at k1 + 256 k2.
        #C arrives transposed from stage1, so the tile loads straight into
        #the dot with no in loop transpose (the transpose measured 5x the
        #cost of the whole forward stage1)
        b = tl.program_id(0)
        r1 = tl.program_id(1) * BM + tl.arange(0, BM)
        rk = tl.program_id(2) * BN + tl.arange(0, BN)
        accr = tl.zeros((BM, BN), tl.float32)
        acci = tl.zeros((BM, BN), tl.float32)
        for n0 in range(0, 256, BK):
            rn = n0 + tl.arange(0, BK)
            ar = tl.load(CRE + b * 65536 + r1[:, None] * 256 + rn[None, :])
            ai = tl.load(CIM + b * 65536 + r1[:, None] * 256 + rn[None, :])
            fr = tl.load(F2R + rn[:, None] * 256 + rk[None, :])
            fi = tl.load(F2I + rn[:, None] * 256 + rk[None, :])
            accr += tl.dot(ar, fr, input_precision=PREC) - tl.dot(ai, fi, input_precision=PREC)
            acci += tl.dot(ar, fi, input_precision=PREC) + tl.dot(ai, fr, input_precision=PREC)
        if IS_INV:
            off = b * 65536 + r1[:, None] + 256 * rk[None, :]
            tl.store(XO + off, accr * inv_scale)
        else:
            off = b * 65536 + r1[:, None] * 256 + rk[None, :]
            tl.store(DRE + off, accr)
            tl.store(DIM + off, acci)

    @triton.jit
    def _block_direct(DRE, DIM, FRRE, FRIM, KR, KI, PIDX, WRE, WIM,
                      MATR, MATI, GDV, POS, TASK, n_d, n_fr, n_k,
                      IS_FWD: tl.constexpr, PREC: tl.constexpr):
        #one dft matrix per block, forward gathers from the spectrum matrix,
        #windows and stores the block, inverse loads the block, dfts, applies
        #the dual window and atomically scatters into the half spectrum
        row = tl.program_id(0)
        b = tl.program_id(1)
        t = TASK + row * 6
        pb = tl.load(t + 0)
        mb = tl.load(t + 1)
        ob = tl.load(t + 2)
        M = tl.load(t + 3)
        nb = tl.load(t + 4)
        n0 = tl.load(t + 5)
        rb = tl.arange(0, 16)
        rn = n0 + tl.arange(0, 64)
        mb_ = rb < nb
        mn = rn < M
        accr = tl.zeros((16, 64), tl.float32)
        acci = tl.zeros((16, 64), tl.float32)
        for k0 in range(0, 256, 32):
            rk = k0 + tl.arange(0, 32)
            mk = rk < M
            aoff = pb + rb[:, None] * M + rk[None, :]
            m2 = mb_[:, None] & mk[None, :]
            if IS_FWD:
                pi = tl.load(PIDX + aoff, mask=m2, other=0)
                ar = tl.load(DRE + b * n_d + pi, mask=m2, other=0.0) \
                    * tl.load(WRE + aoff, mask=m2, other=0.0)
                ai = tl.load(DIM + b * n_d + pi, mask=m2, other=0.0) \
                    * tl.load(WIM + aoff, mask=m2, other=0.0)
            else:
                coff = ob + rb[:, None] * M + rk[None, :]
                ar = tl.load(KR + b * n_k + coff, mask=m2, other=0.0)
                ai = tl.load(KI + b * n_k + coff, mask=m2, other=0.0)
            moff = mb + rk[:, None] * M + rn[None, :]
            mm = mk[:, None] & mn[None, :]
            br = tl.load(MATR + moff, mask=mm, other=0.0)
            bi = tl.load(MATI + moff, mask=mm, other=0.0)
            accr += tl.dot(ar, br, input_precision=PREC) - tl.dot(ai, bi, input_precision=PREC)
            acci += tl.dot(ar, bi, input_precision=PREC) + tl.dot(ai, br, input_precision=PREC)
        m2 = mb_[:, None] & mn[None, :]
        eoff = rb[:, None] * M + rn[None, :]
        if IS_FWD:
            tl.store(KR + b * n_k + ob + eoff, accr, mask=m2)
            tl.store(KI + b * n_k + ob + eoff, acci, mask=m2)
        else:
            g = tl.load(GDV + pb + eoff, mask=m2, other=0.0)
            p = tl.load(POS + pb + eoff, mask=m2, other=0)
            tl.atomic_add(FRRE + b * n_fr + p, accr * g, mask=m2)
            tl.atomic_add(FRIM + b * n_fr + p, acci * g, mask=m2)

    @triton.jit
    def _block_two(DRE, DIM, FRRE, FRIM, KR, KI, PIDX, WRE, WIM,
                   F1R, F1I, TWR, TWI, F2R, F2I, GDV, POS, TASK,
                   n_d, n_fr, n_k, scale,
                   N1: tl.constexpr, N2: tl.constexpr, SPLIT: tl.constexpr,
                   ATOMIC: tl.constexpr, IS_FWD: tl.constexpr, PREC: tl.constexpr):
        #two matmul stages per block, forward runs the inverse dft through
        #the conj trick, inverse runs the forward dft. SPLIT programs share
        #one bin, each computing a slice of the second stage's columns, the
        #first stage is recomputed per slice (it is tiny). one program per
        #bin gave 33 programs on 108 sms for the 2048 blocks, measured as
        #the second worst kernel in the pipeline
        pid = tl.program_id(0)
        b = tl.program_id(1)
        row = pid // SPLIT
        sp = pid % SPLIT
        t = TASK + row * 2
        ab = tl.load(t + 0)
        ob = tl.load(t + 1)
        r2 = tl.arange(0, N2)
        r1 = tl.arange(0, N1)
        j = ab + r2[:, None] + N2 * r1[None, :]
        if IS_FWD:
            pi = tl.load(PIDX + j)
            ar = tl.load(DRE + b * n_d + pi) * tl.load(WRE + j)
            ai = -(tl.load(DIM + b * n_d + pi) * tl.load(WIM + j))
        else:
            coff = ob + r2[:, None] + N2 * r1[None, :]
            ar = tl.load(KR + b * n_k + coff)
            ai = tl.load(KI + b * n_k + coff)
        f1r = tl.load(F1R + r1[:, None] * N1 + r1[None, :])
        f1i = tl.load(F1I + r1[:, None] * N1 + r1[None, :])
        br = tl.dot(ar, f1r, input_precision=PREC) - tl.dot(ai, f1i, input_precision=PREC)
        bi = tl.dot(ar, f1i, input_precision=PREC) + tl.dot(ai, f1r, input_precision=PREC)
        twr = tl.load(TWR + r2[:, None] * N1 + r1[None, :])
        twi = tl.load(TWI + r2[:, None] * N1 + r1[None, :])
        cr = br * twr - bi * twi
        ci = br * twi + bi * twr
        ar2 = tl.trans(cr)
        ai2 = tl.trans(ci)
        #this program's slice of the output columns
        rk2 = sp * (N2 // SPLIT) + tl.arange(0, N2 // SPLIT)
        f2r = tl.load(F2R + r2[:, None] * N2 + rk2[None, :])
        f2i = tl.load(F2I + r2[:, None] * N2 + rk2[None, :])
        dr = tl.dot(ar2, f2r, input_precision=PREC) - tl.dot(ai2, f2i, input_precision=PREC)
        di = tl.dot(ar2, f2i, input_precision=PREC) + tl.dot(ai2, f2r, input_precision=PREC)
        oo = r1[:, None] + N1 * rk2[None, :]
        if IS_FWD:
            tl.store(KR + b * n_k + ob + oo, dr * scale)
            tl.store(KI + b * n_k + ob + oo, -di * scale)
        else:
            g = tl.load(GDV + ab + oo)
            p = tl.load(POS + ab + oo)
            if ATOMIC:
                tl.atomic_add(FRRE + b * n_fr + p, dr * g)
                tl.atomic_add(FRIM + b * n_fr + p, di * g)
            else:
                #the colour classes guarantee no other program touches these
                #bins, plain read add write, no contention
                vr = tl.load(FRRE + b * n_fr + p)
                vi = tl.load(FRIM + b * n_fr + p)
                tl.store(FRRE + b * n_fr + p, vr + dr * g)
                tl.store(FRIM + b * n_fr + p, vi + di * g)
