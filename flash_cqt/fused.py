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
import pathlib
import torch

#the raw cuda inverse block kernel, built on first use when nvcc exists.
#triton past 3.1 compiles that kernel 2 to 6x slower than its own 3.1
#output (measured, bench/endings.py), the cuda source is immune. falls
#back to the triton path silently when there is no compiler around
_EXT = None


def _blocks_ext():
    global _EXT
    if _EXT is None:
        try:
            from torch.utils.cpp_extension import load
            src = pathlib.Path(__file__).resolve().parent / "csrc" / "blocks.cu"
            _EXT = load(name="flash_cqt_blocks", sources=[str(src)], verbose=False)
        except Exception:
            _EXT = False
    return _EXT


#the wmma stage kernels, tensor cores at near fp32 accuracy for tf32x3
_SEXT = None


def _stages_ext():
    global _SEXT
    if _SEXT is None:
        try:
            from torch.utils.cpp_extension import load
            src = pathlib.Path(__file__).resolve().parent / "csrc" / "stages.cu"
            _SEXT = load(name="flash_cqt_stages", sources=[str(src)], verbose=False)
        except Exception:
            _SEXT = False
    return _SEXT

#the packed big fft, three radix 32 passes, first layer of the factored path
_FEXT = None


def _packedfft_ext():
    global _FEXT
    if _FEXT is None:
        try:
            from torch.utils.cpp_extension import load
            src = pathlib.Path(__file__).resolve().parent / "csrc" / "packedfft.cu"
            _FEXT = load(name="flash_cqt_packedfft", sources=[str(src)], verbose=False)
        except Exception:
            _FEXT = False
    return _FEXT


#the router layer of the factored path
_REXT = None


def _router_ext():
    global _REXT
    if _REXT is None:
        try:
            from torch.utils.cpp_extension import load
            src = pathlib.Path(__file__).resolve().parent / "csrc" / "router.cu"
            _REXT = load(name="flash_cqt_router", sources=[str(src)], verbose=False)
        except Exception:
            _REXT = False
    return _REXT


#the radix band engine, mixed radix warp fft bands for the factored path
_XEXT = None


def _radix_ext():
    global _XEXT
    if _XEXT is None:
        try:
            from torch.utils.cpp_extension import load
            src = pathlib.Path(__file__).resolve().parent / "csrc" / "radix.cu"
            _XEXT = load(name="flash_cqt_radix", sources=[str(src)], verbose=False)
        except Exception:
            _XEXT = False
    return _XEXT


#the factored resident pair
_P2EXT = None


def _persist2_ext():
    global _P2EXT
    if _P2EXT is None:
        try:
            from torch.utils.cpp_extension import load
            src = pathlib.Path(__file__).resolve().parent / "csrc" / "persist2.cu"
            _P2EXT = load(name="flash_cqt_persist2", sources=[str(src)], verbose=False)
        except Exception:
            _P2EXT = False
    return _P2EXT


#the two resident kernels, the whole transform in one cooperative launch
_PEXT = None


def _persist_ext():
    global _PEXT
    if _PEXT is None:
        try:
            from torch.utils.cpp_extension import load
            src = pathlib.Path(__file__).resolve().parent / "csrc" / "persist.cu"
            _PEXT = load(name="flash_cqt_persist", sources=[str(src)], verbose=False)
        except Exception:
            _PEXT = False
    return _PEXT


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


#the radix engine's factor pairs for the two level lengths
RADIX_FACTORS = {64: (8, 8), 128: (16, 8), 256: (16, 16),
                 512: (32, 16), 1024: (32, 32)}


def _brev(x, q):
    y = torch.zeros_like(x)
    for _ in range(q):
        y = (y << 1) | (x & 1)
        x = x >> 1
    return y


def radix_load_perm(M):
    #tap table order the radix kernel loads in, position (vector, lane)
    #maps to natural input index n, so every warp read is contiguous
    if M == 32:
        return _brev(torch.arange(32), 5)
    if M == 2048:
        q = torch.arange(64)[:, None]
        lane = torch.arange(32)[None, :]
        return (q + 64 * _brev(lane, 5)).reshape(-1)
    R0, R1 = RADIX_FACTORS[M]
    n1 = torch.arange(R1)[:, None]
    ell = torch.arange(R0)[None, :]
    return (n1 + R1 * _brev(ell, int(math.log2(R0)))).reshape(-1)


def radix_store_perm(M):
    #gain and position table order the inverse epilogue stores in
    if M == 32:
        return torch.arange(32)
    if M == 2048:
        v = torch.arange(256)[:, None]
        ell = torch.arange(8)[None, :]
        return ((v >> 3) + 32 * (v & 7) + 256 * ell).reshape(-1)
    R0, R1 = RADIX_FACTORS[M]
    k0 = torch.arange(R0)[:, None]
    ell = torch.arange(R1)[None, :]
    return (k0 + R0 * ell).reshape(-1)


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
        d_router = {}
        two = {}
        #radix band engine tables, one descriptor per band {m, tap offset,
        #heap offset, root offset}. the root blob per length holds 32 stage
        #roots, then the level one twiddles tw0[m] for m 64 and up, then
        #tw1[2048] for the three level case, every entry in the exact warp
        #read order and copied out of the float64 built m root table so the
        #bits match the gated values. synthesis sign stored, the analysis
        #path negates the imaginary plane on load
        x_rows, x_off = [], 0
        x_tabs = {nm: [] for nm in ("p1", "p2", "ar", "ai", "br", "bi",
                                    "g", "gdv", "pos", "gav", "dar",
                                    "dai", "gdw", "aidx", "asgn")}
        x_rr, x_ri, x_roff = [], [], {}
        o = 0
        for M in sorted({M for _, M, _ in self.gspec}):
            j = torch.arange(M, dtype=torch.int64)
            ang = -2.0 * math.pi * j.to(torch.float64) / M
            rmr = torch.cos(ang).float()
            rmi = torch.sin(ang).float()
            pr, pi = [rmr[::M // 32]], [rmi[::M // 32]]
            if M == 2048:
                q = torch.arange(64, dtype=torch.int64)[:, None]
                k0 = torch.arange(32, dtype=torch.int64)[None, :]
                e0 = (q * k0).reshape(-1)
                v = torch.arange(256, dtype=torch.int64)[:, None]
                k1 = torch.arange(8, dtype=torch.int64)[None, :]
                e1 = (32 * (v % 8) * k1).reshape(-1)
                pr += [rmr[e0], rmr[e1]]
                pi += [rmi[e0], rmi[e1]]
            elif M >= 64:
                R0, R1 = RADIX_FACTORS[M]
                n1 = torch.arange(R1, dtype=torch.int64)[:, None]
                k0 = torch.arange(R0, dtype=torch.int64)[None, :]
                e0 = (n1 * k0).reshape(-1)
                pr.append(rmr[e0])
                pi.append(rmi[e0])
            x_rr.append(torch.cat(pr))
            x_ri.append(torch.cat(pi))
            x_roff[M] = o
            o += x_rr[-1].numel()
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
            #router planes for the factored path, every tap becomes
            #tap = A Z[p1] + B conj(Z[p2]) on the packed half length
            #spectrum, untangle and window folded into A and B, the conj
            #flagged side taps swap roles. indices pre permuted into the
            #packed fft kernel's storage order
            kk = idx.reshape(-1).to(torch.int64).cpu()
            ang = -2.0 * math.pi * kk.to(torch.float64) / N
            W = torch.complex(torch.cos(ang), torch.sin(ang))
            cA = 0.5 * (1.0 - 1j * W)
            cB = 0.5 * (1.0 + 1j * W)
            cf = conj.reshape(-1).cpu()
            wv = gvf.to(torch.complex128)
            rA = torch.where(cf, wv * cB.conj(), wv * cA).to(torch.complex64)
            rB = torch.where(cf, wv * cA.conj(), wv * cB).to(torch.complex64)
            q1 = kk % (N // 2)
            q2 = (N // 2 - kk) % (N // 2)
            rP1 = torch.where(cf, q2, q1)
            rP2 = torch.where(cf, q1, q2)

            def zperm(q):
                return ((q & 31) * 1024 + ((q >> 5) & 31) * 32
                        + (q >> 10)).to(torch.int32)

            rt = dict(p1=zperm(rP1), p2=zperm(rP2),
                      ar=rA.real.float(), ai=rA.imag.float(),
                      br=rB.real.float(), bi=rB.imag.float(),
                      k=kk.to(torch.int32), cf=cf)
            #dual window taps for the adjoint of the synthesis, same
            #untangle coefficients, the dual window replaces the window
            wvd = gdvf.to(torch.complex128)
            rAd = torch.where(cf, wvd * cB.conj(), wvd * cA).to(torch.complex64)
            #radix engine rows and tables, both families in one layout,
            #taps permuted to the kernel's load order and gains to its
            #store order so every warp access lands contiguous
            for xb in range(b):
                x_rows.append([M, x_off + xb * M, cbase + xb * M, x_roff[M]])
            x_off += b * M
            xlp = radix_load_perm(M)
            xsp = radix_store_perm(M)
            for nm in ("p1", "p2", "ar", "ai", "br", "bi"):
                x_tabs[nm].append(rt[nm].reshape(b, M)[:, xlp].reshape(-1))
            x_tabs["g"].append(gvf.reshape(b, M)[:, xlp].reshape(-1))
            x_tabs["gdv"].append(gdvf.reshape(b, M)[:, xsp].reshape(-1))
            x_tabs["pos"].append(posf.reshape(b, M)[:, xsp].reshape(-1))
            #adjoint tables, the analysis gain rides the scatter engine
            #and the dual window rides the router engine. the scatter
            #side needs the analysis gather indices (they differ from
            #the synthesis positions at the conj side taps) and the
            #conj sign for the imaginary plane
            x_tabs["aidx"].append(
                kk.to(torch.int32).reshape(b, M)[:, xsp].reshape(-1))
            x_tabs["asgn"].append(
                torch.where(cf, -1.0, 1.0).float()
                .reshape(b, M)[:, xsp].reshape(-1))
            x_tabs["gav"].append(gvf.reshape(b, M)[:, xsp].reshape(-1))
            x_tabs["dar"].append(rAd.real.float().reshape(b, M)[:, xlp].reshape(-1))
            x_tabs["dai"].append(rAd.imag.float().reshape(b, M)[:, xlp].reshape(-1))
            x_tabs["gdw"].append(gdvf.reshape(b, M)[:, xlp].reshape(-1))
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
                for nm in rt:
                    d_router.setdefault(nm, []).append(rt[nm])
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
                for nm in rt:
                    t.setdefault("r_" + nm, []).append(rt[nm])
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
        self.d_router = {nm: torch.cat(v).to(dev).contiguous()
                         for nm, v in d_router.items()}
        self.radix = {nm: torch.cat(v).to(dev).contiguous()
                      for nm, v in x_tabs.items()}
        self.radix["desc"] = torch.tensor(x_rows, dtype=torch.int32,
                                          device=dev)
        #compact 16 byte router record per point, q1 | q2 << 16 then the
        #float bits of re a, im a and the window g. the kernel rebuilds
        #re b = g - re a and im b = -im a, one aligned load per tap
        #instead of six streams
        xq = (self.radix["p1"].to(torch.int64)
              | (self.radix["p2"].to(torch.int64) << 16))
        xrec = torch.empty(xq.numel(), 4, dtype=torch.int32, device=dev)
        xrec[:, 0] = xq.to(torch.int32)
        xrec[:, 1] = self.radix["ar"].view(torch.int32)
        xrec[:, 2] = self.radix["ai"].view(torch.int32)
        xrec[:, 3] = self.radix["g"].view(torch.int32)
        self.radix["route"] = xrec.reshape(-1).contiguous()
        xrecd = torch.empty(xq.numel(), 4, dtype=torch.int32, device=dev)
        xrecd[:, 0] = xq.to(torch.int32)
        xrecd[:, 1] = self.radix["dar"].view(torch.int32)
        xrecd[:, 2] = self.radix["dai"].view(torch.int32)
        xrecd[:, 3] = self.radix["gdw"].view(torch.int32)
        self.radix["route_d"] = xrecd.reshape(-1).contiguous()
        #adjoint tables with every verified constant baked in, gates in
        #bench/autograd_check.py, scales n/2 and 2/n proven clean there
        xms = self.radix["desc"][:, 0].repeat_interleave(
            self.radix["desc"][:, 0].to(torch.int64)).float()
        edge_s = ((self.radix["aidx"] == 0)
                  | (self.radix["aidx"] == HALF)).float()
        agr = self.radix["gav"] * (N / 2) / xms * (1.0 + edge_s)
        self.radix["agr"] = agr.contiguous()
        #the adjoint of the real fft ignores the imaginary parts of the
        #dc and nyquist bins, the tail does not, so the edge taps get a
        #zero imaginary gain and the accumulated imag vanishes exactly
        self.radix["agi"] = (agr * self.radix["asgn"]
                             * (1.0 - edge_s)).contiguous()
        #per point band length is the same in either table order
        edge_l = ((self.radix["p1"] == 0)
                  & (self.radix["p2"] == 0)).float()
        cb = (2.0 * xms / N) * (1.0 - 0.5 * edge_l)
        xrecb = torch.empty(xq.numel(), 4, dtype=torch.int32, device=dev)
        xrecb[:, 0] = xq.to(torch.int32)
        xrecb[:, 1] = (self.radix["dar"] * cb).view(torch.int32)
        xrecb[:, 2] = (self.radix["dai"] * cb).view(torch.int32)
        xrecb[:, 3] = (self.radix["gdw"] * cb).view(torch.int32)
        self.radix["route_b"] = xrecb.reshape(-1).contiguous()
        self.radix["rr"] = torch.cat(x_rr).to(dev).contiguous()
        self.radix["ri"] = torch.cat(x_ri).to(dev).contiguous()

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
                router={nm: torch.cat(t["r_" + nm]).to(dev).contiguous()
                        for nm in ("p1", "p2", "ar", "ai", "br", "bi",
                                   "k", "cf")},
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

        #ell slot tables for the atomic free inverse scatter. every tap
        #with a nonzero dual window owns a unique (bin, slot) edge, the
        #spectral overlap degree is at most 3 (proven, asserted on
        #every build), so the reduction reads at most three slots per bin
        #and nothing is ever contended or zeroed
        fam_pos, fam_gdv = [], []
        fam_pos.append(self.d_pos.cpu().to(torch.int64))
        fam_gdv.append(self.d_gdv.cpu())
        for t in self.two:
            fam_pos.append(t["pos"].cpu().to(torch.int64))
            fam_gdv.append(t["gdv"].cpu())
        allpos = torch.cat(fam_pos)
        allg = torch.cat(fam_gdv)
        live = allg != 0
        lp = allpos[live]
        order = torch.argsort(lp, stable=True)
        sp = lp[order]
        first = torch.ones_like(sp, dtype=torch.bool)
        first[1:] = sp[1:] != sp[:-1]
        gid = torch.cumsum(first.to(torch.int64), 0) - 1
        ar = torch.arange(sp.numel())
        rank = ar - ar[first][gid]
        assert int(rank.max()) <= 2, "overlap degree above three"
        slots = torch.zeros(lp.numel(), dtype=torch.int64)
        slots[order] = rank
        eidx_all = torch.zeros(allpos.numel(), dtype=torch.int64)
        eidx_all[live] = allpos[live] * 3 + slots
        deg = torch.zeros(HALF + 1, dtype=torch.int64)
        deg.scatter_add_(0, allpos[live], torch.ones_like(allpos[live]))
        self.ell_deg = deg.to(torch.int32).to(dev).contiguous()
        off = 0
        cuts = [f.numel() for f in fam_pos]
        parts = torch.split(eidx_all.to(torch.int32), cuts)
        self.d_eidx = parts[0].to(dev).contiguous()
        for t, pe in zip(self.two, parts[1:]):
            t["eidx"] = pe.to(dev).contiguous()

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

        #cublas takes the big fft stages at batch, it beat the triton
        #stages 1.9x at fp32 and 6.3x at tf32 for B 32 (bench/stages.py)
        #while losing at B 1, so the switch is batch size. complex tables
        #and the mirror index live here for that path
        self._F1c = torch.complex(self.F1R, self.F1I)
        self._TWc = torch.complex(self.TWR, self.TWI)
        jj = torch.arange(N, device=dev)
        sel = jj <= HALF
        self._midx = torch.where(sel, jj, N - jj)
        self._msgn = torch.where(sel, -1.0, 1.0).float()
        self.stage_blas_bmin = 16

        #the tensor core stage path for tf32x3 at batch. a complex matmul
        #against the fixed dft matrix becomes one real gemm against a 512
        #by 512 block matrix [[Fr, Fi], [-Fi, Fr]], and each real gemm is
        #done as the three way tf32 split, high and low halves, three
        #tensor core gemms per product at near fp32 accuracy. the matrix
        #splits are precomputed here, the input splits per call
        def tf32_hi(t):
            i = t.contiguous().view(torch.int32)
            return ((i + 4096) & -8192).view(torch.float32)

        G1 = torch.cat([self.F1R, self.F1I], dim=1)
        FB = torch.cat([torch.cat([self.F1R, self.F1I], dim=1),
                        torch.cat([-self.F1I, self.F1R], dim=1)], dim=0)
        self._G1h = tf32_hi(G1)
        self._G1l = G1 - self._G1h
        self._FBh = tf32_hi(FB)
        self._FBl = FB - self._FBh
        self._tf32_hi = tf32_hi

        #split tables for the wmma stage kernels, low halves rounded to
        #tf32 as well since the fragments load through tf32 anyway, and
        #negated imaginary tables for the subtraction term of the complex
        #product (wmma cannot subtract)
        #residuals scaled by 2^11 before rounding so their own tf32
        #rounding lands below fp32 precision, the kernel descales the
        #cross products once at the end, the ootomo style split. the
        #unscaled residual scheme measured a 96 db floor
        self._wt = {}
        for nm, mat in (("f1r", self.F1R), ("f1i", self.F1I),
                        ("nf1i", -self.F1I)):
            h = tf32_hi(mat.contiguous())
            self._wt[nm + "h"] = h
            self._wt[nm + "l"] = tf32_hi(((mat - h) * 2048.0).contiguous())

        #packed descriptors for the resident kernels, one table heap and one
        #global tap concatenation across the two stage buckets
        heap = []
        hoff = 0
        cats = {k: [] for k in ("pidx", "wre", "wim", "gdv", "pos")}
        toff = 0
        desc = []
        for t in self.two:
            n1, n2 = t["N1"], t["N2"]
            offs = []
            for pair in (("F1R", "F1I"), ("TWR", "TWI"), ("F2R", "F2I")):
                offs.append(hoff)
                for nmm in pair:
                    v = t[nmm].reshape(-1)
                    heap.append(v)
                    hoff += v.numel()
            for ab, ob in t["task"].tolist():
                desc.append([n1, n2, toff + ab, ob,
                             offs[0], offs[1], offs[2], 0])
            for k in cats:
                cats[k].append(t[k])
            toff += t["pidx"].numel()
        self._pd = dict(
            desc=torch.tensor(desc, dtype=torch.int32, device=dev),
            tabs=torch.cat(heap).contiguous(),
            **{k: torch.cat(v).contiguous() for k, v in cats.items()})

        def fit_shared(cfg, limit):
            #triton stages operand tiles through shared memory, roughly
            #(BM*BK + 2*BK*BN) floats per pipeline stage (measured exact on
            #the t4 crash, 73728 bytes for BM32 BN128 BK64). shrink until it
            #fits the card, stages first, then BN, then BK
            def usage(c):
                return (c["BM"] * c["BK"] + 2 * c["BK"] * c["BN"]) * 4 * c["num_stages"]
            while usage(cfg) > limit and cfg["num_stages"] > 1:
                cfg["num_stages"] -= 1
            while usage(cfg) > limit and cfg["BN"] > 32:
                cfg["BN"] //= 2
            while usage(cfg) > limit and cfg["BK"] > 32:
                cfg["BK"] //= 2
            return cfg

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

            #per block shared memory the card actually allows, sm75 has 64k,
            #sm70 96k, sm80 164k. every stage config gets shrunk to fit so
            #the kernel never dies on a smaller card
            limits = {(7, 5): 64 * 1024, (7, 0): 96 * 1024}
            limit = limits.get(cap, 96 * 1024 if cap < (8, 0) else 164 * 1024)
            for d in (self.s1f, self.s1i, self.s2f, self.s2i):
                for p in d:
                    d[p] = fit_shared(d[p], limit)

            #triton newer than 3.1 compiles the inverse block kernel 2 to
            #6x worse at every configuration (measured, same t4, same
            #source, 3.0 ms on 3.1 against 19.8 on 3.7 at split 4, best
            #case 5.4 at split 2). no single pattern to dodge, so cap the
            #inverse splits at 2 there, the least bad point, until the
            #cuda flagship path replaces these kernels
            if HAVE_TRITON:
                try:
                    ver = tuple(int(v) for v in triton.__version__.split(".")[:2])
                except (ValueError, AttributeError):
                    ver = (9, 9)
                if ver >= (3, 2):
                    for t in self.two:
                        t["spliti"] = min(t["spliti"], 2)
                        t["spliti_big"] = min(t["spliti_big"], 2)

    def persist_fwd(self, x):
        #the whole forward in one resident kernel
        ext = _persist_ext()
        B = x.shape[0]
        w = self._workspace(B)
        pd = self._pd
        ext.persist_fwd(x, w["CR"], w["CI"], w["DR"], w["DI"],
                        w["KR"], w["KI"], w["FR"], w["FI"], w["XO"],
                        self.F1R, self.F1I, self.TWR, self.TWI,
                        pd["desc"], pd["tabs"], pd["pidx"], pd["wre"],
                        pd["wim"], pd["gdv"], pd["pos"],
                        self.d_task, self.d_pidx, self.d_wre, self.d_wim,
                        self.d_wir, self.d_wii, self.d_gdv, self.d_pos,
                        self.ncoef, B)
        return w["KR"], w["KI"]

    def persist_inv(self, KR, KI):
        #the whole inverse in one resident kernel
        ext = _persist_ext()
        B = KR.shape[0]
        w = self._workspace(B)
        pd = self._pd
        ext.persist_inv(KR, w["CR"], w["CI"], w["DR"], w["DI"],
                        KR, KI, w["FR"], w["FI"], w["XO"],
                        self.F1R, self.F1I, self.TWR, self.TWI,
                        pd["desc"], pd["tabs"], pd["pidx"], pd["wre"],
                        pd["wim"], pd["gdv"], pd["pos"],
                        self.d_task, self.d_pidx, self.d_wre, self.d_wim,
                        self.d_fmr, self.d_fmi, self.d_gdv, self.d_pos,
                        self.ncoef, B)
        return w["XO"]

    def _mm3(self, a, bh, bl):
        #three way tf32 split of a real gemm against a fixed matrix whose
        #high and low halves are precomputed, near fp32 accuracy on tensor
        #cores, call under allow_tf32
        ah = self._tf32_hi(a)
        return ah @ bh + ah @ bl + (a - ah) @ bh

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
        if B >= self.stage_blas_bmin and prec == "tf32x3" \
                and getattr(self, "use_cuda_stages", False) and _stages_ext():
            #wmma stage kernels, the three way split in registers
            wt = self._wt
            sx = _stages_ext()
            sx.s1_fwd(x, x, w["CR"], w["CI"],
                      wt["f1rh"], wt["f1rl"], wt["f1ih"], wt["f1il"],
                      wt["nf1ih"], wt["nf1il"], self.TWR, self.TWI, 1.0)
            sx.s2_fwd(w["CR"], w["CI"], w["DR"], w["DI"],
                      wt["f1rh"], wt["f1rl"], wt["f1ih"], wt["f1il"],
                      wt["nf1ih"], wt["nf1il"], self.TWR, self.TWI, 1.0)
        elif B >= self.stage_blas_bmin and prec == "tf32x3" \
                and getattr(self, "stage_mm3", False):
            #tensor core stages through the three way split as torch ops.
            #measured slower than the triton stages it replaced (the split
            #and cat traffic outweighs the tensor core gain, and snr fell
            #to 105 db), kept behind a flag as the reference for the wmma
            #kernel that does the split in registers instead
            keep = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            X = x.view(B, R, R).transpose(1, 2).contiguous()
            CC = self._mm3(X, self._G1h, self._G1l)
            Cr, Ci = CC[..., :R], CC[..., R:]
            Cr2 = Cr * self.TWR - Ci * self.TWI
            Ci2 = Cr * self.TWI + Ci * self.TWR
            E = torch.cat([Cr2.transpose(1, 2),
                           Ci2.transpose(1, 2)], dim=-1).contiguous()
            DD = self._mm3(E, self._FBh, self._FBl)
            w["DR"].copy_(DD[..., :R].reshape(B, N))
            w["DI"].copy_(DD[..., R:].reshape(B, N))
            torch.backends.cuda.matmul.allow_tf32 = keep
        elif B >= self.stage_blas_bmin and prec != "tf32x3":
            #cublas stages, exact fp32 gemms in ieee mode
            keep = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
            A = x.view(B, R, R).transpose(1, 2).to(torch.complex64)
            D = ((A @ self._F1c) * self._TWc).transpose(1, 2) @ self._F1c
            w["DR"].copy_(D.real.reshape(B, N))
            w["DI"].copy_(D.imag.reshape(B, N))
            torch.backends.cuda.matmul.allow_tf32 = keep
        else:
            c1, c2 = self.s1f[prec], self.s2f[prec]
            #DR DI are passed to stage1 as the inverse input planes, unused
            #in the forward branch, any tensor of the right size does
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
        ext = _blocks_ext() if getattr(self, "use_cuda_fwd_blocks", False) else False
        for t in self.two:
            if ext:
                ext.fwd_two(w["DR"], w["DI"], w["KR"], w["KI"],
                            t["pidx"], t["wre"], t["wim"],
                            t["F1R"], t["F1I"], t["TWR"], t["TWI"],
                            t["F2R"], t["F2I"], t["task"], N, self.ncoef,
                            t["N1"], t["N2"], t["scale"], B)
                continue
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
        #inverse scatter for one bucket. the cuda extension takes it when a
        #compiler is around (fp32 exact, so every precision mode may use
        #it), otherwise the triton path below. atomic mode is one launch
        #over all bands, coloured mode is one launch per collision free
        #class with plain read add write. which wins depends on the card
        #(sm70 fp32 atomics are slow, sm80 ones are fine), a measured flag
        if getattr(self, "use_cuda_blocks", True) and prec == "ieee":
            ext = _blocks_ext()
            if ext:
                ext.inv_two(KR, KI, w["FR"], w["FI"],
                            t["F1R"], t["F1I"], t["TWR"], t["TWI"],
                            t["F2R"], t["F2I"], t["gdv"], t["pos"],
                            t["task"], HALF + 1, self.ncoef,
                            t["N1"], t["N2"], B)
                return
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

    def _scatter(self, KR, KI, w, B, prec):
        #band dfts and the dual window scatter into the half spectrum,
        #shared by the shipped inverse and the factored inverse tail
        w["FR"].zero_()
        w["FI"].zero_()
        _block_direct[(self.d_task.shape[0], B)](
            w["DR"], w["DI"], w["FR"], w["FI"], KR, KI,
            self.d_pidx, self.d_wre, self.d_wim, self.d_fmr, self.d_fmi,
            self.d_gdv, self.d_pos, self.d_task, N, HALF + 1, self.ncoef,
            IS_FWD=False, PREC=prec, num_warps=4, num_stages=2)
        for t in self.two:
            self._inv_two(t, B, w, KR, KI, prec)

    def bwd(self, KR, KI):
        B = KR.shape[0]
        w = self._workspace(B)
        prec = getattr(self, "prec", "ieee")
        self._scatter(KR, KI, w, B, prec)
        if B >= self.stage_blas_bmin and prec == "tf32x3" \
                and getattr(self, "use_cuda_stages", False) and _stages_ext():
            #mirror to the full conjugated planes, then the wmma pair
            wt = self._wt
            sx = _stages_ext()
            _mirror[(B, N // 1024)](
                w["FR"], w["FI"], w["DR"], w["DI"], HALF, BLOCK=1024)
            sx.s1_inv(w["DR"], w["DI"], w["CR"], w["CI"],
                      wt["f1rh"], wt["f1rl"], wt["f1ih"], wt["f1il"],
                      wt["nf1ih"], wt["nf1il"], self.TWR, self.TWI, 1.0)
            sx.s2_inv(w["CR"], w["CI"], w["XO"], w["DR"],
                      wt["f1rh"], wt["f1rl"], wt["f1ih"], wt["f1il"],
                      wt["nf1ih"], wt["nf1il"], self.TWR, self.TWI, 1.0 / N)
        elif B >= self.stage_blas_bmin and prec != "tf32x3":
            keep = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
            XR = w["FR"][:, self._midx]
            XI = w["FI"][:, self._midx] * self._msgn
            A = torch.complex(XR, XI).view(B, R, R).transpose(1, 2)
            D = ((A @ self._F1c) * self._TWc).transpose(1, 2) @ self._F1c
            w["XO"].copy_(D.real.transpose(1, 2).reshape(B, N) / N)
            torch.backends.cuda.matmul.allow_tf32 = keep
        else:
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
            #graph replayed median. eager medians misled five rounds of
            #work, launch tax made slow kernels look decisive and cheap
            #ones invisible, the shipped path is a graph so measure one
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            ts = []
            st, e = torch.cuda.Event(True), torch.cuda.Event(True)
            for _ in range(reps):
                st.record()
                g.replay()
                e.record()
                torch.cuda.synchronize()
                ts.append(st.elapsed_time(e))
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
        bext = _blocks_ext() if getattr(self, "use_cuda_blocks", True) else False
        for t in self.two:
            if bext:
                fn = lambda t=t: bext.fwd_two(
                    w["DR"], w["DI"], w["KR"], w["KI"],
                    t["pidx"], t["wre"], t["wim"],
                    t["F1R"], t["F1I"], t["TWR"], t["TWI"],
                    t["F2R"], t["F2I"], t["task"], N, self.ncoef,
                    t["N1"], t["N2"], t["scale"], B)
            else:
                sp = max(1, min(t["split"], t["N2"] // 16))
                fn = lambda t=t, sp=sp: _block_two[(t["task"].shape[0] * sp, B)](
                    w["DR"], w["DI"], w["FR"], w["FI"], w["KR"], w["KI"],
                    t["pidx"], t["wre"], t["wim"],
                    t["F1R"], t["F1I"], t["TWR"], t["TWI"], t["F2R"], t["F2I"],
                    t["gdv"], t["pos"], t["task"], N, HALF + 1, self.ncoef,
                    t["scale"], N1=t["N1"], N2=t["N2"], SPLIT=sp,
                    ATOMIC=True, IS_FWD=True, PREC=prec,
                    num_warps=4, num_stages=2)
            parts.append((f"fwd two {t['N1']}x{t['N2']}", med(fn)))
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
            #lanes in the window gap carry g 0 and pos 0, unmasked they all
            #hit bin 0 together and serialise, mask them out entirely
            m3 = m2 & (g != 0.0)
            tl.atomic_add(FRRE + b * n_fr + p, accr * g, mask=m3)
            tl.atomic_add(FRIM + b * n_fr + p, acci * g, mask=m3)

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
            #lanes in the window gap carry g 0 and pos 0. unmasked, every
            #such lane in every program does its memory op on bin 0, the
            #same address across the whole launch, which serialises at the
            #cache and grows with batch. for the 2048 blocks the gap is up
            #to half the tile. mask them out, they contribute nothing
            gm = g != 0.0
            if ATOMIC:
                tl.atomic_add(FRRE + b * n_fr + p, dr * g, mask=gm)
                tl.atomic_add(FRIM + b * n_fr + p, di * g, mask=gm)
            else:
                #the colour classes guarantee no other program touches these
                #bins, plain read add write, no contention
                vr = tl.load(FRRE + b * n_fr + p, mask=gm, other=0.0)
                vi = tl.load(FRIM + b * n_fr + p, mask=gm, other=0.0)
                tl.store(FRRE + b * n_fr + p, vr + dr * g, mask=gm)
                tl.store(FRIM + b * n_fr + p, vi + di * g, mask=gm)
