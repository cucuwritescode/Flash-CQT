"""adjoint gates for training support, cpu, no kernels involved.

  python bench/autograd_check.py

the transform is linear so its gradient is its adjoint. candidate one,
backward of the analysis, is the synthesis plumbing with the analysis
window in place of the dual. candidate two, backward of the synthesis,
is the analysis plumbing with the dual window in place of the window.
both are checked against torch autograd through the eager transform.
any clean constant scale between candidate and reference is measured,
asserted identical across draws, and reported so the kernel wrapper
can bake it.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import (FusedOctCQT, N, HALF, radix_load_perm,
                             radix_store_perm)

FS = 44100
L = N // 2


def snr(a, b):
    p = a.double().square().sum()
    e = (a.double() - b.double()).square().sum()
    return float("inf") if float(e) == 0 else float(10 * torch.log10(p / e))


def packed_spectrum(x):
    z = torch.complex(x[..., 0::2].contiguous(), x[..., 1::2].contiguous())
    Z = torch.fft.fft(z)
    q = torch.arange(L, device=x.device)
    zperm = (q & 31) * 1024 + ((q >> 5) & 31) * 32 + (q >> 10)
    zp = torch.empty_like(Z)
    zp[..., zperm] = Z
    return zp


def main():
    torch.manual_seed(11)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=N, device="cpu")
    f = FusedOctCQT(cqt, "cpu")
    rx = f.radix
    rows = rx["desc"].tolist()
    B = 2

    for draw in range(2):
        #reference one, gradient of the analysis by autograd
        x = torch.randn(B, N, requires_grad=True)
        blocks, side = cqt.fwd(x)
        allc = list(blocks) + list(side)
        ys = [torch.randn_like(torch.view_as_real(c)) for c in allc]
        loss = sum((torch.view_as_real(c) * y).sum()
                   for c, y in zip(allc, ys))
        loss.backward()
        gref = x.grad.detach().clone()

        #candidate one, scatter plumbing with the analysis window at the
        #analysis gather indices, conj taps conjugate their contribution,
        #the per band 1 over m is the adjoint of the analysis ifft
        groups = list(cqt.groups) + list(cqt.side_groups)
        frr = torch.zeros(B, HALF + 1)
        fri = torch.zeros(B, HALF + 1)
        for y, (idx, gv, pos, gdv, conj) in zip(ys, groups):
            m = idx.shape[1]
            d = torch.fft.fft(torch.view_as_complex(y)) / m
            sgn = torch.where(conj, -1.0, 1.0)
            val_r = (d.real * gv).reshape(B, -1)
            val_i = (d.imag * gv * sgn).reshape(B, -1)
            flat = idx.reshape(-1)
            frr.index_add_(1, flat, val_r)
            fri.index_add_(1, flat, val_i)
        #adjoint of rfft weighs dc and nyquist half of what irfft does,
        #double those rows before the tail so one clean scale remains
        frr[:, 0] *= 2.0
        frr[:, HALF] *= 2.0
        fri[:, 0] *= 2.0
        fri[:, HALF] *= 2.0
        cand = torch.fft.irfft(torch.complex(frr, fri), N)
        a1 = float((gref.double() * cand.double()).sum()
                   / cand.double().square().sum())
        s1 = snr(gref, a1 * cand)

        #reference two, gradient of the synthesis by autograd
        blocks2, side2 = cqt.fwd(torch.randn(B, N))
        leaves = [c.detach().clone().requires_grad_(True)
                  for c in list(blocks2) + list(side2)]
        ng = len(cqt.groups)
        u = torch.randn(B, N)
        xr = cqt.bwd(leaves[:ng], leaves[ng:])
        (xr * u).sum().backward()
        grefs = [lv.grad.detach().clone().reshape(B, -1) for lv in leaves]
        grefh = torch.cat(grefs, dim=-1)

        #candidate two, analysis plumbing with the dual window taps,
        #the per band m is the adjoint of the synthesis fft
        zp = packed_spectrum(u)
        candh = torch.zeros_like(grefh)
        for m, ab, ob, _ in rows:
            lp = radix_load_perm(m)
            Ad = torch.empty(m, dtype=torch.complex64)
            gd = torch.empty(m)
            i1 = torch.empty(m, dtype=torch.long)
            i2 = torch.empty(m, dtype=torch.long)
            Ad[lp] = torch.complex(rx["dar"][ab:ab + m],
                                   rx["dai"][ab:ab + m])
            gd[lp] = rx["gdw"][ab:ab + m]
            i1[lp] = rx["p1"][ab:ab + m].long()
            i2[lp] = rx["p2"][ab:ab + m].long()
            Bd = torch.complex(gd - Ad.real, -Ad.imag)
            #the adjoint of irfft halves the dc and nyquist bins, those
            #taps are exactly the ones with both indices at zero
            edge = (i1 == 0) & (i2 == 0)
            v = Ad * zp[:, i1] + Bd * zp[:, i2].conj()
            v[:, edge] *= 0.5
            candh[:, ob:ob + m] = torch.fft.ifft(v) * m
        a2 = float((torch.view_as_real(grefh).double()
                    * torch.view_as_real(candh).double()).sum()
                   / torch.view_as_real(candh).double().square().sum())
        s2 = snr(torch.view_as_real(grefh),
                 a2 * torch.view_as_real(candh))

        print(f"draw {draw}  adj fwd {s1:7.2f} db scale {a1:.10g}   "
              f"adj bwd {s2:7.2f} db scale {a2:.10g}")
        if draw == 0:
            a1p, a2p = a1, a2

    same = abs(a1 - a1p) < 1e-6 * abs(a1) + 1e-12 \
        and abs(a2 - a2p) < 1e-6 * abs(a2) + 1e-12
    clean = abs(a1 - N / 2) < 1e-3 and abs(a2 - 2.0 / N) < 1e-9
    ok = s1 >= 120 and s2 >= 120 and same and clean
    print(f"scales stable across draws: {same}, clean "
          f"(N/2 and 2/N): {clean}")
    print("ADJOINT-GATES-" + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
