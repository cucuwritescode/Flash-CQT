"""window, position and dual frame helpers.

taken from repo-refs/CQT_pytorch/cqt_nsgt_pytorch (thomas grill's nsgt lineage,
eloi moliner's torch port), unchanged apart from trimming what we do not use.
kept verbatim so the bank maths stays bit identical to the offline reference.
"""
import numpy as np
import torch


def _win_dtype(device):
    #mps has no float64 and windows get cast to the target dtype anyway, so float32 on mps is harmless
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def hannwin(l, device="cpu"):
    r = torch.arange(l, dtype=_win_dtype(device), device=torch.device(device))
    r *= np.pi * 2. / l
    r = torch.cos(r)
    r += 1.
    r *= 0.5
    return r


def calcwinrange(g, rfbas, Ls, device="cpu"):
    shift = np.concatenate(((np.mod(-rfbas[-1], Ls),), rfbas[1:] - rfbas[:-1]))

    timepos = np.cumsum(shift)
    nn = timepos[-1]
    timepos -= shift[0]  #positions from the shift vector

    wins = []
    for gii, tpii in zip(g, timepos):
        Lg = len(gii)
        win_range = torch.arange(-(Lg // 2) + tpii, Lg - (Lg // 2) + tpii,
                                 dtype=int, device=torch.device(device))
        win_range %= nn
        wins.append(win_range)

    return wins, nn


def nsdual(g, wins, nn, M=None, dtype=torch.float32, device="cpu"):
    #diagonal of the frame operator, painless case
    x = torch.zeros((nn,), dtype=dtype, device=torch.device(device))
    for gi, mii, sl in zip(g, M, wins):
        xa = torch.square(torch.fft.fftshift(gi))
        xa *= mii
        x[sl] += xa

    gd = [gi / torch.fft.ifftshift(x[wi]) for gi, wi in zip(g, wins)]
    return gd
