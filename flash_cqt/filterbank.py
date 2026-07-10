"""flex oct nsgt filterbank, the analysis and dual windows for the streamable cqt.

builds ONE filterbank covering all octave groups (mr cqtdiff uses one cqt per
resolution group, each with its own full fft, this bank shares a single fft).
window recipes follow repo-refs/CQT_pytorch nsgfwin so the per octave block
sizes reproduce what multi_CQT produces, group locally
  first bin of a group    M = round(fbas/q) with the group log scale q
  every other bin         M = round(fbas[k+1] - fbas[k-1]) with group local
                          neighbours, including the group's dropped top octave
  per octave              M rounded up to the next power of 2, mirrored
dc and nyquist windows are kept as exact side bands (the offline oct mode drops
them, which is fine offline but costs in band accuracy once the signal is
sliced, the slice window leaks energy into them).
"""
import math
import numpy as np
import torch

from .ref_util import hannwin, calcwinrange, nsdual


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


class FlexOctBank:
    """analysis windows, dual windows and octave layout for one audio_len"""

    def __init__(self, num_octs, bins_per_oct, fs, audio_len, min_win=4,
                 device="cpu", dtype=torch.float32):
        if isinstance(num_octs, int):
            num_octs = [num_octs]
        if isinstance(bins_per_oct, int):
            bins_per_oct = [bins_per_oct]
        assert len(num_octs) == len(bins_per_oct)

        self.num_octs = list(num_octs)
        self.bins_per_oct = list(bins_per_oct)
        self.fs = fs
        self.Ls = audio_len
        self.device = torch.device(device)
        self.dtype = dtype

        nf = fs / 2.0
        eps = 1e-6
        scale = float(audio_len) / fs  #hz to bin units

        #top frequency of each group, the consumer rule from mr cqtdiff
        K = len(num_octs)
        fmaxs = [fs * 2.0 ** (-1 - sum(num_octs[j + 1:])) for j in range(K)]

        frq = []   #kept centre frequencies, bin units
        Ms = []    #kept window lengths, bin units, float for now
        for j in range(K):
            n, b = num_octs[j], bins_per_oct[j]
            #groups below nyquist get one extra octave which is then dropped,
            #exactly how multi_CQT builds its sub cqts
            if fmaxs[j] < nf:
                fmax_g = 2.0 * fmaxs[j] - eps
                n_g = n + 1
            else:
                fmax_g = fmaxs[j] - eps
                n_g = n
            fmin_g = fmax_g / 2.0 ** n_g
            nb_g = n_g * b
            odiv = (math.log2(fmax_g) - math.log2(fmin_g)) / (nb_g - 1)
            p = 2.0 ** odiv
            q = math.sqrt(p) / (p - 1.0) / 2.0
            f_all = fmin_g * p ** np.arange(nb_g + 1)  #one beyond for the last spacing
            f_all[-1] = min(f_all[-1], nf)             #last group runs into nyquist
            f_all *= scale
            keep = n * b
            M_g = np.zeros(keep)
            M_g[0] = np.round(f_all[0] / q)
            for k in range(1, keep):
                M_g[k] = np.round(f_all[k + 1] - f_all[k - 1])
            frq.extend(f_all[:keep])
            Ms.extend(M_g)

        lbas = len(frq)
        frq = np.array(frq)
        Ms = np.array(Ms)

        #dc and nyquist windows, offline nsgfwin recipe
        M0 = np.round(2.0 * frq[0])
        Mnyq = np.round(nf * scale - frq[-2])

        #centre positions, the last cqt bin is moved midway to nyquist before
        #rounding, again the offline recipe
        fbas = np.concatenate(((0.0,), frq, (nf * scale,)))
        fbas = np.concatenate((fbas, audio_len - fbas[-2:0:-1]))
        fbas[lbas] = (fbas[lbas - 1] + fbas[lbas + 1]) / 2.0
        fbas[lbas + 2] = audio_len - fbas[lbas]
        rfbas = np.round(fbas).astype(int)

        #full window length vector with the mirrored half, windows are built from
        #these raw lengths, only the block size vector below gets pow2 rounded
        M_raw = np.concatenate(((M0,), Ms, (Mnyq,), Ms[::-1]))
        np.clip(M_raw, min_win, None, out=M_raw)

        #per octave block sizes rounded up to a power of 2, both spectrum halves
        self.oct_bins = [b for n, b in zip(num_octs, bins_per_oct) for _ in range(n)]
        self.size_per_oct = []
        M = M_raw.copy()
        idx = 1
        for b in self.oct_bins:
            value = next_power_of_2(M_raw[idx:idx + b].max())
            self.size_per_oct.append(int(value))
            M[idx:idx + b] = value
            M[len(M) - idx - b:len(M) - idx] = value
            idx += b

        self.M = torch.as_tensor(M, dtype=torch.int64, device=self.device)
        self.g = [hannwin(int(m), device=device).to(dtype) for m in M_raw]
        self.wins, self.nn = calcwinrange(self.g, rfbas, audio_len, device=device)
        assert self.nn == audio_len
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, dtype=dtype, device=device)

        self.fbins = lbas                 #kept cqt bins, one half spectrum
        self.frqs_hz = frq / scale        #centre frequencies in hz
        self.M0 = int(M[0])
        self.Mnyq = int(M[lbas + 1])
