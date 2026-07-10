"""streamable sliced oct cqt.

the signal is cut into 50 percent overlapped slices of length sl_len under a
tukey window (flat top, hann transitions), each slice goes through the oct cqt
of oct_cqt.py, and the inverse overlap adds the per slice inverses. the tukey
windows sum to one at this overlap so the chain is invertible end to end. the
slicing, window and coefficient arrangement follow xumx sliCQ (slicing.py,
unslicing.py, slicq.py) but tensorised, no python generators per band.

latency is the physics floor of the scheme, one full slice, sl_len samples.
a lower fmin needs a longer lowest window, which needs a longer slice, use
latency_table to see the tradeoff before picking sl_len.

per slice each octave keeps its own time resolution, so a slice yields the
same per octave block list that mr cqtdiff consumes, plus exact dc and nyquist
side blocks that keep the round trip near perfect (drop them and you get the
usual oct mode seams below fmin and near nyquist).
"""
import math
import torch

from .ref_util import hannwin
from .oct_cqt import OctCQT


def makewnd(sl_len, tr_area, device="cpu", dtype=torch.float32):
    #tukey window within one slice, transitions centred at sl_len/4 and 3*sl_len/4,
    #same construction as xumx sliCQ makewnd
    hhop = sl_len // 4
    htr = tr_area // 2
    w = hannwin(2 * tr_area, device=device).to(dtype)
    tw = torch.empty(sl_len, dtype=dtype, device=torch.device(device))
    tw[:hhop - htr] = 0
    tw[hhop - htr:hhop + htr] = w[tr_area:]
    tw[hhop + htr:3 * hhop - htr] = 1
    tw[3 * hhop - htr:3 * hhop + htr] = w[:tr_area]
    tw[3 * hhop + htr:] = 0
    return tw


class SlicedCQT:
    def __init__(self, num_octs=[3, 4, 2], bins_per_oct=[8, 16, 32], fs=44100,
                 sl_len=65536, tr_area=None, device="cpu", dtype=torch.float32):
        assert sl_len % 4 == 0
        if tr_area is None:
            tr_area = sl_len // 4
        assert tr_area % 2 == 0 and 0 < tr_area * 2 <= sl_len

        self.fs = fs
        self.sl_len = sl_len
        self.tr_area = tr_area
        self.H = sl_len // 4      #quarter slice
        self.hop = sl_len // 2    #slice advance
        self.device = torch.device(device)
        self.dtype = dtype

        self.cqt = OctCQT(num_octs, bins_per_oct, fs=fs, audio_len=sl_len,
                          device=device, dtype=dtype)
        self.oct_bins = self.cqt.oct_bins
        self.size_per_oct = self.cqt.size_per_oct
        self.frqs_hz = self.cqt.frqs_hz
        self.tw = makewnd(sl_len, tr_area, device=device, dtype=dtype)

        #odd slices are rolled the other way, precompute both window layouts
        #slice content is rolled a quarter so the tukey plateau alternates,
        #and the coefficients are rolled back by a quarter of each block
        self._roll = [m // 4 for m in self.size_per_oct]

    #latency helpers

    def latency_ms(self, sl_len=None):
        #bounded algorithmic latency of the scheme, one full slice
        return 1000.0 * (sl_len if sl_len is not None else self.sl_len) / self.fs

    def lowest_fmin(self, sl_len=None):
        #lowest fmin whose analysis window still fits comfortably in a slice,
        #the sliCQ rule sl_len >= 8 fs q / f inverted
        sl = sl_len if sl_len is not None else self.sl_len
        n0 = 1
        while n0 < len(self.oct_bins) and self.oct_bins[n0] == self.oct_bins[0]:
            n0 += 1
        b0 = self.oct_bins[0]
        odiv = (n0 + 1) / ((n0 + 1) * b0 - 1)  #lowest group includes a dropped octave
        p = 2.0 ** odiv
        q0 = math.sqrt(p) / (p - 1.0) / 2.0
        return 8.0 * self.fs * q0 / sl

    def latency_table(self, sl_lens=(8192, 16384, 32768, 65536, 131072)):
        rows = []
        f_lowest_needed = float(self.frqs_hz[0])
        for sl in sl_lens:
            fmin = self.lowest_fmin(sl)
            clean = fmin <= f_lowest_needed
            rows.append((sl, self.latency_ms(sl), fmin, clean))
        lines = [f"{'sl_len':>8} {'latency_ms':>11} {'lowest_fmin_hz':>15}   covers fmin {f_lowest_needed:.1f} hz"]
        for sl, ms, fmin, clean in rows:
            lines.append(f"{sl:>8} {ms:>11.1f} {fmin:>15.1f}   {'yes' if clean else 'no'}")
        return "\n".join(lines)

    #slicing

    def _num_slices(self, L):
        nb = -(-L // self.H)  #quarter blocks of input
        return (nb + 1) // 2 + 1

    def _slice(self, x):
        #[B, C, L] to windowed rolled slices [B, C, S, sl_len]
        B, C, L = x.shape
        H = self.H
        S = self._num_slices(L)
        nb = -(-L // H)
        pad = torch.zeros(B, C, 2 * H, dtype=x.dtype, device=x.device)
        tail = torch.zeros(B, C, nb * H - L + 3 * H, dtype=x.dtype, device=x.device)
        xp = torch.cat([pad, x, tail], dim=-1)
        sl = xp.unfold(-1, 4 * H, 2 * H)[:, :, :S, :] * self.tw
        even = (torch.arange(S, device=x.device) % 2 == 0).view(1, 1, -1, 1)
        return torch.where(even, torch.roll(sl, -H, -1), torch.roll(sl, +H, -1))

    def _arrange(self, blocks, S, fwd):
        #undo (fwd) or redo (bwd) the quarter roll in the coefficient domain,
        #even and odd slices roll opposite ways. side blocks stay rolled, they
        #are internal and nothing consumes their time axis
        even = None
        out = []
        for c, r in zip(blocks, self._roll):
            if even is None or even.shape[2] != c.shape[2]:
                even = (torch.arange(S, device=c.device) % 2 == 0).view(1, 1, -1, 1, 1)
            s = 1 if fwd else -1
            out.append(torch.where(even, torch.roll(c, s * r, -1), torch.roll(c, -s * r, -1)))
        return out

    #whole buffer api

    def fwd(self, x):
        """x [B, C, L] real. returns (blocks, side)
        blocks[i] is [B, C, oct_bins[i], S*size_per_oct[i]] complex, slice
        coefficients in time order along the last axis. side is (dc, nyq)
        kept per slice, [B, C, S, 1, M]"""
        B, C, L = x.shape
        sl = self._slice(x)
        blocks, side = self.cqt.fwd(sl)
        S = sl.shape[2]
        blocks = self._arrange(blocks, S, fwd=True)
        blocks = [c.permute(0, 1, 3, 2, 4).reshape(B, C, b, S * m)
                  for c, b, m in zip(blocks, self.oct_bins, self.size_per_oct)]
        return blocks, side

    def bwd(self, blocks, side=None, length=None):
        """inverse of fwd, length trims the tail padding"""
        B, C = blocks[0].shape[:2]
        m0 = self.size_per_oct[0]
        S = blocks[0].shape[-1] // m0
        blocks = [c.reshape(B, C, b, S, m).permute(0, 1, 3, 2, 4)
                  for c, b, m in zip(blocks, self.oct_bins, self.size_per_oct)]
        blocks = self._arrange(blocks, S, fwd=False)
        y = self.cqt.bwd(blocks, side)  #[B, C, S, sl_len]
        H = self.H
        even = (torch.arange(S, device=y.device) % 2 == 0).view(1, 1, -1, 1)
        y = torch.where(even, torch.roll(y, +H, -1), torch.roll(y, -H, -1))
        #overlap add, first and second slice halves land 2H apart
        out = torch.zeros(B, C, (S + 1) * 2 * H, dtype=y.dtype, device=y.device)
        out[..., :S * 2 * H] += y[..., :2 * H].reshape(B, C, -1)
        out[..., 2 * H:] += y[..., 2 * H:].reshape(B, C, -1)
        out = out[..., 2 * H:]  #drop the leading zero pad region
        if length is not None:
            out = out[..., :length]
        return out

    #streaming api, bounded state

    def stream_fwd(self, chunks):
        """chunks yields [B, C, hop] pieces of the signal (pad the last one),
        yields one (blocks, side) per slice, blocks[i] [B, C, oct_bins[i], M_i].
        state is one half slice of input, nothing grows with stream length"""
        prev = None
        k = 0
        def one(prev, chunk, k):
            sl = torch.cat([prev, chunk], dim=-1) * self.tw
            sl = torch.roll(sl, -self.H if k % 2 == 0 else +self.H, -1)
            blocks, side = self.cqt.fwd(sl.unsqueeze(2))  #[B, C, 1, b, M]
            blocks = self._arrange(blocks, 1, fwd=(k % 2 == 0))
            #arrange with S=1 sees every slice as even, flip the direction for odd k
            blocks = [c[:, :, 0] for c in blocks]
            side = [c[:, :, 0] for c in side]
            return blocks, side
        for chunk in chunks:
            if prev is None:
                prev = torch.zeros_like(chunk)  #the leading zero pad
            yield one(prev, chunk, k)
            prev = chunk
            k += 1
        yield one(prev, torch.zeros_like(prev), k)  #flush the tail

    def stream_bwd(self, slice_coeffs):
        """slice_coeffs yields (blocks, side) per slice as stream_fwd emits,
        yields [B, C, hop] output pieces, one slice behind the input.
        state is one half slice of overlap add tail"""
        tail = None
        k = 0
        for blocks, side in slice_coeffs:
            blocks = [c.unsqueeze(2) for c in blocks]
            side = [c.unsqueeze(2) for c in side] if side is not None else None
            blocks = self._arrange(blocks, 1, fwd=not (k % 2 == 0))
            y = self.cqt.bwd(blocks, side)[:, :, 0]
            y = torch.roll(y, +self.H if k % 2 == 0 else -self.H, -1)
            if tail is not None:
                yield tail + y[..., :self.hop]
            tail = y[..., self.hop:]
            k += 1
