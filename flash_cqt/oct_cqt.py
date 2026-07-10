"""invertible oct mode cqt over one buffer, the per slice engine of the sliced cqt.

forward is one rfft then gather and multiply per octave, inverse is one flat
index_add then one irfft, both taken from the optimised oct path in
repo-refs/CQT_pytorch (bit identical for the octave blocks). on top of that the
dc and nyquist bands are kept as exact side blocks. their windows wrap around
bin 0 and bin nn/2 where the spectrum of a real signal is conjugate symmetric,
so the forward gathers the mirrored bins with a conjugate mask and the inverse
scatters only the positions inside [0, nn/2] (the rest is implied by the
hermitian symmetry that irfft enforces). no approximation is involved, unlike
the oct_complete mode of the reference which is only accurate to about 60 db.
"""
import torch

from .filterbank import FlexOctBank


class OctCQT:
    def __init__(self, num_octs, bins_per_oct, fs=44100, audio_len=44100,
                 device="cpu", dtype=torch.float32):
        self.bank = FlexOctBank(num_octs, bins_per_oct, fs, audio_len,
                                device=device, dtype=dtype)
        bank = self.bank
        self.Ls = audio_len
        self.nn = bank.nn
        self.device = torch.device(device)
        self.dtype = dtype
        self.numocts = len(bank.oct_bins)
        self.oct_bins = bank.oct_bins
        self.size_per_oct = bank.size_per_oct
        self.frqs_hz = bank.frqs_hz

        nn = self.nn
        half = nn // 2  #nyquist bin index

        def enc_dec_group(band_idx, Lo):
            #gather index, window values, scatter index and dual values for a
            #group of bands that share one block length Lo
            nb = len(band_idx)
            idx = torch.zeros((nb, Lo), dtype=torch.int64, device=self.device)
            gv = torch.zeros((nb, Lo), dtype=self.dtype, device=self.device)
            pos = torch.zeros((nb, Lo), dtype=torch.int64, device=self.device)
            gdv = torch.zeros((nb, Lo), dtype=self.dtype, device=self.device)
            for k, bi in enumerate(band_idx):
                wr = bank.wins[bi]
                Lg = wr.shape[0]
                r, l = (Lg + 1) // 2, Lg // 2
                gs = torch.fft.fftshift(bank.g[bi]).to(self.dtype)
                #forward layout, first half of the support then the wrapped tail
                idx[k, :r] = wr[l:]
                idx[k, Lo - l:] = wr[:l]
                gv[k, :r] = gs[l:]
                gv[k, Lo - l:] = gs[:l]
                #inverse layout, dual window split as [left, zeros, right] times Lo
                gd = bank.gd[bi].to(self.dtype) * Lo
                gdv[k, :r] = gd[:r]
                gdv[k, Lo - l:] = gd[r:]
                pos[k, :r] = wr[l:]
                pos[k, Lo - l:] = wr[:l]
            #bins past nyquist belong to the conjugate half of a real spectrum
            conj = idx > half
            idx = torch.where(conj, nn - idx, idx)
            #scatter only inside [0, nn/2], the rest is implied by hermitian symmetry
            valid = pos <= half
            gdv = gdv * valid
            pos = torch.where(valid, pos, torch.zeros_like(pos))
            return idx, gv, pos, gdv, conj

        #octave groups, then dc and nyquist as single band side groups
        self.groups = []
        start = 1
        for i, b in enumerate(self.oct_bins):
            grp = enc_dec_group(range(start, start + b), self.size_per_oct[i])
            assert not grp[4].any()  #octave windows never wrap
            self.groups.append(grp)
            start += b
        self.side_groups = [enc_dec_group([0], bank.M0),
                            enc_dec_group([bank.fbins + 1], bank.Mnyq)]

        #one flat scatter index over everything for a single index_add inverse
        self.pos_flat = torch.cat([g[2].reshape(-1) for g in self.groups]
                                  + [g[2].reshape(-1) for g in self.side_groups])

    def fwd(self, x):
        """x [..., Ls] real, returns (blocks, side)
        blocks[i] is [..., oct_bins[i], size_per_oct[i]] complex
        side is (dc, nyq), each [..., 1, M] complex"""
        assert x.shape[-1] == self.Ls
        ft = torch.fft.rfft(x)
        blocks = [torch.fft.ifft(ft[..., idx] * gv) for idx, gv, _, _, _ in self.groups]
        side = []
        for idx, gv, _, _, conj in self.side_groups:
            a = ft[..., idx]
            a = torch.where(conj, a.conj(), a) * gv
            side.append(torch.fft.ifft(a))
        return blocks, side

    def bwd(self, blocks, side=None):
        """inverse of fwd, side may be None to discard the dc and nyquist bands"""
        lead = blocks[0].shape[:-2]
        n = 1
        for s in lead:
            n *= s
        srcs = [(torch.fft.fft(c) * gdv).reshape(*lead, -1)
                for c, (_, _, _, gdv, _) in zip(blocks, self.groups)]
        if side is not None:
            srcs += [(torch.fft.fft(c) * gdv).reshape(*lead, -1)
                     for c, (_, _, _, gdv, _) in zip(side, self.side_groups)]
            pos = self.pos_flat
        else:
            pos = self.pos_flat[:sum(s.shape[-1] for s in srcs)]
        src = torch.cat(srcs, dim=-1).reshape(n, -1)
        fr = torch.zeros(n, self.nn // 2 + 1, dtype=src.dtype, device=src.device)
        #real and imag added apart so it also runs on mps, no complex index_add there
        torch.view_as_real(fr).index_add_(1, pos, torch.view_as_real(src))
        sig = torch.fft.irfft(fr, n=self.nn)
        return sig.reshape(*lead, self.nn)[..., :self.Ls]
