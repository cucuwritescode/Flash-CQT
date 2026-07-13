"""roundtrip snr of the sliced oct cqt under reduced precision.

run   python bench/precision.py

emulates what a tensor core kernel would do to the numbers by rounding every
intermediate of the roundtrip to the target mantissa width, tf32 keeps 10
mantissa bits, bf16 keeps 7, fp32 is the untouched reference. two scenarios

  storage  only the coefficients between forward and inverse are rounded,
           all arithmetic stays fp32. this is the floor for keeping
           coefficients in a smaller dtype
  compute  every intermediate is rounded, windowed slice, spectrum, weighted
           gather, coefficients, and the same stages on the way back. this
           approximates a kernel whose matmuls take rounded inputs and
           accumulate in fp32, which is how tf32 and bf16 tensor cores work

a native bf16 fft would round inside every butterfly stage and land below the
compute row here, so treat the bf16 compute number as its best case. runs on
cpu, the rounding is exact bit arithmetic and does not depend on the device.
"""
import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import SlicedCQT

FS = 44100
LS = 262144
SL = 65536
NUM_OCTS = [3, 4, 2]
BINS = [8, 16, 32]


def round_mantissa(x, bits):
    #round a float32 tensor to the given number of mantissa bits, to nearest
    if x.is_complex():
        return torch.view_as_complex(round_mantissa(torch.view_as_real(x), bits))
    drop = 23 - bits
    i = x.contiguous().view(torch.int32)
    r = (i + (1 << (drop - 1))) & -(1 << drop)
    return r.view(torch.float32)


def q_of(name):
    if name == "fp32":
        return lambda x: x
    if name == "tf32":
        return lambda x: round_mantissa(x, 10)
    if name == "bf16":
        return lambda x: round_mantissa(x, 7)
    raise ValueError(name)


def snr_db(x, xh):
    return (10 * torch.log10((x ** 2).sum() / ((x - xh) ** 2).sum())).item()


def snr_inband_db(x, xh, flo, fhi):
    X = torch.fft.rfft(x, dim=-1)
    Xh = torch.fft.rfft(xh, dim=-1)
    n = x.shape[-1]
    i0, i1 = int(flo * n / FS), int(fhi * n / FS)
    return (10 * torch.log10((X[..., i0:i1].abs() ** 2).sum()
                             / ((X - Xh)[..., i0:i1].abs() ** 2).sum())).item()


def qfwd(s, x, q):
    #the whole buffer forward with q applied after every compute stage
    sl = q(s._slice(x))
    ft = q(torch.fft.rfft(sl))
    blocks = [q(torch.fft.ifft(q(ft[..., idx] * gv)))
              for idx, gv, _, _, _ in s.cqt.groups]
    side = []
    for idx, gv, _, _, cj in s.cqt.side_groups:
        a = torch.where(cj, ft[..., idx].conj(), ft[..., idx]) * gv
        side.append(q(torch.fft.ifft(q(a))))
    blocks = s._arrange(blocks, sl.shape[2], fwd=True)
    return blocks, side


def qbwd(s, blocks, side, length, q):
    #mirror of SlicedCQT.bwd and OctCQT.bwd with q after every compute stage,
    #the index_add accumulation stays fp32 like a tensor core accumulator
    cqt = s.cqt
    S = blocks[0].shape[2]
    blocks = s._arrange(blocks, S, fwd=False)
    lead = blocks[0].shape[:-2]
    n = 1
    for d in lead:
        n *= d
    srcs = [q(torch.fft.fft(c) * gdv).reshape(*lead, -1)
            for c, (_, _, _, gdv, _) in zip(blocks, cqt.groups)]
    srcs += [q(torch.fft.fft(c) * gdv).reshape(*lead, -1)
             for c, (_, _, _, gdv, _) in zip(side, cqt.side_groups)]
    src = torch.cat(srcs, dim=-1).reshape(n, -1)
    fr = torch.zeros(n, cqt.nn // 2 + 1, dtype=src.dtype, device=src.device)
    torch.view_as_real(fr).index_add_(1, cqt.pos_flat, torch.view_as_real(src))
    y = q(torch.fft.irfft(q(fr), n=cqt.nn))
    y = y.reshape(*lead, cqt.nn)[..., :cqt.Ls]
    B, C = lead[0], lead[1]
    H = s.H
    even = (torch.arange(S) % 2 == 0).view(1, 1, -1, 1)
    y = torch.where(even, torch.roll(y, +H, -1), torch.roll(y, -H, -1))
    out = torch.zeros(B, C, (S + 1) * 2 * H, dtype=y.dtype)
    out[..., :S * 2 * H] += y[..., :2 * H].reshape(B, C, -1)
    out[..., 2 * H:] += y[..., 2 * H:].reshape(B, C, -1)
    return out[..., 2 * H:][..., :length]


def main():
    torch.manual_seed(0)
    print(f"torch={torch.__version__} fs={FS} ls={LS} sl_len={SL} "
          f"num_octs={NUM_OCTS} bins={BINS} device=cpu\n")
    s = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device="cpu")
    x = torch.randn(1, 1, LS)
    fmin = float(s.frqs_hz[0])
    flo, fhi = 2 * fmin, 0.95 * FS / 2

    print(f"{'mode':<6} {'scenario':<8} | {'snr_raw_db':>10} {'inband_db':>10}")
    print("-" * 42)
    with torch.no_grad():
        for name in ("fp32", "tf32", "bf16"):
            q = q_of(name)
            #storage, round only what crosses the fwd to bwd boundary
            blocks, side = s.fwd(x)
            xh = s.bwd([q(c) for c in blocks], [q(c) for c in side], length=LS)
            print(f"{name:<6} {'storage':<8} | {snr_db(x, xh):>10.2f} "
                  f"{snr_inband_db(x, xh, flo, fhi):>10.2f}")
            #compute, round every intermediate on both directions
            blocks, side = qfwd(s, x, q)
            xh = qbwd(s, blocks, side, LS, q)
            print(f"{name:<6} {'compute':<8} | {snr_db(x, xh):>10.2f} "
                  f"{snr_inband_db(x, xh, flo, fhi):>10.2f}")


if __name__ == "__main__":
    main()
