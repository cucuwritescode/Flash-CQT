"""peak memory of one roundtrip call per path, a100.

  modal run bench/cloud.py --cuda --cmd "python bench/memory_bars.py"

for each batch size and path, the transient peak above the resting
allocation while one roundtrip executes, plus each engine's resident
table and workspace footprint. same objects as factored_bars.py.
"""
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import FusedOctCQT, N, _packedfft_ext, _radix_ext
import packedfft as pf

FS = 44100
L = N // 2


def peak_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


def main():
    pext = _packedfft_ext()
    xext = _radix_ext()
    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=N, device="cuda")
    f = FusedOctCQT(cqt, "cuda")
    f.prec = "ieee"
    rx = f.radix
    print("gpu", torch.cuda.get_device_name(0))

    #engine state, the tables each path keeps resident
    tab = sum(t.numel() * t.element_size() for t in rx.values()) / 1e6
    print(f"radix engine tables resident {tab:.1f} MB")

    F, WL, W1k = pf.tables("cuda")
    FR, FI = F.real.contiguous(), F.imag.contiguous()
    b1024 = torch.arange(1024, device="cuda")
    lane = torch.arange(32, device="cuda")
    WLp = WL[(b1024[:, None] * lane[None, :]) % L].reshape(-1)
    g32 = torch.arange(32, device="cuda")
    WKp = W1k[(g32[:, None] * lane[None, :]) % 1024].reshape(-1)
    WLR, WLI = WLp.real.contiguous(), WLp.imag.contiguous()
    WKR, WKI = WKp.real.contiguous(), WKp.imag.contiguous()
    qq = torch.arange(L, device="cuda", dtype=torch.float64)
    WN = torch.exp(-2j * math.pi * qq / N).to(torch.complex64)
    WNR, WNI = WN.real.contiguous(), WN.imag.contiguous()

    print(f"{'B':>4} | {'cufft_MB':>9} | {'flash_MB':>9}")
    for B in (1, 8, 32, 128):
        x = torch.randn(B, N, device="cuda")

        def cu_rt():
            bl, sd = cqt.fwd(x)
            cqt.bwd(bl, sd)

        #every buffer the factored path touches is allocated inside the
        #measured call, so the number is the whole cost of running this
        #batch, nothing the shipped triton path needs is charged to it
        def flash_rt():
            zr = torch.empty(B, L, device="cuda")
            zi = torch.empty(B, L, device="cuda")
            t1r, t1i = torch.empty_like(zr), torch.empty_like(zr)
            t2r, t2i = torch.empty_like(zr), torch.empty_like(zr)
            zpr, zpi = torch.empty_like(zr), torch.empty_like(zr)
            kr = torch.empty(B, f.ncoef, device="cuda")
            ki = torch.empty(B, f.ncoef, device="cuda")
            fr = torch.zeros(B, L + 1, device="cuda")
            fi = torch.zeros(B, L + 1, device="cuda")
            xot = torch.empty(B, N, device="cuda")
            zr.copy_(x[..., 0::2])
            zi.copy_(x[..., 1::2])
            pext.fft3(zr, zi, t1r, t1i, t2r, t2i, zpr, zpi,
                      FR, FI, WLR, WLI, WKR, WKI, B)
            xext.fwd_bands(zpr, zpi, kr, ki,
                           rx["route"], rx["desc"], rx["rr"],
                           rx["ri"], f.ncoef, B)
            xext.inv_bands(kr, ki, fr, fi,
                           rx["gdv"], rx["pos"], rx["desc"], rx["rr"],
                           rx["ri"], f.ncoef, B, 2)
            pext.inv_tail(fr, fi, zr, zi, t1r, t1i, t2r, t2i,
                          zpr, zpi, xot, FR, FI, WLR, WLI, WKR, WKI,
                          WNR, WNI, B)

        with torch.no_grad():
            cu_rt()
            flash_rt()
            a = peak_mb(cu_rt)
            b_ = peak_mb(flash_rt)
        print(f"{B:>4} | {a:>9.1f} | {b_:>9.1f}")
    print("MEM-DONE")


if __name__ == "__main__":
    main()
