"""the packed big fft as three radix 32 passes, sim and kernel gate.

the factored transform wants F_32768 of the packed signal as three uniform
radix 32 layers (32768 = 32^3). this file holds the exact index scheme as a
torch sim, validated against torch.fft, and the gate for the cuda kernel
that transcribes it.

run   python bench/packedfft.py            local sim gates, cpu, no gpu
run   modal run bench/cloud.py --cuda --cmd "python bench/packedfft.py --kernel"
"""
import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

L = 32768
R = 32
N = 65536


def snr_db(x, xh):
    e = x.abs().double().square().sum()
    d = (x - xh).abs().double().square().sum()
    return float(10 * torch.log10(e / d)) if d > 0 else float("inf")


def tables(device="cpu"):
    #the 32 point dft matrix and the two inter pass twiddle tables
    j = torch.arange(R, dtype=torch.float64)
    F = torch.exp(-2j * math.pi * j[:, None] * j[None, :] / R).to(torch.complex64)
    kL = torch.arange(L, dtype=torch.float64)
    WL = torch.exp(-2j * math.pi * kL / L).to(torch.complex64)
    k1k = torch.arange(1024, dtype=torch.float64)
    W1k = torch.exp(-2j * math.pi * k1k / 1024).to(torch.complex64)
    return F.to(device), WL.to(device), W1k.to(device)


def fft3(z, F, WL, W1k):
    """F_32768 as three radix 32 passes, dit, n = a 1024 + a2 32 + b2,
    k = k1 + 32 k2 + 1024 k3. pass one sums a, pass two a2, pass three b2"""
    B = z.shape[0]
    #pass one, for each b in [0, 1024), c1[b, k1] = sum_a z[a*1024+b] F[a,k1] WL[b*k1]
    zv = z.view(B, R, 1024)                     #[a, b]
    c1 = torch.einsum("zab,ak->zbk", zv, F)     #[b, k1]
    b = torch.arange(1024)
    k1 = torch.arange(R)
    c1 = c1 * WL[(b[:, None] * k1[None, :]) % L]
    #pass two, b = a2 32 + b2, for each (k1, b2),
    #c2[k1, b2, k2] = sum_a2 c1[a2 32 + b2, k1] F[a2, k2] W1k[b2 k2]
    c1v = c1.view(B, R, R, R)                   #[a2, b2, k1]
    c2 = torch.einsum("zabk,aj->zkbj", c1v, F)  #[k1, b2, k2]
    b2 = torch.arange(R)
    c2 = c2 * W1k[(b2[:, None] * k1[None, :]) % 1024]
    #pass three, Z[k1 + 32 k2 + 1024 k3] = sum_b2 c2[k1, b2, k2] F[b2, k3].
    #the [k3, k2, k1] order flattens row major straight to k, no permute
    Z = torch.einsum("zkbj,bm->zmjk", c2, F)    #[k3, k2, k1]
    return Z.reshape(B, L)


def sim_gate():
    torch.manual_seed(0)
    F, WL, W1k = tables()
    z = torch.randn(2, L, dtype=torch.complex64)
    ref = torch.fft.fft(z)
    out = fft3(z, F, WL, W1k)
    v = snr_db(ref, out)
    print(f"fft3 sim vs torch.fft      {v:8.2f} db  ->",
          "PASS" if v > 120 else "FAIL")
    #and the packed real path end to end
    x = torch.randn(2, N)
    zp = torch.complex(x[..., 0::2], x[..., 1::2])
    Z = fft3(zp, F, WL, W1k)
    k = torch.arange(L + 1)
    a = Z[..., k % L]
    bb = Z[..., (-k) % L].conj()
    w = torch.exp(-2j * math.pi * torch.arange(L + 1, dtype=torch.float64) / N
                  ).to(torch.complex64)
    X = 0.5 * ((a + bb) - 1j * w * (a - bb))
    vr = snr_db(torch.fft.rfft(x), X)
    print(f"packed rfft via fft3       {vr:8.2f} db  ->",
          "PASS" if vr > 120 else "FAIL")
    return v > 120 and vr > 120


def kernel_gate():
    from flash_cqt.fused import _packedfft_ext
    ext = _packedfft_ext()
    if not ext:
        print("packedfft extension failed to build")
        sys.exit(1)
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    torch.manual_seed(0)
    F, WL, W1k = tables("cuda")
    FR, FI = F.real.contiguous(), F.imag.contiguous()
    #twiddles reordered into the access order of the kernel, [transform,
    #lane], so the per lane reads coalesce instead of scattering over the
    #table
    b = torch.arange(1024, device="cuda")
    lane = torch.arange(32, device="cuda")
    WLp = WL[(b[:, None] * lane[None, :]) % L].reshape(-1)
    g2 = torch.arange(32, device="cuda")
    WKp = W1k[(g2[:, None] * lane[None, :]) % 1024].reshape(-1)
    WLR, WLI = WLp.real.contiguous(), WLp.imag.contiguous()
    WKR, WKI = WKp.real.contiguous(), WKp.imag.contiguous()
    for B in (1, 32):
        x = torch.randn(B, N, device="cuda")
        zr = x[..., 0::2].contiguous()
        zi = x[..., 1::2].contiguous()
        ref = torch.fft.fft(torch.complex(zr, zi))
        outr = torch.empty(B, L, device="cuda")
        outi = torch.empty(B, L, device="cuda")
        t1r = torch.empty(B, L, device="cuda")
        t1i = torch.empty(B, L, device="cuda")
        t2r = torch.empty(B, L, device="cuda")
        t2i = torch.empty(B, L, device="cuda")
        ext.fft3(zr, zi, t1r, t1i, t2r, t2i, outr, outi,
                 FR, FI, WLR, WLI, WKR, WKI, B)
        #the kernel stores the permuted layout Zp[k1][k2][k3], the router
        #tables absorb it, the gate compares through the permutation
        out = torch.complex(outr, outi).view(B, 32, 32, 32).permute(0, 3, 2, 1)
        v = snr_db(ref.view(B, 32, 32, 32), out)
        print(f"B {B:>3}  kernel fft3 vs torch.fft {v:8.2f} db  ->",
              "PASS" if v > 120 else "FAIL")

        def med(fn, reps=50):
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

        tk = med(lambda: ext.fft3(zr, zi, t1r, t1i, t2r, t2i, outr, outi,
                                  FR, FI, WLR, WLI, WKR, WKI, B))
        tc = med(lambda: torch.fft.rfft(x))
        print(f"B {B:>3}  kernel {tk:7.3f} ms   cufft rfft {tc:7.3f} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="store_true")
    args = ap.parse_args()
    if args.kernel:
        if not sim_gate():
            sys.exit(1)
        kernel_gate()
    else:
        sys.exit(0 if sim_gate() else 1)
