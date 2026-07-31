"""level by level reference for the mixed radix band schedules.

pure torch, no torch.fft inside the band transforms, every butterfly
and twiddle written out the way the cuda kernel executes them, so the
table gates in radix_gpu.py test the exact schedule and not just the
end result. lengths 32 to 2048 as products of radices 8 16 32, the in
place shared layouts mirrored by the same index maps.
"""
import math

import torch

#factor order per length, r0 is the high input digit, transformed first
FACTORS = {32: (32,), 64: (8, 8), 128: (16, 8), 256: (16, 16),
           512: (32, 16), 1024: (32, 32), 2048: (32, 8, 8)}

#physical shared address is logical xor logical shifted, a permutation
SWIZZLE_SHIFT = {64: 3, 128: 4, 256: 4, 512: 4, 1024: 5}


def brev(r, device=None):
    #bit reversed indices for a power of two radix
    q = int(math.log2(r))
    x = torch.arange(r, dtype=torch.int64, device=device)
    y = torch.zeros_like(x)
    for _ in range(q):
        y = (y << 1) | (x & 1)
        x >>= 1
    return y


def roots(m, sign, device):
    j = torch.arange(m, dtype=torch.float64, device=device)
    a = sign * 2.0 * math.pi * j / m
    return torch.cos(a).float(), torch.sin(a).float()


def cmul(ar, ai, br, bi):
    #literal four multiply complex product in float32
    return ar * br - ai * bi, ar * bi + ai * br


def warp_dit(xr, xi, sign):
    #branch free dit network, bit reversed load then ascending levels,
    #natural output order, the same primitive the kernel runs per warp
    r = xr.shape[-1]
    b = brev(r, xr.device)
    yr, yi = xr[..., b].clone(), xi[..., b].clone()
    m = 2
    while m <= r:
        h = m // 2
        shape = yr.shape
        rr = yr.reshape(*shape[:-1], -1, m)
        ii = yi.reshape(*shape[:-1], -1, m)
        ur, ui = rr[..., :h].clone(), ii[..., :h].clone()
        vr, vi = rr[..., h:].clone(), ii[..., h:].clone()
        wr, wi = roots(m, sign, yr.device)
        tr, ti = cmul(vr, vi, wr[:h], wi[:h])
        yr = torch.cat((ur + tr, ur - tr), dim=-1).reshape(shape)
        yi = torch.cat((ui + ti, ui - ti), dim=-1).reshape(shape)
        m *= 2
    return yr, yi


def swz(logical, shift):
    return logical ^ (logical >> shift)


def phys2048(k0, n2, d):
    #conflict free bit linear shared map for the three level schedule,
    #the high six bits store d and n2, the bank bits fold k0 in
    b0 = ((k0 >> 0) ^ (d >> 0)) & 1
    b1 = ((k0 >> 1) ^ (d >> 1)) & 1
    b2 = ((k0 >> 2) ^ (d >> 2) ^ (n2 >> 2)) & 1
    b3 = ((k0 >> 3) ^ (n2 >> 0)) & 1
    b4 = ((k0 >> 4) ^ (n2 >> 1)) & 1
    return 32 * (d + 8 * n2) + (b0 | (b1 << 1) | (b2 << 2)
                                | (b3 << 3) | (b4 << 4))


def _transform(x, sign, scale):
    xr = x.real.reshape(-1, x.shape[-1]).contiguous()
    xi = x.imag.reshape(-1, x.shape[-1]).contiguous()
    m = x.shape[-1]
    f = FACTORS[m]
    dev = x.device

    if len(f) == 1:
        yr, yi = warp_dit(xr, xi, sign)

    elif len(f) == 2:
        r0, r1 = f
        sh = SWIZZLE_SHIFT[m]
        sr = torch.empty(xr.shape[0], m, device=dev)
        si = torch.empty_like(sr)
        wr, wi = roots(m, sign, dev)
        #level 0, vector per n1, lanes own k0 after the dit
        for n1 in range(r1):
            n = n1 + r1 * torch.arange(r0, device=dev)
            lr, li = warp_dit(xr[:, n], xi[:, n], sign)
            k0 = torch.arange(r0, device=dev)
            tr, ti = cmul(lr, li, wr[(n1 * k0) % m], wi[(n1 * k0) % m])
            ph = swz(k0 * r1 + n1, sh)
            sr[:, ph] = tr
            si[:, ph] = ti
        #level 1, vector per k0, natural k out
        yr = torch.empty_like(xr)
        yi = torch.empty_like(xi)
        for k0 in range(r0):
            n1 = torch.arange(r1, device=dev)
            ph = swz(k0 * r1 + n1, sh)
            lr, li = warp_dit(sr[:, ph], si[:, ph], sign)
            out = k0 + r0 * torch.arange(r1, device=dev)
            yr[:, out] = lr
            yi[:, out] = li

    else:
        r0, r1, r2 = f
        sr = torch.empty(xr.shape[0], m, device=dev)
        si = torch.empty_like(sr)
        wmr, wmi = roots(m, sign, dev)
        for n1 in range(r1):
            for n2 in range(r2):
                q = n2 + r2 * n1
                n = q + r1 * r2 * torch.arange(r0, device=dev)
                lr, li = warp_dit(xr[:, n], xi[:, n], sign)
                k0 = torch.arange(r0, device=dev)
                tr, ti = cmul(lr, li, wmr[(q * k0) % m], wmi[(q * k0) % m])
                ph = phys2048(k0, n2, n1)
                sr[:, ph] = tr
                si[:, ph] = ti
        #level 1 runs in place, each vector rewrites its own eight cells
        wbr, wbi = roots(r1 * r2, sign, dev)
        for k0 in range(r0):
            for n2 in range(r2):
                d = torch.arange(r1, device=dev)
                pin = phys2048(k0, n2, d)
                lr, li = warp_dit(sr[:, pin], si[:, pin], sign)
                k1 = torch.arange(r1, device=dev)
                e = (n2 * k1) % (r1 * r2)
                tr, ti = cmul(lr, li, wbr[e], wbi[e])
                pout = phys2048(k0, n2, k1)
                sr[:, pout] = tr
                si[:, pout] = ti
        yr = torch.empty_like(xr)
        yi = torch.empty_like(xi)
        for k0 in range(r0):
            for k1 in range(r1):
                n2 = torch.arange(r2, device=dev)
                ph = phys2048(k0, n2, k1)
                lr, li = warp_dit(sr[:, ph], si[:, ph], sign)
                out = k0 + r0 * k1 + r0 * r1 * torch.arange(r2, device=dev)
                yr[:, out] = lr
                yi[:, out] = li

    if scale != 1.0:
        yr, yi = yr * scale, yi * scale
    return torch.complex(yr, yi).reshape(x.shape)


def scheduled_fft(x):
    #synthesis side forward dft
    return _transform(x, -1, 1.0)


def scheduled_ifft(x):
    #analysis side inverse dft with the 1 over m fold
    return _transform(x, 1, 1.0 / x.shape[-1])
