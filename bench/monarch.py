"""per slice oct cqt as pure matmuls, the step before writing a fused kernel.

run   python bench/monarch.py            on a cuda machine, paste back the output
run   python bench/monarch.py --check    anywhere, correctness gates only, no gpu

every fft in the per slice transform is replaced by cooley tukey matmul stages,
the 65536 point fft becomes two passes of a 256 by 256 dft matrix with a
twiddle in between, blocks of 512 and up get the same two stage split, blocks
of 256 and under multiply one dft matrix directly. that is the flashfftconv
trick, an fft the tensor cores can eat. this script measures whether the
matmul form beats the cufft form at our sizes on real hardware, and at which
precision, before any triton is written. the precision ladder is

  fp32    plain fp32 matmuls, the correctness reference
  tf32    tensor core matmuls with 10 mantissa bits on the inputs
  3xtf32  each fp32 operand split into two tf32 halves, three matmuls per
          product, near fp32 accuracy at roughly a third of tf32 rate
  bf16    7 mantissa bits, expected to be poor, measured anyway

pointwise steps (window, twiddle, gather, scatter) stay fp32 in every mode,
which is what a real tensor core kernel would do too. it prints roundtrip snr
and per slice ms per mode, cuda graph baselines of the cufft path, and a
batch scaling sweep, then a one line verdict.
"""
import sys
import math
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import SlicedCQT

FS = 44100
SL = 65536
NUM_OCTS = [3, 4, 2]
BINS = [8, 16, 32]
REPS = 200
DIRECT_MAX = 256  #dft sizes up to this multiply one matrix, larger ones split


def snr_db(x, xh):
    d = (x - xh).abs() ** 2
    return (10 * torch.log10((x.abs() ** 2).sum() / d.sum())).item()


#precision controlled complex matmul, complex64 in and out, four real gemms
#so the tf32 flag applies unambiguously

def tf32_round(x):
    #round a float32 tensor to 10 mantissa bits, what tensor cores do to inputs
    i = x.contiguous().view(torch.int32)
    return ((i + (1 << 12)) & -(1 << 13)).view(torch.float32)


def mm3(a, b):
    #3xtf32, split each operand into tf32 high and low halves, three products
    ah, bh = tf32_round(a), tf32_round(b)
    al, bl = a - ah, b - bh
    return ah @ bh + ah @ bl + al @ bh


def cmm(a, b, mode):
    ar, ai = a.real.contiguous(), a.imag.contiguous()
    br, bi = b.real.contiguous(), b.imag.contiguous()
    if mode == "bf16":
        ar, ai, br, bi = (t.to(torch.bfloat16) for t in (ar, ai, br, bi))
        return torch.complex((ar @ br - ai @ bi).float(), (ar @ bi + ai @ br).float())
    if mode == "3xtf32":
        return torch.complex(mm3(ar, br) - mm3(ai, bi), mm3(ar, bi) + mm3(ai, br))
    return torch.complex(ar @ br - ai @ bi, ar @ bi + ai @ br)


def set_mode(mode):
    #fp32 must not silently ride the tensor cores, tf32 modes must
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = mode in ("tf32", "3xtf32")


#dft matrices and the two stage cooley tukey fft

def dft_mat(M, device):
    #F[n, k] = exp(-2 pi i n k / M), angles built from integer residues in
    #float64 so the matrix itself is exact to complex64
    n = torch.arange(M, dtype=torch.int64)
    e = (n[:, None] * n[None, :]) % M
    ang = -2.0 * math.pi * e.to(torch.float64) / M
    return torch.polar(torch.ones_like(ang), ang).to(torch.complex64).to(device)


def twiddle_mat(N1, N2, device):
    #T[n2, k1] = exp(-2 pi i n2 k1 / (N1 N2))
    n2 = torch.arange(N2, dtype=torch.int64)[:, None]
    k1 = torch.arange(N1, dtype=torch.int64)[None, :]
    e = (n2 * k1) % (N1 * N2)
    ang = -2.0 * math.pi * e.to(torch.float64) / (N1 * N2)
    return torch.polar(torch.ones_like(ang), ang).to(torch.complex64).to(device)


def split(M):
    #M = N1 * N2 with N1 <= N2, both powers of two
    N1 = 1 << (int(math.log2(M)) // 2)
    return N1, M // N1


class MatFFT:
    #holds the dft and twiddle matrices for every size the transform needs
    def __init__(self, sizes, device):
        self.direct = {}
        self.two = {}
        for M in sorted(set(sizes)):
            if M <= DIRECT_MAX:
                self.direct[M] = dft_mat(M, device)
            else:
                N1, N2 = split(M)
                self.two[M] = (dft_mat(N1, device), dft_mat(N2, device),
                               twiddle_mat(N1, N2, device))

    def fft(self, x, mode):
        #x [..., M] complex, forward dft along the last axis
        M = x.shape[-1]
        if M in self.direct:
            return cmm(x, self.direct[M], mode)
        F1, F2, T = self.two[M]
        N1, N2 = F1.shape[0], F2.shape[0]
        #cooley tukey, dft over n1, twiddle, dft over n2, output D[k1, k2]
        #with X[k1 + N1 k2] = D[k1, k2]
        A = x.reshape(*x.shape[:-1], N1, N2).transpose(-1, -2).contiguous()
        B = cmm(A, F1, mode) * T
        D = cmm(B.transpose(-1, -2).contiguous(), F2, mode)
        return D.transpose(-1, -2).reshape(*x.shape[:-1], M)

    def ifft(self, x, mode):
        return self.fft(x.conj(), mode).conj() / x.shape[-1]


class MatmulOct:
    #the per slice oct cqt of flash_cqt with every fft in matmul form,
    #gathers, windows and scatters unchanged and in fp32
    def __init__(self, cqt, mode):
        self.cqt = cqt
        self.mode = mode
        sizes = list(cqt.size_per_oct) + [g[1].shape[1] for g in cqt.side_groups]
        self.mats = MatFFT(sizes + [cqt.nn], cqt.device)

    def fwd(self, x):
        #x [N, sl_len] real
        c = self.cqt
        xf = torch.complex(x, torch.zeros_like(x))
        ftf = self.mats.fft(xf, self.mode)
        ft = ftf[..., :c.nn // 2 + 1]
        blocks = [self.mats.ifft(ft[..., idx] * gv, self.mode)
                  for idx, gv, _, _, _ in c.groups]
        side = []
        for idx, gv, _, _, cj in c.side_groups:
            a = torch.where(cj, ft[..., idx].conj(), ft[..., idx]) * gv
            side.append(self.mats.ifft(a, self.mode))
        return blocks, side

    def bwd(self, blocks, side):
        c = self.cqt
        N = blocks[0].shape[0]
        srcs = [(self.mats.fft(b, self.mode) * gdv).reshape(N, -1)
                for b, (_, _, _, gdv, _) in zip(blocks, c.groups)]
        srcs += [(self.mats.fft(b, self.mode) * gdv).reshape(N, -1)
                 for b, (_, _, _, gdv, _) in zip(side, c.side_groups)]
        src = torch.cat(srcs, dim=-1)
        fr = torch.zeros(N, c.nn // 2 + 1, dtype=src.dtype, device=src.device)
        torch.view_as_real(fr).index_add_(1, c.pos_flat, torch.view_as_real(src))
        #rebuild the conjugate half so one full length inverse recovers the signal
        full = torch.cat([fr, fr[..., 1:-1].conj().flip(-1)], dim=-1)
        return self.mats.ifft(full, self.mode).real


def check(device):
    #correctness gates, fp32 matmul form against torch.fft and flash_cqt
    print(f"== correctness gates on {device}, fp32 ==")
    torch.manual_seed(0)
    set_mode("fp32")
    s = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device=device)
    m = MatmulOct(s.cqt, "fp32")
    ok = True

    x = torch.randn(2, SL, device=device)
    xf = torch.complex(x, torch.zeros_like(x))
    v = snr_db(torch.fft.fft(xf), m.mats.fft(xf, "fp32"))
    print(f"fft 65536 two stage vs torch.fft   {v:8.2f} db")
    ok &= v > 100
    for M in sorted(set(s.cqt.size_per_oct)):
        a = torch.randn(2, 8, M, device=device, dtype=torch.complex64)
        v = snr_db(torch.fft.ifft(a), m.mats.ifft(a, "fp32"))
        print(f"ifft {M:>5} matmul form vs torch.fft {v:8.2f} db")
        ok &= v > 100

    with torch.no_grad():
        br, sr = s.cqt.fwd(x)
        bm, sm = m.fwd(x)
        v = min(snr_db(a, b) for a, b in zip(br + sr, bm + sm))
        print(f"forward blocks vs flash_cqt, worst  {v:8.2f} db")
        ok &= v > 100
        xh = m.bwd(bm, sm)[..., :SL]
        v = snr_db(x, xh)
        print(f"per slice roundtrip matmul form     {v:8.2f} db")
        ok &= v > 100
    print("gates", "PASSED" if ok else "FAILED")
    return ok


def ev_ms(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
    ts.sort()
    return ts[len(ts) // 2]


def capture(fn):
    #warm up on a side stream then record one cuda graph of fn
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def peak_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def main():
    if not torch.cuda.is_available():
        print("no cuda device found, run with --check for the cpu gates")
        sys.exit(1)
    dev = "cuda"
    print(f"gpu {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"cuda {torch.version.cuda}")
    print(f"config fs {FS} sl_len {SL} num_octs {NUM_OCTS} bins {BINS}\n")
    if not check(dev):
        print("aborting, the fp32 gates must pass before any timing means anything")
        sys.exit(1)

    torch.manual_seed(0)
    s = SlicedCQT(NUM_OCTS, BINS, fs=FS, sl_len=SL, device=dev)
    x = torch.randn(1, SL, device=dev)

    def flash_rt(inp):
        b, sd = s.cqt.fwd(inp)
        return s.cqt.bwd(b, sd)

    #the cufft baselines the matmul form has to beat
    print("\n== per slice roundtrip, B 1 ==")
    print(f"{'path':<16} | {'snr_db':>8} | {'eager_ms':>9} {'graph_ms':>9} | {'peak_MB':>8}")
    print("-" * 62)
    with torch.no_grad():
        e = ev_ms(lambda: flash_rt(x), REPS)
        g = capture(lambda: flash_rt(x))
        gm = ev_ms(g.replay, REPS)
        v = snr_db(x, flash_rt(x)[..., :SL])
        p = peak_mb(lambda: flash_rt(x))
        print(f"{'cufft (flash)':<16} | {v:>8.2f} | {e:>9.3f} {gm:>9.3f} | {p:>8.1f}")
        base_graph = gm

        best = None
        for mode in ("fp32", "3xtf32", "tf32", "bf16"):
            set_mode(mode)
            m = MatmulOct(s.cqt, mode)
            rt = lambda: m.bwd(*m.fwd(x))[..., :SL]
            v = snr_db(x, rt())
            e = ev_ms(rt, REPS)
            g = capture(rt)
            gm = ev_ms(g.replay, REPS)
            p = peak_mb(rt)
            print(f"{'matmul ' + mode:<16} | {v:>8.2f} | {e:>9.3f} {gm:>9.3f} | {p:>8.1f}")
            if v > 100 and (best is None or gm < best[1]):
                best = (mode, gm, v)
        set_mode("fp32")

    #batch scaling, where launch bound turns into bandwidth bound
    print("\n== batch scaling, per slice roundtrip, graphed ==")
    print(f"{'B':>4} | {'cufft_ms':>9} {'slices/s':>9} | ", end="")
    print(f"{'mm_fp32_ms':>10} {'slices/s':>9} | {'mm_tf32_ms':>10} {'slices/s':>9}")
    print("-" * 78)
    with torch.no_grad():
        for B in (1, 8, 32, 128):
            xb = torch.randn(B, SL, device=dev)
            g = capture(lambda: flash_rt(xb))
            a = ev_ms(g.replay, REPS)
            set_mode("fp32")
            m = MatmulOct(s.cqt, "fp32")
            g = capture(lambda: m.bwd(*m.fwd(xb)))
            b = ev_ms(g.replay, REPS)
            set_mode("tf32")
            m = MatmulOct(s.cqt, "tf32")
            g = capture(lambda: m.bwd(*m.fwd(xb)))
            c = ev_ms(g.replay, REPS)
            set_mode("fp32")
            print(f"{B:>4} | {a:>9.3f} {1000 * B / a:>9.0f} | "
                  f"{b:>10.3f} {1000 * B / b:>9.0f} | {c:>10.3f} {1000 * B / c:>9.0f}")

    print()
    if best is None:
        print("verdict no matmul mode held 100 db, the matmul form is not usable "
              "as built, fix precision before any triton")
    elif best[1] < base_graph:
        print(f"verdict matmul form wins, {best[0]} graphed {best[1]:.3f} ms vs "
              f"cufft graphed {base_graph:.3f} ms at {best[2]:.1f} db, fusing it "
              "into triton is worth it and these matmuls are the kernel's shape")
    else:
        print(f"verdict matmul form loses eager for eager, best {best[0]} graphed "
              f"{best[1]:.3f} ms vs cufft graphed {base_graph:.3f} ms at "
              f"{best[2]:.1f} db, a triton kernel must win on fusion (fewer "
              "launches, less hbm traffic), not on matmul speed alone")


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check("cpu") else 1)
    main()
