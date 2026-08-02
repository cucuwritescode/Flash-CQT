"""gradient gates for the trainable wrapper, gpu.

  modal run bench/cloud.py --cuda --cmd "python bench/autograd_gpu.py"

values and gradients of the factored transform against the eager
transform and its automatic gradients, both directions and the full
chain, then a training style timing, forward plus loss plus backward.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from flash_cqt import OctCQT
from flash_cqt.fused import N
from flash_cqt.trainable import TrainableCQT

FS = 44100


def snr(a, b):
    p = a.double().square().sum()
    e = (a.double() - b.double()).square().sum()
    return float("inf") if float(e) == 0 else float(10 * torch.log10(p / e))


def csnr(a, b):
    return snr(torch.view_as_real(a), torch.view_as_real(b))


def med(fn, reps=30):
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


def main():
    torch.manual_seed(0)
    cqt = OctCQT([3, 4, 2], [8, 16, 32], fs=FS, audio_len=N, device="cuda")
    t = TrainableCQT(cqt)
    print("gpu", torch.cuda.get_device_name(0), " torch", torch.__version__)
    B = 4
    ok = True

    #values first, the wrapper must match the eager transform
    x = torch.randn(B, N, device="cuda")
    with torch.no_grad():
        rb, rs = cqt.fwd(x)
        cb, cs = t.fwd(x)
        vdb = min(csnr(a, b) for a, b in zip(list(rb) + list(rs),
                                             list(cb) + list(cs)))
        rdb = snr(x, t.bwd(cb, cs))
    print(f"forward values vs eager {vdb:7.2f} db  roundtrip {rdb:7.2f} db")
    ok &= vdb >= 118 and rdb >= 118

    #gradient of the analysis
    ys = [torch.randn_like(torch.view_as_real(c)) for c in list(rb) + list(rs)]

    def an_loss(fwd_fn, xx):
        bl, sd = fwd_fn(xx)
        return sum((torch.view_as_real(c) * y).sum()
                   for c, y in zip(list(bl) + list(sd), ys))

    xe = x.clone().requires_grad_(True)
    an_loss(cqt.fwd, xe).backward()
    xf = x.clone().requires_grad_(True)
    an_loss(t.fwd, xf).backward()
    g1 = snr(xe.grad, xf.grad)
    print(f"analysis gradient vs autograd {g1:7.2f} db")
    ok &= g1 >= 118

    #gradient of the synthesis
    u = torch.randn(B, N, device="cuda")
    le = [c.detach().clone().requires_grad_(True) for c in list(rb) + list(rs)]
    ng = len(cqt.groups)
    (cqt.bwd(le[:ng], le[ng:]) * u).sum().backward()
    lf = [c.detach().clone().requires_grad_(True) for c in list(rb) + list(rs)]
    (t.bwd(lf[:ng], lf[ng:]) * u).sum().backward()
    g2 = min(csnr(a.grad, b.grad) for a, b in zip(le, lf))
    print(f"synthesis gradient vs autograd {g2:7.2f} db")
    ok &= g2 >= 118

    #the full chain, a training shaped loss through both directions
    tgt = torch.randn(B, N, device="cuda")
    xe = x.clone().requires_grad_(True)
    ((cqt.bwd(*cqt.fwd(xe)) - tgt) ** 2).mean().backward()
    xf = x.clone().requires_grad_(True)
    ((t.bwd(*t.fwd(xf)) - tgt) ** 2).mean().backward()
    g3 = snr(xe.grad, xf.grad)
    print(f"roundtrip chain gradient vs autograd {g3:7.2f} db")
    ok &= g3 >= 115

    #training style timing, forward plus loss plus backward
    print(f"\n{'B':>4} | {'eager_ms':>9} | {'flash_ms':>9} | speedup")
    for Bt in (1, 8, 32):
        xb = torch.randn(Bt, N, device="cuda")

        def step(fwd_fn, bwd_fn):
            xg = xb.clone().requires_grad_(True)
            ((bwd_fn(*fwd_fn(xg)) - tgt[:1]) ** 2).mean().backward()

        a = med(lambda: step(cqt.fwd, cqt.bwd))
        b_ = med(lambda: step(t.fwd, t.bwd))
        print(f"{Bt:>4} | {a:>9.3f} | {b_:>9.3f} | {a / b_:>6.2f}x")

    print("\nGRAD-GATES-" + ("OK" if ok else "FAIL"))


if __name__ == "__main__":
    main()
