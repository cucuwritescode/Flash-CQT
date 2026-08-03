"""training step profile for mr cqtdiff, where does the time go.

run from MR-CQTdiff/src with this repo on the path, one command

  cp <flash-cqt>/bench/profile_step.py .
  PYTHONPATH=<flash-cqt> python profile_step.py
  PYTHONPATH=<flash-cqt> python profile_step.py --conf conf_OpenSinger \\
      --batches 1,4,8 --reps 15 --hours 120

no dataset and no checkpoint needed, random weights and random audio
time the same as trained ones. the script builds the network exactly
as train.py does, runs full training iterations, and splits the step
with cuda events into transform analysis, transform synthesis, the
network forward, the synthesis backward (tensor hooks at the transform
boundary), and the rest of backward plus the optimiser. the loss here
is the plain mean of the squared error, the sigma weighting is a
pointwise multiply and does not move the clock.

if flash_cqt imports, the same in step transform work is also timed
in isolation for both transforms (the legacy multi cqt against the
sliced trainable flash path, roundtrip gated before timing counts)
and a projected step time is printed. the projection swaps transform
costs only and assumes the network cost unchanged, the sliced block
layout has a different time resolution per slice than the offline
transform, so treat the projection as the amdahl bound, the real
number needs the swap wired in.
"""
import argparse
import sys

import torch


def med_ms(fn, reps):
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


class Seg:
    """event pairs around every call of one transform direction, plus
    hooks that bracket the synthesis backward at the tensor boundary"""

    def __init__(self, host, name, hook_backward=False):
        self.host = host
        self.name = name
        self.orig = getattr(host, name)
        self.hook_backward = hook_backward
        self.reset()
        setattr(host, name, self)

    def reset(self):
        self.pairs = []
        self.bwd_start = []
        self.bwd_end = []

    def __call__(self, *a, **k):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = self.orig(*a, **k)
        e.record()
        self.pairs.append((s, e))
        if self.hook_backward:
            #grad of the output is ready just before the synthesis
            #adjoint runs, grads of the inputs just after it
            def mark_start(_g):
                ev = torch.cuda.Event(True)
                ev.record()
                self.bwd_start.append(ev)

            def mark_end(_g):
                ev = torch.cuda.Event(True)
                ev.record()
                self.bwd_end.append(ev)

            if torch.is_tensor(out) and out.requires_grad:
                out.register_hook(mark_start)
            ins = a[0] if a else []
            for t_ in (ins if isinstance(ins, (list, tuple)) else [ins]):
                if torch.is_tensor(t_) and t_.requires_grad:
                    t_.register_hook(mark_end)
        return out

    def ms(self):
        return sum(s.elapsed_time(e) for s, e in self.pairs)

    def bwd_ms(self):
        if not self.bwd_start or not self.bwd_end:
            return 0.0
        return max(s.elapsed_time(e) for s in self.bwd_start
                   for e in self.bwd_end)

    def restore(self):
        setattr(self.host, self.name, self.orig)


def build(conf, overrides):
    import hydra
    from hydra import compose, initialize
    with initialize(config_path="conf", version_base=None):
        args = compose(config_name=conf, overrides=overrides)
    net = hydra.utils.instantiate(args.network).cuda()
    dp = hydra.utils.instantiate(args.diff_params)
    opt = hydra.utils.instantiate(args.exp.optimizer, params=net.parameters())
    return args, net, dp, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="conf_FMA")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--hours", type=float, default=120.0)
    ap.add_argument("overrides", nargs="*", default=[])
    args_cli = ap.parse_args()

    args, net, dp, opt = build(args_cli.conf, args_cli.overrides)
    L = int(args.exp.audio_len)
    print(f"gpu {torch.cuda.get_device_name(0)}  audio_len {L}  "
          f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

    segf = Seg(net.CQTransform, "fwd", hook_backward=False)
    segb = Seg(net.CQTransform, "bwd", hook_backward=True)

    for B in [int(b) for b in args_cli.batches.split(",")]:
        x = torch.randn(B, L, device="cuda")

        def step():
            opt.zero_grad(set_to_none=True)
            segf.reset()
            segb.reset()
            err2, _ = dp.loss_fn(net, x)
            err2.mean().backward()
            opt.step()

        total = med_ms(step, args_cli.reps)
        torch.cuda.synchronize()
        cf, cb, sb = segf.ms(), segb.ms(), segb.bwd_ms()
        share = 100.0 * (cf + cb + sb) / total
        print(f"\nB {B}  training step {total:.1f} ms  "
              f"(medians of {args_cli.reps}, last step segmented)")
        print(f"  analysis fwd      {cf:8.2f} ms")
        print(f"  synthesis fwd     {cb:8.2f} ms")
        print(f"  synthesis bwd     {sb:8.2f} ms")
        print(f"  everything else   {total - cf - cb - sb:8.2f} ms  "
              f"(network fwd, network bwd, loss, optimiser)")
        print(f"  transform share   {share:8.1f} %")

        #isolated transform work at the same shapes, both engines
        try:
            from flash_cqt import OctCQT, SlicedCQT
            from flash_cqt.trainable import TrainableCQT
        except ImportError as ex:
            print("  flash_cqt not importable, no projection,", ex)
            continue
        xin = torch.randn(B, 1, L, device="cuda")

        def iso(tf_fwd, tf_bwd):
            bl = tf_fwd(xin)
            if isinstance(bl, tuple):
                lv = [c.detach().requires_grad_(True) for c in
                      list(bl[0]) + list(bl[1])]
                ng = len(bl[0])

                def run():
                    tf_fwd(xin)
                    y = tf_bwd(lv[:ng], lv[ng:])
                    y.backward(torch.randn_like(y))
            else:
                lv = [c.detach().requires_grad_(True) for c in bl]

                def run():
                    tf_fwd(xin)
                    y = tf_bwd(lv)
                    y.backward(torch.randn_like(y))
            return med_ms(run, args_cli.reps)

        t_leg = iso(segf.orig, segb.orig)

        num_octs = list(args.network.cqt.num_octs)
        bins = list(args.network.cqt.bins_per_oct)
        fs = int(getattr(args.exp, "sample_rate", 44100))
        c = SlicedCQT(num_octs, bins, fs=fs,
                      sl_len=65536, device="cuda")
        c.cqt = TrainableCQT(c.cqt)
        with torch.no_grad():
            xh = c.bwd(*c.fwd(xin), length=L)
            p = xin.double().square().sum()
            err = (xin.double() - xh.double()).square().sum()
            gdb = float(10 * torch.log10(p / err))
        if gdb < 118:
            print(f"  flash sliced gate {gdb:.1f} db FAIL, projection void")
            continue

        def flash_fwd(u):
            return c.fwd(u)

        def flash_bwd(bl, sd):
            return c.bwd(bl, sd, length=L)

        t_fla = iso(flash_fwd, flash_bwd)
        proj = total - t_leg + t_fla
        print(f"  isolated transform, legacy {t_leg:.1f} ms, flash "
              f"{t_fla:.1f} ms (sliced gate {gdb:.1f} db)")
        print(f"  projected step {proj:.1f} ms, {total / proj:.2f}x, "
              f"{args_cli.hours:.0f} h run -> "
              f"{args_cli.hours * proj / total:.0f} h")

    segf.restore()
    segb.restore()
    print("\nPROFILE-DONE")


if __name__ == "__main__":
    sys.exit(main())
