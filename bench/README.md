# bench notes

everything here is cpu or mps on my mac, torch 2.7.1. no cuda numbers yet,
profile_gpu.py is the script to run on the cluster when someone has an a100.

`python bench/baselines.py` for the reference transforms,
`python bench/sliced.py` for our transform. worker.py runs one config per
process so peak rss means something.

## baselines (fs 44100, ls 262144, fp32, B1)

```
engine  device |  snr_raw  snr_hpf   inband |   fwd_ms   inv_ms    rt_ms |  peak_MB
oct     cpu    |    19.74    35.59   131.79 |    17.04    71.47   101.38 |   412.70
slicq   cpu    |   130.13        -   128.83 |   134.80   282.71   376.09 |   172.40
stft    cpu    |   138.72        -   133.31 |     7.15    10.43    17.43 |   251.60
oct     mps    |    19.67    35.66   123.90 |     3.23     6.05     9.51 |   380.00
slicq   mps    |   117.33        -   117.20 |  2535.65  2199.39  4899.56 |   655.50
stft    mps    |   131.11        -   129.57 |     0.81     3.86     4.07 |   223.80
```

the oct raw snr looks alarming but is not, oct mode drops dc and nyquist by
design. in float64 the in band error is 1e-31, everything below fmin (43 hz)
and above 21.5 khz. so quote the inband column, that is where the transform is
defined. slicq keeps dc and nyquist and gets 130 db full band, but its
generator plumbing is unusable on mps, thousands of tiny ops. that overhead is
the whole reason this repo exists.

the xumx slicq needed float64 to float32 fixes in util.py, slicq.py,
unslicing.py (mps has no float64), cpu snr unchanged (130.24 before, 130.13
after). those edits live in the nested xumx-sliCQ repo.

## our transform (flash_cqt, `python bench/sliced.py`)

offline flex oct at 262144 gives the exact multi_CQT block list,
8x128 8x256 8x512 16x512 16x1024 16x2048 16x4096 32x4096 32x8192, and the
octave blocks are bit identical to the repo-refs oct forward (max diff 0.0).
one shared fft instead of one per resolution group.

round trips on white noise, fp32
- offline 131.7 db full band with the dc and nyquist side blocks, 22.1 without
  (same seams as multi_CQT, which round trips at 19 db raw)
- sliced at sl_len 65536, 132.7 db raw, 130.7 inband. the slice window leaks
  into each slice's local dc and nyquist bands so the side blocks are not
  optional here, dropping them caps inband at 64 db
- streaming chunk by chunk equals the whole buffer path, diff 0.0 on cpu,
  state is 2 x sl_len/2 floats per stream, latency is one slice
- grads reach the input

latency table (the slicq rule, sl_len >= 8 fs q0 / fmin)

```
  sl_len  latency_ms  lowest_fmin_hz   covers 43.1 hz
    4096        92.9           481.4   no
    8192       185.8           240.7   no
   16384       371.5           120.3   no
   32768       743.0            60.2   no
   65536      1486.1            30.1   yes
  131072      2972.2            15.0   yes
```

1.49 s to reach 43 hz is physics, not implementation. shorter slices still
invert fine (133 db at 16384), they just blur the lowest octaves.

speed, same machine, cpu rows are stable, mps wall times wobble between runs

```
engine    device |  snr_raw   inband |   fwd_ms   inv_ms    rt_ms |  peak_MB
flash     cpu    |   132.74   130.65 |    17.92    23.75    42.27 |    292.2
flashoct  cpu    |   131.66   131.89 |     6.80    22.31    28.33 |    267.2
multicqt  cpu    |    19.33    22.34 |   105.80    73.62   154.50 |    812.4
slicq     cpu    |   130.13   128.83 |    25.60    48.57    88.60 |    301.1
flash     mps    |   125.36   124.93 |    25.13    30.76    48.71 |    400.4
flashoct  mps    |   124.81   124.52 |     1.73     4.15     5.35 |    866.6
slicq     mps    |   117.33   117.20 |  2326.57  1008.13  2071.94 |    878.7
```

multicqt does not run on mps (float64 in its window code). flash beats slicq
2x on cpu and by a silly margin on mps, and beats multicqt 5x at 3x less
memory offline. the thing to notice is flash on mps barely beating cpu while
flashoct is 5x faster there, the sliced path drowns in small kernel launches.
whether that holds on cuda decides if the fused kernel is worth writing, do
not decide that from mps numbers.
