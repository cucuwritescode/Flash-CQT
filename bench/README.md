# bench notes

everything below the cuda section is cpu or mps on my mac, torch 2.7.1.
the cuda section is profile_gpu.py run on the aalto cluster by eloi.

`python bench/baselines.py` for the reference transforms,
`python bench/sliced.py` for our transform, `python bench/precision.py` for
the reduced precision roundtrips. worker.py runs one config per process so
peak rss means something. `python bench/monarch.py` is the next cluster run,
the per slice transform with every fft as cooley tukey matmul stages, timed
against the graphed cufft path at fp32, 3xtf32, tf32 and bf16. its fp32 gates
pass on cpu (fft 65536 at 127 db vs torch.fft, roundtrip 125 db), speed
numbers pending a cuda run. `python bench/compare_gpu.py` puts every
implementation we know of in one table on one gpu (ours, multi_CQT, the
repo-refs oct, sevagh's sliCQ, archinetai cqt-pytorch, nnAudio, torch.stft)
and times the per slice unit eagerly and through flash_cqt.Graphed, checking
the graph replay is bit identical before trusting its time. missing packages
or checkouts skip with a note. cpu correctness of all eight rows checked
here, cuda timings pending.

back of envelope for what should come next, so the cluster numbers have
something to be checked against. the matmul form does about 50x the flops of
cufft (0.7 gflop per slice roundtrip, the price of trading fft recursion for
two matmul stages), which is 5 us of tf32 or 37 us of fp32 on an a100, so
arithmetic never limits it. but as separate torch ops it launches more
kernels than the cufft path (each complex matmul is four real gemms), so at
B 1 it should lose or tie, and only pull ahead at large batch where gemm
throughput takes over.

the fused kernel is where the win lives, roughly 6 launches instead of 22
and about 5 MB of hbm traffic instead of 15 to 20. note the graphed path
runs at about 1.5 percent of hbm bandwidth (0.587 ms for under 20 MB), it is
latency bound, not bandwidth bound, so the win at B 1 comes from escaping
per kernel latency, not from the traffic ratio. projected per slice
roundtrip at B 1 against the 0.587 ms graphed path, by precision

  fp32 fused    cuda cores, no reconstruction risk   ~4 to 6x
  3xtf32 fused  if it holds near fp32 snr            ~8 to 15x
  tf32 fused    at the measured 68 to 79 db          ~15 to 30x

at large batch the path saturates bandwidth and the traffic ratio is what
survives, ~3 to 5x for every mode. rooflines, not promises, the kernel gets
measured like everything else. and to be plain about motive, streaming at
B 1 is already hundreds of times realtime, the B 1 numbers are about being
the fastest, the batch numbers are what training throughput actually feels.

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

## cuda (profile_gpu.py, run by eloi on the cluster, fp32)

v100 sxm2 16GB, torch 2.5.1 cu121, and a100 sxm4 80GB, torch 2.12.0 cu126.
numbers relayed from his run, not measured by me on this machine.

```
                              v100      a100
whole buffer rt B1 (ms)       4.27      2.38   snr 124.6 / 127.1 db
whole buffer rt B4 (ms)       4.75      2.75   peak 246 / 247 MB
per slice roundtrip (ms)      2.330     1.325
cufft floor (ms)              0.857     0.390
graph replay (ms)             0.644     0.587
launch overhead (ms, share)   1.686 72% 0.738 56%
real time margin              319x      561x
```

both cards come back launch bound. the give away is the v100 row where the
graph replayed roundtrip (0.644 ms, every fft plus all the glue) is faster
than the eagerly timed cufft floor alone (0.857 ms), so even the floor timer
was mostly dispatch. on the a100 the gpu work inside the graphed roundtrip
splits into 0.390 ms of ffts and about 0.197 ms of glue.

two honest caveats before anyone gets excited about a kernel. cuda graphs
already recover the 56 to 72 percent launch share for free, so the bar a
fused kernel must clear is the graphed path, 0.59 ms on a100, not the eager
2.3 ms. and the real time margins are silly (300 to 560x), single stream
inference does not need any of this, the win is training throughput where
thousands of slices per second matter.

## precision (`python bench/precision.py`, exact bit arithmetic, device independent)

every intermediate of the sliced roundtrip rounded to the target mantissa,
storage rounds only the coefficients, compute rounds every stage. fp32
compute reproduces the reference 132.74 db exactly so the mirror is faithful.

```
mode   scenario | snr_raw_db  inband_db
fp32   storage  |     132.74     130.65
fp32   compute  |     132.74     130.65
tf32   storage  |      78.70      79.86
tf32   compute  |      67.69      67.76
bf16   storage  |      60.49      61.83
bf16   compute  |      49.81      49.88
```

so tf32 tensor cores keep about 68 db through the whole roundtrip, bf16 about
50 (and a native bf16 fft would land lower still, this is its best case).
neither is transparent, whether 68 db is enough depends on what mr cqtdiff
tolerates in the coefficients. fp32 stays the reference, any tensor core path
must quote these numbers next to its speed.
