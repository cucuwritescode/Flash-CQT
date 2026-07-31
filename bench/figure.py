"""results figure for sharing.

  python bench/figure.py        writes flashcqt_results.png next to it

three panels. end to end speedup over the cufft based cqt pipeline
(same run, graphed, gated, bench/factored_bars.py), peak memory of one
roundtrip call (bench/memory_bars.py), and the head to head against
slicq (bench/scoreboard.py, same card, same run). every number is a
measurement, medians of 50 runs after three warmup calls.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BATCH = [1, 8, 32, 128]
#roundtrip ms, same run a100, graphed and gated
CUFFT = [0.509, 0.740, 0.947, 3.342]
FLASH = [0.135, 0.236, 0.467, 1.739]
#peak MB of one roundtrip call
CUFFT_MB = [4.2, 34.9, 135.7, 543.9]
FLASH_MB = [2.6, 21.1, 84.4, 337.5]
#head to head, whole 262144 buffer, B 1, same run, ms
SLICQ_MS = 177.554
OURS_EAGER = 3.255
OURS_GRAPH = 0.751

BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#666666"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def render(sketch):
    fig, (ax, mx, bx) = plt.subplots(1, 3, figsize=(15.5, 4.9))

    #panel a, end to end speedup
    sp = [c / f for c, f in zip(CUFFT, FLASH)]
    xs = range(len(BATCH))
    ax.bar(xs, sp, color=BLUE, width=0.6)
    for xi, (v, c, f) in enumerate(zip(sp, CUFFT, FLASH)):
        ax.text(xi, v + 0.34, f"{v:.1f}\u00d7", ha="center", fontsize=14,
                weight="bold", color=BLUE)
        ax.text(xi, v + 0.09, f"{f:.2f} vs {c:.2f} ms", ha="center",
                fontsize=7.5, color=GREY)
    ax.axhline(1.0, color=RED, ls="--", lw=1.2)
    ax.text(3.42, 1.06, "1\u00d7", color=RED, fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(b) for b in BATCH])
    ax.set_xlabel("batch size")
    ax.set_ylabel("round-trip speedup vs cuFFT-based CQT pipeline")
    ax.set_ylim(0, 4.9)
    ax.set_title("A. End-to-end speedup", loc="left", fontsize=11,
                 weight="bold")
    style(ax)

    #panel b, peak memory
    w = 0.38
    bars1 = mx.bar([x - w / 2 for x in xs], CUFFT_MB, width=w, color=RED,
                   label="cuFFT-based pipeline")
    bars2 = mx.bar([x + w / 2 for x in xs], FLASH_MB, width=w, color=BLUE,
                   label="Flash-CQT")
    mx.set_yscale("log")
    for b in list(bars1) + list(bars2):
        mx.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.12,
                f"{b.get_height():.0f}", ha="center", fontsize=8,
                color=GREY)
    mx.set_xticks(list(xs))
    mx.set_xticklabels([str(b) for b in BATCH])
    mx.set_xlabel("batch size")
    mx.set_ylabel("peak MB per round-trip call")
    mx.set_ylim(1.5, 2200)
    mx.text(0.03, 0.97, "1.61\u00d7 lower peak memory at every batch size",
            transform=mx.transAxes, fontsize=9, va="top", color=BLUE)
    mx.legend(fontsize=9, loc="upper left",
              bbox_to_anchor=(0.0, 0.90), frameon=False)
    mx.set_title("B. Peak memory", loc="left", fontsize=11, weight="bold")
    style(mx)

    #panel c, the previous streamable implementation, linear axis
    names = ["sliCQ", "Flash-CQT\neager", "Flash-CQT\nCUDA Graph"]
    vals = [SLICQ_MS, OURS_EAGER, OURS_GRAPH]
    cols = [RED, BLUE, BLUE]
    ys = [0, 1, 2]
    bx.barh(ys, vals, color=cols, height=0.55)
    bx.set_yticks(ys)
    bx.set_yticklabels(names, fontsize=9)
    bx.set_xlim(0, 225)
    bx.set_xlabel("end-to-end latency, 262,144 samples, B = 1 (ms)")
    bx.text(SLICQ_MS + 5, 0, f"{SLICQ_MS:.1f} ms", va="center",
            fontsize=10, color=RED, weight="bold")
    bx.text(OURS_EAGER + 4, 1,
            f"{OURS_EAGER:.2f} ms   54.5\u00d7 faster", va="center",
            fontsize=9, color=BLUE)
    bx.text(OURS_GRAPH + 4, 2,
            f"{OURS_GRAPH:.3f} ms   236\u00d7 faster", va="center",
            fontsize=9, color=BLUE, weight="bold")
    bx.invert_yaxis()
    bx.set_title("C. Previous streamable implementation", loc="left",
                 fontsize=11, weight="bold")
    bx.spines["top"].set_visible(False)
    bx.spines["right"].set_visible(False)
    bx.grid(True, axis="x", alpha=0.25, lw=0.6)
    bx.set_axisbelow(True)

    fig.suptitle("Flash-CQT: end-to-end performance on NVIDIA A100",
                 y=1.06, fontsize=15, weight="bold")
    fig.text(0.5, 0.985,
             "streamable invertible CQT \u00b7 exact FP32 \u00b7 129.8 dB "
             "round-trip SNR \u00b7 identical CQT configuration",
             ha="center", fontsize=10, color=GREY)
    fig.text(0.5, -0.06,
             "Medians of 50 runs after warm-up. Each comparison measured in "
             "one container on one card. All paths SNR-gated; CUDA Graph "
             "replay verified against eager output before timing. sliCQ has "
             "no graph-capturable path, so 54.5\u00d7 is the like-for-like "
             "eager comparison and 236\u00d7 is against Flash-CQT as "
             "deployed.", ha="center", fontsize=8, color=GREY)
    fig.tight_layout()
    nm = "flashcqt_results.png" if sketch else "flashcqt_results_clean.png"
    out = pathlib.Path(__file__).resolve().parent / nm
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    #the hand drawn one is the main output, the straight laced twin
    #sits beside it for anything more formal
    with plt.xkcd():
        render(True)
    render(False)


if __name__ == "__main__":
    main()
