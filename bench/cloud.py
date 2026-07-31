"""run the gpu benchmarks on a rented a100 through modal, billed by the second.

  modal run bench/cloud.py                                    plain benchmark
  modal run bench/cloud.py --cmd "python bench/probe.py"      any command
  modal run bench/cloud.py --old                              torch 2.5.1 and
                                                              its older triton

the --old image exists for one experiment, the inverse scatter kernel runs
7x slower under the newer triton on the cluster a100 than the identical
source under the old one on a v100, this runs both compilers on the same
card. first invocation builds each image (a few minutes), afterwards runs
start in seconds. code changes never rebuild the image, the repo dirs are
attached at container start.
"""
import os
import pathlib
import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
#the decorators run at import time on the laptop, so the gpu is picked by
#environment variable, FLASH_GPU=T4 modal run bench/cloud.py for a cheap card
GPU = os.environ.get("FLASH_GPU", "A100-40GB")

app = modal.App("flash-cqt")


def make_image(torch_spec):
    #gcc for the triton jit launcher, numpy for the filterbank tables
    return (modal.Image.debian_slim(python_version="3.11")
            .apt_install("gcc", "g++")
            .pip_install(torch_spec, "numpy")
            .add_local_dir(str(ROOT / "flash_cqt"), remote_path="/root/flash_cqt")
            .add_local_dir(str(ROOT / "bench"), remote_path="/root/bench"))


IMG_NEW = make_image("torch")
IMG_OLD = make_image("torch==2.5.1")

#devel image with nvcc for building raw cuda extensions, the flagship path.
#torch pinned to the cu126 wheel so its headers match the image's nvcc
IMG_CUDA = (modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04",
                                      add_python="3.11")
            .apt_install("gcc", "g++")
            .pip_install("numpy", "ninja")
            .pip_install("torch", index_url="https://download.pytorch.org/whl/cu126")
            .add_local_dir(str(ROOT / "flash_cqt"), remote_path="/root/flash_cqt")
            .add_local_dir(str(ROOT / "bench"), remote_path="/root/bench")
            .add_local_dir(str(ROOT / "xumx-sliCQ" / "xumx_slicq_v2"),
                           remote_path="/root/xumx-sliCQ/xumx_slicq_v2"))


def _run(cmd):
    import subprocess
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd="/root", timeout=1500)
    return r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else "")


@app.function(image=IMG_NEW, gpu=GPU, timeout=1600)
def bench_new(cmd: str) -> str:
    return _run(cmd)


@app.function(image=IMG_OLD, gpu=GPU, timeout=1600)
def bench_old(cmd: str) -> str:
    return _run(cmd)


@app.function(image=IMG_CUDA, gpu=GPU, timeout=1600)
def bench_cuda(cmd: str) -> str:
    return _run(cmd)


@app.local_entrypoint()
def main(cmd: str = "python bench/fused_gpu.py", old: bool = False,
         cuda: bool = False):
    fn = bench_cuda if cuda else (bench_old if old else bench_new)
    print(fn.remote(cmd))
