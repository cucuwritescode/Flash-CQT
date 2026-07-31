"""print the toolchain and ptxas resource usage for the cuda extensions.

  modal run bench/cloud.py --cuda --cmd "python bench/build_info.py"

builds persist2.cu, router.cu and packedfft.cu with ptxas verbose so the
register and shared memory counts land in the log. no benchmark, the only
gpu use is the arch probe torch does at build time.
"""
import pathlib
import subprocess

import torch
from torch.utils.cpp_extension import load

CSRC = pathlib.Path(__file__).resolve().parents[1] / "flash_cqt" / "csrc"


def main():
    print(subprocess.run("nvcc --version", shell=True, capture_output=True,
                         text=True).stdout)
    print("torch", torch.__version__)
    print("gpu", torch.cuda.get_device_name(0),
          "cc", torch.cuda.get_device_capability(0))
    #fresh names so the cached builds are not reused, verbose prints the
    #full nvcc command line including the gencode flags torch picks
    for src in ("radix.cu", "persist2.cu", "router.cu", "packedfft.cu"):
        print("\n==== " + src + " ====")
        load(name="ptxas_probe_" + src.split(".")[0],
             sources=[str(CSRC / src)],
             extra_cuda_cflags=["-Xptxas", "-v"], verbose=True)


if __name__ == "__main__":
    main()
