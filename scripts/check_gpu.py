#!/usr/bin/env python3
"""check_gpu.py -- device and throughput diagnostic for the EBW pipeline.

Reports the Torch/CUDA build, GPU availability, and a fp32 matmul throughput
comparison (CPU vs GPU). For this benchmark the models are tiny (n<=2000) and
do not benefit from the GPU; all reported campaign runs use EBW_DEVICE=cpu.
"""
import time
import numpy as np
import torch


def bench(device, n=4096, reps=3):
    a = torch.randn(n, n, device=device); b = torch.randn(n, n, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        c = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    flop = 2 * n ** 3
    return flop / dt / 1e12  # TFLOP/s


def main():
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    cpu = bench("cpu")
    print(f"CPU  fp32 matmul: {cpu:6.2f} TFLOP/s")
    if torch.cuda.is_available():
        gpu = bench("cuda")
        print(f"GPU  fp32 matmul: {gpu:6.2f} TFLOP/s  (speedup {gpu/cpu:.1f}x)")


if __name__ == "__main__":
    main()
