# GPU vs CPU on this server — measured, not assumed

Diagnostic run (`check_gpu.py`, 2026-06-01) on the A100 host:

```
Host driver       : 595.58.03   CUDA 13.2
torch             : 2.12.0+cu130   (built for CUDA 13.0)
torch.cuda.is_available : True   device 0 = A100-PCIE-40GB (39.5 GB, SM 8.0)
4096x4096 fp32 matmul : CPU 82.9 ms  (1657 GFLOP/s, 32 cores)
                        GPU 10.8 ms  (12729 GFLOP/s, TF32 off)
                        speedup 7.7x
xgboost GPU fit : OK     catboost GPU fit : OK
lightgbm GPU    : FAILED (PyPI wheel built without CUDA)
```

## What this means

* **The CUDA stack is healthy. Do not reinstall PyTorch.** The wheel is
  `cu130` and the host driver is `13.2`; a newer driver runs an older CUDA
  runtime fine. There is no `cu12x` mismatch — the earlier hypothesis was
  wrong. Downgrading to a `cu121` wheel would be a pointless regression.
* **The A100 is genuinely fast** (10.8 ms, ~65% of A100 fp32 peak with TF32
  disabled). The 7.7x ratio is "low" only because this 32-core Xeon is also
  fast at dense matmul — not because the GPU is slow or overhead-bound on a
  4096³ op.

## When the GPU helps here — and when it does not

The 7.7x above is the **best case** (one large dense matmul). The S-7 NN
models do not look like that:

* train sets are tiny (n ≤ 2000);
* batch size 32, 4 input features, 2 outputs, small hidden layers;
* hundreds of independent short `fit` calls per configuration (HPO trials ×
  CV folds), each dominated by Python / kernel-launch / host↔device transfer.

On such workloads the per-op GPU time is mostly launch + transfer overhead, so
the effective speedup is far below 7.7x and is frequently **< 1x** (slower on
GPU than CPU). **Conclusion: run these models on CPU.** Getting the GPU "more
involved" would not have rescued bucket D; the bottleneck was never the GPU.

GPU *would* help if/when we run: large dense models on n ≥ ~10⁴ without the
sub-sample cap, big batch sizes, or many epochs on wide networks. None of the
D-family models at n ≤ 2000 qualify.

Boosting: `xgboost` and `catboost` GPU fits work, but on n ≤ 2000 the CPU fit
is already milliseconds, so GPU is not worth the context cost. Keep
`EBW_USE_GPU_BOOSTING` unset. `lightgbm` has no CUDA in the PyPI wheel — leave
it on CPU (already handled in `models/trees.py`).

## How to force CPU (patch v5)

`FitContext.resolve_device()` now honours the **`EBW_DEVICE`** environment
variable above everything else. Setting `EBW_DEVICE=cpu` pins the *entire*
campaign — including the HPO inner loop, which previously ignored `--device` —
to CPU. With `device="auto"` and no override, the resolver runs a one-time
cached probe (a tiny CUDA matmul) and falls back to CPU automatically if the
kernel fails. So "auto-fallback on GPU failure" is handled in one place
(`models/base.py`) for every torch model, without per-model edits.

## Threading discipline — the part that actually matters

The real fix for D is parallelism across the 32 cores. The trap to avoid is
**oversubscription**: torch/numpy/MKL each spawn their own thread pool, so if
you launch N parallel worker processes and each spawns 32 BLAS threads you get
N×32 threads fighting over 32 cores — slower than single-threaded.

Rule for the parallel launch (`bucket_D_prime.sh` sets all of these):

```
export EBW_DEVICE=cpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

One thread per worker process, N worker processes ≈ cores. We use 12 shards on
32 cores by default (headroom for the OS, memory, and the occasional model
that still grabs 2 threads).
