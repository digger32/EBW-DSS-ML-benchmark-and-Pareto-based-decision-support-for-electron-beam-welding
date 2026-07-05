# patch_v5 — apply / stop D / run D′

This patch is built on top of the v3 runner/run_s7 that is already live on the
server (`max_train_size` / `per_config_timeout_s` present in
`runs/s7_D/config.json`) and the baseline `models/base.py`. Five files:

```
ebw_ml/models/base.py            (replaced) EBW_DEVICE override + auto CPU fallback
ebw_ml/experiment/runner.py      (replaced) HARD timeout + grid sharding + status col
run_s7.py                        (replaced) --shard / --num-shards
GPU_NOTES.md                     (new)      diagnosis + threading rules
bucket_D_prime.sh                (new)      CPU sharded launcher + merge
```

Nothing else is touched. `models/nn.py`, optimisers, metrics, storage,
merge_runs.py are unchanged.

## 1. Stop the stalled bucket D

```bash
# find and stop the D runner (it will resume cleanly later from done.csv)
pgrep -af "run_s7.py.*s7_D"        # confirm the PID(s)
pkill -f  "run_s7.py.*s7_D"        # stop it
```
Its 577 completed configs in `runs/s7_D/` stay intact and are merged back in
later — nothing is lost.

## 2. Back up and apply the patch

```bash
cd ~/Documents/ebw_ml_s7
cp ebw_ml/models/base.py            ebw_ml/models/base.py.bak
cp ebw_ml/experiment/runner.py      ebw_ml/experiment/runner.py.bak
cp run_s7.py                        run_s7.py.bak

# unzip the patch over the repo (paths inside the zip mirror the repo layout)
unzip -o patch_v5_gpu_or_cpu_fallback.zip -d ~/Documents/ebw_ml_s7
```

Smoke-test that imports still work and the new flags parse:

```bash
.venv/bin/python run_s7.py --out-dir /tmp/v5_smoke \
    --models vote --optimisers random --generators copula \
    --sample-sizes 1000 --ablation-modes synth --n-hpo-trials 2 \
    --num-shards 4 --shard 0 --dry-run
# expect: full grid = 1 ; this shard (0/4) = ... ; first/last config lines
```

(Optional, ~1 min) a real 1-config CPU run to confirm a row is written:

```bash
EBW_DEVICE=cpu .venv/bin/python run_s7.py --out-dir /tmp/v5_smoke \
    --models vote --optimisers random --generators copula \
    --sample-sizes 10000 --ablation-modes synth --n-hpo-trials 2 \
    --max-train-size 1000 --per-config-timeout 120
cat /tmp/v5_smoke/results_aggregated.csv   # should have a status=ok row
```

## 3. Run bucket D′

```bash
bash bucket_D_prime.sh
```

It dry-runs first (expect full grid = 540), then launches 12 CPU shards into
`runs/s7_Dp_sh0 … sh11`, waits, and merges everything into
`runs/s7_merged_Dp/`.

Watch progress:
```bash
grep -H runner runs/logs/s7_Dp_sh*.log | tail -n 40
```

Tunables (env vars, defaults in brackets): `NUM_SHARDS` [24],
`N_HPO_TRIALS` [10], `MAX_TRAIN_SIZE` [2000], `PER_CONFIG_TIMEOUT` [7200],
`PY` [.venv/bin/python].

## 4. After the first progress line

The first `[runner] 5/45 …` line per shard gives the real per-config rate.
Multiply: `eta_wall ≈ (45 configs/shard) / (rate per shard)`. If a shard's ETA
is uncomfortable, raise `NUM_SHARDS` (e.g. 30, RAM permitting) and re-launch — `--resume` makes
it pick up where it stopped. Send me the first few progress lines and I'll
sanity-check the projection rather than guessing.
