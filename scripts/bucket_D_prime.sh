#!/usr/bin/env bash
#
# bucket_D_prime.sh  (v2 -- CORRECTED)
#
# Fix vs v1: now passes --n-folds 3 explicitly. v1 inherited the S7Config
# default of 5, which (a) did not match the 3-fold campaign and (b) caused the
# deep-NN timeouts. This version writes to FRESH dirs (s7_Dp3_sh*) so the
# broken 5-fold dirs (s7_Dp_sh*) are NOT merged in -- discard those.
#
# Run under tmux so the wrapper (and the auto-merge) survive disconnect:
#     tmux new -s dp3
#     bash bucket_D_prime.sh
#     Ctrl-b d                       # detach;  tmux attach -t dp3  to return
#
# Grid: 5 models x 3 generators x 3 ablations x 1 size x 12 optimisers = 540.

set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="python"

# ---- grid -----------------------------------------------------------------
MODELS="${MODELS:-ngb,mdn,vote,tabnet,ftt}"
GENERATORS="tvae,copula,physics"
ABLATIONS="synth,real_plus_synth,tstr"
SAMPLE_SIZES="10000"
N_FOLDS="${N_FOLDS:-3}"             # <-- THE FIX: match the 3-fold campaign
N_HPO_TRIALS="${N_HPO_TRIALS:-10}"
MAX_TRAIN_SIZE="${MAX_TRAIN_SIZE:-2000}"
PER_CONFIG_TIMEOUT="${PER_CONFIG_TIMEOUT:-7200}"
SEED="${SEED:-42}"

# ---- parallelism / threading ----------------------------------------------
# THREADS = intra-op CPU threads per worker. Keep THREADS * NUM_SHARDS <= 32.
#   * Light models (ngb, mdn, vote): THREADS=1, NUM_SHARDS=24  (default).
#   * Heavy NN (ftt, tabnet) if they still time out at 3-fold: re-run just
#     those with  MODELS=ftt,tabnet THREADS=5 NUM_SHARDS=6  so each fit gets
#     5 cores of torch intra-op parallelism.
THREADS="${THREADS:-1}"
NUM_SHARDS="${NUM_SHARDS:-24}"
export EBW_DEVICE=cpu
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export OMP_DYNAMIC=FALSE
export TOKENIZERS_PARALLELISM=false

OUT_PREFIX="${OUT_PREFIX:-runs/s7_Dp3_sh}"   # fresh dirs (3-fold)
mkdir -p runs/logs

echo "=== bucket D' v2 : n_folds=$N_FOLDS, threads/worker=$THREADS, shards=$NUM_SHARDS ==="
$PY run_s7.py --out-dir "${OUT_PREFIX}0" \
    --models "$MODELS" --generators "$GENERATORS" \
    --ablation-modes "$ABLATIONS" --sample-sizes "$SAMPLE_SIZES" \
    --n-folds "$N_FOLDS" --n-hpo-trials "$N_HPO_TRIALS" \
    --max-train-size "$MAX_TRAIN_SIZE" \
    --num-shards "$NUM_SHARDS" --shard 0 --dry-run
echo

PIDS=()
for ((k=0; k<NUM_SHARDS; k++)); do
    nohup $PY run_s7.py \
        --out-dir "${OUT_PREFIX}${k}" \
        --models "$MODELS" --generators "$GENERATORS" \
        --ablation-modes "$ABLATIONS" --sample-sizes "$SAMPLE_SIZES" \
        --n-folds "$N_FOLDS" --n-hpo-trials "$N_HPO_TRIALS" \
        --max-train-size "$MAX_TRAIN_SIZE" \
        --per-config-timeout "$PER_CONFIG_TIMEOUT" \
        --device cpu \
        --num-shards "$NUM_SHARDS" --shard "$k" \
        --seed "$SEED" --resume --progress-every 5 \
        > "runs/logs/$(basename ${OUT_PREFIX})${k}.log" 2>&1 &
    PIDS+=("$!")
    sleep 1
done
echo "shard PIDs: ${PIDS[*]}"
echo "watch: grep -H runner runs/logs/$(basename ${OUT_PREFIX})*.log | tail -n 40"

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || { echo "WARN: pid $pid non-zero" >&2; FAIL=1; }; done
echo "=== shards finished (fail=$FAIL) ==="

# ---- merge: original 3-fold buckets + the NEW 3-fold D' dirs ---------------
# NOTE: the old 5-fold dirs (runs/s7_Dp_sh*) are deliberately NOT included.
echo "=== merging -> runs/s7_merged_Dp3 ==="
$PY merge_runs.py \
    --inputs runs/s7_A runs/s7_B runs/s7_C runs/s7_D ${OUT_PREFIX}* \
    --output runs/s7_merged_Dp3
echo "Done -> runs/s7_merged_Dp3"
