#!/bin/bash
# =============================================================================
# run_all_parallel.sh — scAMUSE benchmark runner, PARALLEL version
#
# Runs ALL remaining benchmark methods with a global job pool:
#   scSemiAAE, scGPT+scSemiAAE, scAGCL, scGPT+scAGCL, scGPT+scCCTR, scCCTR
#   x 8 datasets x 5 seeds (11 42 79 239 528)
#
# Usage:
#   MAX_JOBS=8 nohup bash run_all_parallel.sh > run_parallel.log 2>&1 &
#
# - MAX_JOBS (default 8): how many runs at once. Each gets 16 OMP threads.
# - Resume-safe: finished runs (log has final metrics) are skipped.
# - Small datasets first (quick results), Mucosal_Atlas last.
# - Does NOT touch: scANVI rerun, simple baselines (separate).
# =============================================================================

MAX_JOBS=${MAX_JOBS:-8}

SCAMUSE=~/scAMUSE7
BM=~/benchmark_methods
SP=$SCAMUSE/splits
IN=$SCAMUSE/baseline_inputs
GI=$SCAMUSE/scgpt_inputs

SEEDS="11 42 79 239 528"
# ordered small -> large (cells): quick wins first
DATASETS="Brain Breast_carcinoma Fat Quadriceps_tendon Cortex Retina Innate_lymphocytes Mucosal_Atlas"

export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""

already_done () {  # $1 = logfile
  [ -f "$1" ] && grep -qE "FINAL_METRICS|Final result" "$1"
}

get_K () {  # n_classes from the splits npz
  python3 -c "import numpy as np; print(int(np.load('$SP/$1_s$2.npz')['n_classes']))"
}

submit () {  # $1 = tag, $2 = logfile, rest = command
  local tag=$1 log=$2; shift 2
  if already_done "$log"; then echo "SKIP $tag"; return 0; fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 20; done
  (
    echo "=== [$tag] START $(date) ===" >> "$log"
    if "$@" >> "$log" 2>&1; then
      if already_done "$log"; then echo "=== [$tag] OK $(date) ===" >> "$log";
      else echo "=== [$tag] FAILED (no metrics) $(date) ===" >> "$log"; fi
    else
      echo "=== [$tag] FAILED (exit $?) $(date) ===" >> "$log"
    fi
  ) &
  sleep 1
}

mkdir -p $BM/scSemiAAE/logs $BM/scGPT_scSemiAAE/logs $BM/scAGCL/logs \
         $BM/scGPT_scAGCL/logs $BM/scGPT_scCCTR/logs $BM/scCCTR/logs

# scAGCL reads ./data/<file> — symlink all baseline inputs once
mkdir -p $BM/scAGCL/data
for d in $DATASETS; do ln -sf $IN/${d}.h5 $BM/scAGCL/data/${d}.h5; done

echo "PARALLEL RUN START: $(date) | MAX_JOBS=$MAX_JOBS"

for s in $SEEDS; do
for d in $DATASETS; do
  K=$(get_K $d $s)

  # ---- scSemiAAE ----
  submit "SCSEMIAEE_${d}_s${s}" $BM/scSemiAAE/logs/${d}_s${s}.log \
    bash -c "cd $BM/scSemiAAE && source .venv_scsemiaae/bin/activate && \
      python scSemiAAE.py --data_path $IN/${d}.h5 \
        --splits_npz $SP/${d}_s${s}.npz --seed $s \
        --label_cells 0.05 --batch_size 128 \
        --pretrain_epochs 100 --fit_epochs 100 --save_dir results/${d}_s${s}/"

  # ---- scGPT + scSemiAAE ----
  submit "SCGPT_SCSEMIAEE_${d}_s${s}" $BM/scGPT_scSemiAAE/logs/${d}_s${s}.log \
    bash -c "cd $BM/scGPT_scSemiAAE && source .venv_scsemiaae/bin/activate && \
      python scSemiAAE.py --data_path $GI/${d}_s${s}.h5 --use_scgpt \
        --splits_npz $SP/${d}_s${s}.npz --seed $s \
        --label_cells 0.05 --batch_size 128 --fit_epochs 100 \
        --save_dir results/${d}_s${s}/"

  # ---- scAGCL ----
  submit "SCAGCL_${d}_s${s}" $BM/scAGCL/logs/${d}_s${s}.log \
    bash -c "cd $BM/scAGCL && source .venv_scagcl/bin/activate && \
      python scAGCL.py --data_file ${d}.h5 --num_cluster $K \
        --random_seed $s --splits_npz $SP/${d}_s${s}.npz"

  # ---- scGPT + scAGCL ----
  submit "SCGPT_SCAGCL_${d}_s${s}" $BM/scGPT_scAGCL/logs/${d}_s${s}.log \
    bash -c "cd $BM/scGPT_scAGCL && source .venv_scagcl/bin/activate && \
      python scAGCL.py --data_file $GI/${d}_s${s}.h5 --use_scgpt \
        --num_cluster $K --random_seed $s --splits_npz $SP/${d}_s${s}.npz"

  # ---- scGPT + scCCTR (direct -> transformer chain) ----
  submit "SCGPT_SCCCTR_${d}_s${s}" $BM/scGPT_scCCTR/logs/${d}_s${s}.log \
    bash -c "cd $BM/scGPT_scCCTR && source .venv_sccctr/bin/activate && \
      python run_scgpt_sccctr_direct.py --data_path $GI/${d}_s${s}.h5 \
        --out_dir out_scamuse/${d}_s${s} --seed $s --splits_npz $SP/${d}_s${s}.npz && \
      python run_scgpt_sccctr_transformer.py --data_path $GI/${d}_s${s}.h5 \
        --pseudo_path out_scamuse/${d}_s${s}/scgpt_sccctr_direct_consensus_clusters.csv \
        --out_dir out_scamuse/${d}_s${s}_tr --seed $s --splits_npz $SP/${d}_s${s}.npz"

  # ---- scCCTR (patched: seed + splits + test-only metrics) ----
  submit "SCCCTR_${d}_s${s}" $BM/scCCTR/logs/${d}_s${s}.log \
    bash -c "cd $BM/scCCTR && source .venv_sccctr/bin/activate && \
      python scCCTR_patched.py --data_h5 $IN/${d}.h5 \
        --splits_npz $SP/${d}_s${s}.npz --seed $s \
        --out_dir result_scamuse/${d}_s${s}"

done
done

wait
echo "ALL PARALLEL JOBS FINISHED $(date)"