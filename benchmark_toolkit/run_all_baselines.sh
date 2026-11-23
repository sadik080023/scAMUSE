#!/bin/bash
# =============================================================================
# run_all_baselines.sh — scAMUSE benchmark runner (Phase 5)
#
# Runs every baseline with the scAMUSE protocol:
#   seeds 11 42 79 239 528 | 5% labels | scAMUSE splits | test-only metrics
#
# Usage:
#   bash run_all_baselines.sh scsemiaae        # one method
#   bash run_all_baselines.sh scagcl
#   bash run_all_baselines.sh sccctr
#   bash run_all_baselines.sh scgpt_scsemiaae
#   bash run_all_baselines.sh scgpt_scagcl
#   bash run_all_baselines.sh scgpt_sccctr
#   bash run_all_baselines.sh simple           # scGPT+KMeans, scGPT+Leiden,
#                                              # PCA+KMeans, Scanpy+Louvain
#   bash run_all_baselines.sh all              # everything, sequentially
#
# Resume-safe: a run is skipped when its log already contains a final
# metrics line. Re-run the same command after a crash.
#
# Adjust the paths below if your layout differs.
# =============================================================================

SCAMUSE=~/scAMUSE7
BM=~/benchmark_methods
SP=$SCAMUSE/splits            # splits/{DATASET}_s{SEED}.npz
IN=$SCAMUSE/baseline_inputs   # baseline_inputs/{DATASET}.h5
GI=$SCAMUSE/scgpt_inputs      # scgpt_inputs/{DATASET}_s{SEED}.h5

# Python that has scanpy (for the simple baselines). The scANVI venv works:
SIMPLE_PY=${SIMPLE_PY:-~/scanvi_env/bin/python}

SEEDS="11 42 79 239 528"
DATASETS="Brain Fat Retina Breast_carcinoma Quadriceps_tendon Cortex Mucosal_Atlas Innate_lymphocytes"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""

get_K () {  # read n_classes from the splits npz
  python3 -c "import numpy as np; print(int(np.load('$SP/$1_s$2.npz')['n_classes']))"
}

already_done () {  # $1 = logfile
  [ -f "$1" ] && grep -qE "FINAL_METRICS|Final result" "$1"
}

run () {  # $1 = tag, $2 = logfile, rest = command
  local tag=$1 log=$2; shift 2
  if already_done "$log"; then echo "SKIP $tag (done)"; return 0; fi
  echo "=== [$tag] START $(date) ===" | tee -a "$log"
  if "$@" >> "$log" 2>&1; then
    if already_done "$log"; then echo "=== [$tag] OK $(date) ===" | tee -a "$log";
    else echo "=== [$tag] FAILED (no metrics) $(date) ===" | tee -a "$log"; fi
  else
    echo "=== [$tag] FAILED (exit $?) $(date) ===" | tee -a "$log"
  fi
}

# ---------------------------------------------------------------- scSemiAAE --
m_scsemiaae () {
  cd $BM/scSemiAAE || return 1
  source .venv_scsemiaae/bin/activate
  mkdir -p logs results
  for s in $SEEDS; do for d in $DATASETS; do
    run "SCSEMIAEE_${d}_s${s}" logs/${d}_s${s}.log \
      python scSemiAAE.py --data_path $IN/${d}.h5 \
        --splits_npz $SP/${d}_s${s}.npz --seed $s \
        --label_cells 0.05 --batch_size 128 \
        --pretrain_epochs 100 --fit_epochs 100 \
        --save_dir results/${d}_s${s}/
  done; done
}

# ------------------------------------------------------------------ scAGCL --
m_scagcl () {
  cd $BM/scAGCL || return 1
  source .venv_scagcl/bin/activate
  mkdir -p logs result data
  for d in $DATASETS; do ln -sf $IN/${d}.h5 data/${d}.h5; done
  for s in $SEEDS; do for d in $DATASETS; do
    K=$(get_K $d $s)
    run "SCAGCL_${d}_s${s}" logs/${d}_s${s}.log \
      python scAGCL.py --data_file ${d}.h5 --num_cluster $K \
        --random_seed $s --splits_npz $SP/${d}_s${s}.npz
  done; done
}

# ------------------------------------------------------------------ scCCTR --
m_sccctr () {
  cd $BM/scCCTR || return 1
  source .venv_sccctr/bin/activate
  mkdir -p logs result_scamuse
  for s in $SEEDS; do for d in $DATASETS; do
    run "SCCCTR_${d}_s${s}" logs/${d}_s${s}.log \
      python scCCTR_patched.py --data_h5 $IN/${d}.h5 \
        --splits_npz $SP/${d}_s${s}.npz --seed $s \
        --out_dir result_scamuse/${d}_s${s}
  done; done
}

# ----------------------------------------------------------- scGPT+scSemiAAE --
m_scgpt_scsemiaae () {
  cd $BM/scGPT_scSemiAAE || return 1
  source .venv_scsemiaae/bin/activate
  mkdir -p logs results
  for s in $SEEDS; do for d in $DATASETS; do
    run "SCGPT_SCSEMIAEE_${d}_s${s}" logs/${d}_s${s}.log \
      python scSemiAAE.py --data_path $GI/${d}_s${s}.h5 --use_scgpt \
        --splits_npz $SP/${d}_s${s}.npz --seed $s \
        --label_cells 0.05 --batch_size 128 --fit_epochs 100 \
        --save_dir results/${d}_s${s}/
  done; done
}

# -------------------------------------------------------------- scGPT+scAGCL --
m_scgpt_scagcl () {
  cd $BM/scGPT_scAGCL || return 1
  source .venv_scagcl/bin/activate
  mkdir -p logs result
  for s in $SEEDS; do for d in $DATASETS; do
    K=$(get_K $d $s)
    run "SCGPT_SCAGCL_${d}_s${s}" logs/${d}_s${s}.log \
      python scAGCL.py --data_file $GI/${d}_s${s}.h5 --use_scgpt \
        --num_cluster $K --random_seed $s --splits_npz $SP/${d}_s${s}.npz
  done; done
}

# -------------------------------------------------------------- scGPT+scCCTR --
m_scgpt_sccctr () {
  cd $BM/scGPT_scCCTR || return 1
  source .venv_sccctr/bin/activate
  mkdir -p logs out_scamuse
  for s in $SEEDS; do for d in $DATASETS; do
    run "SCGPT_SCCCTR_${d}_s${s}" logs/${d}_s${s}.log \
      bash -c "python run_scgpt_sccctr_direct.py --data_path $GI/${d}_s${s}.h5 \
          --out_dir out_scamuse/${d}_s${s} --seed $s --splits_npz $SP/${d}_s${s}.npz \
        && python run_scgpt_sccctr_transformer.py --data_path $GI/${d}_s${s}.h5 \
          --pseudo_path out_scamuse/${d}_s${s}/scgpt_sccctr_direct_consensus_clusters.csv \
          --out_dir out_scamuse/${d}_s${s}_tr --seed $s --splits_npz $SP/${d}_s${s}.npz"
  done; done
}

# -------------------------------------------------------- simple baselines --
m_simple () {
  mkdir -p $SCAMUSE/benchmark_results/logs
  for s in $SEEDS; do for d in $DATASETS; do
    for m in scgpt_kmeans scgpt_leiden; do
      run "${m}_${d}_s${s}" $SCAMUSE/benchmark_results/logs/${m}_${d}_s${s}.log \
        $SIMPLE_PY $SCAMUSE/benchmark_toolkit/run_simple_baselines.py \
          --method $m --dataset $d --seed $s \
          --splits_npz $SP/${d}_s${s}.npz --scgpt_h5 $GI/${d}_s${s}.h5 \
          --out_dir $SCAMUSE/benchmark_results/$m/${d}_s${s}
    done
    for m in pca_kmeans scanpy_louvain; do
      run "${m}_${d}_s${s}" $SCAMUSE/benchmark_results/logs/${m}_${d}_s${s}.log \
        $SIMPLE_PY $SCAMUSE/benchmark_toolkit/run_simple_baselines.py \
          --method $m --dataset $d --seed $s \
          --splits_npz $SP/${d}_s${s}.npz --raw_h5 $IN/${d}.h5 \
          --out_dir $SCAMUSE/benchmark_results/$m/${d}_s${s}
    done
  done; done
}

# ------------------------------------------------------------------- main --
case "${1:-all}" in
  scsemiaae)       m_scsemiaae ;;
  scagcl)          m_scagcl ;;
  sccctr)          m_sccctr ;;
  scgpt_scsemiaae) m_scgpt_scsemiaae ;;
  scgpt_scagcl)    m_scgpt_scagcl ;;
  scgpt_sccctr)    m_scgpt_sccctr ;;
  simple)          m_simple ;;
  all)
    m_simple; m_scsemiaae; m_scgpt_scsemiaae; m_scagcl; m_scgpt_scagcl;
    m_scgpt_sccctr; m_sccctr ;;
  *) echo "unknown method: $1"; exit 1 ;;
esac
echo "ALL REQUESTED JOBS FINISHED $(date)"
