#!/bin/bash
# ============================================================
# scAMUSE loss-ablation v2 — proto-মুক্ত cleaned কোডে চূড়ান্ত রান
# 9 configs: full (= final model) + A/B/D/E (add-one-in) + minus4 (leave-one-out)
# 9 × 8 datasets × 5 seeds = 360 runs. Resume-safe.
# Usage:  cd ~/scAMUSE7 && MAX_JOBS=6 nohup bash run_loss_ablation_v2.sh > run_loss_ablation_v2.log 2>&1 &
# ============================================================
cd ~/scAMUSE7 || exit 1
mkdir -p logs_lossabl_v2

MAX_JOBS=${MAX_JOBS:-6}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""

SEEDS="11 42 79 239 528"
DATASETS="Brain Fat Retina Breast_carcinoma Quadriceps_tendon Cortex Mucosal_Atlas Innate_lymphocytes"

dparams() {
  case "$1" in
    Brain)              echo "--epochs 100 --batch-size 256  --memory-temperature 0.5  --w-dcam 0.05  --lr 5e-4 --patience 10" ;;
    Fat)                echo "--epochs 150 --batch-size 512  --memory-temperature 0.4  --w-dcam 0.04  --lr 5e-4 --patience 15" ;;
    Retina)             echo "--epochs 150 --batch-size 512  --memory-temperature 0.3  --w-dcam 0.03  --lr 5e-4 --patience 15" ;;
    Breast_carcinoma)   echo "--epochs 150 --batch-size 512  --memory-temperature 0.5  --w-dcam 0.05  --lr 5e-4 --patience 15" ;;
    Quadriceps_tendon)  echo "--epochs 150 --batch-size 512  --memory-temperature 0.25 --w-dcam 0.025 --lr 5e-4 --patience 10 --n-memories 22" ;;
    Cortex)             echo "--epochs 150 --batch-size 512  --memory-temperature 0.5  --w-dcam 0.05  --lr 5e-4 --patience 15" ;;
    Mucosal_Atlas)      echo "--epochs 100 --batch-size 1024 --memory-temperature 0.45 --w-dcam 0.045 --lr 5e-4 --patience 15" ;;
    Innate_lymphocytes) echo "--epochs 150 --batch-size 512  --memory-temperature 0.35 --w-dcam 0.045 --lr 3e-4 --patience 15" ;;
  esac
}

CONFIGS="full A_ce_only B_ce_triplet D_ce_pairwise E_ce_supcon minus_triplet minus_pairwise minus_supcon minus_ce"
cflags() {
  case "$1" in
    full)           echo "" ;;
    A_ce_only)      echo "--w-triplet 0 --w-pairwise 0 --w-supcon 0" ;;
    B_ce_triplet)   echo "--w-pairwise 0 --w-supcon 0" ;;
    D_ce_pairwise)  echo "--w-triplet 0 --w-supcon 0" ;;
    E_ce_supcon)    echo "--w-triplet 0 --w-pairwise 0" ;;
    minus_triplet)  echo "--w-triplet 0" ;;
    minus_pairwise) echo "--w-pairwise 0" ;;
    minus_supcon)   echo "--w-supcon 0" ;;
    minus_ce)       echo "--w-ce 0" ;;
  esac
}

COMMON="--labeled-ratio 0.05 --memory-update-strength 0.3 --w-dcam-attraction 1.0 --w-dcam-diversity 0.5 --w-dcam-util 0.1 --w-dcam-energy 0.05 --w-dcam-entropy 0.2 --warmup-epochs 50 --ramp-epochs 30 --weight-decay 1e-4"

submit() {  # $1=config $2=dataset $3=seed
  local cfg=$1 d=$2 s=$3
  local out="result/LossAblation_v2/${cfg}/${d}/seed${s}"
  local log="logs_lossabl_v2/${cfg}__${d}__s${s}.log"
  if [ -f "$out/results.json" ]; then
    echo "SKIP ${cfg}__${d}__s${s} (done)"
    return
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 15; done
  (
    echo "=== [${cfg}__${d}__s${s}] START $(date) ==="
    python3 run_scAMUSE.py --data "data/${d}.h5ad" --output-dir "$out" --seed "$s" \
      $(dparams "$d") $COMMON $(cflags "$cfg") > "$log" 2>&1
    if [ -f "$out/results.json" ]; then
      echo "=== [${cfg}__${d}__s${s}] OK $(date) ==="
    else
      echo "=== [${cfg}__${d}__s${s}] FAILED $(date) ==="
    fi
  ) &
}

echo "LOSS ABLATION v2 START $(date)  (MAX_JOBS=$MAX_JOBS)"
for s in $SEEDS; do
  for d in $DATASETS; do
    for cfg in $CONFIGS; do
      submit "$cfg" "$d" "$s"
    done
  done
done
wait
echo "ALL LOSS-ABLATION-V2 RUNS FINISHED $(date)"