#!/bin/bash
# ============================================================
# DCAM Ablation v2 (PROTO-FREE code) — Sheets 4a + 4c
# 9 configs x 8 datasets x 5 seeds = 360 runs
#   4c leave-one-out : minus_attraction / minus_entropy / minus_diversity / minus_utilization / minus_energy
#   4a add-one-in    : a_attraction_only / b_plus_entropy / c_plus_diversity / d_plus_utilization
#   (4a config E = full DCAM = already have as LossAblation_v2 "full" — not repeated)
# Supervised losses: CE+Triplet+Pairwise+SupCon at defaults (no flags needed).
# Launch:  cd ~/scAMUSE7 && MAX_JOBS=6 nohup bash run_dcam_ablation_v2.sh > run_dcamabl_v2.log 2>&1 &
# Progress: grep -c "] OK " run_dcamabl_v2.log   (target 360)
# Resume-safe: existing results.json => SKIP. Just rerun the same command.
# ============================================================
set -u
cd ~/scAMUSE7
MAX_JOBS=${MAX_JOBS:-6}
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p logs_dcamabl_v2

SEEDS="11 42 79 239 528"
DATASETS="Brain Fat Retina Breast_carcinoma Quadriceps_tendon Cortex Mucosal_Atlas Innate_lymphocytes"

# Per-dataset hyperparameters: copied EXACTLY from run_all_phase1.sh
dparams() {
  case "$1" in
    Brain)              echo "--epochs 100 --batch-size 256  --memory-temperature 0.5  --w-dcam 0.05  --lr 5e-4 --weight-decay 1e-4 --patience 10";;
    Fat)                echo "--epochs 150 --batch-size 512  --memory-temperature 0.4  --w-dcam 0.04  --lr 5e-4 --weight-decay 1e-4 --patience 15";;
    Retina)             echo "--epochs 150 --batch-size 512  --memory-temperature 0.3  --w-dcam 0.03  --lr 5e-4 --weight-decay 1e-4 --patience 15";;
    Breast_carcinoma)   echo "--epochs 150 --batch-size 512  --memory-temperature 0.5  --w-dcam 0.05  --lr 5e-4 --weight-decay 1e-4 --patience 15";;
    Quadriceps_tendon)  echo "--epochs 150 --batch-size 512  --memory-temperature 0.25 --w-dcam 0.025 --lr 5e-4 --weight-decay 1e-4 --patience 10 --n-memories 22";;
    Cortex)             echo "--epochs 150 --batch-size 512  --memory-temperature 0.5  --w-dcam 0.05  --lr 5e-4 --weight-decay 1e-4 --patience 15";;
    Mucosal_Atlas)      echo "--epochs 100 --batch-size 1024 --memory-temperature 0.45 --w-dcam 0.045 --lr 5e-4 --weight-decay 1e-4 --patience 15";;
    Innate_lymphocytes) echo "--epochs 150 --batch-size 512  --memory-temperature 0.35 --w-dcam 0.045 --lr 5e-4 --weight-decay 3e-4 --patience 15";;
  esac
}

# DCAM loss weights per config (full = attraction 1.0, diversity 0.5, util 0.1, energy 0.05, entropy 0.2)
dcam_weights() {
  case "$1" in
    minus_attraction)   echo "--w-dcam-attraction 0   --w-dcam-diversity 0.5 --w-dcam-util 0.1 --w-dcam-energy 0.05 --w-dcam-entropy 0.2";;
    minus_entropy)      echo "--w-dcam-attraction 1.0 --w-dcam-diversity 0.5 --w-dcam-util 0.1 --w-dcam-energy 0.05 --w-dcam-entropy 0";;
    minus_diversity)    echo "--w-dcam-attraction 1.0 --w-dcam-diversity 0   --w-dcam-util 0.1 --w-dcam-energy 0.05 --w-dcam-entropy 0.2";;
    minus_utilization)  echo "--w-dcam-attraction 1.0 --w-dcam-diversity 0.5 --w-dcam-util 0   --w-dcam-energy 0.05 --w-dcam-entropy 0.2";;
    minus_energy)       echo "--w-dcam-attraction 1.0 --w-dcam-diversity 0.5 --w-dcam-util 0.1 --w-dcam-energy 0    --w-dcam-entropy 0.2";;
    a_attraction_only)  echo "--w-dcam-attraction 1.0 --w-dcam-diversity 0   --w-dcam-util 0   --w-dcam-energy 0    --w-dcam-entropy 0";;
    b_plus_entropy)     echo "--w-dcam-attraction 1.0 --w-dcam-diversity 0   --w-dcam-util 0   --w-dcam-energy 0    --w-dcam-entropy 0.2";;
    c_plus_diversity)   echo "--w-dcam-attraction 1.0 --w-dcam-diversity 0.5 --w-dcam-util 0   --w-dcam-energy 0    --w-dcam-entropy 0.2";;
    d_plus_utilization) echo "--w-dcam-attraction 1.0 --w-dcam-diversity 0.5 --w-dcam-util 0.1 --w-dcam-energy 0    --w-dcam-entropy 0.2";;
  esac
}

CONFIGS="minus_attraction minus_entropy minus_diversity minus_utilization minus_energy a_attraction_only b_plus_entropy c_plus_diversity d_plus_utilization"
COMMON="--labeled-ratio 0.05 --memory-update-strength 0.3 --warmup-epochs 50 --ramp-epochs 30"

echo "MASTER START: $(date)"
for cfg in $CONFIGS; do
  for d in $DATASETS; do
    for s in $SEEDS; do
      out="result/DCAMAblation_v2/${cfg}/${d}/seed${s}"
      name="${cfg}__${d}__s${s}"
      if [ -f "${out}/results.json" ]; then
        echo "SKIP ${name}"
        continue
      fi
      while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 30; done
      echo "=== [${name}] START $(date) ==="
      ( python3 run_scAMUSE.py --data "data/${d}.h5ad" --output-dir "$out" --seed "$s" \
          $COMMON $(dparams "$d") $(dcam_weights "$cfg") \
          > "logs_dcamabl_v2/${name}.log" 2>&1 \
        && echo "=== [${name}] OK $(date) ===" \
        || echo "=== [${name}] FAILED (exit $?) $(date) ===" ) &
      sleep 2
    done
  done
done
wait
echo "ALL DCAM-ABLATION-V2 RUNS FINISHED $(date)"