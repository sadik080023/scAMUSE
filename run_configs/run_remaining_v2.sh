#!/bin/bash
# ============================================================
# REMAINING v2 runs (PROTO-FREE code) — Sheets 3 / 5 / 9 refresh
#   Part 1: Component ablation v2  (Sheet 3 internal rows)
#           refinement_only / no_anchor_init / no_free_slots / no_adaptive_weights
#           4 configs x 8 datasets x 5 seeds = 160 runs
#           NOTE: Quadriceps_tendon no_free_slots is SKIPPED (full config already
#                 uses --n-memories 22 = n_classes, i.e. 0 free slots) -> 155 runs
#   Part 2: Label selection v2     (Sheet 5)
#           full config, split seeds 1001-1004 -> 8 datasets x 4 seeds = 32 runs
#   Part 3: Low label v2           (Sheet 9)
#           ratios 0.01 0.02 0.10 0.20 x 8 datasets x 5 seeds = 160 runs
#           (5% row = existing v2 full, not repeated)
#   Part 4: K-sensitivity v2       (Sheet 6) — NO retraining needed!
#           post-hoc KMeans at K-2..K+2 on saved fused_embeddings.npy
#           runs automatically at the end via k_sensitivity_v2.py
# TOTAL training runs: 347   (CPU, ~1-3 min each)
#
# Launch:  cd ~/scAMUSE7 && MAX_JOBS=6 nohup bash run_remaining_v2.sh > run_remaining_v2.log 2>&1 &
# Progress: grep -c "] OK " run_remaining_v2.log   (target 347)
# Resume-safe: existing results.json => SKIP. Just rerun the same command.
# ============================================================
set -u
cd ~/scAMUSE7
MAX_JOBS=${MAX_JOBS:-6}
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p logs_remaining_v2

SEEDS="11 42 79 239 528"
DATASETS="Brain Fat Retina Breast_carcinoma Quadriceps_tendon Cortex Mucosal_Atlas Innate_lymphocytes"
COMMON="--labeled-ratio 0.05 --memory-update-strength 0.3 --warmup-epochs 50 --ramp-epochs 30"

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

# True number of classes per dataset (for no_free_slots: n_memories = n_classes => 0 free slots)
nclasses() {
  case "$1" in
    Brain) echo 22;; Fat) echo 34;; Retina) echo 71;; Breast_carcinoma) echo 19;;
    Quadriceps_tendon) echo 22;; Cortex) echo 11;; Mucosal_Atlas) echo 22;; Innate_lymphocytes) echo 9;;
  esac
}

# Extra flags per component-ablation config (full model otherwise)
comp_flags() {
  case "$1" in
    refinement_only)     echo "--w-dcam 0";;
    no_anchor_init)      echo "--no-init-from-labels";;
    no_adaptive_weights) echo "--no-adaptive-weights";;
    no_free_slots)       echo "NOMEM";;
  esac
}

submit() { # $1=outdir $2=logname $3=seed $4=dataset $5=extra-args
  local out="$1" name="$2" s="$3" d="$4" extra="$5"
  if [ -f "${out}/results.json" ]; then
    echo "SKIP ${name}"
    return
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 30; done
  echo "=== [${name}] START $(date) ==="
  ( python3 run_scAMUSE.py --data "data/${d}.h5ad" --output-dir "$out" --seed "$s" \
      $COMMON $(dparams "$d") $extra \
      > "logs_remaining_v2/${name}.log" 2>&1 \
    && echo "=== [${name}] OK $(date) ===" \
    || echo "=== [${name}] FAILED (exit $?) $(date) ===" ) &
  sleep 2
}

echo "MASTER START: $(date)"

# ---------------- Part 1: Component ablation v2 ----------------
echo "##### PART 1: COMPONENT ABLATION V2 #####"
for cfg in refinement_only no_anchor_init no_free_slots no_adaptive_weights; do
  for d in $DATASETS; do
    if [ "$cfg" = "no_free_slots" ] && [ "$d" = "Quadriceps_tendon" ]; then
      echo "SKIP no_free_slots__Quadriceps_tendon (full config already has 0 free slots)"
      continue
    fi
    extra=$(comp_flags "$cfg")
    if [ "$extra" = "NOMEM" ]; then extra="--n-memories $(nclasses "$d")"; fi
    for s in $SEEDS; do
      submit "result/ComponentAblation_v2/${cfg}/${d}/seed${s}" "${cfg}__${d}__s${s}" "$s" "$d" "$extra"
    done
  done
done

# ---------------- Part 2: Label selection v2 ----------------
echo "##### PART 2: LABEL SELECTION V2 #####"
for d in $DATASETS; do
  for s in 1001 1002 1003 1004; do
    submit "result/LabelSelection_v2/${d}/seed${s}" "labelsel__${d}__s${s}" "$s" "$d" ""
  done
done

# ---------------- Part 3: Low label v2 ----------------
echo "##### PART 3: LOW LABEL V2 #####"
for ratio in 0.01 0.02 0.10 0.20; do
  for d in $DATASETS; do
    for s in $SEEDS; do
      submit "result/LowLabel_v2/ratio${ratio}/${d}/seed${s}" "lowlabel${ratio}__${d}__s${s}" "$s" "$d" "--labeled-ratio ${ratio}"
    done
  done
done

wait
echo "ALL TRAINING RUNS FINISHED $(date)"

# ---------------- Part 4: collect + K-sensitivity (post-hoc) ----------------
echo "##### PART 4: COLLECT + K-SENSITIVITY (post-hoc, no training) #####"
python3 collect_remaining_v2.py
python3 k_sensitivity_v2.py
echo "MASTER DONE: $(date)"