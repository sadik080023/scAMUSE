# scAMUSE

scAMUSE is a label-efficient semi-supervised single-cell RNA-seq clustering framework. It refines frozen scGPT-derived cell embeddings with a lightweight MLP head and a dynamic context-aware associative memory (DCAM), trained with a composite objective (cross-entropy + triplet + pairwise + supervised contrastive) plus five memory regularizers, using only 5% labeled cells. Training is CPU-only and takes minutes per dataset.

Companion code for: *"scAMUSE: Associative Memory-Based Deep Clustering for Single-Cell Multi-Omics"* (manuscript).

## Overview

scAMUSE performs:

1. data loading and preprocessing from `.h5ad` files (normalization, HVG, z-score — train-only statistics);
2. stratified splitting into labeled (5%), unlabeled (65%), validation (10%), and test (20%) partitions (exact indices in `splits/`);
3. inductive scGPT embedding extraction (PCA and scaling fitted on training cells only — no test information leakage);
4. semi-supervised refinement: a 2-layer MLP head producing L2-normalized 128-d embeddings;
5. dynamic context-aware associative memory: K anchor slots initialized from labeled class centroids plus free slots, 3-step iterative softmax retrieval, and five regularizers (attraction, diversity, utilization, energy, entropy);
6. K-means clustering at known K on test embeddings and evaluation with ARI, NMI, AMI, and ACC.

## Project structure

```
scAMUSE/
├── run_configs/               # launcher scripts for every experiment in the paper
├── analysis/                  # ablation runners, K-sensitivity, result collectors
├── splits/                    # stratified split indices (.npz) for 8 datasets × 5 seeds
├── results/                   # master result tables (CSV) for all experiments
├── benchmark_toolkit/         # baseline runners/patches (11 baseline methods)
├── scgpt/                     # scGPT source code (included)
├── scGPT_pretrained/          # scGPT checkpoint files (best_model.pt: see download note below)
├── data/                      # input .h5ad files (links below; not included)
├── run_scAMUSE.py             # main pipeline script
├── model.py                   # refinement head, semi-supervised losses, trainer
├── dcam_memory.py             # DCAM associative memory module
├── extract_scgpt_embeddings.py
├── utils_splits.py
├── get_data_partitions.py
├── requirements.txt
├── requirements_full.txt
└── README.md
```

## Installation

```
pip install -r requirements.txt
```

(`requirements_full.txt` pins the exact package versions used in our experiments.)

The scGPT source code (`scgpt/`, from https://github.com/bowanglab/scGPT) is already included in this repository.

## scGPT checkpoint

The scGPT whole-human checkpoint file `best_model.pt` is too large for GitHub and is therefore **not included**. Download it from:

https://huggingface.co/MohamedMabrouk/scGPT/commit/72555feb2fdff37022f9ffddb23fa09f3b7067c3

and place it in the `scGPT_pretrained/` folder (which already contains `args.json` and `vocab.json`).

## Input data

Place the `.h5ad` files in the `data/` folder. The AnnData object must contain a cell-type label column (e.g., `cell_type`). The eight benchmark datasets are publicly available:

| Dataset | Source |
| --- | --- |
| Brain | https://cellxgene.cziscience.com/e/4b8b5fbd-a24e-4d34-864d-16dfd4650d20.cxg/ |
| Fat | https://cellxgene.cziscience.com/e/8f67108f-1f78-4148-a14a-bff85c58f19b.cxg/ |
| Retina (AC subset) | https://doi.org/10.5281/zenodo.14648190 |
| Breast carcinoma | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7782698 |
| Quadriceps tendon | https://cellxgene.cziscience.com/e/acd544d0-4d8b-46c7-98b3-fc48f7e6fdb7.cxg/ |
| Cortex | https://cellxgene.cziscience.com/e/4b83f213-b96b-4c87-a884-9b81faddb194.cxg/ |
| Mucosal Atlas | https://cellxgene.cziscience.com/e/a22a3aeb-5e68-4084-8a45-94b42015adf8.cxg/ |
| Innate lymphocytes | https://cellxgene.cziscience.com/e/4057e00c-0678-454a-90e4-b341a162d007.cxg/ |

## Usage (main 5%-label setting)

```
python3 run_scAMUSE.py \
  --data data/Brain.h5ad \
  --output-dir result/Brain_s42 \
  --labeled-ratio 0.05 --seed 42 --epochs 100 \
  --memory-temperature 0.1 --memory-update-strength 0.5 \
  --w-dcam 0.1 --w-dcam-attraction 1.0 --w-dcam-diversity 0.5 \
  --w-dcam-util 0.1 --w-dcam-energy 0.05 --w-dcam-entropy 0.2 \
  --warmup-epochs 50 --ramp-epochs 30 \
  --lr 1e-3 --weight-decay 1e-5 --patience 10
```

## Key arguments

| Argument | Description |
| --- | --- |
| `--data` | Path to the input `.h5ad` dataset. |
| `--output-dir` | Directory where results are saved. |
| `--labeled-ratio` | Fraction of **all** cells used as labeled (0.05 = 5%). |
| `--seed` | Model/init and data-split seed. |
| `--epochs` | Maximum training epochs (early stopping: patience 10). |
| `--memory-temperature` | Softmax temperature of DCAM retrieval (0.1 in main experiments). |
| `--memory-update-strength` | Interpolation strength η of the iterative refinement (0.5). |
| `--memory-n-iters` | Number of iterative retrieval steps (3). |
| `--n-memories` / `--extra-memory-slots` | Total slots / free slots beyond K anchors (default: K + max(3, K//5)). |
| `--w-dcam` | Weight of the total DCAM memory loss (0.1). |
| `--no-init-from-labels` | Ablation: disable anchor initialization from labeled centroids. |
| `--no-adaptive-weights` | Ablation: disable the adaptive weight schedule. |
| `--warmup-epochs` / `--ramp-epochs` | Adaptive schedule: warm-up then linear ramp (50/30 in main experiments). |
| `--early-stop-metric` | `loss` or `ari` on the validation split. |

## Output

Each run writes `results.json` (ARI/NMI/AMI/ACC on the test partition, plus runtime and memory statistics) and the refined embeddings (`fused_embeddings.npy`).

## Reproducing the paper

All experiment families (main benchmark, loss ablation, DCAM/component ablation, label selection, low-label, K-sensitivity) are launched by the scripts in `run_configs/`; collectors in `analysis/` aggregate run outputs into the master tables in `results/`. All runs are deterministic given the published seeds (11, 42, 79, 239, 528) and take minutes per run on CPU.

## Main files

| File | Description |
| --- | --- |
| `run_scAMUSE.py` | Main pipeline script. |
| `model.py` | Refinement head, semi-supervised losses, trainer. |
| `dcam_memory.py` | DCAM associative memory (anchor/free slots, iterative retrieval, regularizers). |
| `utils_splits.py` | Stratified splits, partial labels, leakage checks. |
| `extract_scgpt_embeddings.py` | Inductive scGPT embedding extraction (train-fitted PCA/scaler). |

## Citation

If you use this code, please cite the associated scAMUSE manuscript.
