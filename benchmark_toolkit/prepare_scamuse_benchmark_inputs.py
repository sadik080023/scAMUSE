#!/usr/bin/env python3
"""
prepare_scamuse_benchmark_inputs.py

Run INSIDE ~/scAMUSE7 with the same Python environment used for run_scAMUSE.py.

Generates, for all 8 datasets and seeds [11, 42, 79, 239, 528]:

  splits/{DATASET}_s{SEED}.npz
      y_full, idx_train_labeled, idx_train_unlabeled, idx_val, idx_test, n_classes
      -> consumed by every patched baseline via --splits_npz

  baseline_inputs/{DATASET}.h5
      X = raw count matrix of the exact kept cells (float32), Y = int labels
      -> input for scSemiAAE, scAGCL, scCCTR

  scgpt_inputs/{DATASET}_s{SEED}.h5
      X = inductive scGPT embeddings (cells x dim, original cell order),
      Y = int labels, labeled_mask (bool), attrs: n_classes
      -> input for scGPT+scSemiAAE, scGPT+scAGCL, scGPT+scCCTR, scGPT+KMeans/Leiden

  benchmark_config.json
      per-dataset n_cells / n_classes (after scAMUSE filtering)

Preprocessing + rare-class filtering + splitting replicate run_scAMUSE.py EXACTLY:
  - var_names from var['feature_name'] if present
  - filter_cells(min_genes=200), filter_genes(min_cells=3)
  - reuse existing HVG flags, else normalize_total(1e4)+log1p+seurat HVG(2000)
  - label column: first match of Group/cell_type1/cell_type/celltype/
                  cell_ontology_class/label/Label/cluster
  - remove classes with < 2 cells
  - create_stratified_splits(labeled=0.05, val=0.1, test=0.2, random_state=seed)

Existing output files are skipped (safe to re-run / resume).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import scanpy as sc
import h5py

SCAMUSE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCAMUSE_DIR)

from utils_splits import create_stratified_splits  # noqa: E402

DATASETS = [
    "Brain", "Fat", "Retina", "Breast_carcinoma",
    "Quadriceps_tendon", "Cortex", "Mucosal_Atlas", "Innate_lymphocytes",
]
SEEDS = [11, 42, 79, 239, 528]

LABELED_RATIO = 0.05
VAL_RATIO = 0.1
TEST_RATIO = 0.2

CANDIDATE_LABELS = [
    "Group", "cell_type1", "cell_type", "celltype",
    "cell_ontology_class", "label", "Label", "cluster",
]


# ---------------------------------------------------------------------------
# Exact copy of run_scAMUSE.load_and_preprocess_data
# ---------------------------------------------------------------------------
def load_and_preprocess_data(data_path: str):
    adata = sc.read_h5ad(data_path)

    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str)
        adata.var_names_make_unique()
        print("   Using adata.var['feature_name'] as gene symbols")

    adata.var_names_make_unique()
    print(f"   Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    if "highly_variable" in adata.var.columns and adata.var["highly_variable"].sum() >= 200:
        print("   Detected existing HVG flags - reusing them")
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()

    print(f"   Retained {adata.n_vars} highly variable genes")
    return adata


def detect_labels(obs):
    for key in CANDIDATE_LABELS:
        if key in obs.columns:
            return key
    raise ValueError("No label column found in adata.obs!")


def to_dense(X):
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(SCAMUSE_DIR, "data"))
    ap.add_argument("--out_dir", default=SCAMUSE_DIR,
                    help="splits/, baseline_inputs/, scgpt_inputs/ go under here")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--skip_scgpt", action="store_true",
                    help="only write splits + baseline_inputs (no embedding extraction)")
    ap.add_argument("--max_baseline_genes", type=int, default=0,
                    help="if >0, keep at most this many most-variable genes in baseline_inputs h5")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "splits"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "baseline_inputs"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "scgpt_inputs"), exist_ok=True)

    extract_fn = None
    if not args.skip_scgpt:
        try:
            from extract_scgpt_embeddings import extract_scgpt_embeddings_inductive
            extract_fn = extract_scgpt_embeddings_inductive
            print("extract_scgpt_embeddings_inductive imported OK")
        except Exception as e:
            print("!" * 70)
            print("Could not import extract_scgpt_embeddings_inductive from")
            print("extract_scgpt_embeddings.py in", SCAMUSE_DIR)
            print("Error:", repr(e))
            print("Re-run with --skip_scgpt to only write splits + baseline_inputs,")
            print("or fix the import and re-run to also write scgpt_inputs.")
            print("!" * 70)
            sys.exit(2)

    config = {}

    for d in args.datasets:
        path = os.path.join(args.data_dir, f"{d}.h5ad")
        print("\n" + "=" * 70)
        print(f"DATASET {d}  ({path})")
        print("=" * 70)
        t0 = time.time()

        # ------------------------------------------------------------------
        # 1) Full scAMUSE preprocessing (identical to run_scAMUSE.py)
        # ------------------------------------------------------------------
        adata = load_and_preprocess_data(path)

        label_key = detect_labels(adata.obs)
        adata.obs["Group"] = adata.obs[label_key].astype(str)
        print(f"   Label column: '{label_key}', classes: {adata.obs['Group'].nunique()}")

        # Rare-class pre-filter (<2 cells) — same order as run_scAMUSE.py
        labels_str = adata.obs["Group"].astype(str).values
        unique_classes = np.unique(labels_str)
        label_map = {v: i for i, v in enumerate(unique_classes)}
        y_full = np.array([label_map[v] for v in labels_str], dtype=np.int64)

        uc, cc = np.unique(y_full, return_counts=True)
        rare = uc[cc < 2]
        if len(rare) > 0:
            keep = ~np.isin(y_full, rare)
            adata = adata[keep].copy()
            y_full = y_full[keep]
            labels_str = labels_str[keep]
            unique_classes = np.unique(labels_str)
            label_map = {v: i for i, v in enumerate(unique_classes)}
            y_full = np.array([label_map[v] for v in labels_str], dtype=np.int64)
            print(f"   Removed {len(rare)} rare class(es); now {adata.n_obs} cells")

        n_cells = adata.n_obs
        n_classes = int(len(unique_classes))
        config[d] = {"n_cells": int(n_cells), "n_classes": n_classes,
                     "label_column": label_key}
        print(f"   FINAL: {n_cells} cells, {n_classes} classes")

        # ------------------------------------------------------------------
        # 2) baseline_inputs/{d}.h5 — raw matrix of the SAME kept cells
        #    (X taken from the ORIGINAL file, before any normalization)
        # ------------------------------------------------------------------
        h5_out = os.path.join(args.out_dir, "baseline_inputs", f"{d}.h5")
        if os.path.exists(h5_out):
            print(f"   [skip] {h5_out} exists")
        else:
            adata_raw = sc.read_h5ad(path)
            # align to kept cells by obs_names (order-preserving)
            adata_raw = adata_raw[adata.obs_names].copy()
            X_raw = to_dense(adata_raw.X)
            if args.max_baseline_genes and X_raw.shape[1] > args.max_baseline_genes:
                vari = np.var(X_raw, axis=0)
                keepg = np.argsort(vari)[-args.max_baseline_genes:]
                keepg = np.sort(keepg)
                X_raw = X_raw[:, keepg]
                print(f"   baseline h5: keeping top {args.max_baseline_genes} variable genes")
            gb = X_raw.nbytes / 1e9
            print(f"   baseline h5: X {X_raw.shape} ({gb:.2f} GB dense float32)")
            with h5py.File(h5_out, "w") as f:
                f.create_dataset("X", data=X_raw, compression="gzip",
                                 compression_opts=1, chunks=True)
                f.create_dataset("Y", data=y_full.astype(np.int64))
                f.attrs["n_classes"] = n_classes
            print(f"   wrote {h5_out}")

        # ------------------------------------------------------------------
        # 3) splits + scGPT embeddings per seed
        # ------------------------------------------------------------------
        for s in args.seeds:
            npz_out = os.path.join(args.out_dir, "splits", f"{d}_s{s}.npz")
            if os.path.exists(npz_out):
                sp = np.load(npz_out)
                idx_labeled = sp["idx_train_labeled"]
                idx_unlabeled = sp["idx_train_unlabeled"]
                idx_val = sp["idx_val"]
                idx_test = sp["idx_test"]
                assert len(sp["y_full"]) == n_cells, "npz cell count mismatch!"
                print(f"   [skip] {npz_out} exists")
            else:
                idx_labeled, idx_unlabeled, idx_val, idx_test = create_stratified_splits(
                    n_samples=n_cells, labels=y_full,
                    labeled_ratio=LABELED_RATIO, val_ratio=VAL_RATIO,
                    test_ratio=TEST_RATIO, random_state=s,
                )
                np.savez_compressed(
                    npz_out,
                    y_full=y_full,
                    idx_train_labeled=idx_labeled.astype(np.int64),
                    idx_train_unlabeled=idx_unlabeled.astype(np.int64),
                    idx_val=idx_val.astype(np.int64),
                    idx_test=idx_test.astype(np.int64),
                    n_classes=np.int64(n_classes),
                )
                print(f"   wrote {npz_out} (labeled={len(idx_labeled)}, test={len(idx_test)})")

            if extract_fn is None:
                continue

            emb_out = os.path.join(args.out_dir, "scgpt_inputs", f"{d}_s{s}.h5")
            if os.path.exists(emb_out):
                print(f"   [skip] {emb_out} exists")
                continue

            idx_train_all = np.concatenate([idx_labeled, idx_unlabeled])
            print(f"   extracting scGPT embeddings (seed={s}) ...")
            t1 = time.time()
            emb_train, emb_val, emb_test = extract_fn(
                adata,
                train_indices=idx_train_all,
                val_indices=idx_val,
                test_indices=idx_test,
                random_state=s,
            )
            dim = emb_train.shape[1]
            emb_full = np.zeros((n_cells, dim), dtype=np.float32)
            emb_full[idx_train_all] = emb_train
            if emb_val is not None and len(idx_val) > 0:
                emb_full[idx_val] = emb_val
            emb_full[idx_test] = emb_test

            labeled_mask = np.zeros(n_cells, dtype=bool)
            labeled_mask[idx_labeled] = True

            with h5py.File(emb_out, "w") as f:
                f.create_dataset("X", data=emb_full, compression="gzip",
                                 compression_opts=1, chunks=True)
                f.create_dataset("Y", data=y_full.astype(np.int64))
                f.create_dataset("labeled_mask", data=labeled_mask)
                f.attrs["n_classes"] = n_classes
            print(f"   wrote {emb_out} (dim={dim}, {time.time()-t1:.1f}s)")

        print(f"DATASET {d} done in {(time.time()-t0)/60:.1f} min")

    cfg_path = os.path.join(args.out_dir, "benchmark_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print("\nAll done. Config:", cfg_path)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
