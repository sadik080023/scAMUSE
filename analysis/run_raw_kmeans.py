#!/usr/bin/env python3
"""Raw scGPT embedding + KMeans baseline (no refinement, no DCAM)."""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_scAMUSE import load_and_preprocess_data, evaluate_clustering, set_seed
from utils_splits import create_stratified_splits, verify_no_leakage
from extract_scgpt_embeddings import extract_scgpt_embeddings_inductive
from sklearn.cluster import KMeans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--labeled-ratio", type=float, default=0.05)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    adata = load_and_preprocess_data(args.data)
    labels_str = adata.obs["Group"].astype(str).values
    unique_classes = np.unique(labels_str)
    n_classes = len(unique_classes)
    label_map = {v: i for i, v in enumerate(unique_classes)}
    y_full = np.array([label_map[v] for v in labels_str], dtype=int)
    n_cells = adata.n_obs

    uc, cc = np.unique(y_full, return_counts=True)
    rare = uc[cc < 2]
    if len(rare):
        m = ~np.isin(y_full, rare)
        adata = adata[m].copy(); y_full = y_full[m]
        labels_str = labels_str[m]
        unique_classes = np.unique(labels_str)
        label_map = {v: i for i, v in enumerate(unique_classes)}
        y_full = np.array([label_map[v] for v in labels_str], dtype=int)
        n_cells = adata.n_obs; n_classes = len(unique_classes)

    il, iu, iv, it_ = create_stratified_splits(n_cells, y_full, args.labeled_ratio,
                                               args.val_ratio, args.test_ratio, args.seed)
    verify_no_leakage(il, iu, iv, it_)
    idx_all = np.concatenate([il, iu])

    emb_train, emb_val, emb_test = extract_scgpt_embeddings_inductive(
        adata, train_indices=idx_all, val_indices=iv, test_indices=it_, random_state=args.seed)

    t0 = time.perf_counter()
    km = KMeans(n_clusters=n_classes, n_init="auto", random_state=args.seed)
    y_pred = km.fit_predict(emb_test)
    m = evaluate_clustering(y_full[it_], y_pred)
    res = {"method": "raw_scgpt_kmeans",
           "test_metrics": m, "val_metrics": None,
           "n_classes": n_classes, "n_cells": n_cells,
           "runtime": {"kmeans_clustering_sec": round(time.perf_counter() - t0, 3)}}
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"Raw scGPT+KMeans ARI: {m['ARI']:.4f}  NMI: {m['NMI']:.4f}  ACC: {m['ACC']:.4f}")

if __name__ == "__main__":
    main()
