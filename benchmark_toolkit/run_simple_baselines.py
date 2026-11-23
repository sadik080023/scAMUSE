#!/usr/bin/env python3
"""
run_simple_baselines.py — simple baselines for the scAMUSE benchmark.

Methods:
  scgpt_kmeans    KMeans on inductive scGPT embeddings
  scgpt_leiden    Leiden (scanpy) on scGPT embeddings, resolution searched to K
  pca_kmeans      scanpy HVG(2000)+scale+PCA(50) on raw counts, then KMeans
  scanpy_louvain  same PCA graph + Louvain (falls back to Leiden if the
                  `louvain` package is missing), resolution searched to K

All methods cluster ALL cells and evaluate on the scAMUSE test split only
(same protocol as the patched scSemiAAE/scAGCL/scCCTR baselines).

Usage (run with the scAMUSE python env, needs scanpy + scikit-learn):
  python run_simple_baselines.py --method scgpt_kmeans --dataset Brain --seed 11 \
      --splits_npz ~/scAMUSE7/splits/Brain_s11.npz \
      --scgpt_h5 ~/scAMUSE7/scgpt_inputs/Brain_s11.h5 \
      --raw_h5 ~/scAMUSE7/baseline_inputs/Brain.h5 \
      --out_dir ~/scAMUSE7/benchmark_results/scgpt_kmeans/Brain_s11
"""

import argparse
import json
import os
import time

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score)


def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return w[r, c].sum() / y_pred.size


def evaluate(y_true, y_pred):
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "AMI": float(adjusted_mutual_info_score(y_true, y_pred)),
        "ACC": float(cluster_acc(y_true, y_pred)),
    }


def build_knn_graph(ad, X, n_neighbors=15, seed=0):
    """Build an exact 15-NN graph with sklearn and hand it to scanpy.

    Bypasses sc.pp.neighbors, which crashes in this env (scanpy expects
    csr_matrix.getnnz but receives scipy csr_array). Exact kNN with
    euclidean distance is a standard, deterministic choice for a baseline.
    """
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X)
    conn = sparse.csr_matrix(nn.kneighbors_graph(X, mode="connectivity"))
    dist = sparse.csr_matrix(nn.kneighbors_graph(X, mode="distance"))
    ad.obsp["connectivities"] = conn
    ad.obsp["distances"] = dist
    ad.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": n_neighbors, "method": "sklearn",
                   "metric": "euclidean", "use_rep": "X",
                   "random_state": seed},
    }


def resolution_search(adata, n_clusters, seed, flavor="leiden", n_iter=40):
    """Binary-search the clustering resolution to hit exactly n_clusters."""
    import scanpy as sc

    lo, hi, best = 0.01, 10.0, None
    for _ in range(n_iter):
        res = (lo + hi) / 2
        if flavor == "louvain":
            sc.tl.louvain(adata, resolution=res, random_state=seed, key_added="cl")
        else:
            sc.tl.leiden(adata, resolution=res, random_state=seed, key_added="cl")
        k = adata.obs["cl"].nunique()
        best = adata.obs["cl"].astype(int).values
        if k == n_clusters:
            break
        if k < n_clusters:
            lo = res
        else:
            hi = res
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["scgpt_kmeans", "scgpt_leiden",
                             "pca_kmeans", "scanpy_louvain"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits_npz", required=True)
    ap.add_argument("--scgpt_h5", default=None, help="needed for scgpt_* methods")
    ap.add_argument("--raw_h5", default=None, help="needed for pca_*/scanpy_* methods")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_pcs", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    sp = np.load(args.splits_npz, allow_pickle=True)
    y_full = sp["y_full"].astype(np.int64)
    idx_test = sp["idx_test"]
    K = int(sp["n_classes"])

    print("=" * 70)
    print(f"{args.method} | dataset={args.dataset} seed={args.seed} K={K}")

    # ---------------- load features ----------------
    if args.method.startswith("scgpt"):
        with h5py.File(os.path.expanduser(args.scgpt_h5), "r") as f:
            X = np.asarray(f["X"][:], dtype=np.float32)
            Y = np.asarray(f["Y"][:], dtype=np.int64)
        assert len(X) == len(y_full) and np.array_equal(Y, y_full)
    else:
        import scanpy as sc
        with h5py.File(os.path.expanduser(args.raw_h5), "r") as f:
            Xraw = np.asarray(f["X"][:], dtype=np.float32)
            Y = np.asarray(f["Y"][:], dtype=np.int64)
        assert len(Xraw) == len(y_full) and np.array_equal(Y, y_full)
        adata = sc.AnnData(Xraw)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=args.n_pcs, random_state=args.seed)
        X = adata.obsm["X_pca"].astype(np.float32)

    # ---------------- cluster ----------------
    if args.method.endswith("kmeans"):
        km = KMeans(n_clusters=K, n_init="auto", random_state=args.seed)
        pred = km.fit_predict(X)
    else:
        import scanpy as sc
        ad = sc.AnnData(X)
        build_knn_graph(ad, X, n_neighbors=15, seed=args.seed)
        flavor = "leiden"
        if args.method == "scanpy_louvain":
            try:
                import louvain  # noqa: F401
                flavor = "louvain"
            except ImportError:
                print("WARNING: `louvain` package not installed — "
                      "falling back to Leiden for this run")
        pred = resolution_search(ad, K, args.seed, flavor=flavor)

    scores = evaluate(y_full[idx_test], pred[idx_test])
    runtime_min = (time.time() - t0) / 60

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"method": args.method, "dataset": args.dataset,
                   "seed": args.seed, "K": K, "test_metrics": scores,
                   "runtime_min": round(runtime_min, 2)}, f, indent=2)

    print(f"FINAL_METRICS method={args.method} dataset={args.dataset} "
          f"seed={args.seed} ARI={scores['ARI']:.4f} NMI={scores['NMI']:.4f} "
          f"AMI={scores['AMI']:.4f} ACC={scores['ACC']:.4f} "
          f"runtime_min={runtime_min:.1f}")


if __name__ == "__main__":
    main()
