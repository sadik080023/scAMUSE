"""
run_raw_scgpt.py
Raw scGPT baseline - aligned with clean main pipeline
"""

import numpy as np
import scanpy as sc
import argparse
import warnings
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

from extract_scgpt_embeddings import extract_scgpt_embeddings_inductive
from utils_splits import create_stratified_splits, verify_no_leakage


def load_and_preprocess_data(data_path: str):
    """Same preprocessing as main pipeline."""
    adata = sc.read_h5ad(data_path)

    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].astype(str)
        adata.var_names_make_unique()

    adata.var_names_make_unique()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    if "highly_variable" in adata.var.columns and adata.var["highly_variable"].sum() >= 200:
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
        adata = adata[:, adata.var.highly_variable].copy()

    candidate_labels = [
        "Group", "cell_type1", "cell_type", "celltype",
        "cell_ontology_class", "label", "Label", "cluster"
    ]

    found = None
    for key in candidate_labels:
        if key in adata.obs.columns:
            found = key
            break

    if found:
        adata.obs["Group"] = adata.obs[found].astype(str)
    else:
        raise ValueError("No label column found!")

    return adata


def evaluate_clustering(true_labels, pred_labels):
    """Same evaluation as main pipeline."""
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)

    def cluster_accuracy(true_labels_inner, pred_labels_inner):
        cont = metrics.cluster.contingency_matrix(true_labels_inner, pred_labels_inner)
        row_ind, col_ind = linear_sum_assignment(-cont)
        return cont[row_ind, col_ind].sum() / np.sum(cont)

    acc = cluster_accuracy(true_labels, pred_labels)
    return {"ARI": ari, "NMI": nmi, "AMI": ami, "ACC": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load and preprocess
    adata = load_and_preprocess_data(args.data)

    labels_str = adata.obs["Group"].astype(str).values
    unique_classes = np.unique(labels_str)
    label_map = {v: i for i, v in enumerate(unique_classes)}
    y_full = np.array([label_map[v] for v in labels_str], dtype=int)
    n_classes = len(unique_classes)

    # Rare-class filtering
    unique_classes_counts, class_counts = np.unique(y_full, return_counts=True)
    rare_classes = unique_classes_counts[class_counts < 2]
    if len(rare_classes) > 0:
        valid_mask = ~np.isin(y_full, rare_classes)
        adata = adata[valid_mask].copy()
        y_full = y_full[valid_mask]
        labels_str = labels_str[valid_mask]
        unique_classes = np.unique(labels_str)
        label_map = {v: i for i, v in enumerate(unique_classes)}
        y_full = np.array([label_map[v] for v in labels_str], dtype=int)
        n_classes = len(unique_classes)

    # Split
    idx_train_labeled, idx_train_unlabeled, idx_val, idx_test = create_stratified_splits(
        n_samples=len(y_full),
        labels=y_full,
        labeled_ratio=args.labeled_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )
    verify_no_leakage(idx_train_labeled, idx_train_unlabeled, idx_val, idx_test)

    idx_train_all = np.concatenate([idx_train_labeled, idx_train_unlabeled])

    # Extract embeddings (INDUCTIVE - aligned with main pipeline)
    emb_train, emb_val, emb_test = extract_scgpt_embeddings_inductive(
        adata,
        train_indices=idx_train_all,
        val_indices=idx_val,
        test_indices=idx_test,
        random_state=args.seed,
    )

    emb = np.zeros((len(y_full), emb_train.shape[1]), dtype=np.float32)
    emb[idx_train_all] = emb_train
    if emb_val is not None:
        emb[idx_val] = emb_val
    emb[idx_test] = emb_test

    # Clustering on test set
    kmeans_test = KMeans(n_clusters=n_classes, random_state=args.seed, n_init=10)
    y_test_pred = kmeans_test.fit_predict(emb[idx_test])
    metrics_test = evaluate_clustering(y_full[idx_test], y_test_pred)

    print(f"\nTEST SET RESULTS ({len(idx_test)} cells):")
    print(f"   ARI:  {metrics_test['ARI']:.4f}")
    print(f"   NMI:  {metrics_test['NMI']:.4f}")
    print(f"   AMI:  {metrics_test['AMI']:.4f}")
    print(f"   ACC:  {metrics_test['ACC']:.4f}")

    # Validation evaluation
    if len(idx_val) > 0:
        kmeans_val = KMeans(n_clusters=n_classes, random_state=args.seed, n_init=10)
        y_val_pred = kmeans_val.fit_predict(emb[idx_val])
        metrics_val = evaluate_clustering(y_full[idx_val], y_val_pred)

        print(f"\nVALIDATION SET RESULTS ({len(idx_val)} cells):")
        print(f"   ARI:  {metrics_val['ARI']:.4f}")
        print(f"   NMI:  {metrics_val['NMI']:.4f}")
        print(f"   AMI:  {metrics_val['AMI']:.4f}")
        print(f"   ACC:  {metrics_val['ACC']:.4f}")


if __name__ == "__main__":
    main()