"""
run_scAMUSE.py

Main orchestration script for scAMUSE - Enhanced DCAM
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import warnings
import argparse
import json
import time

# ========== RUNTIME MEASUREMENT ADDITION ==========
import psutil
# =================================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import scanpy as sc
import torch

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from extract_scgpt_embeddings import extract_scgpt_embeddings_inductive
from model import EnhancedFusionTrainer
from utils_splits import create_stratified_splits, create_partial_labels, verify_no_leakage


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_clustering(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)

    def cluster_accuracy(true_labels_inner, pred_labels_inner):
        cont = metrics.cluster.contingency_matrix(true_labels_inner, pred_labels_inner)
        row_ind, col_ind = linear_sum_assignment(-cont)
        return cont[row_ind, col_ind].sum() / np.sum(cont)

    acc = cluster_accuracy(true_labels, pred_labels)
    return {"ARI": ari, "NMI": nmi, "AMI": ami, "ACC": acc}


def load_and_preprocess_data(data_path: str):
    print("[1/5] Loading and preprocessing data...")
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
        print(f"   Detected label column: '{found}'")
        print(f"   Total classes: {adata.obs['Group'].nunique()}")
    else:
        raise ValueError("No label column found in adata.obs!")

    return adata


def main():
    # ========== RUNTIME MEASUREMENT: START ==========
    program_start_time = time.perf_counter()
    start_cpu = time.process_time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    # =================================================
    
    parser = argparse.ArgumentParser(description="scAMUSE Pipeline - Enhanced DCAM")

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--labeled-ratio", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="result/scAMUSE_output")
    parser.add_argument("--lr", type=float, default=1e-3)

    # Training hyperparameters
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate in refinement head")
    parser.add_argument("--patience", type=int, default=10,help="Patience for early stopping")
    parser.add_argument("--val-eval-interval", type=int, default=5,
                        help="Compute validation KMeans ARI every N epochs (val loss every epoch)")

    # DCAM parameters
    parser.add_argument("--n-memories", type=int, default=None)
    parser.add_argument("--extra-memory-slots", type=int, default=None)
    parser.add_argument("--memory-temperature", type=float, default=0.1)
    parser.add_argument("--memory-update-strength", type=float, default=0.5)
    parser.add_argument("--memory-n-iters", type=int, default=3)

    # DCAM loss weights
    parser.add_argument("--w-dcam", type=float, default=0.1)
    parser.add_argument("--w-dcam-energy", type=float, default=0.05)
    parser.add_argument("--w-dcam-entropy", type=float, default=0.2)
    parser.add_argument("--w-dcam-diversity", type=float, default=0.5)
    parser.add_argument("--w-dcam-util", type=float, default=0.1)
    parser.add_argument("--w-dcam-attraction", type=float, default=1.0)

    # Supervised loss weights
    parser.add_argument("--w-triplet", type=float, default=1.0)
    parser.add_argument("--w-pairwise", type=float, default=0.1)
    parser.add_argument("--w-supcon", type=float, default=0.5)
    parser.add_argument("--w-ce", type=float, default=1.0)

    # Enhancement features
    parser.add_argument("--init-from-labels", action="store_true", default=True)
    parser.add_argument("--no-init-from-labels", dest="init_from_labels", action="store_false")
    parser.add_argument("--adaptive-weights", action="store_true", default=True)
    parser.add_argument("--no-adaptive-weights", dest="adaptive_weights", action="store_false")
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--ramp-epochs", type=int, default=40)

    parser.add_argument("--early-stop-metric", type=str, default="loss", choices=["loss", "ari"])

    args = parser.parse_args()

    print(f"\n===== Running scAMUSE | seed={args.seed} | labeled_ratio={args.labeled_ratio} | data={args.data} =====")

    runtime_log = {}
    t0_total = time.perf_counter()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("scAMUSE Pipeline - Enhanced DCAM")
    print("   Inductive PCA/Scaler | Semi-supervised Refinement | Enhanced DCAM")
    print("=" * 70)

    # ========== RUNTIME MEASUREMENT: START TIME ==========
    runtime_log["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    runtime_log["start_memory_mb"] = start_memory
    # =====================================================

    # [1/5] Load data
    t0 = time.perf_counter()
    adata = load_and_preprocess_data(args.data)
    runtime_log["load_preprocess_sec"] = round(time.perf_counter() - t0, 3)

    n_cells = adata.n_obs
    labels_str = adata.obs["Group"].astype(str).values
    unique_classes = np.unique(labels_str)
    n_classes = len(unique_classes)
    label_map = {v: i for i, v in enumerate(unique_classes)}
    y_full = np.array([label_map[v] for v in labels_str], dtype=int)

    print(f"\nDataset: {n_cells} cells, {n_classes} classes")

    # [2/5] Create stratified data splits
    print("\n[2/5] Creating stratified data splits...")
    t0 = time.perf_counter()

    unique_classes_arr, class_counts = np.unique(y_full, return_counts=True)
    rare_classes = unique_classes_arr[class_counts < 2]

    if len(rare_classes) > 0:
        print(f"   Pre-filtering {len(rare_classes)} rare class(es) with < 2 samples")
        valid_mask = ~np.isin(y_full, rare_classes)
        adata = adata[valid_mask].copy()
        y_full = y_full[valid_mask]
        labels_str = labels_str[valid_mask]
        unique_classes = np.unique(labels_str)
        label_map = {v: i for i, v in enumerate(unique_classes)}
        y_full = np.array([label_map[v] for v in labels_str], dtype=int)
        n_cells = adata.n_obs
        n_classes = len(unique_classes)
        print(f"   Filtered: {n_cells} cells, {n_classes} classes")

    idx_train_labeled, idx_train_unlabeled, idx_val, idx_test = create_stratified_splits(
        n_samples=n_cells, labels=y_full,
        labeled_ratio=args.labeled_ratio, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, random_state=args.seed
    )

    verify_no_leakage(idx_train_labeled, idx_train_unlabeled, idx_val, idx_test)
    runtime_log["split_sec"] = round(time.perf_counter() - t0, 3)

    idx_train_all = np.concatenate([idx_train_labeled, idx_train_unlabeled])

    # [3/5] Extract scGPT embeddings
    print("\n[3/5] Extracting scGPT embeddings...")
    t0 = time.perf_counter()

    emb_train, emb_val, emb_test = extract_scgpt_embeddings_inductive(
        adata, train_indices=idx_train_all, val_indices=idx_val,
        test_indices=idx_test, random_state=args.seed
    )

    runtime_log["scgpt_embedding_sec"] = round(time.perf_counter() - t0, 3)
    print(f"   Train embeddings: {emb_train.shape}")
    print(f"   Test embeddings: {emb_test.shape}")

    # [4/5] Prepare data and create trainer
    print("\n[4/5] Preparing training data...")
    
    # Create partial labels
    y_partial = create_partial_labels(y_full, idx_train_labeled, idx_train_unlabeled)
    y_partial_train = y_partial[idx_train_all]

    # Calculate memory slots
    if args.n_memories is not None:
        n_memories = args.n_memories
        if args.extra_memory_slots is not None:
            print(f"   ⚠️ Warning: Both --n-memories and --extra-memory-slots set. Using --n-memories.")
    else:
        if args.extra_memory_slots is not None:
            extra_slots = args.extra_memory_slots
        else:
            extra_slots = max(3, n_classes // 5)
        n_memories = n_classes + extra_slots
        print(f"   Auto-calculated: n_memories = {n_classes} + {extra_slots} = {n_memories}")

    n_anchor_slots = n_classes
    if n_anchor_slots > n_memories:
        raise ValueError(f"n_anchor_slots ({n_anchor_slots}) > n_memories ({n_memories})")

    print(f"   Enhanced DCAM: {n_memories} total slots ({n_anchor_slots} anchors + {n_memories - n_anchor_slots} free)")

    # Create trainer
    trainer = EnhancedFusionTrainer(
        input_dim=emb_train.shape[1],
        n_classes=n_classes,
        n_memories=n_memories,
        n_anchor_slots=n_anchor_slots,
        memory_temperature=args.memory_temperature,
        memory_update_strength=args.memory_update_strength,
        memory_n_iters=args.memory_n_iters,
        lr=args.lr,
        device="cpu",
        seed=args.seed,
        w_dcam=args.w_dcam,
        w_dcam_energy=args.w_dcam_energy,
        w_dcam_entropy=args.w_dcam_entropy,
        w_dcam_diversity=args.w_dcam_diversity,
        w_dcam_util=args.w_dcam_util,
        w_dcam_attraction=args.w_dcam_attraction,
        w_triplet=args.w_triplet,
        w_pairwise=args.w_pairwise,
        w_supcon=args.w_supcon,
        w_ce=args.w_ce,
        init_from_labels=args.init_from_labels,
        adaptive_weights=args.adaptive_weights,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        total_epochs=args.epochs,
    )

    # Initialize memories from labeled centroids (Idea 1)
    # FIX: idx_train_labeled are full-dataset indices (0 to n_cells-1)
    # emb_train only contains train set samples (0 to len(idx_train_all)-1)
    # Need to map full-dataset indices to train-set positions
    if args.init_from_labels:
        print("\n   Initializing DCAM memories from labeled centroids...")
        
        # Create mapping: full_dataset_index -> train_set_position
        train_index_map = {full_idx: train_pos for train_pos, full_idx in enumerate(idx_train_all)}
        
        # Convert full-dataset indices to train-set relative indices
        idx_labeled_relative = np.array([
            train_index_map[i] for i in idx_train_labeled
        ])
        
        trainer.initialize_memories_from_labels(
            embeddings=emb_train,
            labels=y_partial_train,
            idx_labeled=idx_labeled_relative,
            n_classes=n_classes
        )

    val_emb = emb_val
    val_labels = y_full[idx_val] if len(idx_val) > 0 else None

    # [5/5] Train
    print("\n[5/5] Training Enhanced scAMUSE model...")
    print(f"   Supervised weights: CE={args.w_ce} | "
          f"Triplet={args.w_triplet} | Pairwise={args.w_pairwise} | SupCon={args.w_supcon}")
    t0 = time.perf_counter()

    final_emb_train = trainer.fit_transform(
        embeddings=emb_train,
        labels=y_partial_train,
        val_embeddings=val_emb,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        val_eval_interval=args.val_eval_interval,
        verbose=True,
    )

    runtime_log["training_sec"] = round(time.perf_counter() - t0, 3)
    print(f"   Final training embeddings shape: {final_emb_train.shape}")

    # Transform validation and test sets
    print("\n   Transforming validation and test sets...")
    final_emb_val = trainer.transform(emb_val, batch_size=args.batch_size) if emb_val is not None else None
    final_emb_test = trainer.transform(emb_test, batch_size=args.batch_size)

    # Build full embedding array
    final_emb = np.zeros((n_cells, final_emb_train.shape[1]), dtype=np.float32)
    final_emb[idx_train_all] = final_emb_train
    if final_emb_val is not None:
        final_emb[idx_val] = final_emb_val
    final_emb[idx_test] = final_emb_test

    np.save(f"{args.output_dir}/fused_embeddings.npy", final_emb)
    print(f"   Full embeddings shape: {final_emb.shape}")

    # Evaluation
    print("\n[6/6] Evaluation (TEST SET)...")
    t0 = time.perf_counter()

    # ========== RUNTIME MEASUREMENT: KMEANS START ==========
    kmeans_start = time.perf_counter()
    # =======================================================

    X_test = final_emb[idx_test]
    y_test_true = y_full[idx_test]

    kmeans = KMeans(n_clusters=n_classes, n_init="auto", random_state=args.seed)
    y_test_pred = kmeans.fit_predict(X_test)
    
    # ========== RUNTIME MEASUREMENT: KMEANS END ==========
    runtime_log["kmeans_clustering_sec"] = round(time.perf_counter() - kmeans_start, 3)
    # =====================================================
    
    metrics_test = evaluate_clustering(y_test_true, y_test_pred)

    print(f"\nTEST SET RESULTS ({len(idx_test)} cells):")
    print(f"   ARI:  {metrics_test['ARI']:.4f}")
    print(f"   NMI:  {metrics_test['NMI']:.4f}")
    print(f"   AMI:  {metrics_test['AMI']:.4f}")
    print(f"   ACC:  {metrics_test['ACC']:.4f}")

    if len(idx_val) > 0:
        # ========== RUNTIME MEASUREMENT: VAL KMEANS START ==========
        val_kmeans_start = time.perf_counter()
        # ===========================================================
        
        X_val = final_emb[idx_val]
        y_val_true = y_full[idx_val]
        y_val_pred = kmeans.fit_predict(X_val)
        
        # ========== RUNTIME MEASUREMENT: VAL KMEANS END ==========
        runtime_log["val_kmeans_clustering_sec"] = round(time.perf_counter() - val_kmeans_start, 3)
        # =========================================================
        
        metrics_val = evaluate_clustering(y_val_true, y_val_pred)
        print(f"\nVALIDATION SET RESULTS ({len(idx_val)} cells):")
        print(f"   ARI:  {metrics_val['ARI']:.4f}")
        print(f"   NMI:  {metrics_val['NMI']:.4f}")
        print(f"   AMI:  {metrics_val['AMI']:.4f}")
        print(f"   ACC:  {metrics_val['ACC']:.4f}")
    else:
        metrics_val = None

    runtime_log["evaluation_sec"] = round(time.perf_counter() - t0, 3)
    
    # ========== RUNTIME MEASUREMENT: FINAL CALCULATION ==========
    end_time = time.perf_counter()
    end_cpu = time.process_time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    runtime_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    runtime_log["wall_clock_sec"] = round(end_time - program_start_time, 3)
    runtime_log["wall_clock_min"] = round((end_time - program_start_time) / 60, 2)
    runtime_log["cpu_time_sec"] = round(end_cpu - start_cpu, 3)
    runtime_log["peak_memory_mb"] = round(end_memory, 2)
    runtime_log["memory_change_mb"] = round(end_memory - start_memory, 2)
    
    if runtime_log["wall_clock_sec"] > 0:
        runtime_log["cpu_efficiency"] = round(runtime_log["cpu_time_sec"] / runtime_log["wall_clock_sec"], 3)
    else:
        runtime_log["cpu_efficiency"] = 0.0
    
    runtime_log["total_runtime_sec"] = round(end_time - program_start_time, 3)
    runtime_log["total_runtime_min"] = round((end_time - program_start_time) / 60, 2)
    # =============================================================

    results = {
        "method": "scAMUSE_enhanced",
        "test_metrics": metrics_test,
        "val_metrics": metrics_val,
        "n_classes": n_classes,
        "n_cells": n_cells,
        "n_memories": n_memories,
        "extra_slots": n_memories - n_classes,
        "runtime": runtime_log,
    }

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ========== RUNTIME MEASUREMENT: PRINT SUMMARY ==========
    print("\n" + "=" * 70)
    print("RUNTIME SUMMARY:")
    print("=" * 70)
    print(f"   Start time:           {runtime_log.get('start_time', 'N/A')}")
    print(f"   End time:             {runtime_log.get('end_time', 'N/A')}")
    print(f"   Wall clock time:      {runtime_log.get('wall_clock_min', 0):.2f} min ({runtime_log.get('wall_clock_sec', 0):.2f} sec)")
    print(f"   CPU time:             {runtime_log.get('cpu_time_sec', 0):.2f} sec")
    print(f"   CPU efficiency:       {runtime_log.get('cpu_efficiency', 0):.2f}x")
    print(f"   Peak memory:          {runtime_log.get('peak_memory_mb', 0):.2f} MB")
    print(f"   Memory change:        {runtime_log.get('memory_change_mb', 0):.2f} MB")
    print("-" * 70)
    print(f"   Load + Preprocess:    {runtime_log.get('load_preprocess_sec', 0):.2f} sec")
    print(f"   Data splitting:       {runtime_log.get('split_sec', 0):.2f} sec")
    print(f"   scGPT embedding:      {runtime_log.get('scgpt_embedding_sec', 0):.2f} sec")
    print(f"   Model training:       {runtime_log.get('training_sec', 0):.2f} sec")
    print(f"   K-means clustering:   {runtime_log.get('kmeans_clustering_sec', 0):.2f} sec")
    print(f"   Evaluation:           {runtime_log.get('evaluation_sec', 0):.2f} sec")
    print("=" * 70)
    # =========================================================

    print(f"\nComplete! Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()