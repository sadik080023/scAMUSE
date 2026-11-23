"""
run_refinement_only.py

Refinement-only ablation for scAMUSE
Shudhu RefinementHead + SemiSupervisedLoss (DCAM memory network chara)
Same pipeline structure as run_scAMUSE.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import warnings
import argparse
import json
import time

# ========== RUNTIME MEASUREMENT ==========
import psutil
# =========================================

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
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from extract_scgpt_embeddings import extract_scgpt_embeddings_inductive
from model import RefinementHead, SemiSupervisedLoss
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


# ============================================================
# Refinement-Only Trainer
# ============================================================

class RefinementOnlyTrainer:
    """
    Refinement-only trainer: shudhu RefinementHead + Classifier + SemiSupervisedLoss.
    DCAM memory network completely removed.
    """

    def __init__(self, input_dim, n_classes, hidden_dim=256, output_dim=128,
                 lr=1e-3, weight_decay=1e-5, dropout=0.2, device="cpu", seed=42,
                 w_triplet=1.0, w_pairwise=0.1, w_supcon=0.5, w_proto=0.5, w_ce=1.0,
                 triplet_margin=0.5, early_stop_metric="loss"):

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.n_classes = n_classes
        self.device = device
        self.early_stop_metric = early_stop_metric

        # Refinement head (same as EnhancedFusionTrainer)
        self.refinement_head = RefinementHead(input_dim, hidden_dim, output_dim, dropout).to(device)

        # Classifier
        self.classifier = nn.Linear(output_dim, n_classes).to(device)

        # Semi-supervised loss (same as main pipeline)
        self.criterion = SemiSupervisedLoss(
            n_classes=n_classes,
            w_triplet=w_triplet,
            w_pairwise=w_pairwise,
            w_supcon=w_supcon,
            w_proto=w_proto,
            w_ce=w_ce,
            triplet_margin=triplet_margin,
            device=device
        )

        # Optimizer: only refinement_head + classifier + criterion (prototypes)
        all_params = (
            list(self.refinement_head.parameters()) +
            list(self.classifier.parameters()) +
            list(self.criterion.parameters())
        )
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

        self.best_state = None

    def fit_transform(self, embeddings, labels, val_embeddings=None, val_labels=None,
                      epochs=100, batch_size=256, patience=10, verbose=True):
        """
        Train refinement-only model and return final embeddings.

        Args:
            embeddings: np.ndarray (n_train, input_dim)
            labels: np.ndarray (n_train,) with -1 for unlabeled
            val_embeddings: optional validation embeddings
            val_labels: optional validation labels
        """
        n_samples = embeddings.shape[0]
        embeddings_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=self.device)

        best_val_loss = float("inf")
        best_val_ari = -1.0
        patience_counter = 0

        if verbose:
            print(f"   Refinement-only mode (no DCAM)")
            print(f"   Early stopping metric: {self.early_stop_metric}")

        self.refinement_head.train()
        self.classifier.train()

        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=self.device)
            epoch_losses = {
                "total": 0, "triplet": 0, "pairwise": 0,
                "supcon": 0, "proto": 0, "ce": 0
            }
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                x_batch = embeddings_t[batch_idx]
                batch_labels = labels_t[batch_idx]

                # Forward: refinement head -> classifier
                z = self.refinement_head(x_batch)
                logits = self.classifier(z)

                # Semi-supervised loss (handles labeled + unlabeled)
                loss_dict = self.criterion(z, logits, batch_labels)
                total_loss = loss_dict["total"]

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 1.0)
                self.optimizer.step()

                epoch_losses["total"] += total_loss.item()
                for k in ["triplet", "pairwise", "supcon", "proto", "ce"]:
                    epoch_losses[k] += loss_dict.get(k, 0)
                n_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            # Validation
            val_loss = None
            val_ari = None
            if val_embeddings is not None and val_labels is not None:
                self.refinement_head.eval()
                self.classifier.eval()

                with torch.no_grad():
                    val_emb_t = torch.tensor(val_embeddings, dtype=torch.float32, device=self.device)
                    val_labels_t = torch.tensor(val_labels, dtype=torch.long, device=self.device)
                    z_val = self.refinement_head(val_emb_t)
                    logits_val = self.classifier(z_val)
                    val_loss = F.cross_entropy(logits_val, val_labels_t).item()

                    # Validation ARI for early stopping
                    z_val_np = z_val.cpu().numpy()
                    kmeans_val = KMeans(n_clusters=self.n_classes, random_state=42, n_init="auto")
                    val_pred = kmeans_val.fit_predict(z_val_np)
                    val_ari = adjusted_rand_score(val_labels, val_pred)

                self.refinement_head.train()
                self.classifier.train()

                # Early stopping
                if self.early_stop_metric == "ari":
                    if val_ari > best_val_ari:
                        best_val_ari = val_ari
                        patience_counter = 0
                        self.best_state = {
                            'refinement_head': self.refinement_head.state_dict().copy(),
                            'classifier': self.classifier.state_dict().copy(),
                            'criterion': self.criterion.state_dict().copy(),
                        }
                    else:
                        patience_counter += 1
                else:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.best_state = {
                            'refinement_head': self.refinement_head.state_dict().copy(),
                            'classifier': self.classifier.state_dict().copy(),
                            'criterion': self.criterion.state_dict().copy(),
                        }
                    else:
                        patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        best_val = best_val_ari if self.early_stop_metric == "ari" else best_val_loss
                        metric_name = "ARI" if self.early_stop_metric == "ari" else "loss"
                        print(f"Early stopping at epoch {epoch + 1} (best val {metric_name}: {best_val:.4f})")
                    # Restore best model
                    if self.best_state is not None:
                        self.refinement_head.load_state_dict(self.best_state['refinement_head'])
                        self.classifier.load_state_dict(self.best_state['classifier'])
                        self.criterion.load_state_dict(self.best_state['criterion'])
                    break

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                ari_str = f"{val_ari:.4f}" if val_ari is not None else "N/A"
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Total: {epoch_losses['total']:.4f} | "
                      f"Triplet: {epoch_losses['triplet']:.4f} | "
                      f"Proto: {epoch_losses['proto']:.4f} | "
                      f"CE: {epoch_losses['ce']:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val ARI: {ari_str}")

        # Generate final embeddings
        self.refinement_head.eval()

        with torch.no_grad():
            final_embeddings = []
            for i in range(0, n_samples, batch_size):
                x_batch = embeddings_t[i:i+batch_size]
                z = self.refinement_head(x_batch)
                final_embeddings.append(z.cpu())

        return torch.cat(final_embeddings, dim=0).numpy()

    def transform(self, embeddings, batch_size=512):
        """Transform new embeddings through refinement head only."""
        self.refinement_head.eval()
        embeddings_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        outputs = []

        with torch.no_grad():
            for i in range(0, embeddings_t.shape[0], batch_size):
                x_batch = embeddings_t[i:i+batch_size]
                z = self.refinement_head(x_batch)
                outputs.append(z.cpu())

        return torch.cat(outputs, dim=0).numpy()


def main():
    # ========== RUNTIME MEASUREMENT: START ==========
    program_start_time = time.perf_counter()
    start_cpu = time.process_time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    # =================================================

    parser = argparse.ArgumentParser(description="scAMUSE Refinement-Only Ablation")

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--labeled-ratio", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="result/refinement_only_output")
    parser.add_argument("--lr", type=float, default=1e-3)

    # Training hyperparameters
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)

    # Semi-supervised loss weights (same as main pipeline)
    parser.add_argument("--w-triplet", type=float, default=1.0)
    parser.add_argument("--w-pairwise", type=float, default=0.1)
    parser.add_argument("--w-supcon", type=float, default=0.5)
    parser.add_argument("--w-proto", type=float, default=0.5)
    parser.add_argument("--w-ce", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.5)

    parser.add_argument("--early-stop-metric", type=str, default="loss", choices=["loss", "ari"])

    args = parser.parse_args()

    print(f"\n===== Running scAMUSE Refinement-Only | seed={args.seed} | labeled_ratio={args.labeled_ratio} | data={args.data} =====")

    runtime_log = {}
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("scAMUSE Refinement-Only Ablation")
    print("   Inductive PCA/Scaler | Semi-supervised Refinement | NO DCAM")
    print("=" * 70)

    # ========== RUNTIME MEASUREMENT ==========
    runtime_log["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    runtime_log["start_memory_mb"] = start_memory
    # =========================================

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

    # [4/5] Prepare data and create refinement-only trainer
    print("\n[4/5] Preparing training data...")

    # Create partial labels
    y_partial = create_partial_labels(y_full, idx_train_labeled, idx_train_unlabeled)
    y_partial_train = y_partial[idx_train_all]

    # Create refinement-only trainer (NO DCAM)
    trainer = RefinementOnlyTrainer(
        input_dim=emb_train.shape[1],
        n_classes=n_classes,
        hidden_dim=256,
        output_dim=128,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        device="cpu",
        seed=args.seed,
        w_triplet=args.w_triplet,
        w_pairwise=args.w_pairwise,
        w_supcon=args.w_supcon,
        w_proto=args.w_proto,
        w_ce=args.w_ce,
        triplet_margin=args.triplet_margin,
        early_stop_metric=args.early_stop_metric,
    )

    val_emb = emb_val
    val_labels = y_full[idx_val] if len(idx_val) > 0 else None

    # [5/5] Train
    print("\n[5/5] Training Refinement-Only model...")
    t0 = time.perf_counter()

    final_emb_train = trainer.fit_transform(
        embeddings=emb_train,
        labels=y_partial_train,
        val_embeddings=val_emb,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
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

    X_test = final_emb[idx_test]
    y_test_true = y_full[idx_test]

    kmeans = KMeans(n_clusters=n_classes, n_init="auto", random_state=args.seed)
    y_test_pred = kmeans.fit_predict(X_test)

    metrics_test = evaluate_clustering(y_test_true, y_test_pred)

    print(f"\nTEST SET RESULTS ({len(idx_test)} cells):")
    print(f"   ARI:  {metrics_test['ARI']:.4f}")
    print(f"   NMI:  {metrics_test['NMI']:.4f}")
    print(f"   AMI:  {metrics_test['AMI']:.4f}")
    print(f"   ACC:  {metrics_test['ACC']:.4f}")

    if len(idx_val) > 0:
        X_val = final_emb[idx_val]
        y_val_true = y_full[idx_val]
        y_val_pred = kmeans.fit_predict(X_val)
        metrics_val = evaluate_clustering(y_val_true, y_val_pred)
        print(f"\nVALIDATION SET RESULTS ({len(idx_val)} cells):")
        print(f"   ARI:  {metrics_val['ARI']:.4f}")
        print(f"   NMI:  {metrics_val['NMI']:.4f}")
        print(f"   AMI:  {metrics_val['AMI']:.4f}")
        print(f"   ACC:  {metrics_val['ACC']:.4f}")
    else:
        metrics_val = None

    runtime_log["evaluation_sec"] = round(time.perf_counter() - t0, 3)

    # ========== RUNTIME MEASUREMENT: FINAL ==========
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
    # =================================================

    results = {
        "method": "scAMUSE_refinement_only",
        "test_metrics": metrics_test,
        "val_metrics": metrics_val,
        "n_classes": n_classes,
        "n_cells": n_cells,
        "runtime": runtime_log,
    }

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ========== RUNTIME SUMMARY ==========
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
    print(f"   Evaluation:           {runtime_log.get('evaluation_sec', 0):.2f} sec")
    print("=" * 70)
    # ======================================

    print(f"\nComplete! Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()