#!/usr/bin/env python3
"""
scCCTR_patched.py — scCCTR baseline aligned with the scAMUSE benchmark protocol.

Faithful re-implementation of scCCTR's two-stage pipeline (main.py +
train_and_predict_model), with the same patch contract as the other baselines:

  --data_h5       h5 file with X (raw counts, cells x genes) and Y (int labels),
                  cells in EXACTLY the same order as the splits npz
  --splits_npz    npz with y_full / idx_train_labeled / idx_test / n_classes
  --seed          controls KMeans / torch / numpy RNGs
  test-only evaluation: metrics computed on idx_test cells only

Stage 1 (Iterative_selecting): VAE -> KMeans -> consensus -> agglomerative,
  early stop when ARI(prev, current) > 0.95, top-40%-per-cluster AE refinement
  (identical hyperparameters to main.py: VAE 1200 epochs, AE 120 epochs,
  latent 64, hidden [256,128,64], max 6 iterations).
Stage 2: Transformer trained on the top-40% pseudo-labeled cells
  (num_heads=32, num_layers=8, lr=1e-3, batch 32, early stop patience 5),
  then predicts all cells.

Final line printed:  FINAL_METRICS dataset=.. seed=.. ARI=.. NMI=.. AMI=.. ACC=..
"""

import argparse
import json
import os
import random
import time

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from AEmodel import AE, VAE, train_autoencoder
from prepross import _prepare_training_data
from Transformer import EarlyStopping, TransformerModel, train_model, validate_model
from tSNE import GeneExpressionDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def extract_top_percent_per_cluster(data, clusters, num_clusters, frac=0.4):
    """Identical selection rule as main.py (comment says 30%, code uses 0.4)."""
    centroids = []
    for i in range(num_clusters):
        cluster_points = data[clusters == i, :]
        centroids.append(np.mean(cluster_points, axis=0) if len(cluster_points) else np.zeros(data.shape[1]))
    centroids = np.array(centroids)

    new_matrices, top_indices = [], []
    for i in range(num_clusters):
        cluster_indices = np.where(clusters == i)[0]
        cluster_points = data[cluster_indices, :]
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
        n_sel = max(1, int(frac * len(cluster_indices)))
        nearest = np.argsort(distances)[:n_sel]
        new_matrices.append(data[cluster_indices[nearest], :])
        top_indices.append(cluster_indices[nearest])
    return new_matrices, top_indices


def process_gene_expression_matrices(matrices, epochs=120, learning_rate=1e-4):
    reconstructed = []
    for i, matrix in enumerate(matrices):
        if matrix.shape[0] == 0:
            reconstructed.append(matrix)
            continue
        print(f"Training autoencoder for cluster {i} ({matrix.shape[0]} cells)...")
        ae = AE(dim=matrix.shape[1])
        ae = train_autoencoder(ae, matrix, epochs=epochs, learning_rate=learning_rate)
        with torch.no_grad():
            rec, _ = ae(torch.tensor(matrix, dtype=torch.float32))
        reconstructed.append(rec.detach().numpy())
    return reconstructed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_h5", required=True)
    ap.add_argument("--splits_npz", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_clusters", type=int, default=None,
                    help="default: n_classes from the splits npz")
    ap.add_argument("--max_iterations", type=int, default=6)
    ap.add_argument("--vae_training_epochs", type=int, default=1200)
    ap.add_argument("--ae_epochs", type=int, default=120)
    ap.add_argument("--ae_learning_rate", type=float, default=1e-4)
    # stage-2 transformer (defaults identical to main.py)
    ap.add_argument("--num_heads", type=int, default=32)
    ap.add_argument("--num_layers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--num_epochs_earlystop", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--delta", type=float, default=1e-2)
    args = ap.parse_args()

    t_start = time.time()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- load data + splits ----------------
    with h5py.File(os.path.expanduser(args.data_h5), "r") as f:
        X_counts = np.asarray(f["X"][:], dtype=np.float32)
        Y = np.asarray(f["Y"][:], dtype=np.int64)

    sp = np.load(args.splits_npz, allow_pickle=True)
    assert X_counts.shape[0] == len(sp["y_full"]), (
        f"cell count mismatch: {X_counts.shape[0]} vs {len(sp['y_full'])}")
    y_full = sp["y_full"].astype(np.int64)
    assert np.array_equal(y_full, Y), "label mismatch between h5 and splits npz!"
    idx_test = sp["idx_test"]
    n_cells = len(y_full)
    num_clusters = args.num_clusters or int(sp["n_classes"])

    est_gb = (n_cells ** 2) * 4 / 1e9
    print("=" * 70)
    print(f"scCCTR (patched) | cells={n_cells} genes={X_counts.shape[1]} "
          f"K={num_clusters} seed={args.seed}")
    print(f"Consensus matrix will need ~{est_gb:.1f} GB RAM (float32 n x n)")
    if est_gb > 24:
        print("WARNING: likely OOM on this machine — consider skipping this dataset")
    print("=" * 70)

    # ---------------- scCCTR preprocessing (prepross._prepare_training_data) ---
    adata = sc.AnnData(pd.DataFrame(X_counts).fillna(0.0))
    adata_hvg, X_hvg = _prepare_training_data(adata)
    cell_names = np.array([f"cell_{i}" for i in range(n_cells)])
    print(f"X_hvg shape: {X_hvg.shape}")

    # ---------------- stage 1: iterative consensus ----------------
    vae = VAE(num_genes=2000, latent_dim=64, device=device,
              hidden_dim=[256, 128, 64], decoder_activation="nn.linear").to(device)

    all_clusterings, previous_clusters = [], None
    stable_clusters = None
    top_indices = None

    for iteration in range(args.max_iterations):
        print(f"\nIteration {iteration + 1}/{args.max_iterations}")
        data_tensor = torch.tensor(X_hvg, dtype=torch.float32).to(device)
        for epoch in range(args.vae_training_epochs):
            losses = vae.train_vae(data_tensor)
            if (epoch + 1) % 200 == 0:
                print(f"Epoch [{epoch + 1}/{args.vae_training_epochs}], "
                      f"recon: {losses['loss_reconstruction']:.4f}")

        vae.eval()
        with torch.no_grad():
            data_z = vae.encoder(data_tensor).cpu().numpy()

        kmeans = KMeans(n_clusters=num_clusters, init="k-means++",
                        random_state=args.seed, n_init=10)
        all_clusterings.append(kmeans.fit_predict(data_z))

        # memory-efficient consensus: co-association via chunking
        consensus = np.zeros((n_cells, n_cells), dtype=np.float32)
        for c in all_clusterings:
            consensus += (c[:, None] == c[None, :])
        consensus /= len(all_clusterings)
        distance_matrix = 1.0 - consensus
        del consensus

        agg = AgglomerativeClustering(n_clusters=num_clusters,
                                      metric="precomputed", linkage="average")
        stable_clusters = agg.fit_predict(distance_matrix)
        all_clusterings[-1] = stable_clusters
        del distance_matrix

        if previous_clusters is not None:
            ari_change = adjusted_rand_score(previous_clusters, stable_clusters)
            print(f"ARI(prev, current) = {ari_change:.4f}")
            if ari_change > 0.95:
                print("Converged (ARI > 0.95), stopping iterations.")
                break
        previous_clusters = stable_clusters

        new_matrices, top_indices = extract_top_percent_per_cluster(
            X_hvg, stable_clusters, num_clusters)
        reconstructed = process_gene_expression_matrices(
            new_matrices, args.ae_epochs, args.ae_learning_rate)
        for idx, rec in zip(top_indices, reconstructed):
            X_hvg[idx, :] = rec

    # consensus clusters (pre-transformer), test-only metrics for the log
    scores_direct = evaluate(y_full[idx_test], stable_clusters[idx_test])
    print("Direct consensus (test): ARI %.4f NMI %.4f AMI %.4f ACC %.4f"
          % (scores_direct["ARI"], scores_direct["NMI"],
             scores_direct["AMI"], scores_direct["ACC"]))

    # ---------------- stage 2: transformer on top-40% pseudo-labels ----------
    if top_indices is None:  # converged at iteration 1
        _, top_indices = extract_top_percent_per_cluster(
            X_hvg, stable_clusters, num_clusters)

    sel = np.concatenate(top_indices)
    X_train = X_hvg[sel]
    y_train = stable_clusters[sel]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    n_pseudo_classes = len(set(y_train.tolist()))
    model = TransformerModel(input_dim=X_hvg.shape[1], num_classes=n_pseudo_classes,
                             num_heads=args.num_heads, num_layers=args.num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    g = torch.Generator().manual_seed(args.seed)
    train_dataset = GeneExpressionDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, generator=g)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, verbose=True)

    for _ in range(args.num_epochs_earlystop):
        train_model(model, train_loader, optimizer, criterion, args.num_epochs, device)
        val_loss = validate_model(model, train_loader, criterion, device)
        if early_stopping(val_loss, model):
            print("Early stopping triggered")
            break

    model.eval()
    preds = []
    with torch.no_grad():
        full = torch.tensor(scaler.transform(X_hvg), dtype=torch.float32)
        for xb in DataLoader(full, batch_size=256, shuffle=False):
            preds.extend(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    preds = np.asarray(preds)

    scores = evaluate(y_full[idx_test], preds[idx_test])
    runtime_min = (time.time() - t_start) / 60

    pd.DataFrame({"cell_index": np.arange(n_cells), "true_label": y_full,
                  "pred_cluster": preds}).to_csv(
        os.path.join(args.out_dir, "predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"seed": args.seed, "n_cells": n_cells, "K": num_clusters,
                   "test_metrics": scores,
                   "direct_consensus_test_metrics": scores_direct,
                   "runtime_min": round(runtime_min, 2)}, f, indent=2)

    dataset_name = os.path.basename(args.data_h5).replace(".h5", "")
    print(f"\nFINAL_METRICS dataset={dataset_name} seed={args.seed} "
          f"ARI={scores['ARI']:.4f} NMI={scores['NMI']:.4f} "
          f"AMI={scores['AMI']:.4f} ACC={scores['ACC']:.4f}")
    print(f"Runtime: {runtime_min:.1f} min | outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
