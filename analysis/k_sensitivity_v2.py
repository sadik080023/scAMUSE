#!/usr/bin/env python3
"""K-sensitivity v2 (Sheet 6 refresh) — POST-HOC, no retraining.
Re-clusters the SAVED refined test embeddings (fused_embeddings.npy from the
v2 full-model runs) with KMeans at K-2..K+2, n_init=10, per seed.

Run from ~/scAMUSE7 :  python3 k_sensitivity_v2.py
Needs: data/<DATASET>.h5ad and result/**/<DATASET>/seed<S>/fused_embeddings.npy
Writes: collected_v2/k_sensitivity_v2_results.csv
"""
import os, sys, csv, glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.expanduser("~/scAMUSE7")
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from run_scAMUSE import load_and_preprocess_data          # noqa: E402
from utils_splits import create_stratified_splits          # noqa: E402

SEEDS = [11, 42, 79, 239, 528]
DATASETS = ["Brain", "Fat", "Retina", "Breast_carcinoma",
            "Quadriceps_tendon", "Cortex", "Mucosal_Atlas", "Innate_lymphocytes"]

def find_embeddings(d, s):
    cands = glob.glob(f"result/**/{d}/seed{s}/fused_embeddings.npy", recursive=True)
    if not cands:
        return None
    full = [c for c in cands if "full" in c.lower()]
    phase1 = [c for c in cands if "Phase1" in c]
    for pool in (full, phase1, cands):
        if pool:
            return sorted(pool)[0]
    return None

os.makedirs("collected_v2", exist_ok=True)
out = open("collected_v2/k_sensitivity_v2_results.csv", "w", newline="")
w = csv.writer(out)
w.writerow(["dataset", "seed", "K_offset", "K_value", "ARI"])

for d in DATASETS:
    adata = load_and_preprocess_data(f"data/{d}.h5ad")
    labels_str = adata.obs["Group"].astype(str).values
    unique_classes = np.unique(labels_str)
    label_map = {v: i for i, v in enumerate(unique_classes)}
    y_full = np.array([label_map[v] for v in labels_str], dtype=int)
    # same rare-class pre-filtering as run_scAMUSE.py
    uc, counts = np.unique(y_full, return_counts=True)
    rare = uc[counts < 2]
    if len(rare):
        mask = ~np.isin(y_full, rare)
        y_full = y_full[mask]
        labels_str = labels_str[mask]
        unique_classes = np.unique(labels_str)
        label_map = {v: i for i, v in enumerate(unique_classes)}
        y_full = np.array([label_map[v] for v in labels_str], dtype=int)
    n_cells = len(y_full)
    K = len(unique_classes)

    per_offset = {off: [] for off in (-2, -1, 0, 1, 2)}
    for s in SEEDS:
        emb_path = find_embeddings(d, s)
        if emb_path is None:
            print(f"!! {d} seed{s}: fused_embeddings.npy NOT FOUND — skipped")
            continue
        _, _, _, idx_test = create_stratified_splits(
            n_samples=n_cells, labels=y_full,
            labeled_ratio=0.05, val_ratio=0.1, test_ratio=0.2, random_state=s)
        emb = np.load(emb_path)
        X_test = emb[idx_test]
        y_test = y_full[idx_test]
        for off in (-2, -1, 0, 1, 2):
            k = K + off
            pred = KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X_test)
            ari = adjusted_rand_score(y_test, pred)
            per_offset[off].append(ari)
            w.writerow([d, s, off, k, f"{ari:.4f}"])
        print(f"{d} seed{s}: done (emb from {emb_path})")
    print(f"\n### {d} (true K={K}) ###")
    for off in (-2, -1, 0, 1, 2):
        v = per_offset[off]
        if v:
            print(f"  K{'+' if off >= 0 else ''}{off} ({K + off}): ARI {np.mean(v):.4f} ± {np.std(v):.4f} (n={len(v)})")

out.close()
print("\nWROTE collected_v2/k_sensitivity_v2_results.csv")