#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_scAMUSE import load_and_preprocess_data
import numpy as np

for ds in ["Brain", "Fat", "Retina", "Breast_carcinoma", "Quadriceps_tendon", "Cortex", "Mucosal_Atlas", "Innate_lymphocytes"]:
    adata = load_and_preprocess_data(f"data/{ds}.h5ad")
    labels = adata.obs["Group"].astype(str).values
    unique, counts = np.unique(labels, return_counts=True)

    print(f"\n=== {ds} ===")
    print(f"Total cells: {len(labels)}")
    print(f"Classes: {len(unique)}")
    print(f"Min class size: {counts.min()}")
    print(f"Max class size: {counts.max()}")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")