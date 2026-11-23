#!/usr/bin/env python3
"""Collect DCAM ablation v2 results (proto-free rerun, Sheets 4a + 4c).
Usage:  cd ~/scAMUSE7 && python3 collect_dcam_ablation_v2.py
Reads result/DCAMAblation_v2/{config}/{dataset}/seed*/results.json
Writes result/DCAMAblation_v2/dcam_ablation_v2_results.csv and prints mean±std.
"""
import json, glob, csv, os
import numpy as np

BASE = "result/DCAMAblation_v2"
ORDER = ["minus_attraction", "minus_entropy", "minus_diversity", "minus_utilization", "minus_energy",
         "a_attraction_only", "b_plus_entropy", "c_plus_diversity", "d_plus_utilization"]
DATASETS = ["Brain", "Fat", "Retina", "Breast_carcinoma", "Quadriceps_tendon", "Cortex",
            "Mucosal_Atlas", "Innate_lymphocytes"]
METRICS = ["ARI", "NMI", "AMI", "ACC"]

def get(rec, k):
    # metrics live under "test_metrics" in scAMUSE results.json
    src = rec.get("test_metrics", rec)
    for kk in (k, k.lower(), k.upper()):
        if kk in src:
            return src[kk]
    raise KeyError(k)

rows = []
for cfg in ORDER:
    for d in DATASETS:
        for f in sorted(glob.glob(os.path.join(BASE, cfg, d, "seed*", "results.json"))):
            with open(f) as fh:
                r = json.load(fh)
            seed = int(os.path.basename(os.path.dirname(f)).replace("seed", ""))
            rows.append({"config": cfg, "dataset": d, "seed": seed,
                         **{m: get(r, m) for m in METRICS}})

out_csv = os.path.join(BASE, "dcam_ablation_v2_results.csv")
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["config", "dataset", "seed"] + METRICS)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out_csv} ({len(rows)} runs)\n")

print(f"{'dataset':<20} {'config':<20} n  ARI            NMI            AMI            ACC  (mean±std)")
for d in DATASETS:
    for cfg in ORDER:
        sub = [r for r in rows if r["dataset"] == d and r["config"] == cfg]
        if not sub:
            print(f"{d:<20} {cfg:<20} 0  MISSING")
            continue
        cells = []
        for m in METRICS:
            v = np.array([r[m] for r in sub], dtype=float)
            cells.append(f"{v.mean():.4f}±{v.std():.4f}")
        print(f"{d:<20} {cfg:<20} {len(sub)}  " + "  ".join(cells))