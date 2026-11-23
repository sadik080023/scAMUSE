#!/usr/bin/env python3
"""Collect v2 loss-ablation results: mean±std for ALL metrics.
Run:  cd ~/scAMUSE7 && python3 collect_loss_ablation_v2.py
"""
import json, glob, os, csv
import numpy as np

rows = []
for path in sorted(glob.glob("result/LossAblation_v2/*/*/seed*/results.json")):
    cfg, dataset, seed = path.split(os.sep)[2:5]
    with open(path) as f:
        t = json.load(f)["test_metrics"]
    rows.append(dict(config=cfg, dataset=dataset, seed=int(seed.replace("seed", "")),
                     ARI=t["ARI"], NMI=t["NMI"], AMI=t["AMI"], ACC=t["ACC"]))

os.makedirs("result/LossAblation_v2", exist_ok=True)
with open("result/LossAblation_v2/loss_ablation_v2_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "dataset", "seed", "ARI", "NMI", "AMI", "ACC"])
    w.writeheader(); w.writerows(rows)
print(f"wrote result/LossAblation_v2/loss_ablation_v2_results.csv ({len(rows)} runs)\n")

ORDER = ["full", "A_ce_only", "B_ce_triplet", "D_ce_pairwise", "E_ce_supcon",
         "minus_triplet", "minus_pairwise", "minus_supcon", "minus_ce"]
DATASETS = ["Brain", "Fat", "Retina", "Breast_carcinoma", "Quadriceps_tendon",
            "Cortex", "Mucosal_Atlas", "Innate_lymphocytes"]

def ms(vals):
    return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

print(f"{'dataset':<20}{'config':<16}n  ARI              NMI              AMI              ACC")
for d in DATASETS:
    for cfg in ORDER:
        sel = [r for r in rows if r["dataset"] == d and r["config"] == cfg]
        if not sel:
            continue
        print(f"{d:<20}{cfg:<16}{len(sel)}  "
              f"{ms([r['ARI'] for r in sel]):<17}"
              f"{ms([r['NMI'] for r in sel]):<17}"
              f"{ms([r['AMI'] for r in sel]):<17}"
              f"{ms([r['ACC'] for r in sel])}")