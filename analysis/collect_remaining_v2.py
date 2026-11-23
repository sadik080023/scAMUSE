#!/usr/bin/env python3
"""Collect ComponentAblation_v2 / LabelSelection_v2 / LowLabel_v2 results into CSVs.
Run from ~/scAMUSE7. Prints a mean-ARI summary per config."""
import json, csv, glob, os
from statistics import mean, pstdev

ROOT = os.path.expanduser("~/scAMUSE7")

def get(rec, k):
    src = rec.get("test_metrics", rec)
    for kk in (k, k.lower(), k.upper()):
        if kk in src:
            return src[kk]
    return None

def harvest(pattern, tag, writer):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, pattern))):
        rec = json.load(open(f))
        parts = f.split(os.sep)
        # .../<ROOT>/result/<Group>/<cfg...>/<dataset>/seed<S>/results.json
        seed = parts[-2]
        dataset = parts[-3]
        cfg = parts[-4] if len(parts) >= 5 else ""
        rows.append({
            "group": tag, "config": cfg, "dataset": dataset, "seed": seed.replace("seed", ""),
            "ARI": get(rec, "ARI"), "NMI": get(rec, "NMI"),
            "AMI": get(rec, "AMI"), "ACC": get(rec, "ACC"),
        })
        writer.writerow(rows[-1])
    return rows

def summarize(rows, keys):
    groups = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        groups.setdefault(k, []).append(r["ARI"])
    print(f"\n== {', '.join(keys)} ==")
    for k in sorted(groups):
        v = [x for x in groups[k] if x is not None]
        if v:
            print(f"  {' | '.join(k)}: ARI {mean(v):.4f} ± {pstdev(v):.4f} (n={len(v)})")

os.makedirs(os.path.join(ROOT, "collected_v2"), exist_ok=True)
jobs = [
    ("result/ComponentAblation_v2/*/*/seed*/results.json", "component", "component_ablation_v2_results.csv", ["config"]),
    ("result/LabelSelection_v2/*/seed*/results.json", "label_selection", "label_selection_v2_results.csv", ["dataset"]),
    ("result/LowLabel_v2/*/*/seed*/results.json", "lowlabel", "lowlabel_v2_results.csv", ["config"]),
]
for pattern, tag, outname, keys in jobs:
    out = os.path.join(ROOT, "collected_v2", outname)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "config", "dataset", "seed", "ARI", "NMI", "AMI", "ACC"])
        w.writeheader()
        rows = harvest(pattern, tag, w)
    print(f"{outname}: {len(rows)} runs collected")
    summarize(rows, keys)