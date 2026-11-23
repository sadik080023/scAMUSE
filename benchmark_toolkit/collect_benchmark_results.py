#!/usr/bin/env python3
"""
collect_benchmark_results.py — aggregate all baseline logs into a table.

Scans:
  ~/benchmark_methods/{scSemiAAE,scAGCL,scCCTR,scGPT_scSemiAAE,scGPT_scAGCL,scGPT_scCCTR}/logs/*.log
  ~/benchmark_methods/scANVI/results/**/  (metrics.json)
  ~/scAMUSE7/benchmark_results/**/        (metrics.json + logs/*.log)

Outputs:
  ~/scAMUSE7/benchmark_results/benchmark_results.csv   (one row per run)
  prints mean ± std per dataset x method, ready for Sheet 7_Benchmark
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

BM = os.path.expanduser("~/benchmark_methods")
SC = os.path.expanduser("~/scAMUSE7")

METHOD_DIRS = {
    "scSemiAAE": f"{BM}/scSemiAAE/logs",
    "scAGCL": f"{BM}/scAGCL/logs",
    "scCCTR": f"{BM}/scCCTR/logs",
    "scGPT+scSemiAAE": f"{BM}/scGPT_scSemiAAE/logs",
    "scGPT+scAGCL": f"{BM}/scGPT_scAGCL/logs",
    "scGPT+scCCTR": f"{BM}/scGPT_scCCTR/logs",
}

METRIC_RE = re.compile(
    r"(ARI|NMI|AMI|ACC)\s*[=:]?\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", re.I)
NAME_RE = re.compile(r"(.+?)_s(\d+)")


def parse_log(path):
    """Return (dataset, seed, metrics) from the LAST metrics-bearing line."""
    best = None
    try:
        with open(path, errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in lines:
        found = dict((k.upper(), float(v)) for k, v in METRIC_RE.findall(line))
        if {"ARI", "NMI", "ACC"} <= set(found):
            best = found
    if best is None:
        return None
    base = os.path.basename(path).replace(".log", "")
    m = NAME_RE.search(base)
    if not m:
        return None
    dataset, seed = m.group(1), int(m.group(2))
    # some logs are named e.g. Breast_s11 / Quad_s11 — normalise nothing here
    return dataset, seed, best


def from_metrics_json(path):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    met = d.get("test_metrics", d)
    if not {"ARI", "NMI", "ACC"} <= set(met):
        return None
    return (d.get("dataset"), d.get("seed"),
            {k.upper(): float(met[k.upper()]) for k in ("ari", "nmi", "ami", "acc")
             if k.upper() in met or k in met})


def main():
    rows = []

    for method, d in METHOD_DIRS.items():
        for log in glob.glob(os.path.join(d, "*.log")):
            r = parse_log(log)
            if r:
                ds, seed, met = r
                rows.append({"method": method, "dataset": ds, "seed": seed, **met})

    # simple baselines + patched scCCTR + scANVI metrics.json files
    for pattern, method in [
        (f"{SC}/benchmark_results/*/*/metrics.json", None),
        (f"{BM}/scCCTR/result_scamuse/*/metrics.json", "scCCTR"),
        (f"{BM}/scANVI/results/*/metrics.json", "scANVI"),
        (f"{BM}/scANVI/results/*/*.json", "scANVI"),
    ]:
        for p in glob.glob(pattern):
            r = from_metrics_json(p)
            if not r:
                continue
            ds, seed, met = r
            mname = method
            if mname is None:  # simple baselines keep method inside json
                try:
                    with open(p) as f:
                        mname = json.load(f).get("method")
                except Exception:
                    pass
            if mname is None:
                mname = os.path.basename(os.path.dirname(os.path.dirname(p)))
            if ds is None or seed is None:
                m = NAME_RE.search(os.path.basename(os.path.dirname(p)))
                if m:
                    ds = ds or m.group(1)
                    seed = seed if seed is not None else int(m.group(2))
            rows.append({"method": mname, "dataset": ds, "seed": int(seed), **met})

    # de-duplicate (same method/dataset/seed can appear via log + json)
    seen = {}
    for r in rows:
        if r["dataset"] is None:
            continue
        seen[(r["method"], r["dataset"], r["seed"])] = r
    rows = list(seen.values())

    os.makedirs(f"{SC}/benchmark_results", exist_ok=True)
    csv_path = f"{SC}/benchmark_results/benchmark_results.csv"
    keys = ["method", "dataset", "seed", "ARI", "NMI", "AMI", "ACC"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in sorted(rows, key=lambda x: (x["dataset"], x["method"], x["seed"])):
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print("wrote", csv_path, f"({len(rows)} runs)")

    # mean ± std table
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("ARI", "NMI", "AMI", "ACC"):
            if k in r:
                agg[(r["dataset"], r["method"])][k].append(r[k])

    print("\n%-22s %-16s %5s  %s" % ("dataset", "method", "n", "ARI  NMI  AMI  ACC  (mean±std)"))
    for (ds, m), v in sorted(agg.items()):
        n = len(v["ARI"])
        cells = []
        for k in ("ARI", "NMI", "AMI", "ACC"):
            a = np.array(v[k]) if v[k] else None
            cells.append(f"{a.mean():.4f}±{a.std():.4f}" if a is not None and len(a) else "-")
        print("%-22s %-16s %5d  %s" % (ds, m, n, "  ".join(cells)))


if __name__ == "__main__":
    main()
