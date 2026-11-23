#!/usr/bin/env python3
import os, glob, re
import numpy as np
from collections import defaultdict

def parse_log_time(log_file):
    """Extract wall clock time from log file."""
    try:
        with open(log_file) as f:
            content = f.read()
    except:
        return None

    # Pattern 1: "Wall clock time: 0.53 min (31.93 sec)"
    m = re.search(r'Wall clock time:\s+[\d.]+ min \(([\d.]+) sec\)', content)
    if m:
        return float(m.group(1))

    # Pattern 2: "Wall clock time: X sec" (no min)
    m = re.search(r'Wall clock time:\s+([\d.]+)\s*sec', content, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Pattern 3: "Total time: X seconds"
    m = re.search(r'Total time:\s+([\d.]+)\s*sec', content, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None

# Check Phase 1 logs specifically
print("=== Checking Phase 1 logs ===")
for ds in ["Brain","Fat","Retina","Breast_carcinoma","Quadriceps_tendon","Cortex","Mucosal_Atlas","Innate_lymphocytes"]:
    times = []
    for s in [11, 42, 79, 239, 528]:
        # Try different log name patterns
        for pattern in [f"logs/P1_{ds}__s{s}.log", f"logs/Phase1_{ds}__s{s}.log", f"logs/{ds}__s{s}.log"]:
            if os.path.exists(pattern):
                t = parse_log_time(pattern)
                if t:
                    times.append(t)
                break

    if times:
        a = np.array(times)
        print(f"{ds:<20} {a.mean():.1f}±{a.std():.1f}s (n={len(times)})")
    else:
        # Try any log file for this dataset
        all_logs = glob.glob(f"logs/*{ds}*.log")
        for logf in all_logs[:3]:
            t = parse_log_time(logf)
            if t:
                print(f"{ds:<20} {t:.1f}s (from {logf})")
                break
        else:
            print(f"{ds:<20} NO TIME FOUND")

# If no Phase 1 logs have runtime, try to estimate from file timestamps
print("\n=== Fallback: file timestamps ===")
for ds in ["Brain","Fat","Retina","Breast_carcinoma","Quadriceps_tendon","Cortex","Mucosal_Atlas","Innate_lymphocytes"]:
    for s in [11]:
        rj = f"result/Phase1/{ds}/full__s{s}/results.json"
        if os.path.exists(rj):
            mtime = os.path.getmtime(rj)
            # Find associated log
            log_files = glob.glob(f"logs/*{ds}*s{s}*.log")
            if log_files:
                log_mtime = os.path.getmtime(log_files[0])
                duration = abs(mtime - log_mtime)
                print(f"{ds:<20} ~{duration:.0f}s (file timestamp diff)")
                break