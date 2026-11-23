# scAMUSE Benchmark Toolkit (Phase 5)

All baselines run with the **scAMUSE protocol**: seeds `11 42 79 239 528`,
5% labels (of TOTAL cells), identical stratified splits, test-set-only metrics
(ARI / NMI / AMI / ACC).

## Files

| file | goes to | purpose |
|---|---|---|
| `prepare_scamuse_benchmark_inputs.py` | `~/scAMUSE7/` | builds `splits/`, `baseline_inputs/`, `scgpt_inputs/`, `benchmark_config.json` |
| `run_all_baselines.sh` | `~/scAMUSE7/benchmark_toolkit/` | master runner, resume-safe |
| `run_simple_baselines.py` | `~/scAMUSE7/benchmark_toolkit/` | scGPT+KMeans, scGPT+Leiden, PCA+KMeans, Scanpy+Louvain |
| `collect_benchmark_results.py` | `~/scAMUSE7/benchmark_toolkit/` | aggregates every log → CSV + mean±std table |
| `scCCTR_patched.py` | `~/benchmark_methods/scCCTR/` | scCCTR with `--seed`, `--splits_npz`, h5 input, test-only metrics |

## Step 0 — install

```bash
mkdir -p ~/scAMUSE7/benchmark_toolkit
cd ~/scAMUSE7/benchmark_toolkit && unzip ~/scAMUSE_benchmark_toolkit.zip
cp scAMUSE_benchmark_toolkit/prepare_scamuse_benchmark_inputs.py ~/scAMUSE7/
cp scAMUSE_benchmark_toolkit/scCCTR_patched.py ~/benchmark_methods/scCCTR/
```

## Step 1 — generate inputs (once)

Use the same Python environment as `run_scAMUSE.py`:

```bash
cd ~/scAMUSE7
python prepare_scamuse_benchmark_inputs.py 2>&1 | tee logs/prepare_benchmark_inputs.log
```

- Safe to re-run: existing files are skipped.
- If scGPT import fails, first run with `--skip_scgpt`, fix, then re-run.
- Check `benchmark_config.json` afterwards: per-dataset cell counts / K.

## Step 2 — run baselines

Each method is independent; run them one at a time (CPU!). `nohup` the whole thing:

```bash
cd ~/scAMUSE7/benchmark_toolkit
nohup bash run_all_baselines.sh simple > run_simple.log 2>&1 &        # fast, do first
nohup bash run_all_baselines.sh scsemiaae > run_scsemiaae.log 2>&1 &
bash run_all_baselines.sh scagcl          # or nohup, after scANVI finishes
bash run_all_baselines.sh scgpt_scsemiaae
bash run_all_baselines.sh scgpt_scagcl
bash run_all_baselines.sh scgpt_sccctr
bash run_all_baselines.sh sccctr
```

Resume-safe: finished runs (log contains a final metrics line) are skipped.

For the simple baselines, `SIMPLE_PY` must point to a python with scanpy
(default: `~/scanvi_env/bin/python`). Override if needed:

```bash
SIMPLE_PY=~/scanvi_env/bin/python bash run_all_baselines.sh simple
```

`scanpy_louvain` needs the `louvain` package; if missing it falls back to
Leiden and says so in the log (`pip install louvain` to get true Louvain).

## Step 3 — collect

```bash
python collect_benchmark_results.py
# -> ~/scAMUSE7/benchmark_results/benchmark_results.csv
# -> prints mean±std per dataset × method (paste back to fill Sheet 7)
```

## Memory warning (big datasets)

scAGCL and scCCTR build O(n²) cell×cell matrices:

| dataset | cells | approx RAM for n×n float32 |
|---|---|---|
| Cortex | 22,360 | ~2 GB — OK |
| Innate_lymphocytes | 64,793 | ~17 GB — risky |
| Mucosal_Atlas | 86,725 | ~30 GB — likely OOM |

If they OOM, the run is marked FAILED in the log — that is itself a
reportable scalability result for the paper. Do not silently subsample;
if you want numbers for those cells, ask and we'll add a documented
subsampled variant.
