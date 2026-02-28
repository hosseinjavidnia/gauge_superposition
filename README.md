## Gauge Theory of Superposition

### Folder structure

```text
atlasA/
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_extract_acts.py
│   ├── 03_cluster_graph.py
│   ├── 04_local_bases.py
│   ├── 05_edge_transports.py
│   ├── 06_holonomy_gauge_demo.py
│   ├── 07_plot_holonomy.py
│   ├── 08_shearing_bound.py
│   ├── 09_plot_shearing_bound.py
│   ├── 10_collect_cluster_xgrad.py
│   ├── 11_compute_jamming_metrics.py
│   ├── 12_plot_jamming_C.py
│   ├── 13_audit_C.py
│   ├── 14_summarize_C_runs.py
│   ├── 15_cache_nn12.py
│   ├── 16_bootstrap_AB.py
│   ├── 17_samplecurve_shearing.py
│   └── 18_samplecurve_holonomy.py
├── data/
│   └── datasets/                 # generated JSONL datasets
│       ├── wikitext103.jsonl
│       ├── c4tiny.jsonl
│       └── stack_smol_sample.jsonl
├── runs/                         # generated artifacts
│   ├── acts/                     # activation memmaps + meta (script 02)
│   ├── graph/                    # clustering + kNN graph + NN caches (scripts 03/15)
│   ├── bases/                    # local bases (script 04)
│   ├── edges_base/               # output of script 05 for smin_thresh=0.0
│   │   ├── edge_ops.npz
│   │   ├── edge_index.json
│   │   └── edge_stats.jsonl
│   ├── edges_persist/            # output of script 05 for smin_thresh=0.015
│   │   ├── edge_ops.npz
│   │   ├── edge_index.json
│   │   └── edge_stats.jsonl
│   ├── C_samples/                # per-cluster (x, grad) samples for Result C
│   └── results/                  # plots + JSON/JSONL summaries
│       ├── holonomy_*.json
│       ├── *holonomy_defects*.npy
│       ├── holonomy_compare_*.pdf
│       ├── shearing_*.jsonl
│       ├── shearing_*_*.pdf
│       ├── jamming_*.jsonl
│       ├── jamming_C_*.pdf
│       ├── C_summary.csv
│       ├── D_bootstrap_*.json
│       └── D_curve_*.json
├── requirements.txt
└── README.md
```

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Hugging Face cache (recommended on clusters)
```bash
export HF_HOME=$PWD/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_HOME
```

### Gated datasets / models
Some datasets/models are gated on Hugging Face. Ensure you have access and are logged in:
```bash
huggingface-cli login
```

## End-to-end pipeline (baseline + persistent; Results A–D)
Commands below are the canonical sequence used in the A–D runs (baseline and persistent subsystems). Run from the repo root directory.

### 1) Download datasets (prose / web / code)
```bash
python3 scripts/01_download_data.py --n_wiki 20000 --n_c4 20000 --n_code 20000
```
Expected outputs: `data/datasets/wikitext103.jsonl`, `data/datasets/c4tiny.jsonl`, `data/datasets/stack_smol_sample.jsonl`

### 2) Extract activations from the frozen LLM
```bash
python3 scripts/02_extract_acts.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --inputs data/datasets/wikitext103.jsonl data/datasets/c4tiny.jsonl data/datasets/stack_smol_sample.jsonl \
  --label prose web code \
  --layer 16 --n_tokens 200000
```

### 3) Build the context graph (clusters + kNN)
```bash
python3 scripts/03_cluster_graph.py --n_clusters 128 --knn 6
```

### 4) Compute local chart bases (shared support dimension k=32)
```bash
python3 scripts/04_local_bases.py --k 32
```

### 5) Estimate edge transports (baseline subsystem; no conditioning filter)
```bash
rm -rf runs/edges_base && mkdir -p runs/edges_base
python3 scripts/05_edge_transports.py --k 32 --smin_thresh 0.0 --out runs/edges_base
```

### 6) Result A (baseline): holonomy + spanning-tree gauge identity checks
```bash
python3 scripts/06_holonomy_gauge_demo.py --k 32 \
  --edges_dir runs/edges_base \
  --out_json runs/results/holonomy_base.json \
  --save_arrays
```

### 7) Result B (baseline): compute shearing bound table
```bash
python3 scripts/08_shearing_bound.py \
  --edges_dir runs/edges_base \
  --out_jsonl runs/results/shearing_base.jsonl \
  --k 32
```

### 8) Result B (baseline): shearing bound plots
```bash
python3 scripts/09_plot_shearing_bound.py \
  --infile runs/results/shearing_base.jsonl \
  --outprefix runs/results/shearing_base \
  --k 32
```

### 9) Estimate edge transports (persistent subsystem; well-conditioned edges)
```bash
rm -rf runs/edges_persist && mkdir -p runs/edges_persist
python3 scripts/05_edge_transports.py --k 32 --smin_thresh 0.015 --out runs/edges_persist
```

### 10) Result A (persistent): holonomy + spanning-tree gauge identity checks
```bash
python3 scripts/06_holonomy_gauge_demo.py --k 32 \
  --edges_dir runs/edges_persist \
  --out_json runs/results/holonomy_persist.json \
  --save_arrays
```

### 11) Result B (persistent): compute shearing bound table + plots
```bash
python3 scripts/08_shearing_bound.py \
  --edges_dir runs/edges_persist \
  --out_jsonl runs/results/shearing_persist.jsonl \
  --k 32

python3 scripts/09_plot_shearing_bound.py \
  --infile runs/results/shearing_persist.jsonl \
  --outprefix runs/results/shearing_persist \
  --k 32
```

### 12) Result C + Result D: jamming metrics + stability

### 12.1) Result C: collect per-cluster samples (activations + gradients)
```bash
rm -rf runs/C_samples && mkdir -p runs/C_samples
python3 scripts/10_collect_cluster_xgrad.py \
  --model meta-llama/Llama-3.2-3B-Instruct --layer 16 \
  --inputs data/datasets/wikitext103.jsonl data/datasets/c4tiny.jsonl data/datasets/stack_smol_sample.jsonl \
  --per_cluster 512 --max_len 256 --every 8 \
  --out runs/C_samples
```

### 12.2) Result C: compute certified jamming metrics
```bash
python3 scripts/11_compute_jamming_metrics.py \
  --in_dir runs/C_samples \
  --out_jsonl runs/results/jamming_C_seed0.jsonl \
  --m 256 --alpha 1.0 --max_clusters 40
```

### 12.3) Result C: plotting + audits (optional but recommended)
```bash
python3 scripts/12_plot_jamming_C.py \
  --infile runs/results/jamming_C_seed0.jsonl \
  --outprefix runs/results/jamming_C

python3 scripts/13_audit_C.py \
  --infile runs/results/jamming_C_seed0.jsonl

python3 scripts/14_summarize_C_runs.py \
  --glob "runs/results/jamming_C_*.jsonl" \
  --out_csv runs/results/C_summary.csv
```

### 12.4) Result D (recommended prep): cache NN1/NN2 assignments
This speeds up bootstrap + sample-curve scripts that repeatedly form overlap pools.
```bash
python3 scripts/15_cache_nn12.py
```

### 12.5) Result D: global bootstrap stability (shearing + holonomy)
```bash
python3 scripts/16_bootstrap_AB.py \
  --edges_dir runs/edges_base \
  --out_json runs/results/D_bootstrap_AB_base.json \
  --B 200 --boot_n 2000 --n_edges 50 --n_cycles 25

python3 scripts/16_bootstrap_AB.py \
  --edges_dir runs/edges_persist \
  --out_json runs/results/D_bootstrap_AB_persist.json \
  --B 200 --boot_n 2000 --n_edges 50 --n_cycles 25
```

### 12.6) Result D: within-edge sample curve for shearing (example edge 3-33)
```bash
python3 scripts/17_samplecurve_shearing.py \
  --edge 3-33 \
  --sizes 256 512 1024 2000 4000 \
  --reps 200 \
  --out_json runs/results/D_curve_edge3-33.json
```

### 12.7) Result D: within-loop sample curve for holonomy (example chord 40-102)
```bash
python3 scripts/18_samplecurve_holonomy.py \
  --edges_dir runs/edges_base \
  --chord 40-102 \
  --sizes 256 512 1024 2000 4000 \
  --reps 200 \
  --out_json runs/results/D_curve_holonomy_40-102.json
```

### (Optional) Holonomy comparison plots (baseline vs persistent)
If you ran script 06 with `--save_arrays` for both subsystems, you can compare defect distributions:
```bash
python3 scripts/07_plot_holonomy.py \
  --base runs/results/holonomy_base_holonomy_defects.npy \
  --persist runs/results/holonomy_persist_holonomy_defects.npy \
  --k 32 \
  --outdir runs/results
```

## Script reference (what each script does + parameters)
This section documents intent and CLI parameters as used in your runs. If you later rename flags, update this table.

| Script | What it does | Inputs | Outputs | Key params |
|---|---|---|---|---|
| `scripts/01_download_data.py` | Downloads/creates JSONL datasets (WikiText-103 prose, C4-tiny web, Stack-smol code). | HF datasets access (may require `huggingface-cli login`). | `data/datasets/wikitext103.jsonl`, `data/datasets/c4tiny.jsonl`, `data/datasets/stack_smol_sample.jsonl` | `--out`, `--n_wiki`, `--n_c4`, `--n_code` |
| `scripts/02_extract_acts.py` | Extracts token activations at a chosen layer into memmaps, subsampling positions until `--n_tokens`. Labels samples by source (prose/web/code). | Model weights (`--model`), JSONL datasets (`--inputs`). | `runs/acts/acts.f16`, `runs/acts/labels.i32`, `runs/acts/src.i32`, `runs/acts/meta.json` | `--model`, `--inputs`, `--label`, `--layer`, `--n_tokens`, `--max_len`, `--stride`, `--every`, `--out` |
| `scripts/03_cluster_graph.py` | Clusters activations into chart vertices and builds a centroid kNN graph (context graph / 1-skeleton). | `runs/acts/*` from script 02. | `runs/graph/centroids.npy`, `runs/graph/assign.npy`, `runs/graph/edges.json` | `--n_clusters`, `--knn`, `--acts_dir`, `--out` |
| `scripts/04_local_bases.py` | Computes a local orthonormal basis `B_c ∈ R^{d×k}` per cluster (SVD/PCA) defining the k-dim support used by transports/holonomy. | Clustered activations + assignments (from script 03). | `runs/bases/bases.npy` | `--k`, `--max_per_cluster`, `--seed`, plus `--acts_dir`, `--graph_dir`, `--out` |
| `scripts/05_edge_transports.py` | For each graph edge `(u,v)`: forms overlap, estimates ridge transport + polar factor, logs conditioning, filters edges by `--smin_thresh`, and writes edge operators + JSONL stats. | Graph + bases + activations (from scripts 02–04). | `runs/edges_*/edge_ops.npz`, `runs/edges_*/edge_index.json`, `runs/edges_*/edge_stats.jsonl` | `--k`, `--lambda_ridge`, `--min_overlap`, `--max_overlap`, `--smin_thresh`, `--seed`, `--out` |
| `scripts/06_holonomy_gauge_demo.py` | Result A: spanning-tree gauge, tree/chord residuals, and fundamental-cycle holonomy defects; optionally saves arrays for plotting. | `runs/edges_*/edge_ops.npz` (script 05) + graph edges. | `runs/results/holonomy_*.json` and optional `*.npy` arrays (`--save_arrays`). | `--k`, `--edges_dir`, `--graph_dir`, `--out_json`, `--save_arrays` |
| `scripts/07_plot_holonomy.py` | Plots holonomy defect distributions (hist + ECDF) from arrays saved by script 06; useful for baseline vs persistent comparison. | `runs/results/*holonomy_defects*.npy` | `runs/results/holonomy_compare_hist.pdf`, `runs/results/holonomy_compare_ecdf.pdf` | `--base`, `--persist`, `--k`, `--outdir` |
| `scripts/08_shearing_bound.py` | Result B (compute): computes per-edge shearing quantities (`Δ`, bound `LB`, slack, `D_shear`, covariance eigenvalues) and writes a JSONL table used for plotting. | Edges + bases + activations (from scripts 02–05). | `runs/results/shearing_*.jsonl` | `--edges_dir`, `--out_jsonl`, `--k`, `--lambda_ridge`, `--min_overlap`, `--max_overlap`, `--lcc_only`, `--seed` |
| `scripts/09_plot_shearing_bound.py` | Result B (plot): reads `shearing_*.jsonl` and produces PDFs validating the bound (Δ vs LB, Δ vs `D_shear`, slack histogram). | `runs/results/shearing_*.jsonl` | `runs/results/shearing_*_*.pdf` | `--infile`, `--outprefix`, `--k`, `--min_lam` |
| `scripts/10_collect_cluster_xgrad.py` | Result C (data collection): collects per-cluster samples of (activation `x`, gradient `∂L/∂x`) for next-token NLL, saved per cluster. | Frozen model + datasets + `runs/graph/centroids.npy`. | `runs/C_samples/cluster_*.npz` (+ `meta.json`). | `--model`, `--layer`, `--inputs`, `--per_cluster`, `--max_len`, `--every`, `--graph_dir`, `--out`, `--seed` |
| `scripts/11_compute_jamming_metrics.py` | Result C (metrics + certification): fits a local dictionary per cluster, computes jamming index `J` and energy/projection metrics, outputs per-cluster JSONL. | `runs/C_samples/` (script 10). | `runs/results/jamming_*.jsonl` | `--in_dir`, `--out_jsonl`, `--m`, `--alpha`, `--tau`, `--eta_p`, `--max_clusters`, `--seed` |
| `scripts/12_plot_jamming_C.py` | Plots Result C summaries from jamming JSONL (e.g. certified energy vs bound, energy vs `J`). | `runs/results/jamming_*.jsonl` | `runs/results/jamming_C_*.pdf` | `--infile`, `--outprefix` |
| `scripts/13_audit_C.py` | Audits Result C numerics: counts certified clusters, reports slack stats, flags violations under tolerance. | `runs/results/jamming_*.jsonl` | Console audit summary. | `--infile`, `--tol` |
| `scripts/14_summarize_C_runs.py` | Aggregates multiple Result C runs (`jamming_C_*.jsonl`) into a single CSV summary. | Many JSONLs matched by `--glob`. | `runs/results/C_summary.csv` (or `--out_csv`). | `--glob`, `--out_csv`, `--tol` |
| `scripts/15_cache_nn12.py` | Precomputes nearest and second-nearest centroid indices for every activation (NN1/NN2 cache). | Activations + centroids (`runs/acts/*`, `runs/graph/centroids.npy`). | `runs/graph/nn1.npy`, `runs/graph/nn2.npy` | `--acts_dir`, `--graph_dir` |
| `scripts/16_bootstrap_AB.py` | Result D global stability: bootstraps random edges/cycles to summarize stability of `D_shear` and `D_hol`. | `runs/edges_*/` + graph + bases + NN caches. | `runs/results/D_bootstrap_*.json` | `--edges_dir`, `--out_json`, `--B`, `--boot_n`, `--n_edges`, `--n_cycles`, `--k`, `--lambda_ridge`, `--max_overlap`, `--seed` |
| `scripts/17_samplecurve_shearing.py` | Result D within-edge curve: for one edge `u-v`, bootstraps at multiple sample sizes and records `D_shear` vs `n`. | Activations + bases + NN caches. | `runs/results/D_curve_edge*.json` | `--edge`, `--sizes`, `--reps`, `--out_json`, `--k`, `--lambda_ridge`, `--max_overlap`, `--seed` |
| `scripts/18_samplecurve_holonomy.py` | Result D within-loop curve: for one chord `u-v`, bootstraps the induced fundamental cycle and records `D_hol` vs `n`. | `runs/edges_*/` + graph + bases + NN caches. | `runs/results/D_curve_holonomy*.json` | `--edges_dir`, `--chord`, `--sizes`, `--reps`, `--out_json`, `--k`, `--lambda_ridge`, `--max_overlap`, `--seed` |

## Outputs (where files appear, what they mean)

### Activation directory: `runs/acts/`
* Activation memmaps and metadata produced by script 02 (e.g. `acts.f16`, `labels.i32`, `src.i32`, `meta.json`).

### Graph directory: `runs/graph/`
* `centroids.npy`, `assign.npy`, `edges.json`: produced by script 03.
* `nn1.npy`, `nn2.npy`: produced by script 15 (optional but recommended for Result D).

### Bases directory: `runs/bases/`
* `bases.npy`: produced by script 04; per-cluster orthonormal bases.

### Edge directories: `runs/edges_base`, `runs/edges_persist`
* `edge_ops.npz`: operators needed for holonomy demo and bootstrap.
* `edge_index.json`: edge list/order used by downstream scripts.
* `edge_stats.jsonl`: per-edge conditioning stats (e.g. smallest singular value).

### Results directory: `runs/results/`
* `holonomy_*.json`: summaries from script 06 (components/LCC, residuals, holonomy stats).
* `*holonomy_defects*.npy`: arrays saved by script 06 when `--save_arrays`.
* `holonomy_compare_*.pdf`: comparisons from script 07.
* `shearing_*.jsonl`: per-edge shearing tables from script 08.
* `shearing_*_*.pdf`: bound-validation plots from script 09.
* `jamming_*.jsonl`: per-cluster jamming/certificate metrics from script 11.
* `jamming_C_*.pdf`: plots from script 12.
* `C_summary.csv`: aggregated Result C summary from script 14.
* `D_bootstrap_*.json`: bootstrap stability summaries from script 16.
* `D_curve_*.json`: within-edge / within-loop concentration curves (scripts 17/18).

## Troubleshooting (common issues)

### Missing dataset files
If a file like `data/datasets/c4tiny.jsonl` is missing, re-run Step 1 and confirm the output directory exists.

### Gated dataset/model error (HF access)
If you see “Dataset is gated” or a 401/403, ensure you have access and run:
```bash
huggingface-cli login
```

### Graph becomes tree-like under persistence filtering
If script 06 reports “No chord edges in LCC (LCC is a tree)”, your `--smin_thresh` is too high. Reduce it (e.g. 0.015 → 0.01) or increase graph density (`--knn`) so cycles survive.

### Bootstrap/sample-curve scripts are slow
Run script 15 once to cache NN1/NN2:
```bash
python3 scripts/15_cache_nn12.py
```
