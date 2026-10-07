# SGNet-SR

**SGNet-SR** (internally called Surface Residual V2) is the final model in the manuscript.
It predicts protein-protein binding affinity
by adding a reliability-gated molecular-surface correction to a frozen ESM2
sequence baseline:

```text
prediction = sequence_prediction + reliability_gate * surface_residual
```

The surface encoder consumes only cross-partner edges from an
`interface_patch_full v5` graph. The training target is the sequence model's
five-fold out-of-fold (OOF) residual, not an in-sample residual.

This repository is a paper-reproducibility release. It includes the model and
training code, portable protocol metadata, feature-building code, locked
checkpoints, per-sample predictions, and statistical analysis. It does not
redistribute licensed protein structures or multi-gigabyte generated tensors.

A Chinese guide is available in [USAGE_ZH.md](USAGE_ZH.md).

## Scientific scope

The fixed SSIF-Affinity protocol contains:

| Split | Complexes | Role |
|---|---:|---|
| train | 2,350 | optimization and OOF residual generation |
| validation | 80 | checkpoint/epoch selection only |
| test1 | 79 | external Affinity Benchmark v5.5 evaluation |
| test2 | 81 | held-out PDBbind v2020 evaluation |
| S166 WT | 26 records / 26 bases | supportive SKEMPI evaluation |
| S166 mutant | 140 records / 16 bases | mutation-aware ESM with shared WT surfaces |

The reported evidence is intentionally mixed: SGNet-SR improved Pearson
correlation on test1, but significantly worsened MAE on test2. S166 point
estimates were favorable overall, while base-complex cluster intervals crossed
zero. The shuffled-target control was not significantly separated from the
real arm in the external evaluations. The code and snapshots retain these
negative results.

## Repository layout

```text
sgnet/
  data.py                  data validation and train-only imputation
  sequence.py              frozen ESM2 sequence predictor
  model.py                 cross-edge encoder and reliability gate
sgnet_graph/
  interface_patch.py       v5 patch nodes, features, and cross edges
  surface.py               PLY parsing
  structure.py             residue/chain parsing
scripts/
  build_sequence_artifacts.py
  build_surfaces.py
  audit_surfaces.py
  build_interface_patch_graphs.py
  pack_fixed_splits.py
  train_sequence.py
  train_sgnet.py
  evaluate_s166.py
  analyze_results.py
  smoke_test.py
protocol/
  splits/                  verbatim public SSIF-Affinity lists
  metadata/                frozen, audited manifests with portable paths
  results/                 frozen tables and v5 feature schema
model/                     two deployable seed-20 checkpoints
results_snapshot/          OOF and per-sample locked predictions
```

## Quick verification

The reference environment used Python 3.8.20, PyTorch 2.2.0, CUDA 11.8,
NumPy 1.24.4, SciPy 1.10.1, scikit-learn 1.3.2, Transformers 4.38.0,
PyG 2.6.1, and torch-scatter 2.1.2.

Create an environment and install a PyTorch/torch-scatter build compatible
with the local CUDA driver:

```bash
conda create -n sgnet python=3.8 -y
conda activate sgnet

# Example for CUDA 11.8. Use the official PyTorch/PyG wheel selectors when
# the local CUDA version differs.
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install -r requirements.txt
```

Run the architecture and checkpoint test from the repository root:

```bash
python scripts/smoke_test.py
```

It checks all of the following on CPU:

- the untrained residual model is exactly equal to the sequence baseline;
- perturbing intra-partner edge attributes cannot change the encoder output;
- the sequence and real-surface checkpoints load strictly;
- both checkpoints produce finite outputs with the expected shapes.

## Reproduce reported metrics

No raw structures, ESM model download, surface software, or GPU is needed to
recompute metrics from the locked predictions:

```bash
python scripts/analyze_results.py \
  --snapshot-root results_snapshot \
  --output-dir reproduced_results \
  --seed 20 \
  --sample-bootstrap 50000 \
  --cluster-bootstrap 20000
```

Outputs:

```text
reproduced_results/main_metrics.csv
reproduced_results/statistical_checks.csv
reproduced_results/analysis_protocol.json
```

Validation/test1/test2 use paired sample bootstrap. S166 resamples the 26
`base_pdb_id` clusters rather than treating 166 WT/mutant records as
independent. The script also reports the sensitivity analysis excluding
`1YCS`, which overlaps the training PDB list.

Monte Carlo interval endpoints can differ slightly from the archived table in
`protocol/results/statistical_checks.csv`; point metrics must match apart
from CSV rounding.

## Packed training format

Training consumes four pickle files:

```text
train_dataset_save.pkl   2350 complexes
val_dataset_save.pkl       80 complexes
test1_dataset_save.pkl     79 complexes
test2_dataset_save.pkl     81 complexes
```

Each file stores a list of dictionaries:

| Field | Required content |
|---|---|
| `pdb_id` | unique complex ID |
| `sequence_a.x` | partner-A ESM2 tensor, shape `[L_a, 1280]` |
| `sequence_b.x` | partner-B ESM2 tensor, shape `[L_b, 1280]` |
| `target.y` | scalar `pK = -log10(K [M])` |
| `patch` | `interface_patch_full v5` PyG graph |

The v5 patch graph contains 23 node features and 42 edge features. Its required
attributes are `x`, `edge_index`, `edge_attr`, `edge_type`,
`patch_area`, `chain_id`, `residue_ids`, `residue_weights`,
`residue_counts`, `patch_graph_version`, and `surface_available`.
The exact feature order and thresholds are frozen in
`protocol/results/interface_patch_full_metadata.json`.

## Train from packed data

Train the sequence baseline:

```bash
python scripts/train_sequence.py \
  --data-root /path/to/packed_data \
  --output-dir runs/sequence \
  --device cuda:0
```

Generate train-only OOF predictions and train both the real and
shuffled-target surface arms:

```bash
python scripts/train_sgnet.py \
  --data-root /path/to/packed_data \
  --sequence-checkpoint runs/sequence/best_sequence.pt \
  --output-dir runs/sgnet \
  --device cuda:0 \
  --target-mode both
```

Reference defaults are seed 20, batch size 4, gradient accumulation 8,
five OOF folds, 30 fixed OOF epochs taken from the sequence checkpoint,
100 maximum surface epochs, validation-MSE early stopping, AdamW
learning rate `1e-4`, and no scheduler.

To reuse the archived OOF values instead of rerunning five sequence fits:

```bash
python scripts/train_sgnet.py \
  --data-root /path/to/packed_data \
  --sequence-checkpoint model/best_sequence.pt \
  --output-dir runs/sgnet \
  --device cuda:0 \
  --reuse-oof results_snapshot/oof_sequence_predictions.csv \
  --target-mode both
```

Do not use test1, test2, or S166 to choose epochs or tune hyperparameters.

## Full preprocessing

### External inputs

Obtain these inputs from their original providers and comply with their
licenses:

1. PDBbind v2020 protein-protein structures and
   `INDEX_general_PP.2020`.
2. Affinity Benchmark v5.5 structures and `ABv3.xlsx`.
3. PPB-Affinity/SKEMPI metadata and SKEMPI v2.0 structures for S166.
4. The corrected chain table used to audit partner assignments.
5. MaSIF, MSMS 2.6.1, APBS, Reduce, and MaSIF's mesh dependencies.
6. Hugging Face model `facebook/esm2_t33_650M_UR50D`.

Raw PDB files, PLY meshes, ESM tensors, packed pickle files, and MaSIF binaries
are not included.

### 1. Resolve the frozen manifest

Published manifests use tokens instead of author-machine paths. Resolve them
against local data roots:

```bash
python scripts/resolve_manifest_paths.py \
  --manifest protocol/metadata/all.tsv \
  --pdbbind-dir "/data/PDBbind v2020" \
  --affinity-benchmark-dir "/data/Affinity Benchmark v5.5" \
  --replacement-dir "/data/replacement_structures" \
  --out work/manifests/all.tsv \
  --check

python scripts/audit_manifest.py \
  --manifest work/manifests/all.tsv \
  --report work/manifests/all.audit.json
```

The replacement directory must contain `5BXQ.pdb` and `7SQ2.pdb`. These
structures retain labels from `1A2K` and `3OHM`, respectively.

To reconstruct rather than reuse the frozen manifest, run:

```bash
python scripts/build_manifest.py \
  --corrected-xlsx /data/PPI_dataset_2283.xlsx \
  --ppb-csv /data/PPB-Affinity.csv \
  --pdbbind-dir "/data/PDBbind v2020" \
  --pdbbind-index "/data/PDBbind v2020/index/INDEX_general_PP.2020" \
  --affinity-benchmark-dir "/data/Affinity Benchmark v5.5" \
  --affinity-benchmark-xlsx "/data/Affinity Benchmark v5.5/ABv3.xlsx" \
  --replacement-dir /data/replacement_structures \
  --out-dir work/manifests
```

### 2. Prepare structures and ESM2 tensors

```bash
python scripts/prepare_structures.py \
  --manifest work/manifests/all.tsv \
  --out-dir work/raw_pdb \
  --copy

python scripts/build_sequence_artifacts.py \
  --metadata work/manifests/all.tsv \
  --pdb-dir work/raw_pdb \
  --out-dir work/sequence_graphs \
  --device cuda:0 \
  --num-shards 1 \
  --shard-index 0
```

For long chains, ESM2 uses 1,000-residue windows with 100-residue overlap and
averages overlapping representations. Sequence graph edges are deliberately
empty because the final sequence model uses only residue embeddings.

### 3. Generate molecular surfaces

```bash
python scripts/build_surfaces.py \
  --manifest work/manifests/all.tsv \
  --raw-pdb-dir work/raw_pdb \
  --out-dir work/surfaces \
  --masif-root /opt/masif \
  --num-shards 1 \
  --shard-index 0 \
  --strict

python scripts/audit_surfaces.py \
  --manifest work/manifests/all.tsv \
  --surface-root work/surfaces \
  --out work/surfaces/surface_index.tsv \
  --allow-missing 3
```

The wrapper restricts each staged structure to the two manifest partner
groups, uses the first MODEL, creates an independent scratch directory, records
APBS fallback, and retries MSMS across the frozen probe/density schedule.
`--allow-missing 3` reproduces the paper protocol; only training complexes
`5YWO`, `2DSQ`, and `5KVE` may be represented by marked placeholders.

### 4. Build v5 interface patches

```bash
python scripts/build_interface_patch_graphs.py \
  --surface-dir work/surfaces/complete_ply \
  --surface-index work/surfaces/surface_index.tsv \
  --atom-root work/sequence_graphs \
  --out-root work/patch_graphs \
  --graph-name interface_patch_full \
  --workers 4 \
  --strict
```

The expected graph directory is
`work/patch_graphs/surfgraph_4/interface_patch_full`.

### 5. Pack fixed splits

```bash
python scripts/pack_fixed_splits.py \
  --manifest work/manifests/all.tsv \
  --graph-root work/sequence_graphs \
  --patch-root work/patch_graphs/surfgraph_4/interface_patch_full \
  --out-dir work/training_pkls
```

Packing validates exact split counts, labels, ESM/residue alignment, graph
version, node/edge dimensions, and cross-edge availability. Missing-surface
values are materialized later from training-only statistics.

## S166 preparation and evaluation

Create a local S166 manifest, mutation-aware ESM tensors, shared WT patches,
and packed data:

```bash
python scripts/build_skempi_manifest.py \
  --ppb-csv /data/PPB-Affinity.csv \
  --pdb-dir "/data/SKEMPI v2.0" \
  --out work/skempi/skempi_s166.tsv

python scripts/build_skempi_sequence_artifacts.py \
  --manifest work/skempi/skempi_s166.tsv \
  --pdb-dir "/data/SKEMPI v2.0" \
  --out-dir work/skempi/sequence_graphs \
  --atom-root work/skempi/wt_atom_graphs \
  --device cuda:0
```

Build WT surfaces and v5 patches with the same surface/patch commands, using the
26 base PDB IDs. Then pack and evaluate:

```bash
python scripts/pack_skempi_s166.py \
  --manifest work/skempi/skempi_s166.tsv \
  --graph-root work/skempi/sequence_graphs \
  --patch-root work/skempi/patch_graphs/surfgraph_4/interface_patch_full \
  --out work/skempi/skempi_s166_dataset.pkl

python scripts/evaluate_s166.py \
  --data work/skempi/skempi_s166_dataset.pkl \
  --run-dir model \
  --sequence-checkpoint model/best_sequence.pt \
  --device cuda:0
```

This command evaluates the released real arm. Add `--include-shuffled` only
after generating or supplying a shuffled-control checkpoint.

The 140 mutants use mutation-aware ESM2 sequences but share the experimental
WT coordinates and surface of their base complex. S166 must therefore not be
described as mutation-specific surface modeling. It also contains the known
training overlap `1YCS`.

## Released checkpoints

| File | Purpose |
|---|---|
| `model/best_sequence.pt` | locked sequence baseline, best epoch 30 |
| `model/real/best_surface_residual_v2.pt` | real OOF residual arm, best epoch 59 |

The shuffled-control checkpoint is omitted from this deployment-focused release.
Its locked per-sample predictions remain in `results_snapshot/shuffled/` for
paper-result verification. The OOF fold checkpoints are training intermediates and are unnecessary for
inference. Their per-sample predictions and fold summary are retained instead.

## Reproducibility invariants

- Public split lists are never randomized.
- OOF cross-fitting occurs only inside the 2,350-sample training split.
- Surface normalization and missing-surface imputation use training data only.
- Only `edge_type == 1` enters SGNet-SR.
- The residual head is zero-initialized, making epoch 0 exactly the baseline.
- Real and shuffled arms start from the same parameter-state hash.
- Validation selects checkpoints; test1, test2, and S166 are locked evaluation
  sets.
- S166 bootstrap uses `base_pdb_id`, not individual mutation records.

## Before public upload

This directory deliberately does not choose a software license on the authors'
behalf. Add an explicit `LICENSE` file before making the repository public.
Also add the final article citation/DOI, archive a versioned release, and record
the MaSIF/MSMS/APBS/Reduce versions used on the publication run. See
[UPLOAD_CHECKLIST.md](UPLOAD_CHECKLIST.md).

## Data and model limitations

Inference requires ESM2 tensors and a bound-structure-derived molecular
surface patch. The surface workflow depends on external MaSIF/MSMS/APBS tools.
Three training surfaces were imputed; no validation/test1/test2 surface was
imputed. Validation has only 80 complexes, only one full seed-20 SGNet-SR run
is frozen here, and performance is distribution dependent.

