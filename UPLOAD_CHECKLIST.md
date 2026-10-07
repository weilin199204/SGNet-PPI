# Public Release Checklist

Use this list immediately before uploading the paper code.

## Required author decisions

- [ ] Add an explicit `LICENSE` selected by the authors/institution.
- [ ] Add `CITATION.cff` with the final title, author ORCIDs, venue, year, and DOI.
- [x] Use `SGNet` consistently as the repository/software name and `SGNet-SR` as the final model name.
- [ ] Confirm whether the corrected chain workbook and derived affinity labels
      may be redistributed under the source-dataset terms.
- [ ] Add provider links and formal citations for SSIF-Affinity, PDBbind v2020,
      Affinity Benchmark v5.5, PPB-Affinity/SKEMPI, ESM2, MaSIF, MSMS, APBS,
      Reduce, PyTorch, and PyTorch Geometric.

## Reproducibility metadata

- [ ] Record the exact MaSIF commit.
- [ ] Record MSMS, APBS, Reduce, CUDA driver, GPU, operating system, compiler,
      PyTorch, PyG, Transformers, and torch-scatter versions.
- [ ] Record whether APBS/MSMS fallback markers occurred in the frozen run.
- [ ] Preserve seed 20, OOF fold assignments, the real/control initialization
      hash, and the two released checkpoint SHA256 values.
- [ ] Confirm the released feature schema is
      `protocol/results/interface_patch_full_metadata.json`.

## Automated checks

Run from the repository root:

```bash
python -m compileall -q sgnet sgnet_graph scripts
python scripts/smoke_test.py
python scripts/analyze_results.py \
  --snapshot-root results_snapshot \
  --output-dir reproduced_results \
  --seed 20 \
  --sample-bootstrap 50000 \
  --cluster-bootstrap 20000
rg -n '/home/|/Users/|[A-Za-z]:\\\\' \
  --glob '*.py' --glob '*.md' --glob '*.json' --glob '*.sh' .
sha256sum -c MANIFEST.sha256
```

The path scan may find no author-machine paths. Portable manifest tokens such
as `PDBBIND_ROOT` are expected.

## Protocol checks

- [ ] Split sizes are exactly 2350/80/79/81.
- [ ] Early 77-sample test1 results are absent.
- [ ] No random 8:2 or homology-CV result is described as the paper result.
- [ ] Test1 and test2 are reported separately.
- [ ] Shuffled-control results are present, including the stronger validation
      point estimate.
- [ ] S166 is described as 166 records from 26 base complexes.
- [ ] S166 uncertainty uses base-complex cluster bootstrap.
- [ ] The `1YCS` overlap and exclusion sensitivity are disclosed.
- [ ] Mutants are described as mutation-aware ESM plus shared WT surface.

## Archive

- [ ] Remove `__pycache__`, local runs, logs, generated PLY, packed pickle,
      and raw structures.
- [ ] Create a version tag corresponding to the submitted manuscript.
- [ ] Archive that tag in Zenodo or another long-term repository.
- [ ] Put the release URL/DOI in the manuscript Code Availability section.
- [ ] Put raw-data provider/access instructions in Data Availability.

