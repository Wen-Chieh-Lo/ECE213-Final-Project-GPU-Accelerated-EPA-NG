# MLIPPER

`MLIPPER` is a GPU-backed phylogenetic placement and commit tool. In this repo
the main practical workflow is:

- read a reference MSA
- read a query MSA
- read a backbone tree
- read per-gene model parameters
- commit the query taxa back into the tree
- optionally run local SPR refinement

For ROADIES-specific integration notes, use
[`README_ROADIES.md`](README_ROADIES.md).

## Repo Layout

Source code now lives under `src/`:

- `src/main.cpp`: CLI entrypoint
- `src/tree/`: tree build and GPU upload path
- `src/placement/`: placement and branch-length optimization
- `src/spr/`: local SPR refinement
- `src/likelihood/`: likelihood kernels
- `src/pmatrix/`: eigendecomposition and PMAT construction
- `src/io/`: Newick, jplace, parsing, and input validation helpers
- `src/util/`: shared utility headers
- `src/model_utils.*`: model parsing and frequency helpers
- `src/msa_preprocess.*`: alignment preprocessing helpers

Other important directories:

- `docker/`: Docker build definitions
- `scripts/`: ROADIES helpers and wrapper scripts
- `data/`: small tests plus bundle-style inputs

## Build

Local build:

```bash
make -j4 MLIPPER
./MLIPPER --help
```

Important current build identity:

- local `Makefile` default: double precision
- main Docker image build: float precision

Do not assume the host binary and container binary are identical unless you
verify that explicitly.

## Main CLI

The practical per-gene CLI contract is:

```bash
MLIPPER \
  --tree-alignment REF_MSA.fa \
  --query-alignment QUERY_MSA.fa \
  --tree BACKBONE_TREE.nwk \
  --best-model GENE.raxml.bestModel \
  --commit-to-tree OUT_TREE.nwk
```

Common refinement flags:

```bash
  --local-spr \
  --batch-insert-size 5 \
  --local-spr-radius 4 \
  --local-spr-rounds 1
```

You can also pass explicit model flags instead of `--best-model`, but in this
repo the preferred path is direct `bestModel` ingestion.

## Docker Images

This repo currently keeps two image targets.

### Full image

Build:

```bash
docker build -f docker/Dockerfile -t wenchiehlo/mlipper:20260504 .
```

Use this when you want:

- a larger development image
- the extra tooling bundled in the full container

### ROADIES-focused image

Build:

```bash
docker build -f docker/Dockerfile.roadies -t wenchiehlo/mlipper-roadies:20260504 .
```

Use this when you want:

- a smaller per-gene runtime image
- direct `MLIPPER` execution
- `scripts/run_single_gene_MLIPPER.sh`

Pull:

```bash
docker pull wenchiehlo/mlipper-roadies:20260504
```

## Scripts

### Per-gene wrapper

`scripts/run_single_gene_MLIPPER.sh` is the thin per-gene wrapper.

Example:

```bash
scripts/run_single_gene_MLIPPER.sh \
  --ref-msa data/iter_2_placement_legal_bundle_787_compat/gene_1/iter0_output_msa_from_ref.fa \
  --query-msa data/iter_2_placement_legal_bundle_787_compat/gene_1/iter0_output_msa_from_query.fa \
  --backbone-tree data/iter_2_placement_legal_bundle_787_compat/gene_1/gene_1_filtered.fa.aln.raxml.bestTree \
  --best-model data/iter_2_placement_legal_bundle_787_compat/gene_1/gene_1_filtered.fa.aln.raxml.bestModel \
  --out-tree output/wrapper_smoke/out.nwk \
  --gpu-id 0
```

## Model Handling

`--best-model` is supported and is the preferred interface for this repo’s
per-gene model path.

Current supported helper path:

- DNA only
- 4 states only
- `GTR` only

Current `--best-model` behavior:

- overwrites `--states`
- overwrites `--subst-model`
- overwrites `--ncat`
- overwrites `--alpha`
- overwrites `--rates`
- overwrites `--freqs` / `--empirical-freqs`

It does not currently overwrite:

- `--pinv`
- `--rate-weights`
- placement / output flags

Empirical frequencies are estimated from `--tree-alignment`.

Current implementation details:

- partially informative ambiguity codes are distributed across represented
  states
- `N`, `-`, `.`, and `?` are ignored for empirical frequency estimation
- a tiny positive floor is applied if a state would otherwise receive zero
  empirical mass

`pinv` note:

- keep `pinv = 0.0` unless you have explicitly revalidated the nonzero path

## Test Inputs

Simple test inputs exist in:

- `data/test/`
- `data/iter_2_placement_legal_bundle_787_compat/gene_1/`

The second path is the most convenient smoke-test gene because it already
contains:

- reference MSA
- query MSA
- backbone tree
- `bestModel`

## Current Limitations

- the helper path currently assumes split reference/query alignments
- the helper path currently assumes DNA `GTR`
- amino-acid and non-GTR models are not supported in the current helper path
- `MLIPPER` should be treated as one-visible-GPU per invocation
- the ROADIES-focused image and wrapper are intended for per-gene execution
