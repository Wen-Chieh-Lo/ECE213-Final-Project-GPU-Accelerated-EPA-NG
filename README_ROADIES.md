# MLIPPER For ROADIES

This document describes the current MLIPPER interface for the ROADIES
per-gene tree stage.

The intended use is simple:

1. ROADIES prepares one gene’s inputs.
2. ROADIES calls one wrapper script.
3. MLIPPER writes one committed gene tree.

## What ROADIES Should Use

For ROADIES, the intended entrypoint is:

- `scripts/run_single_gene_MLIPPER.sh`

The intended Docker image for that wrapper is:

- `wenchiehlo/mlipper-roadies:20260504`

ROADIES does not need to call the `MLIPPER` binary directly unless you want to
debug the wrapper.

This setup is currently validated on `peregrine`, where MLIPPER uses its own
bundled `libpll`. Please use this on `peregrine`.

## Architecture

The current architecture has four layers:

1. `ROADIES`
   ROADIES prepares one gene’s files and decides which GPU to use.

2. `run_single_gene_MLIPPER.sh`
   This is a thin wrapper. It validates input paths, maps host paths into the
   container, sets Docker GPU options, and translates the per-gene contract
   into one MLIPPER invocation.

3. `wenchiehlo/mlipper-roadies:20260504`
   This image provides the runtime environment and the compiled `MLIPPER`
   binary.

   It also bundles the vendored `lib/libpll-2_fp32` fork from this repo. For
   the current ROADIES setup, that fork is the intended `libpll`
   implementation; the image is not meant to swap in an upstream `libpll` at
   runtime.

4. `MLIPPER`
   The binary reads the reference MSA, query MSA, backbone tree, and bestModel
   file, then writes one committed gene tree.

In other words, ROADIES should think of MLIPPER as:

- one per-gene wrapper
- one per-gene output tree

not as a larger batch orchestration system.

## Input Contract

Per gene, MLIPPER expects these four core inputs:

- reference MSA: `ref.fa`
- query MSA: `query.fa`
- backbone tree: one `*.raxml.bestTree`
- per-gene model file: one `*.raxml.bestModel`

What each file means:

- `ref.fa`
  The reference or backbone alignment. These taxa are already present in the
  backbone tree.

- `query.fa`
  The query alignment. These taxa are the ones MLIPPER will commit back into
  the tree.

- `*.raxml.bestTree`
  The backbone topology for that gene.

- `*.raxml.bestModel`
  The per-gene model description. MLIPPER reads this file directly via
  `--best-model`.

Important limitation:

- MLIPPER expects split reference/query alignments.
- If ROADIES only has one combined full alignment, ROADIES needs an adapter
  step to split it before calling MLIPPER.

## Output Contract

The main output is:

- one committed final gene tree in Newick format

The current wrapper writes it to whatever path ROADIES passes as:

- `--out-tree`

The common filename used in this repo is:

- `mlipper_gene_tree.nwk`

ROADIES downstream should treat that tree as the main artifact from this
stage.

## Wrapper Interface

The wrapper takes these required arguments:

- `--ref-msa`
- `--query-msa`
- `--backbone-tree`
- `--best-model`
- `--out-tree`

Required argument meanings:

- `--ref-msa`
  Path to the reference/backbone alignment.

- `--query-msa`
  Path to the query alignment.

- `--backbone-tree`
  Path to the backbone Newick tree.

- `--best-model`
  Path to the per-gene `bestModel` file.

- `--out-tree`
  Path where the committed output tree should be written.

Optional wrapper arguments:

- `--docker-image`
- `--gpu-id`
- `--docker-gpus`
- `--local-spr`
- `--no-local-spr`
- `--batch-size`
- `--local-spr-radius`
- `--local-spr-rounds`

Optional argument meanings:

- `--docker-image`
  Override the Docker image tag. The wrapper default is
  `wenchiehlo/mlipper-roadies:20260504`.

- `--gpu-id`
  GPU id used when `--docker-gpus` is not provided.

- `--docker-gpus`
  Raw Docker `--gpus` specification. This overrides `--gpu-id`.

- `--local-spr`
  Enable local SPR refinement after query commitment.

- `--no-local-spr`
  Disable local SPR refinement.

- `--batch-size`
  Batch insert size passed to MLIPPER when local SPR is enabled.

- `--local-spr-radius`
  Local SPR radius.

- `--local-spr-rounds`
  Number of local SPR rounds.

## What The Wrapper Actually Runs

Internally, the wrapper runs Docker and then launches `MLIPPER` inside the
container.

At minimum, it forwards these MLIPPER arguments:

- `--tree-alignment`
- `--query-alignment`
- `--tree`
- `--best-model`
- `--commit-to-tree`

If local SPR is enabled, it also forwards:

- `--local-spr`
- `--batch-insert-size`
- `--local-spr-radius`
- `--local-spr-rounds`

The wrapper also does two operational things for ROADIES:

- it converts host paths into container paths
- it owns the Docker GPU selection

## GPU Control

GPU ownership should stay outside the MLIPPER binary.

The intended split is:

- ROADIES decides which GPU to use
- the wrapper translates that into Docker `--gpus`
- MLIPPER runs inside that already-restricted container

Recommended usage:

- one MLIPPER invocation sees one GPU
- ROADIES passes either `--gpu-id N` or `--docker-gpus ...`

MLIPPER itself does not expose a public `--gpu-id` flag.

## Model Handling

The preferred model path for ROADIES is:

- `--best-model`

Current helper-path support is:

- DNA only
- 4 states only
- `GTR` only

What `--best-model` currently overwrites:

- `--states`
- `--subst-model`
- `--ncat`
- `--alpha`
- `--rates`
- `--freqs` / `--empirical-freqs`

What it does not currently overwrite:

- `--pinv`
- `--rate-weights`

Current `pinv` note:

- keep `pinv = 0.0` unless nonzero invariant-site behavior has been explicitly
  revalidated end-to-end

## Empirical Frequencies

If the model implies empirical frequencies, MLIPPER estimates them from the
reference alignment.

Current behavior:

- partially informative ambiguity symbols are distributed across represented
  states
- fully uninformative symbols such as `N`, `-`, `.`, and `?` are ignored
- if any state would otherwise receive zero mass, MLIPPER applies a tiny
  positive floor before renormalization

## How To Use

### 1. Pull the image

```bash
docker pull wenchiehlo/mlipper-roadies:20260504
```

### 2. Run one gene

```bash
scripts/run_single_gene_MLIPPER.sh \
  --ref-msa GENE/ref.fa \
  --query-msa GENE/query.fa \
  --backbone-tree GENE/backbone.nwk \
  --best-model GENE/gene.raxml.bestModel \
  --out-tree GENE/mlipper_gene_tree.nwk \
  --gpu-id 0
```

### 3. Consume the output

ROADIES should use the tree written to:

- `GENE/mlipper_gene_tree.nwk`

or whatever path was passed as `--out-tree`.
Example:

```bash
scripts/run_single_gene_MLIPPER.sh \
  --ref-msa GENE/ref.fa \
  --query-msa GENE/query.fa \
  --backbone-tree GENE/backbone.nwk \
  --best-model GENE/gene.raxml.bestModel \
  --out-tree GENE/mlipper_gene_tree.nwk \
  --gpu-id 0
```

Expected success criteria:

- exit code `0`
- non-empty output Newick tree

## Current Limitations

- the helper path assumes split reference/query alignments
- the helper path assumes DNA with 4 states
- the helper path assumes DNA `GTR`
- amino-acid and non-GTR models are not supported in the current helper path
- `--best-model` does not currently import `pinv`; use `pinv = 0.0` unless that
  path is explicitly revalidated
- MLIPPER should currently be treated as one-visible-GPU per invocation; GPU
  scheduling must be done by ROADIES or the wrapper layer
- the current ROADIES image depends on the vendored `libpll-2_fp32` fork in
  this repo, not an upstream `libpll` release
- the current ROADIES setup is validated on `peregrine`; if ROADIES is moved
  to a different host environment, the image should be re-smoke-tested there
- the ROADIES image and wrapper are intended for per-gene execution
- the current interface guarantees the committed output tree
