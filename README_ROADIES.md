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

- `wenchiehlo/mlipper-roadies:20260507`

ROADIES does not need to call the `MLIPPER` binary directly unless you want to
debug the wrapper.

There are two supported execution modes:

- Docker mode
  This is the default. ROADIES runs `scripts/run_single_gene_MLIPPER.sh`, the
  wrapper starts the Docker image, and the MLIPPER binary comes from inside the
  image.

- Host mode
  ROADIES passes `--no-docker`. The wrapper runs a local `MLIPPER` binary from
  the host and uses `install/setup_host.sh` to install/check host dependencies
  when needed.

This setup is currently validated on `peregrine`, where MLIPPER links against
the distro-packaged `libpll`. Please use this on `peregrine`.

## Architecture

The Docker-mode architecture has four layers:

1. `ROADIES`
   ROADIES prepares one gene’s files and decides which GPU to use.

2. `run_single_gene_MLIPPER.sh`
   This is a thin wrapper. It validates input paths, maps host paths into the
   container, sets Docker GPU options, and translates the per-gene contract
   into one MLIPPER invocation.

3. `wenchiehlo/mlipper-roadies:20260507`
   This image provides the runtime environment and the compiled `MLIPPER`
   binary.

4. `MLIPPER`
   The binary reads the reference MSA, query MSA, backbone tree, and bestModel
   file, then writes one committed gene tree.

In other words, ROADIES should think of MLIPPER as:

- one per-gene wrapper
- one per-gene output tree

not as a larger batch orchestration system.

The host-mode architecture replaces the Docker image layer with:

- `install/setup_host.sh`
  Installs host build/runtime dependencies and builds `MLIPPER` locally.

- host `MLIPPER`
  The wrapper runs the local binary directly. By default this is
  `REPO_ROOT/MLIPPER`, or the path passed via `--local-mlipper`.

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
- In Docker mode, all per-gene input files and the output path should share a
  reasonably small common parent directory. The wrapper mounts that common
  parent into the container as `/workspace/job`.
- Recommended layout: keep `ref.fa`, `query.fa`, the backbone tree, the
  `bestModel`, and the output tree under the same per-gene or per-job directory.
- Avoid spreading inputs and outputs across unrelated filesystem roots, because
  that can force the wrapper to mount an overly broad parent such as `/`.

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
- `--no-docker`
- `--local-mlipper`
- `--local-spr`
- `--no-local-spr`
- `--batch-size`
- `--local-spr-radius`
- `--local-spr-rounds`

Optional argument meanings:

- `--docker-image`
  Override the Docker image tag. The wrapper default is
  `wenchiehlo/mlipper-roadies:20260507`.

- `--gpu-id`
  GPU id used when `--docker-gpus` is not provided.

- `--docker-gpus`
  Raw Docker `--gpus` specification. This overrides `--gpu-id`.

- `--no-docker`
  Run host `MLIPPER` directly instead of Docker.

- `--local-mlipper`
  Override the local `MLIPPER` binary path used with `--no-docker`. The default
  is `REPO_ROOT/MLIPPER`.

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

In Docker mode, the wrapper runs Docker and launches image-internal `MLIPPER` at:

- `/workspace/MLIPPER/MLIPPER`

The wrapper mounts the input/output path root into `/workspace/job`. It does not
mount the host repo over `/workspace/MLIPPER`, so the binary comes from the
Docker image.

In host mode, the wrapper runs the local binary directly:

- default: `REPO_ROOT/MLIPPER`
- override: `--local-mlipper PATH`

Before host execution, the wrapper checks whether host `libpll` headers and
libraries are available. If they are missing, it calls:

- `install/setup_host.sh --skip-mlipper`

If the local binary is missing, it calls:

- `install/setup_host.sh`

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

In Docker mode, the wrapper also does two operational things for ROADIES:

- it converts host paths into container paths
- it owns the Docker GPU selection

## GPU Control

GPU ownership should stay outside the MLIPPER binary.

The intended split is:

- ROADIES decides which GPU to use
- in Docker mode, the wrapper translates that into Docker `--gpus`
- in host mode, ROADIES should restrict visible GPUs before calling the wrapper,
  for example with `CUDA_VISIBLE_DEVICES`
- MLIPPER runs inside the already-restricted GPU environment

Recommended usage:

- one MLIPPER invocation sees one GPU
- Docker mode: ROADIES passes either `--gpu-id N` or `--docker-gpus ...`
- host mode: ROADIES sets `CUDA_VISIBLE_DEVICES=N` and passes `--no-docker`

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

### Docker mode

This is the default mode. Use it when ROADIES should run the binary packaged in
the Docker image.

Example per-gene job layout:

```text
GENE/
  ref.fa
  query.fa
  backbone.nwk
  gene.raxml.bestModel
  mlipper_gene_tree.nwk        # written by MLIPPER
```

With that layout, ROADIES calls:

```bash
scripts/run_single_gene_MLIPPER.sh \
  --ref-msa GENE/ref.fa \
  --query-msa GENE/query.fa \
  --backbone-tree GENE/backbone.nwk \
  --best-model GENE/gene.raxml.bestModel \
  --out-tree GENE/mlipper_gene_tree.nwk \
  --gpu-id 0
```

Internally, the wrapper mounts `GENE/` into the container under
`/workspace/job/...` and rewrites the file paths for MLIPPER. ROADIES should
only pass host paths like `GENE/ref.fa`; it should not pass `/workspace/job`
paths directly.

#### 1. Pull or build the image

```bash
docker pull wenchiehlo/mlipper-roadies:20260507
```

Or build it from this repo:

```bash
docker build -f docker/Dockerfile.roadies -t wenchiehlo/mlipper-roadies:20260507 .
```

During image build, Docker does the setup work:

- builder stage installs compile-time dependencies such as `build-essential`,
  `libpll-dev`, `libblas-dev`, `liblapack-dev`, and `libtbb-dev`
- builder stage runs `make clean && make USE_DOUBLE=1 MLIPPER`
- runtime stage installs runtime libraries such as `libpll0`, `libblas3`,
  `liblapack3`, and `libtbb12`
- runtime stage copies the compiled binary into `/workspace/MLIPPER/MLIPPER`

`install/setup_host.sh` is not used in Docker mode.

#### 2. Run one gene

```bash
scripts/run_single_gene_MLIPPER.sh \
  --ref-msa GENE/ref.fa \
  --query-msa GENE/query.fa \
  --backbone-tree GENE/backbone.nwk \
  --best-model GENE/gene.raxml.bestModel \
  --out-tree GENE/mlipper_gene_tree.nwk \
  --gpu-id 0
```

#### 3. Consume the output

ROADIES should use the tree written to:

- `GENE/mlipper_gene_tree.nwk`

or whatever path was passed as `--out-tree`.

Expected success criteria:

- exit code `0`
- non-empty output Newick tree

### Host mode

Use host mode when ROADIES should run a local host `MLIPPER` binary instead of a
Docker image.

#### 1. Prepare the host

Recommended explicit setup:

```bash
install/setup_host.sh
```

What `setup_host.sh` does:

- installs apt build dependencies used by the host build path
- installs distro package `libpll-dev`
- detects CUDA via `CUDA_HOME`, `nvcc`, or common CUDA install paths
- detects the distro `libpll` library directory
- runs `make clean`
- builds `MLIPPER` with `USE_DOUBLE=1`

What `setup_host.sh` does not do:

- it does not install the CUDA toolkit
- it does not schedule GPUs
- it does not run a gene

If only host dependencies are needed and the local binary should not be rebuilt:

```bash
install/setup_host.sh --skip-mlipper
```

#### 2. Run one gene without Docker

```bash
scripts/run_single_gene_MLIPPER.sh \
  --no-docker \
  --ref-msa GENE/ref.fa \
  --query-msa GENE/query.fa \
  --backbone-tree GENE/backbone.nwk \
  --best-model GENE/gene.raxml.bestModel \
  --out-tree GENE/mlipper_gene_tree.nwk
```

With explicit GPU restriction:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_single_gene_MLIPPER.sh \
  --no-docker \
  --ref-msa GENE/ref.fa \
  --query-msa GENE/query.fa \
  --backbone-tree GENE/backbone.nwk \
  --best-model GENE/gene.raxml.bestModel \
  --out-tree GENE/mlipper_gene_tree.nwk
```

With a custom local binary:

```bash
scripts/run_single_gene_MLIPPER.sh \
  --no-docker \
  --local-mlipper /path/to/MLIPPER \
  --ref-msa GENE/ref.fa \
  --query-msa GENE/query.fa \
  --backbone-tree GENE/backbone.nwk \
  --best-model GENE/gene.raxml.bestModel \
  --out-tree GENE/mlipper_gene_tree.nwk
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
- the current ROADIES image depends on distro packages `libpll-dev` and
  `libpll0`; if the host distribution or package version changes, the image
  should be re-smoke-tested
- host mode requires a working host CUDA toolkit with `nvcc`; `setup_host.sh`
  does not install CUDA
- host mode may need root or `sudo` privileges for apt dependency installation
- the current ROADIES setup is validated on `peregrine`; if ROADIES is moved
  to a different host environment, the image should be re-smoke-tested there
- the ROADIES image and wrapper are intended for per-gene execution
- the current interface guarantees the committed output tree
