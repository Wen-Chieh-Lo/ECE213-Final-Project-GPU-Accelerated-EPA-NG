# MLIPPER

- a self-contained GPU Docker build in [`Dockerfile`](Dockerfile)
- the benchmark driver script in [`run.sh`](run.sh)
- a compressed runtime dataset archive: `data/neotrop_runtime_dataset.tar.gz`

The expected workflow is:

1. build the Docker image
2. extract the runtime dataset archive if needed
3. run `./run.sh`

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA driver installed on the host
- `nvidia-container-toolkit` configured so `docker run --gpus all ...` works

## Docker Setup

Build the image from the repository root:

```bash
docker build -t ece213-mlipper:latest .
```

This image is based on `nvidia/cuda:12.3.2-devel-ubuntu22.04` and installs all required dependencies inside the container, including:

- Miniconda
- `epa-ng`
- `libpll`
- BLAS / LAPACK / TBB
- the float `MLIPPER` binary built with `make float`

Open an interactive container:

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace/MLIPPER \
  -w /workspace/MLIPPER \
  ece213-mlipper:latest bash
```

You can also run the full benchmark directly without entering a shell:

```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace/MLIPPER \
  -w /workspace/MLIPPER \
  ece213-mlipper:latest bash -lc './run.sh'
```

## Runtime Dataset Archive

The benchmark only needs the following input files:

- `data/neotrop/reference.fasta`
- `data/neotrop/tree.newick`
- `data/neotrop/query_1k.fasta`
- `data/neotrop/query_2k.fasta`
- `data/neotrop/query_5k.fasta`

These files are bundled in:

```bash
data/neotrop_runtime_dataset.tar.gz
```

Manual extraction from the repository root:

```bash
tar -xzf data/neotrop_runtime_dataset.tar.gz
```

The archive restores the files under `data/neotrop/`.

[`run.sh`](run.sh) also supports automatic extraction. If any required runtime dataset file is missing and `data/neotrop_runtime_dataset.tar.gz` is present, the script will extract the archive before starting the benchmark.

## Running The Benchmark

From inside the container, run:

```bash
./run.sh
```

The script performs the following steps automatically:

1. extracts `data/neotrop_runtime_dataset.tar.gz` if the required dataset files are missing
2. rebuilds the float binary with `make float`
3. checks that `epa-ng` is available in the configured conda environment
4. generates missing EPA-ng truth placements for 1k, 2k, and 5k query sets
5. runs MLIPPER in `baseline` and `fast` modes
6. compares each MLIPPER result against the EPA-ng truth with [`scripts/compare_jplace.py`](scripts/compare_jplace.py)
7. prints a CLI summary table with runtime, top-1 accuracy, and speedup relative to EPA-ng

Example output format:

```text
dataset  epa_ng_s     base_s       base_acc     base_spd     fast_s       fast_acc     fast_spd
1k       ...          ...          ...          ...          ...          ...          ...
2k       ...          ...          ...          ...          ...          ...          ...
5k       ...          ...          ...          ...          ...          ...          ...
```

Column definitions:

- `epa_ng_s`: EPA-ng wall-clock runtime in seconds
- `base_s`: MLIPPER baseline runtime in seconds
- `base_acc`: baseline top-1 exact edge match accuracy
- `base_spd`: baseline speedup relative to EPA-ng
- `fast_s`: MLIPPER fast runtime in seconds
- `fast_acc`: fast top-1 exact edge match accuracy
- `fast_spd`: fast speedup relative to EPA-ng

## Output Locations

- EPA-ng truth outputs:
  - `output/runtime_benchmarks/epa_ng_reference/query_1k/`
  - `output/runtime_benchmarks/epa_ng_reference/query_2k/`
  - `output/runtime_benchmarks/epa_ng_reference/query_5k/`
- MLIPPER benchmark outputs:
  - `output/run_sh_benchmarks/query_1k/`
  - `output/run_sh_benchmarks/query_2k/`
  - `output/run_sh_benchmarks/query_5k/`

Each benchmark directory contains:

- the MLIPPER run log
- the predicted `.jplace` file
- the comparison log against EPA-ng

## Notes

- The first run can take significantly longer because EPA-ng truth generation is performed when the truth `.jplace` files do not already exist.
- If you want to rerun the EPA-ng truth generation from scratch, remove the corresponding directory under `output/runtime_benchmarks/epa_ng_reference/`.
- The benchmark is intended to be run inside the provided Docker image so the TA does not need to install additional dependencies manually.
