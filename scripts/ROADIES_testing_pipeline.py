#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import statistics
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]

Record = Tuple[str, str]
OPT_BRANCH_LEN_MIN = 1.0e-4
OPT_BRANCH_LEN_MAX = 100.0
DEFAULT_BRANCH_LENGTH = 0.10536051565782628  # -log(0.9)


@dataclass
class ModelSpec:
    subst_model: str
    rates: list[float]
    alpha: float
    ncat: int
    freq_mode: str
    freqs: Optional[list[float]]
    raw_text: str


def read_fasta(path: Path) -> list[Record]:
    records: list[Record] = []
    name = None
    seq_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_parts)))
                name = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        records.append((name, "".join(seq_parts)))
    return records


def write_fasta(path: Path, records: Iterable[Record]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name, seq in records:
            handle.write(f">{name}\n")
            handle.write(f"{seq}\n")


def ensure_unique_names(records: Sequence[Record], label: str) -> None:
    seen = set()
    dup = set()
    for name, _ in records:
        if name in seen:
            dup.add(name)
        seen.add(name)
    if dup:
        preview = ", ".join(sorted(list(dup))[:5])
        raise SystemExit(f"{label} contains duplicate names: {preview}")


def parse_slash_floats(text: str, label: str) -> list[float]:
    parts = [part.strip() for part in text.split("/") if part.strip()]
    if not parts:
        raise SystemExit(f"{label} is empty")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise SystemExit(f"Failed to parse {label}: {text}") from exc


def parse_best_model(path: Path) -> ModelSpec:
    line = ""
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                break
    if not line:
        raise SystemExit(f"bestModel file is empty: {path}")

    model_text = line.split(",", 1)[0].strip()

    model_match = re.search(r"^([A-Za-z0-9_]+)\{([^}]*)\}", model_text)
    if not model_match:
        raise SystemExit(f"Could not parse substitution model from bestModel: {model_text}")
    subst_model = model_match.group(1)
    rates = parse_slash_floats(model_match.group(2), "GTR rates")

    gamma_match = re.search(r"\+G(\d+)[^{}]*\{([^}]*)\}", model_text)
    if gamma_match:
        ncat = int(gamma_match.group(1))
        alpha = float(gamma_match.group(2))
    else:
        ncat = 1
        alpha = 1.0

    if re.search(r"\+FC(?:\+|$)", model_text):
        freq_mode = "empirical"
        freqs = None
    else:
        freq_match = (
            re.search(r"\+FU\{([^}]*)\}", model_text)
            or re.search(r"\+FO\{([^}]*)\}", model_text)
            or re.search(r"\+F\{([^}]*)\}", model_text)
        )
        if freq_match:
            freq_mode = "manual"
            freqs = parse_slash_floats(freq_match.group(1), "base frequencies")
        else:
            freq_mode = "uniform"
            freqs = None

    if subst_model != "GTR":
        raise SystemExit(
            f"Unsupported substitution model in bestModel: {subst_model}. "
            "This helper currently supports only GTR."
        )
    if len(rates) != 6:
        raise SystemExit(f"GTR bestModel must provide 6 rates, got {len(rates)}")
    if freq_mode == "manual" and len(freqs or []) != 4:
        raise SystemExit("Manual DNA frequencies must contain exactly 4 values")

    return ModelSpec(
        subst_model=subst_model,
        rates=rates,
        alpha=alpha,
        ncat=ncat,
        freq_mode=freq_mode,
        freqs=freqs,
        raw_text=model_text,
    )


def build_rate_weights(ncat: int) -> list[float]:
    if ncat <= 0:
        raise SystemExit("ncat must be positive")
    weight = 1.0 / float(ncat)
    return [weight] * ncat

DEFAULT_DOCKER_IMAGE = "ece213-mlipper:latest"
DEFAULT_BUNDLE_ARCHIVE = REPO_ROOT / "data" / "usable_796_genes_bundle.tar.gz"
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "data" / "usable_796_genes_bundle"
DEFAULT_GENE_TREE_OUTDIR = REPO_ROOT / "output" / "usable_796_gene_trees"
DEFAULT_PIPELINE_OUTDIR = REPO_ROOT / "output" / "usable_796_mlipper_r4_astral_pipeline"
DEFAULT_MANIFEST_NAME = "manifest.tsv"
DEFAULT_GENE_TREE_SUMMARY_NAME = "gene_tree_summary.tsv"
DEFAULT_GENE_TREE_RUNTIME_NAME = "gene_tree_runtime_breakdown.tsv"
DEFAULT_GENE_STATE_NAME = "gene_state.json"
DEFAULT_FULL_ALIGNMENT_NAME = "full_alignment.fa"
DEFAULT_MLIPPER_TREE_NAME = "mlipper_gene_tree.nwk"
DEFAULT_MLIPPER_LOG_NAME = "mlipper.log"
DEFAULT_RAXML_PREFIX_RELPATH = "raxml_constraint/gene_tree"
DEFAULT_RAXML_LOG_NAME = "raxml_constraint.log"
DEFAULT_ALL_TREES_NAME = "mlipper_all_gene_trees.tre"
DEFAULT_MAPPING_NAME = "mlipper_astral_mapping.txt"
DEFAULT_SPECIES_TREE_NAME = "mlipper_species_tree.nwk"
DEFAULT_RF_PREFIX_NAME = "mlipper_species_tree_vs_golden_harmonized"
DEFAULT_TREE_SOURCES_NAME = "tree_sources.tsv"
DEFAULT_PIPELINE_SUMMARY_NAME = "summary.tsv"
DEFAULT_PIPELINE_RUNTIME_NAME = "runtime_breakdown.tsv"
DEFAULT_PIPELINE_STATE_NAME = "pipeline_state.json"
DEFAULT_GOLDEN_TREE = (
    REPO_ROOT / "output" / "usable_796_mlipper_gene_trees" / "mammals_reftree_240_latest.nwk"
)
DEFAULT_GOLDEN_RENAMES: Sequence[Tuple[str, str]] = (
    ("Sowerby’s_beaked_whale", "Sowerby_beaked_whale"),
    ("bats2", "Natal_long-fingered_bat"),
    ("bats", "David_myotis"),
)
MANIFEST_FIELDS = [
    "gene",
    "ref_msa",
    "query_msa",
    "backbone_tree",
    "best_model",
    "full_alignment",
    "gene_state",
    "mlipper_tree",
    "raxml_constraint_tree",
]
GPU_WALL_CLOCK_RE = re.compile(r"GPU Wall Clock time = ([0-9.]+) ms")


@dataclass
class GenePaths:
    gene: str
    gene_dir: Path
    best_tree: Path
    best_model: Path
    ref_fa: Path
    query_fa: Path


@dataclass
class GeneLoadEstimate:
    gene: GenePaths
    estimated_seconds: float
    source: str


@dataclass(frozen=True)
class WorkMount:
    host_root: Path
    container_root: str = "/workspace/job"

    def container_path(self, path: Path) -> str:
        rel = path.resolve().relative_to(self.host_root)
        return f"{self.container_root}/{str(rel).replace(os.sep, '/')}"


@contextmanager
def exclusive_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_extracted(bundle_root: Path, bundle_archive: Path) -> None:
    if bundle_root.exists():
        return
    if not bundle_archive.exists():
        raise SystemExit(f"Bundle root missing and archive not found: {bundle_archive}")
    with tarfile.open(bundle_archive, "r:gz") as tar:
        tar.extractall(bundle_root.parent)


def discover_genes(
    bundle_root: Path,
    *,
    best_tree_glob: str = "*.raxml.bestTree",
    best_model_glob: str = "*.raxml.bestModel",
    ref_msa_name: str = "iter0_output_msa_from_ref.fa",
    query_msa_name: str = "iter0_output_msa_from_query.fa",
) -> List[GenePaths]:
    genes: List[GenePaths] = []
    for gene_dir in sorted(p for p in bundle_root.iterdir() if p.is_dir()):
        best_tree = next(gene_dir.glob(best_tree_glob), None)
        best_model = next(gene_dir.glob(best_model_glob), None)
        ref_fa = gene_dir / ref_msa_name
        query_fa = gene_dir / query_msa_name
        if all(p is not None and Path(p).exists() for p in (best_tree, best_model, ref_fa, query_fa)):
            genes.append(
                GenePaths(
                    gene=gene_dir.name,
                    gene_dir=gene_dir,
                    best_tree=Path(best_tree),
                    best_model=Path(best_model),
                    ref_fa=ref_fa,
                    query_fa=query_fa,
                )
            )
    return genes


def resolve_output_path(base_dir: Path, raw_value: str, default_name: str) -> Path:
    text = raw_value.strip()
    target = Path(text) if text else Path(default_name)
    return target.resolve() if target.is_absolute() else (base_dir / target).resolve()


def artifact_suffix_path(prefix: Path, suffix: str) -> Path:
    return Path(str(prefix.resolve()) + suffix)


def default_balance_source_candidates(local_spr_radius: int) -> Sequence[Path]:
    r2_sources = (
        REPO_ROOT / "output" / "usable_796_mlipper_r2_astral_pipeline_timed_rerun_fresh",
        REPO_ROOT / "output" / "usable_796_mlipper_gene_trees",
    )
    r4_sources = (
        REPO_ROOT / "output" / "usable_796_mlipper_r4_astral_pipeline_timed_rerun",
        REPO_ROOT / "output" / "usable_796_mlipper_r4_astral_pipeline",
    )
    if local_spr_radius <= 2:
        return (*r2_sources, *r4_sources)
    return (*r4_sources, *r2_sources)


def resolve_balance_source_outdir(
    explicit: str,
    local_spr_radius: int,
) -> Path | None:
    if explicit.strip():
        path = Path(explicit).resolve()
        if not path.exists():
            raise SystemExit(f"--balance-source-outdir does not exist: {path}")
        return path
    for candidate in default_balance_source_candidates(local_spr_radius):
        if candidate.exists():
            return candidate.resolve()
    return None


def historical_gpu_wall_seconds(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = GPU_WALL_CLOCK_RE.search(text)
    if not match:
        return None
    return float(match.group(1)) / 1000.0


def fallback_gene_size_seconds(gene: GenePaths) -> float:
    return float(
        gene.ref_fa.stat().st_size +
        gene.query_fa.stat().st_size +
        gene.best_tree.stat().st_size +
        gene.best_model.stat().st_size
    )


def estimate_gene_loads(
    genes: Sequence[GenePaths],
    balance_source_outdir: Path | None,
) -> List[GeneLoadEstimate]:
    fallback_sizes = {gene.gene: fallback_gene_size_seconds(gene) for gene in genes}
    historical_seconds_by_gene: dict[str, float] = {}
    ratio_samples: List[float] = []
    if balance_source_outdir is not None:
        for gene in genes:
            historical_seconds = historical_gpu_wall_seconds(
                balance_source_outdir / gene.gene / "mlipper.log"
            )
            if historical_seconds is None or historical_seconds <= 0.0:
                continue
            historical_seconds_by_gene[gene.gene] = historical_seconds
            fallback_size = fallback_sizes[gene.gene]
            if fallback_size > 0.0:
                ratio_samples.append(historical_seconds / fallback_size)
    seconds_per_byte = statistics.median(ratio_samples) if ratio_samples else 1.0

    estimates: List[GeneLoadEstimate] = []
    for gene in genes:
        historical_seconds = historical_seconds_by_gene.get(gene.gene)
        if historical_seconds is not None:
            estimates.append(
                GeneLoadEstimate(
                    gene=gene,
                    estimated_seconds=historical_seconds,
                    source="historical_gpu_wall_seconds",
                )
            )
            continue
        estimates.append(
            GeneLoadEstimate(
                gene=gene,
                estimated_seconds=fallback_sizes[gene.gene] * seconds_per_byte,
                source="scaled_input_file_bytes",
            )
        )
    return estimates


def assign_genes_to_balanced_shards(
    genes: Sequence[GenePaths],
    num_shards: int,
    balance_source_outdir: Path | None,
) -> tuple[List[List[GeneLoadEstimate]], List[float]]:
    shard_items: List[List[GeneLoadEstimate]] = [[] for _ in range(num_shards)]
    shard_loads = [0.0] * num_shards
    estimates = estimate_gene_loads(genes, balance_source_outdir)
    estimates.sort(key=lambda item: (-item.estimated_seconds, item.gene.gene))
    for estimate in estimates:
        shard_idx = min(range(num_shards), key=lambda idx: (shard_loads[idx], idx))
        shard_items[shard_idx].append(estimate)
        shard_loads[shard_idx] += estimate.estimated_seconds
    for items in shard_items:
        items.sort(key=lambda item: item.gene.gene)
    return shard_items, shard_loads


def dynamic_queue_paths(queue_dir: Path) -> dict[str, Path]:
    return {
        "lock": queue_dir / "queue.lock",
        "pending": queue_dir / "pending.tsv",
        "running": queue_dir / "running.tsv",
        "done": queue_dir / "done.tsv",
        "failed": queue_dir / "failed.tsv",
    }


def read_nonempty_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_nonempty_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{line}\n" for line in lines if line)
    path.write_text(payload, encoding="utf-8")


def initialize_dynamic_queue(
    genes: Sequence[GenePaths],
    queue_dir: Path,
    balance_source_outdir: Path | None,
    manifest_path: Path,
    reset_queue: bool,
    clear_manifest: bool,
) -> tuple[int, int, int]:
    queue_dir.mkdir(parents=True, exist_ok=True)
    paths = dynamic_queue_paths(queue_dir)
    with exclusive_file_lock(paths["lock"]):
        if reset_queue:
            for key in ("pending", "running", "done", "failed"):
                if paths[key].exists():
                    paths[key].unlink()
            if clear_manifest and manifest_path.exists():
                manifest_path.unlink()

        queue_exists = any(paths[key].exists() for key in ("pending", "running", "done", "failed"))
        if not queue_exists:
            estimates = estimate_gene_loads(genes, balance_source_outdir)
            estimates.sort(key=lambda item: (-item.estimated_seconds, item.gene.gene))
            write_nonempty_lines(paths["pending"], [item.gene.gene for item in estimates])
            write_nonempty_lines(paths["running"], [])
            write_nonempty_lines(paths["done"], [])
            write_nonempty_lines(paths["failed"], [])

        pending_count = len(read_nonempty_lines(paths["pending"]))
        running_count = len(read_nonempty_lines(paths["running"]))
        done_count = len(read_nonempty_lines(paths["done"]))
    return pending_count, running_count, done_count


def claim_dynamic_gene(
    queue_dir: Path,
    gene_lookup: dict[str, GenePaths],
    worker_label: str,
) -> GenePaths | None:
    paths = dynamic_queue_paths(queue_dir)
    with exclusive_file_lock(paths["lock"]):
        pending_genes = read_nonempty_lines(paths["pending"])
        if not pending_genes:
            return None
        gene_name = pending_genes[0]
        write_nonempty_lines(paths["pending"], pending_genes[1:])
        running_entries = read_nonempty_lines(paths["running"])
        running_entries.append(f"{gene_name}\t{worker_label}")
        write_nonempty_lines(paths["running"], running_entries)
    gene = gene_lookup.get(gene_name)
    if gene is None:
        raise RuntimeError(f"Dynamic queue referenced unknown gene: {gene_name}")
    return gene


def finish_dynamic_gene(
    queue_dir: Path,
    gene_name: str,
    worker_label: str,
    success: bool,
) -> None:
    paths = dynamic_queue_paths(queue_dir)
    destination_key = "done" if success else "failed"
    with exclusive_file_lock(paths["lock"]):
        running_entries = [
            line
            for line in read_nonempty_lines(paths["running"])
            if line.split("\t", 1)[0] != gene_name
        ]
        write_nonempty_lines(paths["running"], running_entries)
        destination_entries = read_nonempty_lines(paths[destination_key])
        destination_entries.append(f"{gene_name}\t{worker_label}")
        write_nonempty_lines(paths[destination_key], destination_entries)


def common_mount_root(paths: Sequence[Path]) -> Path:
    anchors: list[str] = []
    for path in paths:
        resolved = path.resolve()
        anchor = resolved if resolved.exists() and resolved.is_dir() else resolved.parent
        anchors.append(str(anchor))
    if not anchors:
        raise RuntimeError("Cannot compute common mount root for empty path list")
    return Path(os.path.commonpath(anchors)).resolve()


def container_path(mount: WorkMount, path: Path) -> str:
    return mount.container_path(path)


def run_command(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("CMD: " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")


def run_command_capture(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        detail = stderr or stdout or "no output"
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\n{detail}"
        )
    return proc.stdout.strip()


def resolve_docker_gpus_spec(gpu_id: int, docker_gpus: str | None) -> str:
    if docker_gpus is not None and docker_gpus.strip():
        return docker_gpus.strip()
    return f"device={gpu_id}"


def docker_mlipper_cmd(
    repo_root: Path,
    work_mount: WorkMount,
    image: str,
    args: List[str],
    mount_libtbb: bool,
    gpu_spec: str,
    runtime_dir: Path,
) -> List[str]:
    uid = shutil.os.getuid()
    gid = shutil.os.getgid()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        gpu_spec,
        "--user",
        f"{uid}:{gid}",
        "-v",
        f"{repo_root}:/workspace/MLIPPER",
        "-v",
        f"{work_mount.host_root}:{work_mount.container_root}",
        "-v",
        f"{runtime_dir}:/workspace/runtime",
        "-w",
        "/workspace/runtime",
    ]
    if mount_libtbb:
        libtbb = Path("/usr/lib/x86_64-linux-gnu/libtbb.so.2")
        libtbb_link = Path("/usr/lib/x86_64-linux-gnu/libtbb.so")
        if libtbb.exists():
            cmd += ["-v", f"{libtbb}:{libtbb}:ro"]
        if libtbb_link.exists():
            cmd += ["-v", f"{libtbb_link}:{libtbb_link}:ro"]
    cmd += [image, "/workspace/MLIPPER/MLIPPER"]
    cmd += args
    return cmd


def start_mlipper_container(
    repo_root: Path,
    work_mount: WorkMount,
    image: str,
    mount_libtbb: bool,
    gpu_spec: str,
    container_name: str,
    runtime_dir: Path,
) -> str:
    uid = shutil.os.getuid()
    gid = shutil.os.getgid()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "--gpus",
        gpu_spec,
        "--user",
        f"{uid}:{gid}",
        "-v",
        f"{repo_root}:/workspace/MLIPPER",
        "-v",
        f"{work_mount.host_root}:{work_mount.container_root}",
        "-v",
        f"{runtime_dir}:/workspace/runtime",
        "-w",
        "/workspace/runtime",
    ]
    if mount_libtbb:
        libtbb = Path("/usr/lib/x86_64-linux-gnu/libtbb.so.2")
        libtbb_link = Path("/usr/lib/x86_64-linux-gnu/libtbb.so")
        if libtbb.exists():
            cmd += ["-v", f"{libtbb}:{libtbb}:ro"]
        if libtbb_link.exists():
            cmd += ["-v", f"{libtbb_link}:{libtbb_link}:ro"]
    cmd += [image, "tail", "-f", "/dev/null"]
    return run_command_capture(cmd)


def stop_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def docker_mlipper_exec_cmd(
    container_name: str,
    args: List[str],
    visible_devices: str | None = None,
) -> List[str]:
    cmd = [
        "docker",
        "exec",
        "-w",
        "/workspace/runtime",
        container_name,
        "/workspace/MLIPPER/MLIPPER",
    ]
    if visible_devices is not None and visible_devices.strip():
        cmd[2:2] = ["-e", f"CUDA_VISIBLE_DEVICES={visible_devices.strip()}"]
    cmd += args
    return cmd


def docker_raxml_constraint_cmd(
    work_mount: WorkMount,
    image: str,
    msa_path: Path,
    constraint_tree: Path,
    best_model: Path,
    prefix: Path,
    threads_spec: str,
) -> List[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{work_mount.host_root}:{work_mount.container_root}",
        "-w",
        work_mount.container_root,
        image,
        "raxml-ng",
        "--search",
        "--redo",
        "--msa",
        container_path(work_mount, msa_path),
        "--model",
        container_path(work_mount, best_model),
        "--opt-model",
        "off",
        "--threads",
        threads_spec,
        "--blopt",
        "nr_safe",
        "--tree-constraint",
        container_path(work_mount, constraint_tree),
        "--prefix",
        container_path(work_mount, prefix),
    ]


def ensure_full_alignment(ref_fa: Path, query_fa: Path, out_path: Path, redo: bool) -> None:
    if out_path.exists() and not redo:
        return
    ref_records = read_fasta(ref_fa)
    query_records = read_fasta(query_fa)
    ensure_unique_names(ref_records, "reference alignment")
    ensure_unique_names(query_records, "query alignment")
    combined = list(ref_records) + list(query_records)
    ensure_unique_names(combined, "combined alignment")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(out_path, combined)


def build_mlipper_commit_args(
    model: ModelSpec,
    tree_alignment: str,
    query_alignment: str,
    tree_path: str,
    committed_tree: str,
    local_spr: bool,
    batch_size: int,
    local_spr_radius: int,
    local_spr_rounds: int,
    local_spr_dynamic_validation_conflicts: bool,
) -> List[str]:
    args = [
        "--tree-alignment",
        tree_alignment,
        "--query-alignment",
        query_alignment,
        "--tree",
        tree_path,
        "--states",
        "4",
        "--subst-model",
        model.subst_model,
        "--ncat",
        str(model.ncat),
        "--alpha",
        f"{model.alpha:.6f}",
        "--pinv",
        "0.0",
        "--rates",
        ",".join(f"{rate:.6f}" for rate in model.rates),
        "--rate-weights",
        ",".join(f"{weight:.12g}" for weight in build_rate_weights(model.ncat)),
        "--commit-to-tree",
        committed_tree,
        "--commit-collapse-internal-epsilon",
        "-1",
    ]
    if model.freq_mode == "empirical":
        args.append("--empirical-freqs")
    elif model.freq_mode == "manual":
        assert model.freqs is not None
        args.extend(["--freqs", ",".join(f"{freq:.6f}" for freq in model.freqs)])
    else:
        args.extend(["--freqs", "0.25,0.25,0.25,0.25"])

    if local_spr:
        args.append("--local-spr")
        if batch_size > 0:
            args.extend(["--batch-insert-size", str(batch_size)])
        if local_spr_radius >= 0:
            args.extend(["--local-spr-radius", str(local_spr_radius)])
        if local_spr_rounds > 0:
            args.extend(["--local-spr-rounds", str(local_spr_rounds)])
        if local_spr_dynamic_validation_conflicts:
            args.append("--local-spr-dynamic-validation-conflicts")
    return args


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Sequence[Path], extra_tokens: Sequence[str] = ()) -> str:
    digest = hashlib.sha256()
    for token in extra_tokens:
        digest.update(b"token\0")
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    for path in paths:
        resolved = path.resolve()
        digest.update(b"path\0")
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hash_file(resolved).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_json_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON state file: {path}") from exc


def write_json_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_gene_tree_runtime_breakdown(
    path: Path,
    rows: Sequence[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_gene_tree_summary(
    path: Path,
    *,
    pipeline_label: str,
    expected_genes: int,
    manifest_genes: int,
    mlipper_tree_count: int,
    raxml_tree_count: int,
    mlipper_ran_count: int,
    raxml_ran_count: int,
    mlipper_gpu_wall_sum_seconds: float,
    pipeline_wall_clock_seconds: float,
    outdir: Path,
    manifest_path: Path,
    runtime_path: Path,
    dynamic_queue_dir: Path | None,
    run_mlipper: bool,
    run_raxml: bool,
    num_shards: int,
    shard_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "pipeline_label",
                "expected_genes",
                "manifest_genes",
                "mlipper_tree_count",
                "raxml_constraint_tree_count",
                "mlipper_ran_count",
                "raxml_ran_count",
                "mlipper_gpu_wall_sum_seconds",
                "pipeline_wall_clock_seconds",
                "outdir",
                "manifest",
                "runtime_breakdown",
                "dynamic_queue_dir",
                "run_mlipper",
                "run_raxml",
                "num_shards",
                "shard_index",
            ]
        )
        writer.writerow(
            [
                pipeline_label,
                expected_genes,
                manifest_genes,
                mlipper_tree_count,
                raxml_tree_count,
                mlipper_ran_count,
                raxml_ran_count,
                f"{mlipper_gpu_wall_sum_seconds:.6f}",
                f"{pipeline_wall_clock_seconds:.6f}",
                str(outdir),
                str(manifest_path),
                str(runtime_path),
                "" if dynamic_queue_dir is None else str(dynamic_queue_dir),
                "yes" if run_mlipper else "no",
                "yes" if run_raxml else "no",
                num_shards,
                shard_index,
            ]
        )


def write_manifest_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(MANIFEST_FIELDS)


def manifest_fieldnames(path: Path) -> list[str]:
    if not path.exists():
        return list(MANIFEST_FIELDS)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if row:
                return row
    return list(MANIFEST_FIELDS)


def load_manifest_genes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = (row.get("gene") or "").strip()
            if gene:
                seen.add(gene)
    return seen


def append_manifest_row(
    path: Path,
    gene: GenePaths,
    full_alignment: Path,
    gene_state: Path,
    mlipper_tree: Path | None,
    raxml_tree: Path | None,
) -> None:
    fieldnames = manifest_fieldnames(path)
    include_gene_state = "gene_state" in fieldnames
    row = [
        gene.gene,
        str(gene.ref_fa),
        str(gene.query_fa),
        str(gene.best_tree),
        str(gene.best_model),
        str(full_alignment),
    ]
    if include_gene_state:
        row.append(str(gene_state))
    row.extend(
        [
            str(mlipper_tree) if mlipper_tree is not None else "",
            str(raxml_tree) if raxml_tree is not None else "",
        ]
    )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(row)


def append_manifest_row_once(
    path: Path,
    gene: GenePaths,
    full_alignment: Path,
    gene_state: Path,
    mlipper_tree: Path | None,
    raxml_tree: Path | None,
) -> bool:
    lock_path = path.with_suffix(path.suffix + ".lock")
    with exclusive_file_lock(lock_path):
        write_manifest_header(path)
        if gene.gene in load_manifest_genes(path):
            return False
        append_manifest_row(path, gene, full_alignment, gene_state, mlipper_tree, raxml_tree)
        return True


def gene_tree_main(argv: Sequence[str] | None = None) -> dict[str, object]:
    pipeline_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Generate MLIPPER and/or RAxML-NG constrained gene trees from a bundle of per-gene "
            "inputs. The defaults still target usable_796, but discovery patterns, artifact names, "
            "and skip-existing behaviour are now parameterized for reuse in other datasets."
        )
    )
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--bundle-archive", type=Path, default=DEFAULT_BUNDLE_ARCHIVE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_GENE_TREE_OUTDIR)
    parser.add_argument("--best-tree-glob", default="*.raxml.bestTree")
    parser.add_argument("--best-model-glob", default="*.raxml.bestModel")
    parser.add_argument("--ref-msa-name", default="iter0_output_msa_from_ref.fa")
    parser.add_argument("--query-msa-name", default="iter0_output_msa_from_query.fa")
    parser.add_argument("--manifest-name", default="")
    parser.add_argument("--summary-name", default="")
    parser.add_argument("--runtime-name", default="")
    parser.add_argument("--gene-state-name", default="")
    parser.add_argument("--full-alignment-name", default="")
    parser.add_argument("--mlipper-tree-name", default="")
    parser.add_argument("--mlipper-log-name", default="")
    parser.add_argument("--raxml-prefix-relpath", default="")
    parser.add_argument("--raxml-log-name", default="")
    parser.add_argument("--pipeline-label", default="")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--redo", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--only-mlipper", action="store_true", default=False)
    parser.add_argument("--only-raxml", action="store_true", default=False)
    parser.add_argument("--mount-libtbb", action="store_true", default=True)
    parser.add_argument("--no-mount-libtbb", action="store_false", dest="mount_libtbb")
    parser.add_argument("--raxml-threads", default="8")
    parser.add_argument("--mlipper-local-spr", dest="mlipper_local_spr", action="store_true", default=True)
    parser.add_argument("--mlipper-no-local-spr", dest="mlipper_local_spr", action="store_false")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--local-spr-radius", type=int, default=2)
    parser.add_argument("--local-spr-rounds", type=int, default=1)
    parser.add_argument(
        "--local-spr-dynamic-validation-conflicts",
        action="store_true",
        default=False,
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--docker-gpus", default="")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--balance-source-outdir", default="")
    parser.add_argument("--dynamic-queue-dir", default="")
    parser.add_argument("--dynamic-reset-queue", action="store_true", default=False)
    parser.add_argument("--mlipper-reuse-container", action="store_true", default=False)
    args = parser.parse_args(argv)

    bundle_root = args.bundle_root.resolve()
    bundle_archive = args.bundle_archive.resolve()
    outdir = args.outdir.resolve()
    repo_root = REPO_ROOT.resolve()
    manifest_path = resolve_output_path(outdir, args.manifest_name, DEFAULT_MANIFEST_NAME)
    summary_path = resolve_output_path(outdir, args.summary_name, DEFAULT_GENE_TREE_SUMMARY_NAME)
    runtime_path = resolve_output_path(outdir, args.runtime_name, DEFAULT_GENE_TREE_RUNTIME_NAME)
    full_alignment_name = args.full_alignment_name.strip() or DEFAULT_FULL_ALIGNMENT_NAME
    mlipper_tree_name = args.mlipper_tree_name.strip() or DEFAULT_MLIPPER_TREE_NAME
    mlipper_log_name = args.mlipper_log_name.strip() or DEFAULT_MLIPPER_LOG_NAME
    raxml_prefix_relpath = args.raxml_prefix_relpath.strip() or DEFAULT_RAXML_PREFIX_RELPATH
    raxml_log_name = args.raxml_log_name.strip() or DEFAULT_RAXML_LOG_NAME
    gene_state_name = args.gene_state_name.strip() or DEFAULT_GENE_STATE_NAME
    pipeline_label = args.pipeline_label.strip() or outdir.name

    ensure_extracted(bundle_root, bundle_archive)
    genes = discover_genes(
        bundle_root,
        best_tree_glob=args.best_tree_glob,
        best_model_glob=args.best_model_glob,
        ref_msa_name=args.ref_msa_name,
        query_msa_name=args.query_msa_name,
    )
    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0")
    if args.gpu_id < 0:
        raise SystemExit("--gpu-id must be >= 0")
    if args.batch_size < 0:
        raise SystemExit("--batch-size must be >= 0")
    if args.local_spr_radius < 0:
        raise SystemExit("--local-spr-radius must be >= 0")
    if args.local_spr_rounds < 0:
        raise SystemExit("--local-spr-rounds must be >= 0")
    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be > 0")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < num-shards")
    dynamic_queue_dir = Path(args.dynamic_queue_dir).resolve() if args.dynamic_queue_dir.strip() else None
    if dynamic_queue_dir is not None and args.num_shards != 1:
        raise SystemExit("--dynamic-queue-dir cannot be combined with --num-shards > 1")
    if args.only_mlipper and args.only_raxml:
        raise SystemExit("--only-mlipper and --only-raxml are mutually exclusive")
    run_mlipper = not args.only_raxml
    run_raxml = not args.only_mlipper
    if args.start_index:
        genes = genes[args.start_index:]
    if args.max_genes > 0:
        genes = genes[: args.max_genes]
    balance_source_outdir = resolve_balance_source_outdir(
        args.balance_source_outdir,
        args.local_spr_radius,
    )
    if dynamic_queue_dir is None and args.num_shards > 1:
        shard_items, shard_loads = assign_genes_to_balanced_shards(
            genes,
            args.num_shards,
            balance_source_outdir,
        )
        selected_items = shard_items[args.shard_index]
        historical_count = sum(
            1 for item in selected_items if item.source == "historical_gpu_wall_seconds"
        )
        fallback_count = len(selected_items) - historical_count
        source_label = str(balance_source_outdir) if balance_source_outdir is not None else "none"
        load_summary = ", ".join(f"{load:.3f}" for load in shard_loads)
        print(
            f"Balanced shard {args.shard_index}/{args.num_shards}: "
            f"{len(selected_items)} genes, estimated load {shard_loads[args.shard_index]:.3f} s "
            f"(historical={historical_count}, fallback={fallback_count}, source={source_label})."
        )
        print(f"Estimated shard loads (s): [{load_summary}]")
        genes = [item.gene for item in selected_items]
    if not genes:
        raise SystemExit(f"No valid genes found under {bundle_root}")
    expected_genes = len(genes)
    gene_lookup = {gene.gene: gene for gene in genes}
    docker_gpu_spec = resolve_docker_gpus_spec(args.gpu_id, args.docker_gpus)
    outdir.mkdir(parents=True, exist_ok=True)
    work_mount = WorkMount(common_mount_root([bundle_root, outdir]))
    worker_runtime_dir = outdir / ".worker_runtime" / f"gpu{args.gpu_id}-pid{os.getpid()}"
    worker_runtime_dir.mkdir(parents=True, exist_ok=True)
    if dynamic_queue_dir is None:
        if args.redo and manifest_path.exists():
            manifest_path.unlink()
        write_manifest_header(manifest_path)
        manifest_genes = load_manifest_genes(manifest_path)
    else:
        pending_count, running_count, done_count = initialize_dynamic_queue(
            genes,
            dynamic_queue_dir,
            balance_source_outdir,
            manifest_path,
            args.dynamic_reset_queue,
            args.redo,
        )
        write_manifest_header(manifest_path)
        worker_label = f"gpu{args.gpu_id}-pid{os.getpid()}"
        print(
            f"Dynamic queue worker {worker_label}: queue={dynamic_queue_dir}, "
            f"pending={pending_count}, running={running_count}, done={done_count}"
        )
        manifest_genes = set()

    counters = {
        "mlipper_ran_count": 0,
        "raxml_ran_count": 0,
    }

    mlipper_container_name: str | None = None
    if run_mlipper and args.mlipper_reuse_container:
        mlipper_container_name = (
            f"usable796-mlipper-gpu{args.gpu_id}-"
            f"sh{args.shard_index}of{args.num_shards}-{os.getpid()}"
        )
        start_mlipper_container(
            repo_root,
            work_mount,
            args.docker_image,
            args.mount_libtbb,
            docker_gpu_spec,
            mlipper_container_name,
            worker_runtime_dir,
        )

    try:
        def process_gene(gene: GenePaths) -> None:
            gene_out = outdir / gene.gene
            if args.redo and gene_out.exists():
                shutil.rmtree(gene_out, ignore_errors=True)
            gene_out.mkdir(parents=True, exist_ok=True)

            model = parse_best_model(gene.best_model)
            state_path = gene_out / gene_state_name
            prior_state = load_json_state(state_path)

            full_alignment = gene_out / full_alignment_name
            if run_raxml:
                ensure_full_alignment(gene.ref_fa, gene.query_fa, full_alignment, args.redo)

            mlipper_tree = gene_out / mlipper_tree_name
            mlipper_log = gene_out / mlipper_log_name
            mlipper_fingerprint = ""
            if run_mlipper:
                mlipper_fingerprint = fingerprint_files(
                    [gene.ref_fa, gene.query_fa, gene.best_tree, gene.best_model],
                    extra_tokens=[
                        f"docker_image={args.docker_image}",
                        f"mlipper_local_spr={args.mlipper_local_spr}",
                        f"batch_size={args.batch_size}",
                        f"local_spr_radius={args.local_spr_radius}",
                        f"local_spr_rounds={args.local_spr_rounds}",
                        f"dynamic_validation_conflicts={args.local_spr_dynamic_validation_conflicts}",
                    ],
                )
                prior_mlipper_fingerprint = str(prior_state.get("mlipper_input_fingerprint", "")).strip()
                need_mlipper = (
                    args.redo
                    or not args.skip_existing
                    or not mlipper_tree.exists()
                    or (
                        prior_mlipper_fingerprint
                        and prior_mlipper_fingerprint != mlipper_fingerprint
                    )
                )
            else:
                need_mlipper = False
            if need_mlipper:
                mlipper_args = build_mlipper_commit_args(
                    model,
                    container_path(work_mount, gene.ref_fa),
                    container_path(work_mount, gene.query_fa),
                    container_path(work_mount, gene.best_tree),
                    container_path(work_mount, mlipper_tree),
                    args.mlipper_local_spr,
                    args.batch_size,
                    args.local_spr_radius,
                    args.local_spr_rounds,
                    args.local_spr_dynamic_validation_conflicts,
                )
                if mlipper_container_name is not None:
                    run_command(
                        docker_mlipper_exec_cmd(mlipper_container_name, mlipper_args),
                        mlipper_log,
                    )
                else:
                    run_command(
                        docker_mlipper_cmd(
                            repo_root,
                            work_mount,
                            args.docker_image,
                            mlipper_args,
                            args.mount_libtbb,
                            docker_gpu_spec,
                            worker_runtime_dir,
                        ),
                        mlipper_log,
                    )
                counters["mlipper_ran_count"] += 1

            raxml_prefix = gene_out / Path(raxml_prefix_relpath)
            raxml_tree = artifact_suffix_path(raxml_prefix, ".raxml.bestTree")
            raxml_log = gene_out / raxml_log_name
            raxml_fingerprint = ""
            if run_raxml:
                raxml_fingerprint = fingerprint_files(
                    [full_alignment, gene.best_tree, gene.best_model],
                    extra_tokens=[
                        f"docker_image={args.docker_image}",
                        f"raxml_threads={args.raxml_threads}",
                    ],
                )
                prior_raxml_fingerprint = str(prior_state.get("raxml_input_fingerprint", "")).strip()
                need_raxml = (
                    args.redo
                    or not args.skip_existing
                    or not raxml_tree.exists()
                    or (
                        prior_raxml_fingerprint
                        and prior_raxml_fingerprint != raxml_fingerprint
                    )
                )
            else:
                need_raxml = False
            if need_raxml:
                raxml_prefix.parent.mkdir(parents=True, exist_ok=True)
                run_command(
                    docker_raxml_constraint_cmd(
                        work_mount,
                        args.docker_image,
                        full_alignment,
                        gene.best_tree,
                        gene.best_model,
                        raxml_prefix,
                        args.raxml_threads,
                        ),
                        raxml_log,
                    )
                counters["raxml_ran_count"] += 1

            mlipper_ready = (not run_mlipper) or mlipper_tree.exists()
            raxml_ready = (not run_raxml) or raxml_tree.exists()
            write_json_state(
                state_path,
                {
                    "gene": gene.gene,
                    "gene_dir": str(gene.gene_dir),
                    "mlipper_input_fingerprint": mlipper_fingerprint,
                    "raxml_input_fingerprint": raxml_fingerprint,
                    "full_alignment": str(full_alignment),
                    "mlipper_tree": str(mlipper_tree) if mlipper_tree.exists() else "",
                    "raxml_constraint_tree": str(raxml_tree) if raxml_tree.exists() else "",
                    "mlipper_log": str(mlipper_log),
                    "raxml_log": str(raxml_log),
                },
            )
            if not (mlipper_ready and raxml_ready):
                return
            if dynamic_queue_dir is not None:
                append_manifest_row_once(
                    manifest_path,
                    gene,
                    full_alignment,
                    state_path,
                    mlipper_tree if mlipper_tree.exists() else None,
                    raxml_tree if raxml_tree.exists() else None,
                )
                return
            if gene.gene not in manifest_genes:
                append_manifest_row(
                    manifest_path,
                    gene,
                    full_alignment,
                    state_path,
                    mlipper_tree if mlipper_tree.exists() else None,
                    raxml_tree if raxml_tree.exists() else None,
                )
                manifest_genes.add(gene.gene)

        if dynamic_queue_dir is None:
            for gene in genes:
                process_gene(gene)
        else:
            while True:
                gene = claim_dynamic_gene(dynamic_queue_dir, gene_lookup, worker_label)
                if gene is None:
                    break
                print(f"[{worker_label}] Claimed {gene.gene}")
                try:
                    process_gene(gene)
                except Exception:
                    finish_dynamic_gene(dynamic_queue_dir, gene.gene, worker_label, success=False)
                    raise
                finish_dynamic_gene(dynamic_queue_dir, gene.gene, worker_label, success=True)
    finally:
        if mlipper_container_name is not None:
            stop_container(mlipper_container_name)

    manifest_gene_count = len(load_manifest_genes(manifest_path))
    mlipper_tree_count = sum(
        1 for gene in genes if (outdir / gene.gene / mlipper_tree_name).exists()
    )
    raxml_tree_count = sum(
        1 for gene in genes
        if artifact_suffix_path(outdir / gene.gene / Path(raxml_prefix_relpath), ".raxml.bestTree").exists()
    )
    mlipper_gpu_wall_sum_seconds = sum(
        seconds
        for seconds in (
            historical_gpu_wall_seconds(outdir / gene.gene / mlipper_log_name)
            for gene in genes
        )
        if seconds is not None
    )
    pipeline_wall_clock_seconds = time.perf_counter() - pipeline_start
    write_gene_tree_runtime_breakdown(
        runtime_path,
        [
            ("pipeline_label", pipeline_label),
            ("expected_genes", str(expected_genes)),
            ("manifest_genes", str(manifest_gene_count)),
            ("mlipper_tree_count", str(mlipper_tree_count)),
            ("raxml_constraint_tree_count", str(raxml_tree_count)),
            ("mlipper_ran_count", str(counters["mlipper_ran_count"])),
            ("raxml_ran_count", str(counters["raxml_ran_count"])),
            ("mlipper_gpu_wall_sum_seconds", f"{mlipper_gpu_wall_sum_seconds:.6f}"),
            ("pipeline_wall_clock_seconds", f"{pipeline_wall_clock_seconds:.6f}"),
        ],
    )
    write_gene_tree_summary(
        summary_path,
        pipeline_label=pipeline_label,
        expected_genes=expected_genes,
        manifest_genes=manifest_gene_count,
        mlipper_tree_count=mlipper_tree_count,
        raxml_tree_count=raxml_tree_count,
        mlipper_ran_count=counters["mlipper_ran_count"],
        raxml_ran_count=counters["raxml_ran_count"],
        mlipper_gpu_wall_sum_seconds=mlipper_gpu_wall_sum_seconds,
        pipeline_wall_clock_seconds=pipeline_wall_clock_seconds,
        outdir=outdir,
        manifest_path=manifest_path,
        runtime_path=runtime_path,
        dynamic_queue_dir=dynamic_queue_dir,
        run_mlipper=run_mlipper,
        run_raxml=run_raxml,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    return {
        "pipeline_label": pipeline_label,
        "bundle_root": bundle_root,
        "outdir": outdir,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "runtime_path": runtime_path,
        "expected_genes": expected_genes,
        "manifest_genes": manifest_gene_count,
        "mlipper_tree_count": mlipper_tree_count,
        "raxml_tree_count": raxml_tree_count,
        "run_mlipper": run_mlipper,
        "run_raxml": run_raxml,
        "mlipper_tree_name": mlipper_tree_name,
        "mlipper_log_name": mlipper_log_name,
        "raxml_prefix_relpath": raxml_prefix_relpath,
        "full_alignment_name": full_alignment_name,
        "gene_state_name": gene_state_name,
        "pipeline_wall_clock_seconds": pipeline_wall_clock_seconds,
    }


@dataclass(frozen=True)
class GeneTreeInventory:
    paths: list[Path]
    genes: list[str]
    source: str
    manifest_path: Path | None
    tree_column: str | None


@dataclass(frozen=True)
class GeneTreeRecord:
    gene: str
    source: str
    path: Path


@dataclass(frozen=True)
class StageTiming:
    ran: bool
    seconds: float


def run_timed_command(cmd: List[str], log_path: Path) -> float:
    start = time.perf_counter()
    run_command(cmd, log_path)
    return time.perf_counter() - start


def parse_golden_rename(raw_value: str) -> tuple[str, str]:
    if "=" not in raw_value:
        raise SystemExit(f"Invalid --golden-rename value {raw_value!r}; expected OLD=NEW")
    old, new = raw_value.split("=", 1)
    old = old.strip()
    new = new.strip()
    if not old or not new:
        raise SystemExit(f"Invalid --golden-rename value {raw_value!r}; expected OLD=NEW")
    return old, new


def resolve_golden_renames(golden_tree: Path, raw_renames: Sequence[str]) -> list[tuple[str, str]]:
    if raw_renames:
        return [parse_golden_rename(item) for item in raw_renames]
    if golden_tree.resolve() == DEFAULT_GOLDEN_TREE.resolve():
        return list(DEFAULT_GOLDEN_RENAMES)
    return []


def read_tree_leaves(path: Path) -> set[str]:
    leaf_re = re.compile(r"(?<=[,(])([^():;,]+)(?=[:),;])")
    return set(leaf_re.findall(path.read_text(encoding="utf-8")))


def rename_tree_leaves(text: str, renames: Sequence[tuple[str, str]]) -> str:
    if not renames:
        return text
    rename_map = dict(renames)
    leaf_re = re.compile(r"(?<=[,(])([^():;,]+)(?=[:),;])")
    return leaf_re.sub(lambda match: rename_map.get(match.group(1), match.group(1)), text)


def harmonize_golden_tree(src: Path, dst: Path, renames: Sequence[Tuple[str, str]]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    dst.write_text(rename_tree_leaves(text, renames), encoding="utf-8")


def read_rfdist(path: Path) -> Tuple[int, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 4:
            return int(fields[2]), float(fields[3])
    raise RuntimeError(f"Failed to parse RF distances from {path}")


def count_mapping_species(mapping_path: Path) -> int:
    species = set()
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                species.add(row[1].strip())
    return len(species)


def docker_astral_cmd(
    image: str,
    all_trees: Path,
    mapping: Path,
    species_tree: Path,
    threads: int,
) -> List[str]:
    mount = WorkMount(common_mount_root([all_trees, mapping, species_tree]), "/workspace/pipeline")
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{mount.host_root}:{mount.container_root}",
        "-w",
        mount.container_root,
        image,
        "astral-pro3",
        "-t",
        str(threads),
        "-i",
        container_path(mount, all_trees),
        "-a",
        container_path(mount, mapping),
        "-o",
        container_path(mount, species_tree),
    ]


def docker_rfdist_cmd(
    image: str,
    tree_a: Path,
    tree_b: Path,
    prefix: Path,
) -> List[str]:
    mount = WorkMount(common_mount_root([tree_a, tree_b, prefix]), "/workspace/pipeline")
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{mount.host_root}:{mount.container_root}",
        "-w",
        mount.container_root,
        image,
        "raxml-ng",
        "--rfdist",
        "--redo",
        "--tree",
        f"{container_path(mount, tree_a)},{container_path(mount, tree_b)}",
        "--prefix",
        container_path(mount, prefix),
    ]


def load_inventory_from_manifest(manifest_path: Path, tree_column: str) -> GeneTreeInventory:
    paths: list[Path] = []
    genes: list[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if tree_column not in fieldnames:
            raise RuntimeError(
                f"Manifest {manifest_path} is missing requested tree column {tree_column!r}; "
                f"available columns: {fieldnames}"
            )
        for row_index, row in enumerate(reader, start=2):
            tree_text = (row.get(tree_column) or "").strip()
            if not tree_text:
                continue
            gene = (row.get("gene") or "").strip() or f"row_{row_index}"
            genes.append(gene)
            paths.append(Path(tree_text).expanduser().resolve())
    return GeneTreeInventory(
        paths=paths,
        genes=genes,
        source=f"manifest:{manifest_path.name}",
        manifest_path=manifest_path.resolve(),
        tree_column=tree_column,
    )


def discover_inventory_from_glob(gene_tree_outdir: Path, gene_tree_glob: str) -> GeneTreeInventory:
    paths = [path.resolve() for path in sorted(gene_tree_outdir.glob(gene_tree_glob))]
    genes = [path.parent.name for path in paths]
    return GeneTreeInventory(
        paths=paths,
        genes=genes,
        source=f"glob:{gene_tree_glob}",
        manifest_path=None,
        tree_column=None,
    )


def discover_gene_tree_inventory(
    gene_tree_outdir: Path,
    manifest_path: Path | None,
    tree_column: str,
    gene_tree_glob: str,
) -> GeneTreeInventory:
    if manifest_path is not None and manifest_path.exists():
        return load_inventory_from_manifest(manifest_path, tree_column)

    default_manifest = gene_tree_outdir / DEFAULT_MANIFEST_NAME
    if default_manifest.exists():
        try:
            inventory = load_inventory_from_manifest(default_manifest, tree_column)
            if inventory.paths:
                return inventory
        except RuntimeError:
            pass
    return discover_inventory_from_glob(gene_tree_outdir, gene_tree_glob)


def validate_inventory(inventory: GeneTreeInventory, expected_genes: int | None) -> None:
    if not inventory.paths:
        raise RuntimeError(f"No gene trees discovered from {inventory.source}")
    missing_paths = [str(path) for path in inventory.paths if not path.exists()]
    if missing_paths:
        raise RuntimeError(f"Discovered gene trees include missing files: {missing_paths[:10]}")
    duplicate_genes = sorted({gene for gene in inventory.genes if inventory.genes.count(gene) > 1})
    if duplicate_genes:
        raise RuntimeError(f"Duplicate gene labels discovered: {duplicate_genes[:10]}")
    duplicate_paths = sorted({str(path) for path in inventory.paths if inventory.paths.count(path) > 1})
    if duplicate_paths:
        raise RuntimeError(f"Duplicate gene tree paths discovered: {duplicate_paths[:10]}")
    if expected_genes is not None and len(inventory.paths) != expected_genes:
        raise RuntimeError(
            f"Expected {expected_genes} gene trees, but found {len(inventory.paths)} from {inventory.source}"
        )


def inventory_to_records(inventory: GeneTreeInventory) -> list[GeneTreeRecord]:
    return [
        GeneTreeRecord(gene=gene, source=inventory.source, path=path)
        for gene, path in zip(inventory.genes, inventory.paths)
    ]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def wait_for_queue(queue_dir: Path, expected_done: int, poll_seconds: int) -> None:
    while True:
        done = count_lines(queue_dir / "done.tsv")
        running = count_lines(queue_dir / "running.tsv")
        pending = count_lines(queue_dir / "pending.tsv")
        failed = count_lines(queue_dir / "failed.tsv")
        print(
            f"queue status: done={done} running={running} pending={pending} failed={failed}",
            flush=True,
        )
        if failed:
            raise RuntimeError(f"Queue has {failed} failed gene(s): {queue_dir / 'failed.tsv'}")
        if done >= expected_done and running == 0 and pending == 0:
            return
        time.sleep(poll_seconds)


def parse_gene_tree_source(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Combined gene-tree source does not exist: {path}")
    return path


def read_tsv_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or [])


def load_records_from_tree_sources(path: Path) -> list[GeneTreeRecord]:
    records: list[GeneTreeRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        if "gene" not in fieldnames or "tree_path" not in fieldnames:
            raise RuntimeError(
                f"Tree-sources file {path} must contain gene and tree_path columns; "
                f"available columns: {fieldnames}"
            )
        default_source = f"tree_sources:{path.stem}"
        for row_index, row in enumerate(reader, start=2):
            gene = (row.get("gene") or "").strip() or f"row_{row_index}"
            tree_path_text = (row.get("tree_path") or "").strip()
            if not tree_path_text:
                continue
            source = (row.get("source") or "").strip() or default_source
            records.append(
                GeneTreeRecord(
                    gene=gene,
                    source=source,
                    path=Path(tree_path_text).expanduser().resolve(),
                )
            )
    if not records:
        raise RuntimeError(f"No usable tree records found in {path}")
    return records


def discover_source_records(
    source_path: Path,
    tree_sources_name: str,
    tree_column: str,
    gene_tree_glob: str,
) -> list[GeneTreeRecord]:
    if source_path.is_file():
        fieldnames = read_tsv_fieldnames(source_path)
        if {"gene", "tree_path"}.issubset(fieldnames):
            return load_records_from_tree_sources(source_path)
        inventory = load_inventory_from_manifest(source_path, tree_column)
        source_label = f"manifest:{source_path.stem}"
    elif source_path.is_dir():
        tree_sources_path = source_path / tree_sources_name
        if tree_sources_path.exists():
            return load_records_from_tree_sources(tree_sources_path)
        inventory = discover_gene_tree_inventory(source_path, None, tree_column, gene_tree_glob)
        source_label = f"outdir:{source_path.name}"
        inventory = GeneTreeInventory(
            paths=inventory.paths,
            genes=inventory.genes,
            source=source_label,
            manifest_path=inventory.manifest_path,
            tree_column=inventory.tree_column,
        )
    else:
        raise RuntimeError(f"Unsupported combined gene-tree source: {source_path}")
    validate_inventory(inventory, None)
    return inventory_to_records(inventory)


def original_bundle_best_tree_records(
    bundle_root: Path,
    best_tree_glob: str,
    best_model_glob: str,
    ref_msa_name: str,
    query_msa_name: str,
) -> list[GeneTreeRecord]:
    genes = discover_genes(
        bundle_root,
        best_tree_glob=best_tree_glob,
        best_model_glob=best_model_glob,
        ref_msa_name=ref_msa_name,
        query_msa_name=query_msa_name,
    )
    return [
        GeneTreeRecord(
            gene=gene.gene,
            source="original_bundle_best_tree",
            path=gene.best_tree.resolve(),
        )
        for gene in genes
    ]


def combine_iteration_records(
    *,
    include_original_bundle_best_trees: bool,
    bundle_root: Path,
    best_tree_glob: str,
    best_model_glob: str,
    ref_msa_name: str,
    query_msa_name: str,
    include_current_gene_trees: bool,
    current_gene_tree_outdir: Path,
    current_manifest_path: Path | None,
    combine_tree_sources: Sequence[Path],
    tree_sources_name: str,
    tree_column: str,
    gene_tree_glob: str,
) -> list[GeneTreeRecord]:
    merged: dict[str, GeneTreeRecord] = {}
    seen_source_paths: set[str] = set()

    if include_original_bundle_best_trees:
        for record in original_bundle_best_tree_records(
            bundle_root,
            best_tree_glob,
            best_model_glob,
            ref_msa_name,
            query_msa_name,
        ):
            merged[record.gene] = record

    for source_path in combine_tree_sources:
        key = str(source_path.resolve())
        if key in seen_source_paths:
            continue
        seen_source_paths.add(key)
        for record in discover_source_records(source_path, tree_sources_name, tree_column, gene_tree_glob):
            merged[record.gene] = record

    if include_current_gene_trees:
        current_key = str(current_gene_tree_outdir.resolve())
        if current_key not in seen_source_paths:
            current_inventory = discover_gene_tree_inventory(
                current_gene_tree_outdir,
                current_manifest_path if current_manifest_path is not None and current_manifest_path.exists() else None,
                tree_column,
                gene_tree_glob,
            )
            validate_inventory(current_inventory, None)
            current_records = [
                GeneTreeRecord(
                    gene=record.gene,
                    source="current_iteration",
                    path=record.path,
                )
                for record in inventory_to_records(current_inventory)
            ]
            for record in current_records:
                merged[record.gene] = record

    return [merged[gene] for gene in sorted(merged)]


def build_astral_inputs_from_records(
    records: Sequence[GeneTreeRecord],
    all_trees_path: Path,
    mapping_path: Path,
    sources_path: Path,
) -> Tuple[int, int]:
    leaf_re = re.compile(r"(?<=[,(])([^():;,]+)(?=[:),;])")
    copy_re = re.compile(r"^(.*)_([0-9]+)$")
    tree_texts: list[str] = []
    leaf_to_species: dict[str, str] = {}

    all_trees_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.parent.mkdir(parents=True, exist_ok=True)

    with sources_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["gene", "source", "tree_path"])
        for record in records:
            text = record.path.read_text(encoding="utf-8").strip()
            if not text:
                raise RuntimeError(f"Empty gene tree file: {record.path}")
            tree_texts.append(text)
            writer.writerow([record.gene, record.source, str(record.path)])
            for leaf in leaf_re.findall(text):
                match = copy_re.match(leaf)
                species = match.group(1) if match else leaf
                leaf_to_species[leaf] = species

    all_trees_path.write_text("\n".join(tree_texts) + "\n", encoding="utf-8")
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for leaf in sorted(leaf_to_species):
            writer.writerow([leaf, leaf_to_species[leaf]])
    return len(tree_texts), len(set(leaf_to_species.values()))


def write_pipeline_runtime_breakdown(
    path: Path,
    rows: Sequence[tuple[str, bool, float, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["stage", "ran", "wall_clock_seconds", "detail"])
        for stage, ran, seconds, detail in rows:
            writer.writerow([stage, "yes" if ran else "no", f"{seconds:.6f}", detail])


def write_pipeline_summary(
    path: Path,
    *,
    input_mode: str,
    pipeline_label: str,
    artifact_prefix: str,
    tree_column: str,
    expected_genes: int,
    gene_tree_count: int,
    species_count: int,
    rf: int | None,
    nrf: float | None,
    gene_tree_wall_clock_seconds: float,
    astral_pro_wall_clock_seconds: float,
    nrf_wall_clock_seconds: float,
    astral_input_prep_seconds: float,
    golden_prep_seconds: float,
    stage_sum_wall_clock_seconds: float,
    pipeline_wall_clock_seconds: float,
    gene_tree_stage_ran: bool,
    astral_stage_ran: bool,
    nrf_stage_ran: bool,
    skip_golden_compare: bool,
    gene_tree_outdir: Path,
    gene_tree_discovery_source: str,
    gene_tree_manifest: Path | None,
    all_trees: Path,
    mapping: Path,
    tree_sources: Path | None,
    species_tree: Path,
    golden_tree: Path | None,
    harmonized_golden_tree: Path | None,
    state_path: Path,
    runtime_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "expected_genes",
                "gene_tree_count",
                "species_count",
                "rf",
                "nrf",
                "mlipper_wall_clock_seconds",
                "astral_pro_wall_clock_seconds",
                "nrf_wall_clock_seconds",
                "stage_sum_wall_clock_seconds",
                "pipeline_wall_clock_seconds",
                "gene_tree_outdir",
                "all_gene_trees",
                "mapping",
                "tree_sources",
                "species_tree",
                "golden_tree",
                "harmonized_golden_tree",
                "input_mode",
                "pipeline_label",
                "artifact_prefix",
                "tree_column",
                "gene_tree_discovery_source",
                "gene_tree_manifest",
                "gene_tree_stage_ran",
                "astral_stage_ran",
                "nrf_stage_ran",
                "skip_golden_compare",
                "gene_tree_stage_wall_clock_seconds",
                "astral_input_prep_seconds",
                "golden_prep_seconds",
                "runtime_breakdown",
                "pipeline_state",
            ]
        )
        writer.writerow(
            [
                expected_genes,
                gene_tree_count,
                species_count,
                "" if rf is None else rf,
                "" if nrf is None else f"{nrf:.6f}",
                f"{gene_tree_wall_clock_seconds:.6f}",
                f"{astral_pro_wall_clock_seconds:.6f}",
                f"{nrf_wall_clock_seconds:.6f}",
                f"{stage_sum_wall_clock_seconds:.6f}",
                f"{pipeline_wall_clock_seconds:.6f}",
                str(gene_tree_outdir),
                str(all_trees),
                str(mapping),
                "" if tree_sources is None else str(tree_sources),
                str(species_tree),
                "" if golden_tree is None else str(golden_tree),
                "" if harmonized_golden_tree is None else str(harmonized_golden_tree),
                input_mode,
                pipeline_label,
                artifact_prefix,
                tree_column,
                gene_tree_discovery_source,
                "" if gene_tree_manifest is None else str(gene_tree_manifest),
                "yes" if gene_tree_stage_ran else "no",
                "yes" if astral_stage_ran else "no",
                "yes" if nrf_stage_ran else "no",
                "yes" if skip_golden_compare else "no",
                f"{gene_tree_wall_clock_seconds:.6f}",
                f"{astral_input_prep_seconds:.6f}",
                f"{golden_prep_seconds:.6f}",
                str(runtime_path),
                str(state_path),
            ]
        )


def pipeline_main(argv: Sequence[str] | None = None) -> int:
    pipeline_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Unified ROADIES testing pipeline: gene-tree generation, ASTRAL-Pro species tree, "
            "and optional RF comparison in a single script."
        )
    )
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--bundle-archive", type=Path, default=DEFAULT_BUNDLE_ARCHIVE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_PIPELINE_OUTDIR)
    parser.add_argument("--gene-tree-outdir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--best-tree-glob", default="*.raxml.bestTree")
    parser.add_argument("--best-model-glob", default="*.raxml.bestModel")
    parser.add_argument("--ref-msa-name", default="iter0_output_msa_from_ref.fa")
    parser.add_argument("--query-msa-name", default="iter0_output_msa_from_query.fa")
    parser.add_argument("--manifest-name", default="")
    parser.add_argument("--gene-tree-summary-name", default="")
    parser.add_argument("--gene-tree-runtime-name", default="")
    parser.add_argument("--gene-state-name", default="")
    parser.add_argument("--full-alignment-name", default="")
    parser.add_argument("--mlipper-tree-name", default="")
    parser.add_argument("--mlipper-log-name", default="")
    parser.add_argument("--raxml-prefix-relpath", default="")
    parser.add_argument("--raxml-log-name", default="")
    parser.add_argument("--tree-column", default="mlipper_tree")
    parser.add_argument("--gene-tree-glob", default="gene_*/mlipper_gene_tree.nwk")
    parser.add_argument("--expected-genes", type=int, default=0)
    parser.add_argument("--expected-total-genes", type=int, default=0)
    parser.add_argument("--pipeline-label", default="")
    parser.add_argument("--artifact-prefix", default="mlipper")
    parser.add_argument("--all-trees-name", default="")
    parser.add_argument("--mapping-name", default="")
    parser.add_argument("--tree-sources-name", default="")
    parser.add_argument("--species-tree-name", default="")
    parser.add_argument("--rf-prefix-name", default="")
    parser.add_argument("--summary-name", default="")
    parser.add_argument("--runtime-name", default="")
    parser.add_argument("--state-name", default="")
    parser.add_argument("--harmonized-golden-name", default="")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--golden-tree", type=Path, default=DEFAULT_GOLDEN_TREE)
    parser.add_argument("--golden-rename", action="append", default=[])
    parser.add_argument("--astral-threads", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--redo", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.add_argument("--only-mlipper", action="store_true", default=False)
    parser.add_argument("--only-raxml", action="store_true", default=False)
    parser.add_argument("--mount-libtbb", action="store_true", default=True)
    parser.add_argument("--no-mount-libtbb", action="store_false", dest="mount_libtbb")
    parser.add_argument("--raxml-threads", default="8")
    parser.add_argument("--mlipper-local-spr", dest="mlipper_local_spr", action="store_true", default=True)
    parser.add_argument("--mlipper-no-local-spr", dest="mlipper_local_spr", action="store_false")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--local-spr-radius", type=int, default=4)
    parser.add_argument("--local-spr-rounds", type=int, default=1)
    parser.add_argument(
        "--local-spr-dynamic-validation-conflicts",
        action="store_true",
        default=False,
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--docker-gpus", default="")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--balance-source-outdir", default="")
    parser.add_argument("--dynamic-queue-dir", default="")
    parser.add_argument("--dynamic-reset-queue", action="store_true", default=False)
    parser.add_argument("--mlipper-reuse-container", action="store_true", default=False)
    parser.add_argument("--skip-gene-tree-stage", action="store_true", default=False)
    parser.add_argument("--stop-after-gene-trees", action="store_true", default=False)
    parser.add_argument("--combine-iterations", action="store_true", default=False)
    parser.add_argument("--combine-tree-source", action="append", default=[])
    parser.add_argument("--include-current-gene-trees", action="store_true", default=True)
    parser.add_argument("--no-include-current-gene-trees", action="store_false", dest="include_current_gene_trees")
    parser.add_argument("--include-original-bundle-best-trees", action="store_true", default=True)
    parser.add_argument("--no-include-original-bundle-best-trees", action="store_false", dest="include_original_bundle_best_trees")
    parser.add_argument("--wait-for-queue-dir", type=Path, default=None)
    parser.add_argument("--wait-for-queue-expected-done", type=int, default=0)
    parser.add_argument("--wait-for-queue-poll-seconds", type=int, default=60)
    parser.add_argument("--skip-astral-stage", action="store_true", default=False)
    parser.add_argument("--skip-golden-compare", action="store_true", default=False)
    args = parser.parse_args(argv)

    if args.astral_threads <= 0:
        raise SystemExit("--astral-threads must be > 0")
    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0")
    if args.max_genes < 0:
        raise SystemExit("--max-genes must be >= 0")
    if args.expected_genes < 0:
        raise SystemExit("--expected-genes must be >= 0")
    if args.expected_total_genes < 0:
        raise SystemExit("--expected-total-genes must be >= 0")
    if args.wait_for_queue_expected_done < 0:
        raise SystemExit("--wait-for-queue-expected-done must be >= 0")
    if args.wait_for_queue_poll_seconds <= 0:
        raise SystemExit("--wait-for-queue-poll-seconds must be > 0")
    if not args.tree_column.strip():
        raise SystemExit("--tree-column must be non-empty")
    if not args.gene_tree_glob.strip():
        raise SystemExit("--gene-tree-glob must be non-empty")

    outdir = args.outdir.resolve()
    gene_tree_outdir = args.gene_tree_outdir.resolve() if args.gene_tree_outdir is not None else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path is not None
        else resolve_output_path(gene_tree_outdir, args.manifest_name, DEFAULT_MANIFEST_NAME)
    )
    all_trees = resolve_output_path(outdir, args.all_trees_name, DEFAULT_ALL_TREES_NAME)
    mapping = resolve_output_path(outdir, args.mapping_name, DEFAULT_MAPPING_NAME)
    tree_sources = resolve_output_path(outdir, args.tree_sources_name, DEFAULT_TREE_SOURCES_NAME)
    species_tree = resolve_output_path(outdir, args.species_tree_name, DEFAULT_SPECIES_TREE_NAME)
    rf_prefix = resolve_output_path(outdir, args.rf_prefix_name, DEFAULT_RF_PREFIX_NAME)
    summary_path = resolve_output_path(outdir, args.summary_name, DEFAULT_PIPELINE_SUMMARY_NAME)
    runtime_path = resolve_output_path(outdir, args.runtime_name, DEFAULT_PIPELINE_RUNTIME_NAME)
    state_path = resolve_output_path(outdir, args.state_name, DEFAULT_PIPELINE_STATE_NAME)
    harmonized_golden = resolve_output_path(
        outdir,
        args.harmonized_golden_name,
        f"{args.golden_tree.resolve().stem}.harmonized.nwk",
    )
    rf_output = artifact_suffix_path(rf_prefix, ".raxml.rfDistances")
    pipeline_label = args.pipeline_label.strip() or outdir.name
    artifact_prefix = args.artifact_prefix.strip() or "mlipper"

    gene_tree_stage_result: dict[str, object] | None = None
    gene_tree_stage_timing = StageTiming(ran=False, seconds=0.0)
    if not args.skip_gene_tree_stage:
        gene_tree_argv = [
            "--bundle-root",
            str(args.bundle_root.resolve()),
            "--bundle-archive",
            str(args.bundle_archive.resolve()),
            "--outdir",
            str(gene_tree_outdir),
            "--best-tree-glob",
            args.best_tree_glob,
            "--best-model-glob",
            args.best_model_glob,
            "--ref-msa-name",
            args.ref_msa_name,
            "--query-msa-name",
            args.query_msa_name,
            "--manifest-name",
            str(manifest_path),
            "--summary-name",
            args.gene_tree_summary_name.strip() or DEFAULT_GENE_TREE_SUMMARY_NAME,
            "--runtime-name",
            args.gene_tree_runtime_name.strip() or DEFAULT_GENE_TREE_RUNTIME_NAME,
            "--gene-state-name",
            args.gene_state_name.strip() or DEFAULT_GENE_STATE_NAME,
            "--full-alignment-name",
            args.full_alignment_name.strip() or DEFAULT_FULL_ALIGNMENT_NAME,
            "--mlipper-tree-name",
            args.mlipper_tree_name.strip() or DEFAULT_MLIPPER_TREE_NAME,
            "--mlipper-log-name",
            args.mlipper_log_name.strip() or DEFAULT_MLIPPER_LOG_NAME,
            "--raxml-prefix-relpath",
            args.raxml_prefix_relpath.strip() or DEFAULT_RAXML_PREFIX_RELPATH,
            "--raxml-log-name",
            args.raxml_log_name.strip() or DEFAULT_RAXML_LOG_NAME,
            "--pipeline-label",
            f"{pipeline_label}_gene_trees",
            "--docker-image",
            args.docker_image,
            "--start-index",
            str(args.start_index),
            "--max-genes",
            str(args.max_genes),
            "--raxml-threads",
            args.raxml_threads,
            "--batch-size",
            str(args.batch_size),
            "--local-spr-radius",
            str(args.local_spr_radius),
            "--local-spr-rounds",
            str(args.local_spr_rounds),
            "--gpu-id",
            str(args.gpu_id),
            "--docker-gpus",
            args.docker_gpus,
            "--num-shards",
            str(args.num_shards),
            "--shard-index",
            str(args.shard_index),
            "--balance-source-outdir",
            args.balance_source_outdir,
            "--dynamic-queue-dir",
            args.dynamic_queue_dir,
        ]
        if args.redo:
            gene_tree_argv.append("--redo")
        if args.skip_existing:
            gene_tree_argv.append("--skip-existing")
        else:
            gene_tree_argv.append("--no-skip-existing")
        if args.only_mlipper:
            gene_tree_argv.append("--only-mlipper")
        if args.only_raxml:
            gene_tree_argv.append("--only-raxml")
        if args.mount_libtbb:
            gene_tree_argv.append("--mount-libtbb")
        else:
            gene_tree_argv.append("--no-mount-libtbb")
        if args.mlipper_local_spr:
            gene_tree_argv.append("--mlipper-local-spr")
        else:
            gene_tree_argv.append("--mlipper-no-local-spr")
        if args.local_spr_dynamic_validation_conflicts:
            gene_tree_argv.append("--local-spr-dynamic-validation-conflicts")
        if args.dynamic_reset_queue:
            gene_tree_argv.append("--dynamic-reset-queue")
        if args.mlipper_reuse_container:
            gene_tree_argv.append("--mlipper-reuse-container")

        gene_tree_stage_result = gene_tree_main(gene_tree_argv)
        gene_tree_stage_timing = StageTiming(
            ran=True,
            seconds=float(gene_tree_stage_result["pipeline_wall_clock_seconds"]),
        )
        manifest_path = Path(str(gene_tree_stage_result["manifest_path"])).resolve()

    if args.stop_after_gene_trees:
        return 0

    if args.wait_for_queue_dir is not None:
        queue_dir = args.wait_for_queue_dir.resolve()
        expected_done = args.wait_for_queue_expected_done
        if expected_done <= 0 and gene_tree_stage_result is not None:
            expected_done = int(gene_tree_stage_result["expected_genes"])
        if expected_done <= 0 and args.expected_genes > 0:
            expected_done = int(args.expected_genes)
        if expected_done <= 0:
            raise SystemExit(
                "--wait-for-queue-dir requires --wait-for-queue-expected-done, "
                "or a current gene-tree stage / expected-genes value to infer it."
            )
        wait_for_queue(queue_dir, expected_done, args.wait_for_queue_poll_seconds)

    combine_mode = args.combine_iterations or bool(args.combine_tree_source)
    input_mode = "combined_iterations" if combine_mode else "direct_inventory"
    inventory_manifest_for_summary: Path | None = None
    inventory_source_label = ""

    if combine_mode:
        bundle_root = args.bundle_root.resolve()
        if args.include_original_bundle_best_trees:
            ensure_extracted(bundle_root, args.bundle_archive.resolve())
        combine_source_paths = [parse_gene_tree_source(path_text) for path_text in args.combine_tree_source]
        current_manifest_for_inventory = manifest_path if manifest_path.exists() else None
        combined_records = combine_iteration_records(
            include_original_bundle_best_trees=args.include_original_bundle_best_trees,
            bundle_root=bundle_root,
            best_tree_glob=args.best_tree_glob,
            best_model_glob=args.best_model_glob,
            ref_msa_name=args.ref_msa_name,
            query_msa_name=args.query_msa_name,
            include_current_gene_trees=args.include_current_gene_trees,
            current_gene_tree_outdir=gene_tree_outdir,
            current_manifest_path=current_manifest_for_inventory,
            combine_tree_sources=combine_source_paths,
            tree_sources_name=args.tree_sources_name.strip() or DEFAULT_TREE_SOURCES_NAME,
            tree_column=args.tree_column,
            gene_tree_glob=args.gene_tree_glob,
        )
        if not combined_records:
            raise RuntimeError("No gene trees discovered in combined-iterations mode")
        expected_genes = int(args.expected_total_genes) if args.expected_total_genes > 0 else None
        if expected_genes is None and args.include_original_bundle_best_trees:
            expected_genes = len(
                original_bundle_best_tree_records(
                    bundle_root,
                    args.best_tree_glob,
                    args.best_model_glob,
                    args.ref_msa_name,
                    args.query_msa_name,
                )
            )
        if expected_genes is not None and len(combined_records) != expected_genes:
            raise RuntimeError(
                f"Expected {expected_genes} combined gene trees, but found {len(combined_records)}"
            )
        prep_start = time.perf_counter()
        gene_tree_count, species_count = build_astral_inputs_from_records(
            combined_records,
            all_trees,
            mapping,
            tree_sources,
        )
        astral_input_prep_seconds = time.perf_counter() - prep_start
        inventory_source_label = f"combined:{len(combined_records)}"
        expected_genes = gene_tree_count if expected_genes is None else expected_genes
    else:
        inventory = discover_gene_tree_inventory(
            gene_tree_outdir,
            manifest_path if manifest_path.exists() else None,
            args.tree_column,
            args.gene_tree_glob,
        )
        expected_genes = int(args.expected_genes) if args.expected_genes > 0 else None
        if gene_tree_stage_result is not None:
            expected_genes = int(gene_tree_stage_result["expected_genes"])
        if expected_genes is None:
            expected_genes = len(inventory.paths)
        validate_inventory(inventory, expected_genes)
        prep_start = time.perf_counter()
        combined_records = inventory_to_records(inventory)
        gene_tree_count, species_count = build_astral_inputs_from_records(
            combined_records,
            all_trees,
            mapping,
            tree_sources,
        )
        astral_input_prep_seconds = time.perf_counter() - prep_start
        inventory_manifest_for_summary = inventory.manifest_path
        inventory_source_label = inventory.source

    species_count = count_mapping_species(mapping)

    state = load_json_state(state_path)
    astral_input_fingerprint = fingerprint_files(
        [all_trees, mapping],
        extra_tokens=[
            f"docker_image={args.docker_image}",
            f"astral_threads={args.astral_threads}",
            f"tree_column={args.tree_column}",
            f"artifact_prefix={artifact_prefix}",
        ],
    )
    astral_is_stale = (
        state.get("astral_input_fingerprint") != astral_input_fingerprint
        or state.get("species_tree") != str(species_tree)
    )

    astral_stage_timing = StageTiming(ran=False, seconds=0.0)
    if args.skip_astral_stage:
        if not species_tree.exists():
            raise RuntimeError(
                f"--skip-astral-stage was requested, but species tree does not exist: {species_tree}"
            )
    else:
        need_astral = (
            args.redo
            or not args.skip_existing
            or not species_tree.exists()
            or astral_is_stale
        )
        if need_astral:
            species_tree.parent.mkdir(parents=True, exist_ok=True)
            astral_stage_timing = StageTiming(
                ran=True,
                seconds=run_timed_command(
                    docker_astral_cmd(
                        args.docker_image,
                        all_trees,
                        mapping,
                        species_tree,
                        args.astral_threads,
                    ),
                    outdir / "astral_pro.log",
                ),
            )
    if not species_tree.exists():
        raise RuntimeError(f"Species tree not found after ASTRAL stage: {species_tree}")

    rf_abs: int | None = None
    nrf: float | None = None
    golden_prep_seconds = 0.0
    nrf_stage_timing = StageTiming(ran=False, seconds=0.0)
    nrf_input_fingerprint = ""
    golden_tree = args.golden_tree.resolve()
    renames = resolve_golden_renames(golden_tree, args.golden_rename)
    if not args.skip_golden_compare:
        golden_prep_start = time.perf_counter()
        harmonize_golden_tree(golden_tree, harmonized_golden, renames)
        species_tree_leaves = read_tree_leaves(species_tree)
        golden_leaves = read_tree_leaves(harmonized_golden)
        golden_prep_seconds = time.perf_counter() - golden_prep_start
        if species_tree_leaves != golden_leaves:
            only_species = sorted(species_tree_leaves - golden_leaves)
            only_golden = sorted(golden_leaves - species_tree_leaves)
            raise RuntimeError(
                "Species tree and golden reference have mismatched taxa after harmonization: "
                f"only_species={only_species[:10]} only_golden={only_golden[:10]}"
            )

        nrf_input_fingerprint = fingerprint_files(
            [species_tree, harmonized_golden],
            extra_tokens=[
                f"docker_image={args.docker_image}",
                f"rf_prefix={rf_prefix}",
            ],
        )
        nrf_is_stale = (
            state.get("nrf_input_fingerprint") != nrf_input_fingerprint
            or state.get("rf_output") != str(rf_output)
        )
        need_nrf = (
            args.redo
            or not args.skip_existing
            or not rf_output.exists()
            or astral_stage_timing.ran
            or nrf_is_stale
        )
        if need_nrf:
            rf_prefix.parent.mkdir(parents=True, exist_ok=True)
            nrf_stage_timing = StageTiming(
                ran=True,
                seconds=run_timed_command(
                    docker_rfdist_cmd(
                        args.docker_image,
                        species_tree,
                        harmonized_golden,
                        rf_prefix,
                    ),
                    outdir / "rfdist.log",
                ),
            )
        rf_abs, nrf = read_rfdist(rf_output)

    stage_sum_wall_clock_seconds = (
        gene_tree_stage_timing.seconds +
        astral_stage_timing.seconds +
        nrf_stage_timing.seconds
    )
    pipeline_wall_clock_seconds = time.perf_counter() - pipeline_start
    write_pipeline_runtime_breakdown(
        runtime_path,
        [
            (
                "gene_tree_stage",
                gene_tree_stage_timing.ran,
                gene_tree_stage_timing.seconds,
                f"outdir={gene_tree_outdir} expected_genes={expected_genes}",
            ),
            (
                "astral_input_prep",
                True,
                astral_input_prep_seconds,
                f"gene_trees={gene_tree_count} species={species_count} source={inventory_source_label}",
            ),
            (
                "astral_pro",
                astral_stage_timing.ran,
                astral_stage_timing.seconds,
                f"species_tree={species_tree} stale={astral_is_stale}",
            ),
            (
                "golden_prep",
                not args.skip_golden_compare,
                golden_prep_seconds,
                (
                    "comparison skipped"
                    if args.skip_golden_compare
                    else f"golden_tree={golden_tree} renames={len(renames)}"
                ),
            ),
            (
                "rfdist",
                nrf_stage_timing.ran,
                nrf_stage_timing.seconds,
                (
                    "comparison skipped"
                    if args.skip_golden_compare
                    else f"rf_output={rf_output}"
                ),
            ),
            (
                "total_pipeline",
                True,
                pipeline_wall_clock_seconds,
                f"summary={summary_path}",
            ),
        ],
    )
    write_pipeline_summary(
        summary_path,
        input_mode=input_mode,
        pipeline_label=pipeline_label,
        artifact_prefix=artifact_prefix,
        tree_column=args.tree_column,
        expected_genes=expected_genes,
        gene_tree_count=gene_tree_count,
        species_count=species_count,
        rf=rf_abs,
        nrf=nrf,
        gene_tree_wall_clock_seconds=gene_tree_stage_timing.seconds,
        astral_pro_wall_clock_seconds=astral_stage_timing.seconds,
        nrf_wall_clock_seconds=nrf_stage_timing.seconds,
        astral_input_prep_seconds=astral_input_prep_seconds,
        golden_prep_seconds=golden_prep_seconds,
        stage_sum_wall_clock_seconds=stage_sum_wall_clock_seconds,
        pipeline_wall_clock_seconds=pipeline_wall_clock_seconds,
        gene_tree_stage_ran=gene_tree_stage_timing.ran,
        astral_stage_ran=astral_stage_timing.ran,
        nrf_stage_ran=nrf_stage_timing.ran,
        skip_golden_compare=args.skip_golden_compare,
        gene_tree_outdir=gene_tree_outdir,
        gene_tree_discovery_source=inventory_source_label,
        gene_tree_manifest=inventory_manifest_for_summary,
        all_trees=all_trees,
        mapping=mapping,
        tree_sources=tree_sources,
        species_tree=species_tree,
        golden_tree=None if args.skip_golden_compare else golden_tree,
        harmonized_golden_tree=None if args.skip_golden_compare else harmonized_golden,
        state_path=state_path,
        runtime_path=runtime_path,
    )
    write_json_state(
        state_path,
        {
            "input_mode": input_mode,
            "pipeline_label": pipeline_label,
            "artifact_prefix": artifact_prefix,
            "tree_column": args.tree_column,
            "gene_tree_outdir": str(gene_tree_outdir),
            "gene_tree_source": inventory_source_label,
            "gene_tree_manifest": (
                None if inventory_manifest_for_summary is None else str(inventory_manifest_for_summary)
            ),
            "gene_tree_count": gene_tree_count,
            "species_count": species_count,
            "combine_iterations": combine_mode,
            "combine_tree_sources": [str(parse_gene_tree_source(path_text)) for path_text in args.combine_tree_source],
            "include_current_gene_trees": args.include_current_gene_trees,
            "include_original_bundle_best_trees": args.include_original_bundle_best_trees,
            "all_trees": str(all_trees),
            "mapping": str(mapping),
            "tree_sources": str(tree_sources),
            "species_tree": str(species_tree),
            "astral_input_fingerprint": astral_input_fingerprint,
            "rf_output": str(rf_output),
            "nrf_input_fingerprint": nrf_input_fingerprint,
            "golden_tree": None if args.skip_golden_compare else str(golden_tree),
            "harmonized_golden_tree": None if args.skip_golden_compare else str(harmonized_golden),
            "skip_golden_compare": args.skip_golden_compare,
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return pipeline_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
