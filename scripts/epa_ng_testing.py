#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "run_sh_benchmarks"
DEFAULT_DATA_ARCHIVE = ROOT_DIR / "data" / "neotrop_runtime_dataset.tar.gz"
DEFAULT_EPA_NG_MODEL = "GTR{1.0/1.0/1.0/1.0/1.0/1.0}+FU{0.25/0.25/0.25/0.25}+G4{0.3}"
ELAPSED_SECONDS_RE = re.compile(r"INFO Elapsed Time: ([0-9.]+)s")
PRECISION_RE = re.compile(r"^Precision mode:\s+(\S+)", re.MULTILINE)
MAX_MISMATCH_ROWS = 50
MAX_TOPK_GAP_ROWS = 50


BENCH_MODE_ENVS = {
    "baseline": {
        "MLIPPER_FULL_OPT_PASSES": "4",
        "MLIPPER_REFINE_GLOBAL_PASSES": "0",
        "MLIPPER_REFINE_EXTRA_PASSES": "0",
        "MLIPPER_REFINE_DETECT_TOPK": "0",
        "MLIPPER_REFINE_TOPK": "0",
    },
    "fast": {
        "MLIPPER_FULL_OPT_PASSES": "1",
        "MLIPPER_REFINE_GLOBAL_PASSES": "0",
        "MLIPPER_REFINE_EXTRA_PASSES": "0",
        "MLIPPER_REFINE_DETECT_TOPK": "0",
        "MLIPPER_REFINE_TOPK": "0",
    },
}


DATASETS = {
    "1k": {
        "query_id": "query_1k",
        "query_path": ROOT_DIR / "data" / "neotrop" / "query_1k.fasta",
    },
    "2k": {
        "query_id": "query_2k",
        "query_path": ROOT_DIR / "data" / "neotrop" / "query_2k.fasta",
    },
    "5k": {
        "query_id": "query_5k",
        "query_path": ROOT_DIR / "data" / "neotrop" / "query_5k.fasta",
    },
}


REFERENCE_FASTA = ROOT_DIR / "data" / "neotrop" / "reference.fasta"
TREE_PATH = ROOT_DIR / "data" / "neotrop" / "tree.newick"


@dataclass(frozen=True)
class PlacementRow:
    edge_num: int
    likelihood: float
    like_weight_ratio: float | None
    distal_length: float | None
    pendant_length: float | None


@dataclass(frozen=True)
class JplaceData:
    path: Path
    placements_by_name: dict[str, tuple[PlacementRow, ...]]
    split_by_edge: dict[int, tuple[str, ...]]
    leaves: tuple[str, ...]


@dataclass(frozen=True)
class MismatchDetail:
    query_name: str
    truth_edge: int
    pred_edge: int
    truth_likelihood: float
    pred_likelihood: float


@dataclass(frozen=True)
class TopKGapDetail:
    query_name: str
    truth_top1_edge: int
    pred_edge: int
    truth_rank: int
    truth_top1_likelihood: float
    truth_pred_edge_likelihood: float
    likelihood_gap: float


@dataclass(frozen=True)
class CompareMetrics:
    truth_path: Path
    pred_path: Path
    shared_count: int
    truth_only_count: int
    pred_only_count: int
    raw_top1_exact: int
    split_top1_exact: int
    raw_topk_hit: int
    split_topk_hit: int
    truth_rank_histogram: dict[int, int]
    avg_truth_gap: float | None
    mean_abs_distal_all: float | None
    mean_abs_pendant_all: float | None
    mean_abs_distal_matched: float | None
    mean_abs_pendant_matched: float | None
    mismatches: list[MismatchDetail]
    topk_gap_details: list[TopKGapDetail]
    truth_topk: int


@dataclass(frozen=True)
class BenchmarkModeResult:
    dataset_size: str
    mode: str
    truth_jplace: Path
    pred_jplace: Path
    run_log: Path
    compare_log: Path
    gpu_kernel_ms: float | None
    gpu_wall_ms: float | None
    precision_mode: str | None
    metrics: CompareMetrics


@dataclass
class TreeNode:
    label: str | None
    children: list["TreeNode"]
    edge_num: int | None = None


def bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", ""}


def split_env_words(name: str, default: Sequence[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return list(default)
    return [part for part in value.replace(",", " ").split() if part]


def ensure_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing required file: {path}")


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    if log_path is None:
        subprocess.run(list(cmd), cwd=cwd, env=env, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"CMD: {shlex.join(cmd)}\n")
        handle.flush()
        subprocess.run(
            list(cmd),
            cwd=cwd,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def ensure_runtime_dataset(data_archive: Path) -> None:
    required_files = [
        REFERENCE_FASTA,
        TREE_PATH,
        DATASETS["1k"]["query_path"],
        DATASETS["2k"]["query_path"],
        DATASETS["5k"]["query_path"],
    ]
    if all(path.is_file() for path in required_files):
        return

    if not data_archive.is_file():
        raise SystemExit(
            "Required dataset files are missing.\n"
            f"Expected archive for automatic extraction: {data_archive}"
        )

    print(f"Extracting runtime dataset from {data_archive.name}", flush=True)
    (ROOT_DIR / "data" / "neotrop").mkdir(parents=True, exist_ok=True)
    with tarfile.open(data_archive, "r:gz") as archive:
        archive.extractall(ROOT_DIR)

    for path in required_files:
        ensure_file(path)


def ensure_binary(precision: str, build_dir: str) -> Path:
    mlipper_binary = ROOT_DIR / "MLIPPER"
    build_config = ROOT_DIR / build_dir / ".build_config"
    if precision == "float":
        print("Building MLIPPER float binary for benchmark.", flush=True)
        run_command(["make", "float"], cwd=ROOT_DIR)
    elif precision == "double":
        print("Building MLIPPER double binary for benchmark.", flush=True)
        run_command(["make", "double"], cwd=ROOT_DIR)
    else:
        raise SystemExit(f"Unknown MLIPPER_BENCH_PRECISION: {precision}")

    if not os.access(mlipper_binary, os.X_OK):
        raise SystemExit(f"Failed to build MLIPPER binary: {mlipper_binary}")
    if not build_config.is_file():
        raise SystemExit(f"Missing build config stamp after build: {build_config}")

    build_stamp = build_config.read_text(encoding="utf-8", errors="ignore").splitlines()
    expected_stamp = "USE_DOUBLE=1" if precision == "double" else "USE_DOUBLE=0"
    if expected_stamp not in build_stamp:
        raise SystemExit(
            f"Benchmark requested {precision}, but build stamp does not show {expected_stamp}: "
            f"{build_config}"
        )

    print(f"Verified MLIPPER benchmark precision: {precision}", flush=True)
    return mlipper_binary


def ensure_conda() -> None:
    if shutil_which("conda") is None:
        raise SystemExit("conda not found in PATH. Install Miniconda/Anaconda first.")


def shutil_which(program: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / program
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def ensure_epa_ng(epa_ng_env: str, epa_ng_bin: Path | None) -> None:
    if epa_ng_bin is not None:
        if not os.access(epa_ng_bin, os.X_OK):
            raise SystemExit(f"EPA_NG_BIN is not executable: {epa_ng_bin}")
        return

    ensure_conda()
    result = subprocess.run(
        ["conda", "run", "-n", epa_ng_env, "epa-ng", "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"epa-ng is not available in conda env {epa_ng_env!r}.\n"
            f"Install it with: conda install -y -n {epa_ng_env} -c bioconda epa-ng"
        )


def truth_jplace_path(query_id: str) -> Path:
    return ROOT_DIR / "output" / "runtime_benchmarks" / "epa_ng_reference" / query_id / "epa_result.jplace"


def build_epa_ng_command(
    *,
    query_path: Path,
    outdir: Path,
    epa_ng_threads: int,
    epa_ng_model: str,
    epa_ng_env: str,
    epa_ng_bin: Path | None,
) -> list[str]:
    base_cmd: list[str]
    if epa_ng_bin is not None:
        base_cmd = [str(epa_ng_bin)]
    else:
        base_cmd = ["conda", "run", "--no-capture-output", "-n", epa_ng_env, "epa-ng"]

    return base_cmd + [
        "--redo",
        "--no-heur",
        "--tree",
        str(TREE_PATH),
        "--ref-msa",
        str(REFERENCE_FASTA),
        "--query",
        str(query_path),
        "--threads",
        str(epa_ng_threads),
        "--model",
        epa_ng_model,
        "--no-pre-mask",
        "--outdir",
        str(outdir),
    ]


def ensure_truth_jplace(
    *,
    query_path: Path,
    query_id: str,
    epa_ng_threads: int,
    epa_ng_model: str,
    epa_ng_env: str,
    epa_ng_bin: Path | None,
    force_regenerate_truth: bool,
) -> Path:
    truth_jplace = truth_jplace_path(query_id)
    if not force_regenerate_truth and truth_jplace.is_file():
        return truth_jplace

    truth_dir = truth_jplace.parent
    truth_log = truth_dir / "run.log"
    truth_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating EPA-ng truth for {query_id}", flush=True)
    run_command(
        build_epa_ng_command(
            query_path=query_path,
            outdir=truth_dir,
            epa_ng_threads=epa_ng_threads,
            epa_ng_model=epa_ng_model,
            epa_ng_env=epa_ng_env,
            epa_ng_bin=epa_ng_bin,
        ),
        cwd=ROOT_DIR,
        log_path=truth_log,
    )
    ensure_file(truth_jplace)
    return truth_jplace


def parse_ms_metric(log_path: Path, label: str) -> float | None:
    if not log_path.is_file():
        return None
    pattern = re.compile(rf"^{re.escape(label)} = ([0-9.]+) ms$", re.MULTILINE)
    match = pattern.search(log_path.read_text(encoding="utf-8", errors="ignore"))
    return float(match.group(1)) if match else None


def parse_precision_mode(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    match = PRECISION_RE.search(log_path.read_text(encoding="utf-8", errors="ignore"))
    return match.group(1) if match else None


def parse_epa_elapsed_ms(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    match = ELAPSED_SECONDS_RE.search(log_path.read_text(encoding="utf-8", errors="ignore"))
    return (float(match.group(1)) * 1000.0) if match else None


def format_ms(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def format_seconds_from_ms(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value / 1000.0:.3f}"


def format_speedup(reference_ms: float | None, candidate_ms: float | None) -> str:
    if reference_ms is None or candidate_ms is None or candidate_ms == 0:
        return "NA"
    return f"{reference_ms / candidate_ms:.2f}x"


def format_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "NA"
    return f"{(100.0 * count / total):.2f}%"


def format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.12f}"


def parse_jplace_tree(tree_text: str) -> TreeNode:
    index = 0
    length = len(tree_text)
    current_children: list[TreeNode] = []
    stack: list[list[TreeNode]] = []
    root: TreeNode | None = None

    def skip_ws(pos: int) -> int:
        while pos < length and tree_text[pos].isspace():
            pos += 1
        return pos

    def read_label(pos: int) -> tuple[str, int]:
        if pos < length and tree_text[pos] == "'":
            pos += 1
            start = pos
            while pos < length and tree_text[pos] != "'":
                pos += 1
            if pos >= length:
                raise RuntimeError("Unterminated quoted label in jplace tree")
            return tree_text[start:pos], pos + 1

        start = pos
        while pos < length and tree_text[pos] not in ",():;{}":
            pos += 1
        return tree_text[start:pos].strip(), pos

    def read_branch_suffix(pos: int, node: TreeNode) -> int:
        pos = skip_ws(pos)
        if pos < length and tree_text[pos] == ":":
            pos += 1
            while pos < length and tree_text[pos] not in ",);{":
                pos += 1
        pos = skip_ws(pos)
        if pos < length and tree_text[pos] == "{":
            pos += 1
            start = pos
            while pos < length and tree_text[pos] != "}":
                pos += 1
            if pos >= length:
                raise RuntimeError("Unterminated edge annotation in jplace tree")
            node.edge_num = int(tree_text[start:pos])
            pos += 1
        return pos

    while index < length:
        index = skip_ws(index)
        if index >= length:
            break
        token = tree_text[index]
        if token == "(":
            stack.append(current_children)
            current_children = []
            index += 1
            continue
        if token == ",":
            index += 1
            continue
        if token == ")":
            node = TreeNode(label=None, children=current_children)
            if not stack:
                raise RuntimeError("Malformed jplace tree: unmatched ')'")
            current_children = stack.pop()
            current_children.append(node)
            index += 1
            index = skip_ws(index)
            if index < length and tree_text[index] not in ":,);":
                label, index = read_label(index)
                if label:
                    node.label = label
            index = read_branch_suffix(index, node)
            continue
        if token == ";":
            if len(current_children) != 1:
                raise RuntimeError("Malformed jplace tree: expected single root before ';'")
            root = current_children[0]
            index += 1
            break

        label, index = read_label(index)
        if not label:
            raise RuntimeError("Malformed jplace tree: empty label")
        node = TreeNode(label=label, children=[])
        current_children.append(node)
        index = read_branch_suffix(index, node)

    if root is None:
        if len(current_children) != 1:
            raise RuntimeError("Malformed jplace tree: missing root")
        root = current_children[0]
    return root


def build_split_map(root: TreeNode) -> tuple[dict[int, tuple[str, ...]], tuple[str, ...]]:
    def collect_leaves(node: TreeNode) -> set[str]:
        if not node.children:
            if not node.label:
                raise RuntimeError("Leaf node missing label in jplace tree")
            return {node.label}
        leaves: set[str] = set()
        for child in node.children:
            leaves.update(collect_leaves(child))
        return leaves

    leaf_sets: dict[int, set[str]] = {}

    def fill(node: TreeNode) -> set[str]:
        if not node.children:
            assert node.label is not None
            leaves = {node.label}
        else:
            leaves = set()
            for child in node.children:
                leaves.update(fill(child))
        leaf_sets[id(node)] = leaves
        return leaves

    def canonicalize(subset: set[str], all_leaves: set[str]) -> tuple[str, ...]:
        complement = all_leaves - subset
        left = tuple(sorted(subset))
        right = tuple(sorted(complement))
        if len(left) < len(right):
            return left
        if len(right) < len(left):
            return right
        return left if left <= right else right

    all_leaves = collect_leaves(root)
    fill(root)
    split_by_edge: dict[int, tuple[str, ...]] = {}

    def assign(node: TreeNode) -> None:
        for child in node.children:
            if child.edge_num is not None:
                split_by_edge[child.edge_num] = canonicalize(leaf_sets[id(child)], all_leaves)
            assign(child)

    assign(root)
    return split_by_edge, tuple(sorted(all_leaves))


def load_jplace(path: Path) -> JplaceData:
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = raw.get("fields")
    if not isinstance(fields, list):
        raise RuntimeError(f"Invalid jplace file {path}: missing fields")
    field_index = {name: idx for idx, name in enumerate(fields)}
    for required in ("edge_num", "likelihood"):
        if required not in field_index:
            raise RuntimeError(f"Invalid jplace file {path}: missing field {required!r}")

    root = parse_jplace_tree(raw["tree"])
    split_by_edge, leaves = build_split_map(root)
    placements_by_name: dict[str, tuple[PlacementRow, ...]] = {}

    for placement in raw.get("placements", []):
        names = placement.get("n")
        if names is None:
            names = [item[0] for item in placement.get("nm", [])]
        if not names:
            raise RuntimeError(f"Invalid jplace file {path}: placement entry missing names")
        rows = []
        for raw_row in placement.get("p", []):
            rows.append(
                PlacementRow(
                    edge_num=int(raw_row[field_index["edge_num"]]),
                    likelihood=float(raw_row[field_index["likelihood"]]),
                    like_weight_ratio=(
                        float(raw_row[field_index["like_weight_ratio"]])
                        if "like_weight_ratio" in field_index
                        else None
                    ),
                    distal_length=(
                        float(raw_row[field_index["distal_length"]])
                        if "distal_length" in field_index
                        else None
                    ),
                    pendant_length=(
                        float(raw_row[field_index["pendant_length"]])
                        if "pendant_length" in field_index
                        else None
                    ),
                )
            )
        row_tuple = tuple(rows)
        for name in names:
            if name in placements_by_name:
                raise RuntimeError(f"Duplicate query instance {name!r} in {path}")
            placements_by_name[name] = row_tuple

    return JplaceData(
        path=path,
        placements_by_name=placements_by_name,
        split_by_edge=split_by_edge,
        leaves=leaves,
    )


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def compare_jplace_files(truth_path: Path, pred_path: Path, truth_topk: int) -> CompareMetrics:
    truth = load_jplace(truth_path)
    pred = load_jplace(pred_path)
    if truth.leaves != pred.leaves:
        raise RuntimeError(
            "Truth and prediction jplace trees do not have identical reference leaf sets; "
            f"truth leaves={len(truth.leaves)} pred leaves={len(pred.leaves)}"
        )

    truth_names = set(truth.placements_by_name)
    pred_names = set(pred.placements_by_name)
    shared_names = sorted(truth_names & pred_names)
    truth_only_count = len(truth_names - pred_names)
    pred_only_count = len(pred_names - truth_names)

    raw_top1_exact = 0
    split_top1_exact = 0
    raw_topk_hit = 0
    split_topk_hit = 0
    truth_rank_histogram: Counter[int] = Counter()
    truth_gap_values: list[float] = []
    distal_all: list[float] = []
    pendant_all: list[float] = []
    distal_matched: list[float] = []
    pendant_matched: list[float] = []
    mismatches: list[MismatchDetail] = []
    topk_gap_details: list[TopKGapDetail] = []

    for query_name in shared_names:
        truth_rows = truth.placements_by_name[query_name]
        pred_rows = pred.placements_by_name[query_name]
        if not truth_rows or not pred_rows:
            continue

        truth_top1 = truth_rows[0]
        pred_top1 = pred_rows[0]
        truth_topk_rows = truth_rows[:truth_topk]
        truth_raw_to_rank = {row.edge_num: idx + 1 for idx, row in enumerate(truth_topk_rows)}
        truth_raw_to_ll = {row.edge_num: row.likelihood for row in truth_rows}

        truth_top1_split = truth.split_by_edge.get(truth_top1.edge_num)
        pred_top1_split = pred.split_by_edge.get(pred_top1.edge_num)
        truth_topk_splits = {
            truth.split_by_edge.get(row.edge_num)
            for row in truth_topk_rows
            if row.edge_num in truth.split_by_edge
        }

        raw_exact = pred_top1.edge_num == truth_top1.edge_num
        split_exact = (
            truth_top1_split is not None and
            pred_top1_split is not None and
            pred_top1_split == truth_top1_split
        )
        raw_topk = pred_top1.edge_num in truth_raw_to_rank
        split_topk = pred_top1_split is not None and pred_top1_split in truth_topk_splits

        if raw_exact:
            raw_top1_exact += 1
        else:
            mismatches.append(
                MismatchDetail(
                    query_name=query_name,
                    truth_edge=truth_top1.edge_num,
                    pred_edge=pred_top1.edge_num,
                    truth_likelihood=truth_top1.likelihood,
                    pred_likelihood=pred_top1.likelihood,
                )
            )

        if split_exact:
            split_top1_exact += 1
        if raw_topk:
            raw_topk_hit += 1
        if split_topk:
            split_topk_hit += 1

        if truth_top1.distal_length is not None and pred_top1.distal_length is not None:
            distal_error = abs(truth_top1.distal_length - pred_top1.distal_length)
            distal_all.append(distal_error)
            if raw_exact:
                distal_matched.append(distal_error)

        if truth_top1.pendant_length is not None and pred_top1.pendant_length is not None:
            pendant_error = abs(truth_top1.pendant_length - pred_top1.pendant_length)
            pendant_all.append(pendant_error)
            if raw_exact:
                pendant_matched.append(pendant_error)

        if raw_topk and not raw_exact:
            truth_rank = truth_raw_to_rank[pred_top1.edge_num]
            truth_pred_ll = truth_raw_to_ll[pred_top1.edge_num]
            gap = truth_top1.likelihood - truth_pred_ll
            truth_rank_histogram[truth_rank] += 1
            truth_gap_values.append(gap)
            topk_gap_details.append(
                TopKGapDetail(
                    query_name=query_name,
                    truth_top1_edge=truth_top1.edge_num,
                    pred_edge=pred_top1.edge_num,
                    truth_rank=truth_rank,
                    truth_top1_likelihood=truth_top1.likelihood,
                    truth_pred_edge_likelihood=truth_pred_ll,
                    likelihood_gap=gap,
                )
            )

    return CompareMetrics(
        truth_path=truth_path,
        pred_path=pred_path,
        shared_count=len(shared_names),
        truth_only_count=truth_only_count,
        pred_only_count=pred_only_count,
        raw_top1_exact=raw_top1_exact,
        split_top1_exact=split_top1_exact,
        raw_topk_hit=raw_topk_hit,
        split_topk_hit=split_topk_hit,
        truth_rank_histogram=dict(sorted(truth_rank_histogram.items())),
        avg_truth_gap=mean_or_none(truth_gap_values),
        mean_abs_distal_all=mean_or_none(distal_all),
        mean_abs_pendant_all=mean_or_none(pendant_all),
        mean_abs_distal_matched=mean_or_none(distal_matched),
        mean_abs_pendant_matched=mean_or_none(pendant_matched),
        mismatches=mismatches,
        topk_gap_details=topk_gap_details,
        truth_topk=truth_topk,
    )


def render_compare_report(metrics: CompareMetrics) -> str:
    lines = [
        f"truth file: {metrics.truth_path}",
        f"pred file:  {metrics.pred_path}",
        f"shared query instances: {metrics.shared_count}",
        f"truth-only query instances: {metrics.truth_only_count}",
        f"pred-only query instances: {metrics.pred_only_count}",
        "",
        (
            f"top-1 exact edge match (raw edge_num): "
            f"{metrics.raw_top1_exact}/{metrics.shared_count} "
            f"({format_ratio(metrics.raw_top1_exact, metrics.shared_count)})"
        ),
        (
            f"top-1 exact edge match (tree split):   "
            f"{metrics.split_top1_exact}/{metrics.shared_count} "
            f"({format_ratio(metrics.split_top1_exact, metrics.shared_count)})"
        ),
        (
            f"pred top-1 in truth top-{metrics.truth_topk} (raw edge_num): "
            f"{metrics.raw_topk_hit}/{metrics.shared_count} "
            f"({format_ratio(metrics.raw_topk_hit, metrics.shared_count)})"
        ),
        (
            f"pred top-1 in truth top-{metrics.truth_topk} (tree split):   "
            f"{metrics.split_topk_hit}/{metrics.shared_count} "
            f"({format_ratio(metrics.split_topk_hit, metrics.shared_count)})"
        ),
        (
            "avg truth-top1 minus pred-edge truth-lk gap "
            f"(top-{metrics.truth_topk} but not top-1): {format_float(metrics.avg_truth_gap)}"
        ),
        f"truth rank histogram for pred top-1 when inside truth top-k but not top-1:",
    ]

    if metrics.truth_rank_histogram:
        for rank, count in metrics.truth_rank_histogram.items():
            lines.append(f"  rank {rank}: {count}")
    else:
        lines.append("  (none)")

    lines.extend(
        [
            f"mean abs distal length error (all shared):   {format_float(metrics.mean_abs_distal_all)}",
            f"mean abs pendant length error (all shared):  {format_float(metrics.mean_abs_pendant_all)}",
            f"mean abs distal length error (matched):      {format_float(metrics.mean_abs_distal_matched)}",
            f"mean abs pendant length error (matched):     {format_float(metrics.mean_abs_pendant_matched)}",
            "",
            "top-1 edge mismatches:",
        ]
    )

    if metrics.mismatches:
        for detail in metrics.mismatches[:MAX_MISMATCH_ROWS]:
            lines.append(
                "  "
                f"{detail.query_name}: truth edge={detail.truth_edge}, pred edge={detail.pred_edge}, "
                f"truth ll={detail.truth_likelihood:.12f}, pred ll={detail.pred_likelihood:.12f}"
            )
        remaining = len(metrics.mismatches) - MAX_MISMATCH_ROWS
        if remaining > 0:
            lines.append(f"  ... {remaining} more mismatches omitted")
    else:
        lines.append("  (none)")

    lines.extend(
        [
            "",
            f"pred top-1 inside truth top-{metrics.truth_topk} but not truth top-1:",
        ]
    )

    if metrics.topk_gap_details:
        for detail in metrics.topk_gap_details[:MAX_TOPK_GAP_ROWS]:
            lines.append(
                "  "
                f"{detail.query_name}: truth top1 edge={detail.truth_top1_edge}, "
                f"pred edge={detail.pred_edge}, truth rank of pred={detail.truth_rank}, "
                f"truth top1 ll={detail.truth_top1_likelihood:.12f}, "
                f"truth ll at pred edge={detail.truth_pred_edge_likelihood:.12f}, "
                f"gap={detail.likelihood_gap:.12f}"
            )
        remaining = len(metrics.topk_gap_details) - MAX_TOPK_GAP_ROWS
        if remaining > 0:
            lines.append(f"  ... {remaining} more top-k hits omitted")
    else:
        lines.append("  (none)")

    return "\n".join(lines) + "\n"


def run_mode(
    *,
    dataset_size: str,
    query_id: str,
    query_path: Path,
    truth_jplace: Path,
    output_dir: Path,
    mlipper_binary: Path,
    compare_topk: int,
    mode: str,
) -> BenchmarkModeResult:
    if mode not in BENCH_MODE_ENVS:
        raise SystemExit(f"Unknown mode: {mode}")

    run_dir = output_dir / query_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = run_dir / f"{query_id}_{mode}.log"
    pred_jplace = run_dir / f"{query_id}_{mode}.jplace"
    compare_log = run_dir / f"{query_id}_{mode}.compare.log"

    env = os.environ.copy()
    env.update(BENCH_MODE_ENVS[mode])

    print(f"Running {dataset_size} {mode}", flush=True)
    run_command(
        [
            str(mlipper_binary),
            "--tree-alignment",
            str(REFERENCE_FASTA),
            "--query-alignment",
            str(query_path),
            "--tree",
            str(TREE_PATH),
            "--states",
            "4",
            "--subst-model",
            "GTR",
            "--ncat",
            "4",
            "--alpha",
            "0.3",
            "--pinv",
            "0.0",
            "--freqs",
            "0.25,0.25,0.25,0.25",
            "--rates",
            "1.0,1.0,1.0,1.0,1.0,1.0",
            "--rate-weights",
            "0.25,0.25,0.25,0.25",
            "--jplace-out",
            str(pred_jplace),
        ],
        cwd=ROOT_DIR,
        env=env,
        log_path=run_log,
    )

    metrics = compare_jplace_files(truth_jplace, pred_jplace, compare_topk)
    compare_log.write_text(render_compare_report(metrics), encoding="utf-8")

    result = BenchmarkModeResult(
        dataset_size=dataset_size,
        mode=mode,
        truth_jplace=truth_jplace,
        pred_jplace=pred_jplace,
        run_log=run_log,
        compare_log=compare_log,
        gpu_kernel_ms=parse_ms_metric(run_log, "GPU kernel time"),
        gpu_wall_ms=parse_ms_metric(run_log, "GPU Wall Clock time"),
        precision_mode=parse_precision_mode(run_log),
        metrics=metrics,
    )

    print(
        f"Completed {dataset_size:<8} {mode:<8} runtime={format_ms(result.gpu_wall_ms)} ms "
        f"top-1 acc={format_ratio(metrics.split_top1_exact, metrics.shared_count)} "
        f"top-{compare_topk} acc={format_ratio(metrics.split_topk_hit, metrics.shared_count)}",
        flush=True,
    )
    return result


def print_summary_table(
    datasets: Sequence[str],
    epa_ms_by_dataset: dict[str, float | None],
    results_by_key: dict[tuple[str, str], BenchmarkModeResult],
) -> None:
    print()
    print(
        f"{'dataset':<8} {'epa_ng_s':<12} {'base_s':<12} {'base_t1':<12} "
        f"{'base_t5':<12} {'base_spd':<12} {'fast_s':<12} {'fast_t1':<12} "
        f"{'fast_t5':<12} {'fast_spd':<12}"
    )
    print(
        f"{'-------':<8} {'--------':<12} {'------':<12} {'-------':<12} "
        f"{'-------':<12} {'--------':<12} {'------':<12} {'-------':<12} "
        f"{'-------':<12} {'--------':<12}"
    )
    for dataset in datasets:
        epa_ms = epa_ms_by_dataset.get(dataset)
        baseline = results_by_key.get((dataset, "baseline"))
        fast = results_by_key.get((dataset, "fast"))
        baseline_top1 = (
            format_ratio(baseline.metrics.split_top1_exact, baseline.metrics.shared_count)
            if baseline is not None else "NA"
        )
        baseline_top5 = (
            format_ratio(baseline.metrics.split_topk_hit, baseline.metrics.shared_count)
            if baseline is not None else "NA"
        )
        fast_top1 = (
            format_ratio(fast.metrics.split_top1_exact, fast.metrics.shared_count)
            if fast is not None else "NA"
        )
        fast_top5 = (
            format_ratio(fast.metrics.split_topk_hit, fast.metrics.shared_count)
            if fast is not None else "NA"
        )
        baseline_ms = baseline.gpu_wall_ms if baseline is not None else None
        fast_ms = fast.gpu_wall_ms if fast is not None else None
        print(
            f"{dataset:<8} {format_seconds_from_ms(epa_ms):<12} "
            f"{format_seconds_from_ms(baseline_ms):<12} {baseline_top1:<12} "
            f"{baseline_top5:<12} {format_speedup(epa_ms, baseline_ms):<12} "
            f"{format_seconds_from_ms(fast_ms):<12} {fast_top1:<12} "
            f"{fast_top5:<12} {format_speedup(epa_ms, fast_ms):<12}"
        )


def benchmark_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MLIPPER against EPA-ng and compare jplace outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))))
    parser.add_argument("--data-archive", type=Path, default=Path(os.environ.get("DATA_ARCHIVE", str(DEFAULT_DATA_ARCHIVE))))
    parser.add_argument("--epa-ng-env", default=os.environ.get("EPA_NG_ENV", "base"))
    parser.add_argument("--epa-ng-threads", type=int, default=int(os.environ.get("EPA_NG_THREADS", "48")))
    parser.add_argument("--mlipper-bench-precision", choices=["float", "double"], default=os.environ.get("MLIPPER_BENCH_PRECISION", "float"))
    parser.add_argument("--bench-modes", nargs="+", default=split_env_words("BENCH_MODES", ["baseline", "fast"]))
    parser.add_argument("--datasets", nargs="+", default=split_env_words("BENCH_DATASETS", ["1k"]))
    parser.add_argument("--compare-topk", type=int, default=int(os.environ.get("COMPARE_TOPK", "5")))
    parser.add_argument("--build-dir", default=os.environ.get("BUILD_DIR", "build"))
    parser.add_argument("--epa-ng-bin", type=Path, default=(Path(os.environ["EPA_NG_BIN"]) if os.environ.get("EPA_NG_BIN") else None))
    parser.add_argument("--epa-ng-model", default=os.environ.get("EPA_NG_MODEL", DEFAULT_EPA_NG_MODEL))
    parser.add_argument("--force-regenerate-truth", action="store_true", default=bool_env("FORCE_REGENERATE_TRUTH", False))
    return parser.parse_args(argv)


def compare_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two jplace files.")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--truth-topk", type=int, default=5)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser.parse_args(argv)


def benchmark_main(argv: Sequence[str]) -> int:
    args = benchmark_args(argv)
    if args.epa_ng_threads <= 0:
        raise SystemExit("--epa-ng-threads must be > 0")
    if args.compare_topk <= 0:
        raise SystemExit("--compare-topk must be > 0")

    invalid_datasets = [item for item in args.datasets if item not in DATASETS]
    if invalid_datasets:
        raise SystemExit(f"Unknown dataset(s): {invalid_datasets}")
    invalid_modes = [item for item in args.bench_modes if item not in BENCH_MODE_ENVS]
    if invalid_modes:
        raise SystemExit(f"Unknown benchmark mode(s): {invalid_modes}")

    os.chdir(ROOT_DIR)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_dataset(args.data_archive.resolve())
    mlipper_binary = ensure_binary(args.mlipper_bench_precision, args.build_dir)
    ensure_epa_ng(args.epa_ng_env, None if args.epa_ng_bin is None else args.epa_ng_bin.resolve())

    epa_ms_by_dataset: dict[str, float | None] = {}
    results_by_key: dict[tuple[str, str], BenchmarkModeResult] = {}

    for dataset_size in args.datasets:
        dataset = DATASETS[dataset_size]
        query_id = dataset["query_id"]
        query_path = dataset["query_path"]
        truth_jplace = ensure_truth_jplace(
            query_path=query_path,
            query_id=query_id,
            epa_ng_threads=args.epa_ng_threads,
            epa_ng_model=args.epa_ng_model,
            epa_ng_env=args.epa_ng_env,
            epa_ng_bin=None if args.epa_ng_bin is None else args.epa_ng_bin.resolve(),
            force_regenerate_truth=args.force_regenerate_truth,
        )
        epa_ms_by_dataset[dataset_size] = parse_epa_elapsed_ms(truth_jplace.parent / "run.log")
        for mode in args.bench_modes:
            result = run_mode(
                dataset_size=dataset_size,
                query_id=query_id,
                query_path=query_path,
                truth_jplace=truth_jplace,
                output_dir=output_dir,
                mlipper_binary=mlipper_binary,
                compare_topk=args.compare_topk,
                mode=mode,
            )
            results_by_key[(dataset_size, mode)] = result

    print_summary_table(args.datasets, epa_ms_by_dataset, results_by_key)
    return 0


def compare_main(argv: Sequence[str]) -> int:
    args = compare_args(argv)
    if args.truth_topk <= 0:
        raise SystemExit("--truth-topk must be > 0")
    metrics = compare_jplace_files(args.truth.resolve(), args.pred.resolve(), args.truth_topk)
    report = render_compare_report(metrics)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare-jplace":
        return compare_main(argv[1:])
    return benchmark_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
