#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REFERENCE_FASTA="$ROOT_DIR/data/neotrop/reference.fasta"
TREE_PATH="$ROOT_DIR/data/neotrop/tree.newick"
QUERY_1K="$ROOT_DIR/data/neotrop/query_1k.fasta"
QUERY_2K="$ROOT_DIR/data/neotrop/query_2k.fasta"
QUERY_5K="$ROOT_DIR/data/neotrop/query_5k.fasta"

COMPARE_TOPK="${COMPARE_TOPK:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/run_sh_benchmarks}"
DATA_ARCHIVE="${DATA_ARCHIVE:-$ROOT_DIR/data/neotrop_runtime_dataset.tar.gz}"
EPA_NG_ENV="${EPA_NG_ENV:-base}"
EPA_NG_THREADS="${EPA_NG_THREADS:-48}"
EPA_NG_MODEL="${EPA_NG_MODEL:-}"
if [[ -z "$EPA_NG_MODEL" ]]; then
    EPA_NG_MODEL='GTR{1.0/1.0/1.0/1.0/1.0/1.0}+FU{0.25/0.25/0.25/0.25}+G4{0.3}'
fi

mkdir -p "$OUTPUT_DIR"

declare -A GPU_WALL_MS_BY_KEY
declare -A TOP1_ACC_BY_KEY
declare -A EPA_NG_MS_BY_DATASET

ensure_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
}

ensure_runtime_dataset() {
    local file
    local missing=0
    local required_files=(
        "$REFERENCE_FASTA"
        "$TREE_PATH"
        "$QUERY_1K"
        "$QUERY_2K"
        "$QUERY_5K"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing=1
            break
        fi
    done

    if [[ "$missing" -eq 0 ]]; then
        return
    fi

    if [[ ! -f "$DATA_ARCHIVE" ]]; then
        echo "Required dataset files are missing." >&2
        echo "Expected archive for automatic extraction: $DATA_ARCHIVE" >&2
        exit 1
    fi

    echo "Extracting runtime dataset from $(basename "$DATA_ARCHIVE")"
    mkdir -p "$ROOT_DIR/data/neotrop"
    tar -xzf "$DATA_ARCHIVE" -C "$ROOT_DIR"

    for file in "${required_files[@]}"; do
        ensure_file "$file"
    done
}

extract_value() {
    local pattern="$1"
    local file="$2"
    local line
    line="$(grep -F -m1 -- "$pattern" "$file" || true)"
    if [[ -z "$line" ]]; then
        printf 'NA'
        return
    fi
    printf '%s' "$line" | sed -E 's/.*= ([0-9.]+) ms/\1/'
}

extract_percent() {
    local pattern="$1"
    local file="$2"
    local line
    line="$(grep -F -m1 -- "$pattern" "$file" || true)"
    if [[ -z "$line" ]]; then
        printf 'NA'
        return
    fi
    printf '%s' "$line" | sed -E 's/.*\(([0-9.]+%)\)/\1/'
}

extract_precision() {
    local file="$1"
    local line
    line="$(grep -m1 '^Precision mode:' "$file" || true)"
    if [[ -z "$line" ]]; then
        printf 'NA'
        return
    fi
    printf '%s' "$line" | awk '{print $3}'
}

format_ms() {
    local value="$1"
    awk -v value="$value" 'BEGIN {
        if (value == "" || value == "NA") print "NA";
        else printf "%.3f", value + 0.0;
    }'
}

format_seconds_from_ms() {
    local value="$1"
    awk -v value="$value" 'BEGIN {
        if (value == "" || value == "NA") print "NA";
        else printf "%.3f", (value + 0.0) / 1000.0;
    }'
}

format_speedup() {
    local reference_ms="$1"
    local candidate_ms="$2"
    awk -v reference_ms="$reference_ms" -v candidate_ms="$candidate_ms" 'BEGIN {
        if (reference_ms == "" || candidate_ms == "" || reference_ms == "NA" || candidate_ms == "NA" || candidate_ms == 0) print "NA";
        else printf "%.2fx", reference_ms / candidate_ms;
    }'
}

extract_epa_elapsed_ms() {
    local file="$1"
    local line
    local seconds
    line="$(grep -F -m1 -- 'INFO Elapsed Time:' "$file" || true)"
    if [[ -z "$line" ]]; then
        printf 'NA'
        return
    fi
    seconds="$(printf '%s' "$line" | sed -E 's/.*: ([0-9.]+)s/\1/')"
    awk -v seconds="$seconds" 'BEGIN { printf "%.3f", (seconds + 0.0) * 1000.0 }'
}

record_epa_runtime() {
    local dataset_size="$1"
    local truth_jplace="$2"
    local truth_log

    truth_log="$(dirname "$truth_jplace")/run.log"
    EPA_NG_MS_BY_DATASET["$dataset_size"]="$(extract_epa_elapsed_ms "$truth_log")"
}

print_summary_table() {
    local dataset
    local epa_ms
    local baseline_ms
    local fast_ms
    local baseline_acc
    local fast_acc
    local baseline_speedup
    local fast_speedup

    echo
    printf '%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n' \
        "dataset" "epa_ng_s" "base_s" "base_acc" "base_spd" "fast_s" "fast_acc" "fast_spd"
    printf '%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n' \
        "-------" "--------" "------" "--------" "--------" "------" "--------" "--------"
    for dataset in 1k 2k 5k; do
        epa_ms="${EPA_NG_MS_BY_DATASET["$dataset"]:-NA}"
        baseline_ms="${GPU_WALL_MS_BY_KEY["$dataset|baseline"]:-NA}"
        fast_ms="${GPU_WALL_MS_BY_KEY["$dataset|fast"]:-NA}"
        baseline_acc="${TOP1_ACC_BY_KEY["$dataset|baseline"]:-NA}"
        fast_acc="${TOP1_ACC_BY_KEY["$dataset|fast"]:-NA}"
        baseline_speedup="$(format_speedup "$epa_ms" "$baseline_ms")"
        fast_speedup="$(format_speedup "$epa_ms" "$fast_ms")"
        printf '%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n' \
            "$dataset" \
            "$(format_seconds_from_ms "$epa_ms")" \
            "$(format_seconds_from_ms "$baseline_ms")" \
            "$baseline_acc" \
            "$baseline_speedup" \
            "$(format_seconds_from_ms "$fast_ms")" \
            "$fast_acc" \
            "$fast_speedup"
    done
}

ensure_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda not found in PATH. Install Miniconda/Anaconda first." >&2
        exit 1
    fi
}

ensure_epa_ng() {
    ensure_conda
    if ! conda run -n "$EPA_NG_ENV" epa-ng --help >/dev/null 2>&1; then
        echo "epa-ng is not available in conda env '$EPA_NG_ENV'." >&2
        echo "Install it with: conda install -y -n $EPA_NG_ENV -c bioconda epa-ng" >&2
        exit 1
    fi
}

ensure_truth_jplace() {
    local query_path="$1"
    local query_id="$2"
    local truth_jplace="$3"
    local truth_dir
    local truth_log

    if [[ -f "$truth_jplace" ]]; then
        return
    fi

    truth_dir="$(dirname "$truth_jplace")"
    truth_log="$truth_dir/run.log"
    mkdir -p "$truth_dir"

    echo "Generating EPA-ng truth for ${query_id}"
    conda run --no-capture-output -n "$EPA_NG_ENV" epa-ng \
        --redo \
        --no-heur \
        --tree "$TREE_PATH" \
        --ref-msa "$REFERENCE_FASTA" \
        --query "$query_path" \
        --threads "$EPA_NG_THREADS" \
        --model "$EPA_NG_MODEL" \
        --no-pre-mask \
        --outdir "$truth_dir" \
        >"$truth_log" 2>&1

    ensure_file "$truth_jplace"
}


run_mode() {
    local dataset_size="$1"
    local query_path="$2"
    local query_id="$3"
    local truth_jplace="$4"
    local mode="$5"

    local run_dir="$OUTPUT_DIR/$query_id"
    local run_log="$run_dir/${query_id}_${mode}.log"
    local pred_jplace="$run_dir/${query_id}_${mode}.jplace"
    local compare_log="$run_dir/${query_id}_${mode}.compare.log"
    local gpu_kernel_ms
    local gpu_wall_ms
    local top1_exact_pct
    local -a mode_env

    mkdir -p "$run_dir"

    case "$mode" in
        baseline)
            mode_env=(
                MLIPPER_FULL_OPT_PASSES=4
                MLIPPER_REFINE_GLOBAL_PASSES=0
                MLIPPER_REFINE_EXTRA_PASSES=0
                MLIPPER_REFINE_DETECT_TOPK=0
                MLIPPER_REFINE_TOPK=0
            )
            ;;
        fast)
            mode_env=(
                MLIPPER_FULL_OPT_PASSES=1
                MLIPPER_REFINE_GLOBAL_PASSES=0
                MLIPPER_REFINE_EXTRA_PASSES=0
                MLIPPER_REFINE_DETECT_TOPK=0
                MLIPPER_REFINE_TOPK=0
            )
            ;;
        *)
            echo "Unknown mode: $mode" >&2
            exit 1
            ;;
    esac

    echo "Running ${dataset_size} ${mode}"
    env "${mode_env[@]}" \
        "$ROOT_DIR/MLIPPER" \
        --tree-alignment "$REFERENCE_FASTA" \
        --query-alignment "$query_path" \
        --tree "$TREE_PATH" \
        --states 4 \
        --subst-model GTR \
        --ncat 4 \
        --alpha 0.3 \
        --pinv 0.0 \
        --freqs 0.25,0.25,0.25,0.25 \
        --rates 1.0,1.0,1.0,1.0,1.0,1.0 \
        --rate-weights 0.25,0.25,0.25,0.25 \
        --jplace-out "$pred_jplace" \
        >"$run_log" 2>&1

    python3 "$ROOT_DIR/scripts/compare_jplace.py" \
        --truth "$truth_jplace" \
        --pred "$pred_jplace" \
        --truth-topk "$COMPARE_TOPK" \
        >"$compare_log"

    gpu_kernel_ms="$(extract_value 'GPU kernel time =' "$run_log")"
    gpu_wall_ms="$(extract_value 'GPU Wall Clock time =' "$run_log")"
    top1_exact_pct="$(extract_percent 'top-1 exact edge match (tree split):' "$compare_log")"
    GPU_WALL_MS_BY_KEY["$dataset_size|$mode"]="$gpu_wall_ms"
    TOP1_ACC_BY_KEY["$dataset_size|$mode"]="$top1_exact_pct"

    printf 'Completed %-8s %-8s runtime=%s ms acc=%s\n' \
        "$dataset_size" \
        "$mode" \
        "$(format_ms "$gpu_wall_ms")" \
        "$top1_exact_pct"
}

ensure_runtime_dataset

TRUTH_1K="$ROOT_DIR/output/runtime_benchmarks/epa_ng_reference/query_1k/epa_result.jplace"
TRUTH_2K="$ROOT_DIR/output/runtime_benchmarks/epa_ng_reference/query_2k/epa_result.jplace"
TRUTH_5K="$ROOT_DIR/output/runtime_benchmarks/epa_ng_reference/query_5k/epa_result.jplace"

echo "Building float binary"
make float

ensure_epa_ng
ensure_truth_jplace "$QUERY_1K" "query_1k" "$TRUTH_1K"
ensure_truth_jplace "$QUERY_2K" "query_2k" "$TRUTH_2K"
ensure_truth_jplace "$QUERY_5K" "query_5k" "$TRUTH_5K"
record_epa_runtime "1k" "$TRUTH_1K"
record_epa_runtime "2k" "$TRUTH_2K"
record_epa_runtime "5k" "$TRUTH_5K"

for mode in baseline fast; do
    run_mode "1k" "$QUERY_1K" "query_1k" "$TRUTH_1K" "$mode"
    run_mode "2k" "$QUERY_2K" "query_2k" "$TRUTH_2K" "$mode"
    run_mode "5k" "$QUERY_5K" "query_5k" "$TRUTH_5K" "$mode"
done

print_summary_table
