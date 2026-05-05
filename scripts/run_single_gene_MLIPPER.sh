#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DEFAULT_DOCKER_IMAGE="${MLIPPER_DOCKER_IMAGE:-wenchiehlo/mlipper-roadies:20260504}"
DEFAULT_GPU_ID="${MLIPPER_GPU_ID:-0}"

usage() {
  cat <<EOF
Usage:
  $SCRIPT_NAME \\
    --ref-msa REF.fa \\
    --query-msa QUERY.fa \\
    --backbone-tree BACKBONE.nwk \\
    --best-model GENE.raxml.bestModel \\
    --out-tree OUT.nwk \\
    [options]

Required:
  --ref-msa PATH              Reference/backbone alignment
  --query-msa PATH            Query alignment to commit into the backbone tree
  --backbone-tree PATH        Backbone/reference Newick tree
  --best-model PATH           Per-gene bestModel file
  --out-tree PATH             Output committed Newick tree

Optional:
  --docker-image IMAGE        Docker image to run
                              Default: $DEFAULT_DOCKER_IMAGE
  --gpu-id INT                GPU id used when --docker-gpus is not provided
                              Default: $DEFAULT_GPU_ID
  --docker-gpus SPEC          Raw Docker --gpus value (overrides --gpu-id)
  --local-spr                 Enable local SPR refinement
                              Default: enabled
  --no-local-spr              Disable local SPR refinement
  --batch-size INT            Batch insert size
                              Default: 5
  --local-spr-radius INT      Local SPR radius
                              Default: 4
  --local-spr-rounds INT      Local SPR rounds
                              Default: 1
  -h, --help                  Show this message

Notes:
  - This wrapper owns the Docker invocation.
  - ROADIES should decide which GPU to pass in.
  - MLIPPER itself reads the bestModel file via --best-model.
EOF
}

die() {
  echo "$SCRIPT_NAME: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing file: $path"
}

abs_existing_path() {
  python3 - "$1" <<'PY'
import os
import sys

path = sys.argv[1]
if not os.path.exists(path):
    raise SystemExit(f"missing path: {path}")
print(os.path.realpath(path))
PY
}

abs_target_path() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
}

common_root_for_paths() {
  python3 - "$@" <<'PY'
import os
import sys

print(os.path.commonpath(sys.argv[1:]))
PY
}

containerize_path() {
  python3 - "$1" "$2" <<'PY'
import os
import sys

path = os.path.realpath(sys.argv[1])
root = os.path.realpath(sys.argv[2])
rel = os.path.relpath(path, root)
if rel.startswith(".."):
    raise SystemExit(f"path {path} escapes mount root {root}")
print("/workspace/job/" + rel.replace(os.sep, "/"))
PY
}

quote_cmd() {
  local piece
  for piece in "$@"; do
    printf '%q ' "$piece"
  done
  printf '\n'
}

ref_msa=""
query_msa=""
backbone_tree=""
best_model=""
out_tree=""
docker_image="$DEFAULT_DOCKER_IMAGE"
gpu_id="$DEFAULT_GPU_ID"
docker_gpus=""
local_spr=1
batch_size=5
local_spr_radius=4
local_spr_rounds=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref-msa)
      ref_msa="${2:-}"
      shift 2
      ;;
    --query-msa)
      query_msa="${2:-}"
      shift 2
      ;;
    --backbone-tree)
      backbone_tree="${2:-}"
      shift 2
      ;;
    --best-model)
      best_model="${2:-}"
      shift 2
      ;;
    --out-tree)
      out_tree="${2:-}"
      shift 2
      ;;
    --docker-image)
      docker_image="${2:-}"
      shift 2
      ;;
    --gpu-id)
      gpu_id="${2:-}"
      shift 2
      ;;
    --docker-gpus)
      docker_gpus="${2:-}"
      shift 2
      ;;
    --local-spr)
      local_spr=1
      shift
      ;;
    --no-local-spr)
      local_spr=0
      shift
      ;;
    --batch-size)
      batch_size="${2:-}"
      shift 2
      ;;
    --local-spr-radius)
      local_spr_radius="${2:-}"
      shift 2
      ;;
    --local-spr-rounds)
      local_spr_rounds="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$ref_msa" ]] || die "--ref-msa is required"
[[ -n "$query_msa" ]] || die "--query-msa is required"
[[ -n "$backbone_tree" ]] || die "--backbone-tree is required"
[[ -n "$best_model" ]] || die "--best-model is required"
[[ -n "$out_tree" ]] || die "--out-tree is required"

require_file "$ref_msa"
require_file "$query_msa"
require_file "$backbone_tree"
require_file "$best_model"

[[ "$gpu_id" =~ ^[0-9]+$ ]] || die "--gpu-id must be a non-negative integer"
[[ "$batch_size" =~ ^[0-9]+$ ]] || die "--batch-size must be a non-negative integer"
[[ "$local_spr_radius" =~ ^[0-9]+$ ]] || die "--local-spr-radius must be a non-negative integer"
[[ "$local_spr_rounds" =~ ^[1-9][0-9]*$ ]] || die "--local-spr-rounds must be >= 1"

mkdir -p "$(dirname "$out_tree")"

ref_msa="$(abs_existing_path "$ref_msa")"
query_msa="$(abs_existing_path "$query_msa")"
backbone_tree="$(abs_existing_path "$backbone_tree")"
best_model="$(abs_existing_path "$best_model")"
out_tree="$(abs_target_path "$out_tree")"

common_root="$(common_root_for_paths "$ref_msa" "$query_msa" "$backbone_tree" "$best_model" "$out_tree")"
ref_msa_in_container="$(containerize_path "$ref_msa" "$common_root")"
query_msa_in_container="$(containerize_path "$query_msa" "$common_root")"
backbone_tree_in_container="$(containerize_path "$backbone_tree" "$common_root")"
best_model_in_container="$(containerize_path "$best_model" "$common_root")"
out_tree_in_container="$(containerize_path "$out_tree" "$common_root")"

gpu_spec="$docker_gpus"
if [[ -z "$gpu_spec" ]]; then
  gpu_spec="device=$gpu_id"
fi

mlipper_args=(
  --tree-alignment "$ref_msa_in_container"
  --query-alignment "$query_msa_in_container"
  --tree "$backbone_tree_in_container"
  --best-model "$best_model_in_container"
  --commit-to-tree "$out_tree_in_container"
)

if [[ "$local_spr" -eq 1 ]]; then
  mlipper_args+=(
    --local-spr
    --batch-insert-size "$batch_size"
    --local-spr-radius "$local_spr_radius"
    --local-spr-rounds "$local_spr_rounds"
  )
fi

docker_cmd=(
  docker run --rm
  --gpus "$gpu_spec"
  --user "$(id -u):$(id -g)"
  -v "$common_root:/workspace/job"
  -w /workspace/job
  --entrypoint /workspace/MLIPPER/MLIPPER
  "$docker_image"
)
docker_cmd+=("${mlipper_args[@]}")

echo "CMD: $(quote_cmd "${docker_cmd[@]}")" >&2
"${docker_cmd[@]}"
