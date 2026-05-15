#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_DOCKER_IMAGE="${MLIPPER_DOCKER_IMAGE:-wenchiehlo/mlipper-roadies:latest}"
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
  --docker-image IMAGE        Docker image to run (docker mode only)
                              Default: $DEFAULT_DOCKER_IMAGE
  --gpu-id INT                GPU id used when --docker-gpus is not provided
                              Default: $DEFAULT_GPU_ID
  --docker-gpus SPEC          Raw Docker --gpus value (overrides --gpu-id)
  --no-docker                 Run local MLIPPER binary instead of Docker
  --local-mlipper PATH        Local MLIPPER binary path (defaults to REPO_ROOT/MLIPPER)
  --no-local-spr              Disable local SPR refinement
                              Default: local SPR is enabled unless this flag is set
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
  - In non-docker mode, setup_host.sh is used to fetch/check libpll requirements.
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

detect_pll_lib_dir() {
  local multiarch=""
  local candidate=""
  if [[ -n "${PLL_LIB_DIR:-}" ]]; then
    echo "$PLL_LIB_DIR"
    return
  fi
  for candidate in /usr/local/lib /usr/local/lib64; do
    if [[ -e "$candidate/libpll.so" || -e "$candidate/libpll.a" || -e "$candidate/libpll.so.0" ]]; then
      echo "$candidate"
      return
    fi
  done
  if command -v gcc >/dev/null 2>&1; then
    multiarch="$(gcc -print-multiarch 2>/dev/null || true)"
  fi
  if [[ -z "$multiarch" ]] && command -v dpkg-architecture >/dev/null 2>&1; then
    multiarch="$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || true)"
  fi
  if [[ -n "$multiarch" && -d "/usr/lib/$multiarch" ]]; then
    echo "/usr/lib/$multiarch"
    return
  fi
  echo "/usr/lib/x86_64-linux-gnu"
}

detect_pll_inc_dir() {
  local candidate=""
  if [[ -n "${PLL_INC_DIR:-}" ]]; then
    echo "$PLL_INC_DIR"
    return
  fi
  for candidate in /usr/local/include /usr/include; do
    if [[ -f "$candidate/libpll/pll.h" ]]; then
      echo "$candidate"
      return
    fi
  done
  echo "/usr/include"
}

ensure_host_libs_or_install() {
  local setup_script="$REPO_ROOT/install/setup_host.sh"
  local force_build="$1"
  local pll_inc_dir
  local pll_lib_dir
  local have_header=0
  local have_lib=0
  pll_inc_dir="$(detect_pll_inc_dir)"
  pll_lib_dir="$(detect_pll_lib_dir)"

  if [[ -f "$pll_inc_dir/libpll/pll.h" ]]; then
    have_header=1
  fi
  if [[ -e "$pll_lib_dir/libpll.so" || -e "$pll_lib_dir/libpll.a" || -e "$pll_lib_dir/libpll.so.0" ]]; then
    have_lib=1
  fi

  if [[ "$have_header" -eq 1 && "$have_lib" -eq 1 && "$force_build" -eq 0 ]]; then
    return
  fi

  if [[ "$force_build" -eq 1 && -x "$local_mlipper" ]]; then
    return
  fi

  if [[ ! -x "$setup_script" ]]; then
    die "missing or non-executable: $setup_script"
  fi

  echo "Auto-fixing host dependencies via setup_host.sh" >&2
  if [[ "$force_build" -eq 1 ]]; then
    bash "$setup_script"
  else
    bash "$setup_script" --skip-mlipper
  fi
}

ensure_local_binary() {
  if [[ ! -x "$local_mlipper" ]]; then
    die "local MLIPPER binary missing or not executable: $local_mlipper"
  fi
}

ref_msa=""
query_msa=""
backbone_tree=""
best_model=""
out_tree=""
docker_image="$DEFAULT_DOCKER_IMAGE"
gpu_id="$DEFAULT_GPU_ID"
docker_gpus=""
local_mlipper="$REPO_ROOT/MLIPPER"
use_docker=1
# Keep local SPR enabled by default in this wrapper even though the
# underlying MLIPPER binary defaults it to disabled unless --local-spr is set.
local_spr_enabled=1
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
    --no-docker)
      use_docker=0
      shift
      ;;
    --local-mlipper)
      local_mlipper="${2:-}"
      shift 2
      ;;
    --no-local-spr)
      local_spr_enabled=0
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

if [[ "$use_docker" -eq 1 ]]; then
  gpu_spec="$docker_gpus"
  if [[ -z "$gpu_spec" ]]; then
    gpu_spec="device=$gpu_id"
  fi

  common_root="$(common_root_for_paths "$ref_msa" "$query_msa" "$backbone_tree" "$best_model" "$out_tree")"
  ref_msa_in_container="$(containerize_path "$ref_msa" "$common_root")"
  query_msa_in_container="$(containerize_path "$query_msa" "$common_root")"
  backbone_tree_in_container="$(containerize_path "$backbone_tree" "$common_root")"
  best_model_in_container="$(containerize_path "$best_model" "$common_root")"
  out_tree_in_container="$(containerize_path "$out_tree" "$common_root")"
else
  local_mlipper="$(abs_target_path "$local_mlipper")"
  if [[ -x "$local_mlipper" ]]; then
    ensure_host_libs_or_install 0
  else
    ensure_host_libs_or_install 1
  fi
  ensure_local_binary
fi

if [[ "$use_docker" -eq 1 ]]; then
  mlipper_args=(
    --tree-alignment "$ref_msa_in_container"
    --query-alignment "$query_msa_in_container"
    --tree "$backbone_tree_in_container"
    --best-model "$best_model_in_container"
    --commit-to-tree "$out_tree_in_container"
  )
else
  mlipper_args=(
    --tree-alignment "$ref_msa"
    --query-alignment "$query_msa"
    --tree "$backbone_tree"
    --best-model "$best_model"
    --commit-to-tree "$out_tree"
  )
  mlipper_args+=(
    --gpu-id "$gpu_id"
  )
fi

if [[ "$local_spr_enabled" -eq 1 ]]; then
  mlipper_args+=(
    --local-spr
    --batch-insert-size "$batch_size"
    --local-spr-radius "$local_spr_radius"
    --local-spr-rounds "$local_spr_rounds"
  )
fi

if [[ "$use_docker" -eq 1 ]]; then
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
else
  local_cmd=("$local_mlipper")
  local_cmd+=("${mlipper_args[@]}")
fi

if [[ "$use_docker" -eq 1 ]]; then
  echo "CMD: $(quote_cmd "${docker_cmd[@]}")" >&2
  "${docker_cmd[@]}"
else
  echo "CMD: $(quote_cmd "${local_cmd[@]}")" >&2
  "${local_cmd[@]}"
fi
