#!/usr/bin/env bash
# Apply the vendored custom edits to the Megatron-LM-251125 and sglang-v0.5.2
# submodules after a fresh `git clone --recurse-submodules` (or `git pull` +
# `git submodule update --init`).
#
# WHY THIS EXISTS:
#   The submodule origins point at upstream (NVIDIA/Megatron-LM, sgl-project/sglang)
#   which we cannot push to. The parent repo only records each submodule's commit
#   SHA, so working-tree edits (Alpha custom features #1/#2/#3 in CLAUDE.md, plus
#   the sglang alpha model) are NOT carried by a normal clone/pull. This script
#   re-applies them from the .patch files committed alongside it.
#
# USAGE (from anywhere):
#   bash backends/submodule_patches/apply_submodule_patches.sh
#
# Idempotency: if a patch is already applied the script reports it and skips,
# rather than failing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# submodule_path : expected_base_sha : patch_file
TARGETS=(
  "backends/megatron/Megatron-LM-251125:a6d86a6da:Megatron-LM-251125.patch"
  "backends/sglang/sglang-v0.5.2:b0d25e72c:sglang-v0.5.2.patch"
)

apply_one() {
  local sub_rel="$1" base_sha="$2" patch_file="$3"
  local sub_dir="$REPO_ROOT/$sub_rel"
  local patch_path="$SCRIPT_DIR/$patch_file"

  echo "==> $sub_rel"
  if [[ ! -d "$sub_dir/.git" && ! -f "$sub_dir/.git" ]]; then
    echo "    ERROR: submodule not initialized. Run: git submodule update --init $sub_rel" >&2
    return 1
  fi
  if [[ ! -f "$patch_path" ]]; then
    echo "    ERROR: patch not found: $patch_path" >&2
    return 1
  fi

  local head_sha
  head_sha="$(git -C "$sub_dir" rev-parse --short HEAD)"
  if [[ "$head_sha" != "$base_sha"* && "$base_sha" != "$head_sha"* ]]; then
    echo "    WARNING: submodule HEAD is $head_sha but patch was generated against $base_sha." \
         "Apply may fuzz/fail; proceeding with --3way." >&2
  fi

  # Already applied? (reverse-check succeeds => changes are present)
  if git -C "$sub_dir" apply --reverse --check "$patch_path" >/dev/null 2>&1; then
    echo "    already applied — skipping."
    return 0
  fi

  if git -C "$sub_dir" apply --check "$patch_path" >/dev/null 2>&1; then
    git -C "$sub_dir" apply "$patch_path"
    echo "    applied cleanly."
  else
    echo "    clean apply failed; retrying with --3way ..."
    git -C "$sub_dir" apply --3way "$patch_path"
    echo "    applied via --3way."
  fi
}

for t in "${TARGETS[@]}"; do
  IFS=':' read -r sub_rel base_sha patch_file <<< "$t"
  apply_one "$sub_rel" "$base_sha" "$patch_file"
done

echo "All submodule patches processed."
