#!/bin/bash
#
# Create 50% Subset via Symbolic Links
#
# Creates a subset directory with symlinks to:
# - DCLM: Even-numbered shards only (00, 02, 04, ..., 14) = 50%
# - Korean Web: 100% included
#
# Usage:
#   bash create_50pct_subset.sh
#

set -e

# ==================== Configuration ====================

# Source directory (full dataset)
SOURCE_DIR="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/datasets/intermediate/full"

# Target directory (50% subset with symlinks)
TARGET_DIR="/home/work/Datasets/KORMo_processed/intermediate/subset_50pct"

# DCLM shards to include (even numbers = 50%)
DCLM_SHARDS=(00 02 04 06 08 10 12 14)

# ==================== Validation ====================

echo "================================================================================"
echo "Creating 50% Subset (Symbolic Links)"
echo "================================================================================"
echo "Source: ${SOURCE_DIR}"
echo "Target: ${TARGET_DIR}"
echo "DCLM shards: ${DCLM_SHARDS[*]}"
echo "Korean Web: 100% included"
echo "================================================================================"
echo

# Check source directory exists
if [ ! -d "${SOURCE_DIR}" ]; then
    echo "Error: Source directory not found: ${SOURCE_DIR}"
    exit 1
fi

# Check if target already exists
if [ -d "${TARGET_DIR}" ]; then
    echo "Warning: Target directory already exists: ${TARGET_DIR}"
    read -p "Do you want to remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing directory..."
        rm -rf "${TARGET_DIR}"
    else
        echo "Aborted."
        exit 0
    fi
fi

# ==================== Create Symlinks ====================

echo "Creating target directory..."
mkdir -p "${TARGET_DIR}"

echo
echo "Creating DCLM symlinks (8 shards)..."
for shard in "${DCLM_SHARDS[@]}"; do
    src="${SOURCE_DIR}/dclm_shard_${shard}.jsonl"
    dst="${TARGET_DIR}/dclm_shard_${shard}.jsonl"

    if [ ! -f "${src}" ]; then
        echo "  Error: Source file not found: ${src}"
        exit 1
    fi

    ln -s "${src}" "${dst}"
    echo "  Created: dclm_shard_${shard}.jsonl"
done

echo
echo "Creating Korean Web symlink..."
src="${SOURCE_DIR}/korean_web.jsonl"
dst="${TARGET_DIR}/korean_web.jsonl"

if [ ! -f "${src}" ]; then
    echo "  Error: Source file not found: ${src}"
    exit 1
fi

ln -s "${src}" "${dst}"
echo "  Created: korean_web.jsonl"

# ==================== Verification ====================

echo
echo "================================================================================"
echo "Verification"
echo "================================================================================"

# Count files
file_count=$(find "${TARGET_DIR}" -name "*.jsonl" -type l | wc -l)
echo "Total symlinks created: ${file_count}"

# List files with sizes
echo
echo "Files in subset:"
for f in "${TARGET_DIR}"/*.jsonl; do
    fname=$(basename "$f")
    size=$(du -h "$f" | cut -f1)
    echo "  ${fname}: ${size}"
done

# Calculate total size
echo
total_size=$(du -sh "${TARGET_DIR}" | cut -f1)
echo "Total size (via symlinks): ${total_size}"

echo
echo "================================================================================"
echo "50% Subset Created Successfully!"
echo "================================================================================"
echo "Location: ${TARGET_DIR}"
echo
echo "Next step: Run mmap conversion"
echo "  bash preprocess_kormo_subset.sh 50"
echo "================================================================================"
