#!/usr/bin/env python3
"""
Upload benchmark results to WandB as run.summary for easy model comparison.

Usage:
    # Batch upload: scan multiple directories
    python upload_benchmarks_to_wandb.py --scan-dir outputs/dir1 --scan-dir outputs/dir2

    # Single model: called from run_benchmarks.sh after evaluation
    python upload_benchmarks_to_wandb.py --model-path outputs/.../hfmodel_0050000

    # Dry run: print table without uploading
    python upload_benchmarks_to_wandb.py --scan-dir outputs/ --dry-run
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# Task -> metric_key
TASK_METRICS = {
    "mmlu":          "acc,none",
    "hellaswag":     "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy":      "acc,none",
    "winogrande":    "acc,none",
    "boolq":         "acc,none",
    "piqa":          "acc,none",
    "kmmlu":         "acc,none",
    "gsm8k":         "exact_match,flexible-extract",
    "hendrycks_math": "exact_match,none",
    "humaneval":     "pass@1,create_test",
    "mbpp":          "pass_at_1,none",
}

# Tasks used for eng_avg aggregate
ENG_AVG_TASKS = ["mmlu", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "boolq", "piqa"]

# Stage 1 final iteration — used as base for cumulative_iter in stage2
STAGE1_FINAL_ITER = 440_000

# Group definitions:
#   main: stage1 → ablation (440k) → stage2 (continual from ablation)
#   cooldown: deprecated branch (not adopted)
GROUP_MAP = {
    "stage1":   "main",
    "ablation": "main",
    "stage2":   "main",
    "cooldown": "cooldown",
}


def find_result_jsons(base_path: str) -> list[str]:
    """Find all results_*.json files under a path."""
    return sorted(glob.glob(
        os.path.join(base_path, "**", "results_*.json"),
        recursive=True,
    ))


def extract_model_key(json_path: str) -> str | None:
    """Extract the hfmodel directory path from a result JSON path."""
    path = Path(json_path)
    for parent in path.parents:
        if parent.name.startswith("hfmodel_"):
            return str(parent)
    return None


def parse_stage(model_path: str) -> str:
    """Classify model into stage based on path components."""
    path_lower = model_path.lower()
    if "cooldown" in path_lower:
        return "cooldown"
    if "ablation" in path_lower:
        return "ablation"
    if "stage2" in path_lower:
        return "stage2"
    return "stage1"


def compute_cumulative_iter(stage: str, local_iter: int) -> int:
    """Compute cumulative iteration across training stages.

    Main line: stage1 (0~400k) → ablation (440k) → stage2 (440k + local)
    Cooldown: standalone at 440k
    """
    if stage == "stage2":
        return STAGE1_FINAL_ITER + local_iter
    return local_iter


def make_run_name(model_path: str) -> str:
    """Generate a concise WandB run name from a model path.

    Examples:
        .../alpha_baseline_48L_20251219_095156/hfmodel_0050000 → s1_iter050k
        .../alpha_baseline_48L_ablation_.../hfmodel_0440000    → ablation_iter440k
        .../alpha_baseline_48L_stage2_20260301_.../hfmodel_0025000 → s2_iter025k
        .../alpha_baseline_48L_cooldown_.../hfmodel_0440000    → cooldown_iter440k
    """
    parts = Path(model_path).parts
    hfmodel_name = parts[-1]  # hfmodel_NNNNNN
    run_dir = parts[-2]       # alpha_baseline_48L_...

    # Extract iteration
    iter_match = re.search(r"hfmodel_(\d+)", hfmodel_name)
    iteration = int(iter_match.group(1)) if iter_match else 0
    iter_str = f"{iteration // 1000:03d}k"

    stage = parse_stage(model_path)

    if stage == "stage1":
        return f"s1_iter{iter_str}"
    elif stage == "stage2":
        # Include date for stage2 variants to distinguish runs
        date_match = re.search(r"stage2_(\d{8})_", run_dir)
        date_str = date_match.group(1)[4:8] if date_match else ""  # MMDD
        return f"s2_{date_str}_iter{iter_str}"
    else:
        return f"{stage}_iter{iter_str}"


def extract_run_id(model_path: str) -> str:
    """Extract the full run directory name."""
    parts = Path(model_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def collect_metrics(json_files: list[str]) -> dict[str, float]:
    """Merge multiple result JSONs and extract primary metrics for each task."""
    all_results = {}
    for f in json_files:
        try:
            data = json.load(open(f))
            all_results.update(data.get("results", {}))
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Failed to read {f}: {e}", file=sys.stderr)

    metrics = {}
    for task, metric_key in TASK_METRICS.items():
        if task in all_results and metric_key in all_results[task]:
            metrics[task] = all_results[task][metric_key]

    # Compute eng_avg
    eng_values = [metrics[t] for t in ENG_AVG_TASKS if t in metrics]
    if eng_values:
        metrics["eng_avg"] = sum(eng_values) / len(eng_values)

    return metrics


def discover_models(scan_dirs: list[str]) -> dict[str, list[str]]:
    """Discover all models and their result JSONs from scan directories."""
    models = defaultdict(list)
    for scan_dir in scan_dirs:
        all_jsons = find_result_jsons(scan_dir)
        for json_path in all_jsons:
            model_key = extract_model_key(json_path)
            if model_key:
                models[model_key].append(json_path)
    return dict(models)


def print_table(model_data: list[dict]):
    """Print a formatted comparison table."""
    if not model_data:
        print("No results found.")
        return

    # Determine which task columns have data
    all_task_keys = list(TASK_METRICS.keys()) + ["eng_avg"]
    active_tasks = [t for t in all_task_keys if any(t in m["metrics"] for m in model_data)]

    # Header
    name_width = max(len(m["run_name"]) for m in model_data)
    name_width = max(name_width, 10)

    header = f"{'Model':<{name_width}}  {'Stage':<10}  {'Group':<10}  {'CumIter':>8}"
    for task in active_tasks:
        header += f"  {task:>12}"
    print(header)
    print("─" * len(header))

    # Rows — insert separator between groups
    prev_group = None
    for m in model_data:
        group = m["group"]
        if prev_group is not None and group != prev_group:
            print("─" * len(header))
        prev_group = group

        cum_iter = m["cumulative_iter"]
        cum_str = f"{cum_iter // 1000}k"
        row = f"{m['run_name']:<{name_width}}  {m['stage']:<10}  {m['group']:<10}  {cum_str:>8}"
        for task in active_tasks:
            val = m["metrics"].get(task)
            if val is not None:
                row += f"  {val:>12.4f}"
            else:
                row += f"  {'—':>12}"
        print(row)


def upload_to_wandb(model_data: list[dict], project: str, entity: str | None):
    """Upload metrics to WandB — one run per group, multiple log steps per run.

    This creates connected line plots where:
      X-axis = cumulative_iter, Y-axis = metric value
      Each metric (mmlu, kmmlu, eng_avg, ...) becomes one line.
    """
    try:
        from wandb import init as wandb_init
    except (ImportError, AttributeError) as e:
        print(f"Error: wandb import failed ({e}). Run: pip install wandb", file=sys.stderr)
        sys.exit(1)

    # Group models by their group label
    groups = defaultdict(list)
    for m in model_data:
        groups[m["group"]].append(m)

    for group_name, members in groups.items():
        # Sort by cumulative_iter within group
        members.sort(key=lambda m: m["cumulative_iter"])

        run = wandb_init(
            project=project,
            entity=entity,
            name=group_name,
            config={
                "group": group_name,
                "checkpoints": [m["run_name"] for m in members],
            },
            tags=[group_name],
            reinit=True,
        )

        # Log each checkpoint as a step — creates connected line plot
        for m in members:
            log_data = dict(m["metrics"])
            log_data["cumulative_iter"] = m["cumulative_iter"]
            log_data["stage"] = m["stage"]
            run.log(log_data)
            print(f"  ✓ Logged: {m['run_name']} (iter {m['cumulative_iter']//1000}k)")

        run.finish()
        print(f"  → Run '{group_name}' finished ({len(members)} checkpoints)\n")

    print(f"Done. {len(groups)} run(s) uploaded to project '{project}'.")


def main():
    parser = argparse.ArgumentParser(description="Upload benchmark results to WandB summary")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan-dir", action="append", dest="scan_dirs",
                       help="Directory to scan for eval results (can specify multiple)")
    group.add_argument("--model-path", help="Single model hfmodel directory")

    parser.add_argument("--project", default="alpha-benchmarks", help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity (team/user)")
    parser.add_argument("--dry-run", action="store_true", help="Print table only, no upload")

    args = parser.parse_args()

    # Discover models
    if args.scan_dirs:
        print(f"Scanning: {', '.join(args.scan_dirs)}")
        models = discover_models(args.scan_dirs)
    else:
        model_path = os.path.abspath(args.model_path)
        jsons = find_result_jsons(os.path.join(model_path, "eval_results"))
        if not jsons:
            print(f"No eval results found in {model_path}/eval_results/", file=sys.stderr)
            sys.exit(1)
        models = {model_path: jsons}

    if not models:
        print("No eval results found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(models)} model(s)\n")

    # Build model data
    model_data = []
    for model_path, json_files in sorted(models.items()):
        metrics = collect_metrics(json_files)
        if not metrics:
            print(f"  Skipping {model_path} (no recognized metrics)")
            continue

        iter_match = re.search(r"hfmodel_(\d+)", model_path)
        iteration = int(iter_match.group(1)) if iter_match else 0
        stage = parse_stage(model_path)

        model_data.append({
            "model_path": model_path,
            "run_name": make_run_name(model_path),
            "stage": stage,
            "group": GROUP_MAP.get(stage, "other"),
            "run_id": extract_run_id(model_path),
            "iteration": iteration,
            "cumulative_iter": compute_cumulative_iter(stage, iteration),
            "metrics": metrics,
        })

    # Sort: by group (main first), then cumulative_iter
    group_order = {"main": 0, "cooldown": 1, "other": 2}
    model_data.sort(key=lambda m: (group_order.get(m["group"], 9), m["cumulative_iter"]))

    # Print table
    print_table(model_data)
    print()

    # Upload
    if args.dry_run:
        print("(Dry run — no upload)")
    else:
        upload_to_wandb(model_data, args.project, args.entity)


if __name__ == "__main__":
    main()
