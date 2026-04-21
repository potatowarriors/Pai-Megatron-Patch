#!/usr/bin/env python3
"""
wandb_relay.py — Relay train.log metrics to wandb.

Parses Megatron-LM training log lines and uploads metrics to wandb.
Creates a new run and backfills all data from train.log, then switches
to tail-f mode for real-time relay.

Optionally merges exported data (max_attention_logit, lm loss validation)
from a previous run via --export-path.

Usage:
    WANDB_API_KEY=<key> python wandb_relay.py \
        --log-path /path/to/train.log \
        --project alpha-pretraining \
        --entity project-alpha-banana \
        --export-path /tmp/wandb_export.json \
        --run-name "baseline_48L_stage2_2_stage2_20260330_160918"
"""

import argparse
import json
import os
import re
import signal
import sys
import time

import wandb

# Regex for iteration lines:
#  [2026-04-07 19:34:18] iteration   268709/  400000 | key: value | ...
ITER_RE = re.compile(
    r"\[.*?\]\s*iteration\s+(\d+)\s*/\s*(\d+)\s*\|(.+)"
)

# Regex for key: value pairs (value can be int, float, or scientific notation)
KV_RE = re.compile(
    r"^\s*([^:]+?)\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)

# Megatron's training_log() uses different key names for wandb vs console.
# Only include keys that the original training actually logged to wandb.
# Map: console log key → (wandb key, value transform)
#   transform: None = pass through, "ms_to_s" = divide by 1000, "float" = cast to float
METRIC_MAP = {
    "lm loss":                          ("lm loss", None),
    "load_balancing_loss":              ("load_balancing_loss", None),
    "learning rate":                    ("learning-rate", None),
    "grad norm":                        ("grad-norm", None),
    "loss scale":                       ("loss-scale", None),
    "global batch size":                ("batch-size", None),
    "elapsed time per iteration (ms)":  ("iteration-time", "ms_to_s"),
    "throughput per GPU (TFLOP/s/GPU)": ("throughput", None),
    "consumed samples":                 ("samples vs steps", "float"),
}

# Console keys to skip (not logged by original training to wandb)
SKIP_KEYS = {
    "number of skipped iterations",
    "number of nan iterations",
}


def parse_iteration_line(line: str) -> tuple[int, int, dict] | None:
    """Parse a single iteration log line.

    Returns (iteration, total_iterations, metrics_dict) or None.
    """
    m = ITER_RE.search(line)
    if not m:
        return None

    iteration = int(m.group(1))
    total_iterations = int(m.group(2))
    rest = m.group(3)

    metrics = {}
    for part in rest.split("|"):
        kv = KV_RE.match(part)
        if not kv:
            continue

        raw_key = kv.group(1).strip()
        value_str = kv.group(2)

        # Skip keys not logged by original training
        if raw_key in SKIP_KEYS:
            continue

        # Look up mapping
        mapping = METRIC_MAP.get(raw_key)
        if mapping is None:
            # Unknown key — pass through with original name
            wandb_key = raw_key
            transform = None
        else:
            wandb_key, transform = mapping

        # Parse value
        if "." in value_str or "e" in value_str.lower():
            value = float(value_str)
        else:
            value = int(value_str)

        # Apply transform
        if transform == "ms_to_s":
            value = value / 1000.0
        elif transform == "float":
            value = float(value)

        metrics[wandb_key] = value

    return iteration, total_iterations, metrics


def load_export_data(path: str) -> dict:
    """Load exported data from a previous run (max_attention_logit, validation, config)."""
    with open(path, "r") as f:
        data = json.load(f)

    # Convert string keys to int for step lookup
    result = {"max_attention_logit": {}, "lm_loss_validation": {}, "config": {}}

    for step_str, val in data.get("max_attention_logit", {}).items():
        result["max_attention_logit"][int(step_str)] = val

    for step_str, val in data.get("lm_loss_validation", {}).items():
        result["lm_loss_validation"][int(step_str)] = val

    result["config"] = data.get("config", {})
    return result


def tail_f(f, poll_interval: float = 1.0):
    """Generator that yields new lines from a file, polling for new data."""
    while True:
        line = f.readline()
        if line:
            yield line
        else:
            time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Relay train.log to wandb")
    parser.add_argument("--log-path", required=True, help="Path to train.log")
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", required=True, help="wandb entity/team name")
    parser.add_argument("--run-id", default=None, help="wandb run ID (omit for auto)")
    parser.add_argument("--run-name", default=None, help="wandb run display name")
    parser.add_argument("--export-path", default=None,
                        help="Path to exported JSON from previous run")
    parser.add_argument("--start-step", type=int, default=0,
                        help="Skip iterations <= this step")
    parser.add_argument("--poll-interval", type=float, default=1.0,
                        help="Seconds between polls in tail mode (default: 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        print(f"ERROR: Log file not found: {args.log_path}", file=sys.stderr)
        sys.exit(1)

    # --- Load export data ---
    export_data = None
    if args.export_path:
        if not os.path.exists(args.export_path):
            print(f"ERROR: Export file not found: {args.export_path}", file=sys.stderr)
            sys.exit(1)
        export_data = load_export_data(args.export_path)
        mal_count = len(export_data["max_attention_logit"])
        val_count = len(export_data["lm_loss_validation"])
        print(f"Loaded export: {mal_count} max_attention_logit, {val_count} validation points")

    # --- Initialize wandb ---
    init_kwargs = dict(
        project=args.project,
        entity=args.entity,
    )
    if args.run_id:
        init_kwargs["id"] = args.run_id
        init_kwargs["resume"] = "allow"
    if args.run_name:
        init_kwargs["name"] = args.run_name
    if export_data and export_data["config"]:
        init_kwargs["config"] = export_data["config"]

    print(f"Creating wandb run in {args.entity}/{args.project}...")
    run = wandb.init(**init_kwargs)
    print(f"Run ID: {run.id}, URL: {run.url}")

    last_step = args.start_step

    # --- Graceful shutdown ---
    shutdown = False

    def handle_signal(signum, frame):
        nonlocal shutdown
        print(f"\nReceived signal {signum}, finishing wandb run...")
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # --- Process log file ---
    logged_count = 0
    skipped_count = 0
    export_injected = 0

    with open(args.log_path, "r") as f:
        print(f"Reading {args.log_path}...")

        # Phase 1: Backfill — use readline() to avoid Python's read-ahead buffer
        while not shutdown:
            line = f.readline()
            if not line:
                break
            parsed = parse_iteration_line(line)
            if parsed is None:
                continue

            iteration, total_iterations, metrics = parsed
            if iteration <= last_step:
                skipped_count += 1
                continue

            # Inject export data if available for this step
            if export_data:
                mal_val = export_data["max_attention_logit"].get(iteration)
                if mal_val is not None:
                    metrics["max_attention_logit"] = mal_val
                    export_injected += 1
                val_val = export_data["lm_loss_validation"].get(iteration)
                if val_val is not None:
                    metrics["lm loss validation"] = val_val

            wandb.log(metrics, step=iteration)
            logged_count += 1

            if logged_count % 1000 == 0:
                print(f"  Backfill: logged {logged_count} iterations "
                      f"(current: {iteration}, export injected: {export_injected})")

        if not shutdown:
            print(f"Backfill complete: skipped {skipped_count}, logged {logged_count}, "
                  f"export injected {export_injected}.")
            print(f"Switching to tail mode (poll every {args.poll_interval}s)...")

            # Phase 2: Tail — poll for new lines
            for line in tail_f(f, args.poll_interval):
                if shutdown:
                    break
                parsed = parse_iteration_line(line)
                if parsed is None:
                    continue

                iteration, total_iterations, metrics = parsed
                wandb.log(metrics, step=iteration)
                logged_count += 1

                if logged_count % 100 == 0:
                    print(f"  Live: iteration {iteration}/{total_iterations} "
                          f"(total logged: {logged_count})")

                # Detect training completion
                if total_iterations > 0 and iteration >= total_iterations:
                    print(f"Training complete at iteration {iteration}/{total_iterations}.")
                    break

    print(f"Total iterations logged: {logged_count}")
    wandb.finish()
    print("wandb run finished.")


if __name__ == "__main__":
    main()
