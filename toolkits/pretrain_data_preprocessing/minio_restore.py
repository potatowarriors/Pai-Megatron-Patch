#!/usr/bin/env python3
"""
minio_restore.py — download a MinIO prefix back to local disk (restore counterpart
to /home/work/Datasets/LL_datasets/minio_backup.py, which is upload-only).

Reuses the endpoint + credentials + clock-skew patch from minio_backup.py by
importing it (so secrets are NOT duplicated into this repo file).

Stage-2 use (Nemotron CC-HQ, deleted locally, backed up per
LL_datasets/pretraining/stage2/eng/README.md):

  # dry-run: validate connectivity + show object count / total size (no download)
  python minio_restore.py --dry-run \
    --prefix Opensource-data/Text/LLM/pretraining/stage2/eng/Nemotron-CC-HQ-actual \
    --dest /home/work/Datasets/LL_datasets/pretraining/stage2/eng/Nemotron-CC-HQ-actual

  # full restore (resumable: skips files already present with matching size)
  python minio_restore.py --workers 16 \
    --prefix Opensource-data/Text/LLM/pretraining/stage2/eng/Nemotron-CC-HQ-actual \
    --dest /home/work/Datasets/LL_datasets/pretraining/stage2/eng/Nemotron-CC-HQ-actual
"""

import argparse
import fnmatch
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse client config + credentials + time-skew patch from the backup script.
_BACKUP_DIR = "/home/work/Datasets/LL_datasets"
if _BACKUP_DIR not in sys.path:
    sys.path.insert(0, _BACKUP_DIR)
import minio_backup  # noqa: E402  (provides MINIO_CONFIG, BUCKET_NAME, get_s3_client, _patch_botocore_time)


def list_objects(s3, bucket, prefix, pattern):
    """Return [(key, size), ...] under prefix whose basename matches pattern."""
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if pattern and not fnmatch.fnmatch(os.path.basename(key), pattern):
                continue
            out.append((key, obj["Size"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="remote key prefix (under the bucket)")
    ap.add_argument("--dest", required=True, help="local destination directory")
    ap.add_argument("--pattern", default="*", help="basename glob filter (default '*')")
    ap.add_argument("--bucket", default=minio_backup.BUCKET_NAME, help="bucket (default vc-data)")
    ap.add_argument("--workers", type=int, default=16, help="parallel download workers")
    ap.add_argument("--dry-run", action="store_true", help="list + total size only, no download")
    ap.add_argument("--flatten", action="store_true",
                    help="store all files directly in --dest (basename only) instead of mirroring "
                         "the key path below --prefix")
    args = ap.parse_args()

    minio_backup._patch_botocore_time(minio_backup.MINIO_CONFIG["endpoint_url"])
    s3 = minio_backup.get_s3_client()

    print(f"bucket={args.bucket}  prefix={args.prefix}", flush=True)
    print(f"listing objects...", flush=True)
    objs = list_objects(s3, args.bucket, args.prefix, args.pattern)
    total_bytes = sum(sz for _, sz in objs)
    print(f"  {len(objs):,} objects, {total_bytes/1024**4:.3f} TB ({total_bytes/1024**3:.1f} GB)", flush=True)
    if objs:
        for key, sz in objs[:3]:
            print(f"    e.g. {key}  ({sz/1024**2:.1f} MB)", flush=True)

    if args.dry_run:
        print("dry-run: not downloading.", flush=True)
        return
    if not objs:
        print("nothing to download.", flush=True)
        return

    os.makedirs(args.dest, exist_ok=True)
    prefix_norm = args.prefix.rstrip("/") + "/"

    def local_path_for(key):
        if args.flatten:
            return os.path.join(args.dest, os.path.basename(key))
        rel = key[len(prefix_norm):] if key.startswith(prefix_norm) else os.path.basename(key)
        return os.path.join(args.dest, rel)

    state = {"done": 0, "skipped": 0, "bytes": 0, "failed": 0}
    lock = threading.Lock()
    t0 = time.time()

    def fetch(key, size):
        dst = local_path_for(key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # resumable skip: present with matching size
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            with lock:
                state["skipped"] += 1
                state["bytes"] += size
            return
        tmp = dst + ".part"
        s3.download_file(args.bucket, key, tmp)
        os.replace(tmp, dst)
        with lock:
            state["done"] += 1
            state["bytes"] += size

    print(f"downloading {len(objs):,} files → {args.dest}  ({args.workers} workers)", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, k, sz): k for k, sz in objs}
        last = t0
        for i, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            try:
                fut.result()
            except Exception as e:
                with lock:
                    state["failed"] += 1
                print(f"  [FAIL] {key}: {e}", flush=True)
            now = time.time()
            if now - last >= 30 or i == len(objs):
                el = now - t0
                rate = state["bytes"] / el / 1024**2 if el > 0 else 0
                print(f"  [{el:6.0f}s] {i:,}/{len(objs):,}  "
                      f"done={state['done']:,} skip={state['skipped']:,} fail={state['failed']:,}  "
                      f"{state['bytes']/1024**3:.1f} GB  {rate:.1f} MB/s", flush=True)
                last = now

    el = time.time() - t0
    print(f"\nDONE restore in {el/3600:.2f}h", flush=True)
    print(f"  downloaded={state['done']:,}  skipped={state['skipped']:,}  failed={state['failed']:,}", flush=True)
    # verify count on disk
    n_local = sum(
        1 for root, _, files in os.walk(args.dest) for f in files
        if fnmatch.fnmatch(f, args.pattern) and not f.endswith(".part")
    )
    print(f"  local files matching '{args.pattern}': {n_local:,} (remote: {len(objs):,})", flush=True)
    if state["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
