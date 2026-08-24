#!/usr/bin/env python3
"""supplement_if_reasoning.py — 기존 full_translate 산출물에 reasoning 소급 부착.

reasoning 소실 수정(2026-08-24) 이전에 생성된 IF 번역 행은 content 만 있다.
소스의 reasoning_content 원문이 존재하므로 **번역으로 충실 소급**한다 —
재번역(레코드당 ~5콜) 대비 학습 턴 reasoning 번역(~1-2콜)만 지출.

멱등: 산출을 results_think.jsonl 에 append, 이미 처리된 uuid 스킵.
완료 시 --finalize 로 results.jsonl 원자 교체(원본은 .pre_think 백업).

사용:
  python3 supplement_if_reasoning.py --results out/r1_a/results.jsonl \
      --seeds seeds_r1.jsonl --workers 24
  python3 supplement_if_reasoning.py --results ... --seeds ... --finalize
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from translate_regen import Client, translate_message


def load_sources(seeds_path):
    src = {}
    with open(seeds_path) as f:
        for line in f:
            r = json.loads(line)
            src[r["uuid"]] = r
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-31b")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()

    results_path = Path(args.results)
    out_path = results_path.with_name("results_think.jsonl")

    rows = [json.loads(l) for l in open(results_path)]
    if args.finalize:
        done = sum(1 for _ in open(out_path))
        assert done == len(rows), f"미완료: {done}/{len(rows)}"
        backup = results_path.with_suffix(".jsonl.pre_think")
        os.replace(results_path, backup)
        os.replace(out_path, results_path)
        print(f"finalized: {len(rows)} rows → {results_path} (백업 {backup})")
        return

    done_uuids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_uuids.add(json.loads(line)["uuid"])

    sources = load_sources(args.seeds)
    todo = [r for r in rows if r["uuid"] not in done_uuids]
    print(f"todo={len(todo)} (done={len(done_uuids)}) workers={args.workers}", flush=True)

    client = Client(args.base_url, args.model)
    lock = threading.Lock()
    stats = {"ok": 0, "skip": 0, "err": 0}

    def work(row):
        try:
            ks = row["metadata"].get("ko_synthesis", {})
            if ks.get("method") != "full_translate" or ks.get("think"):
                with lock:
                    stats["skip"] += 1
                    with open(out_path, "a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                return
            src_row = sources.get(row["metadata"]["source_uuid"])
            if src_row is None:
                raise RuntimeError("source seed 없음")
            src_msgs = [m for m in src_row["messages"]
                        if not (m["role"] == "system" and not (m["content"] or "").strip())]
            tt = row["metadata"]["train_turns"]
            for i, m in enumerate(row["messages"]):
                if (tt[i] and m["role"] == "assistant"
                        and not m.get("reasoning_content")
                        and src_msgs[i].get("reasoning_content")):
                    m["reasoning_content"] = translate_message(
                        client, "assistant", src_msgs[i]["reasoning_content"], None)
            ks["think"] = True
            ks["think_supplemented"] = "2026-08-24"
            with lock:
                stats["ok"] += 1
                with open(out_path, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n = stats["ok"] + stats["skip"]
                if n % 200 == 0:
                    print(f"progress ok={stats['ok']} skip={stats['skip']} "
                          f"err={stats['err']}", flush=True)
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["err"] += 1
                print(f"ERROR uuid={row.get('uuid')}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    print(json.dumps(stats))
    if stats["err"] == 0 and len(done_uuids) + len(todo) == len(rows):
        print("전량 처리 완료 — --finalize 로 교체하세요")


if __name__ == "__main__":
    main()
