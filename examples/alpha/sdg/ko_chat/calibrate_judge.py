#!/usr/bin/env python3
"""calibrate_judge.py — OxAlpha 독립 심판으로 Gemma 셀프심판 캘리브레이션.

배경 (2026-08-25 사용자 결정): OxAlpha 는 1,000 요청/일 상한이라 생성·전수 심판에는
못 쓴다. 하루 ~900건을 트랙 B(Gemma 생성 + Gemma 심판) 표본의 **재심판**에 써서
Gemma 심판의 관대함을 정량화하고 export 임계를 보정한다 — 3,600건 판정이
B 100k+ 전체의 필터 품질을 좌우하므로 레버리지가 가장 크다.

같은 4축 루브릭(korean_naturalness/helpfulness/factuality/coherence, 1~5)을 그대로 쓴다.
산출: out/judge_calib/<date>.jsonl — {record_uuid, axes, gemma, ox, ox_reasoning}
멱등: 이미 재심판한 uuid 는 건너뜀. 일일 상한 429 감지 시 즉시 종료(exit 0).

사용 (sub1, .env source 후):
  PYTHONPATH=$K $VENV calibrate_judge.py --n 900 --concurrency 8
  PYTHONPATH=$K $VENV calibrate_judge.py --report      # 누적 요약만
"""
import argparse
import glob
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from translate_regen import Client, DailyCapExceeded, extract_json_obj

HERE = Path(__file__).resolve().parent
DIMS = ["korean_naturalness", "helpfulness", "factuality", "coherence"]
RUBRIC = """\
아래 한국어 대화에서 마지막 어시스턴트 응답을 4개 축으로 평가하세요. 각 축 1~5점.

korean_naturalness — 번역투 없이 자연스러운 한국어인가. 존댓말 규약을 지켰는가.
  5 완전히 자연스러운 존댓말 / 4 사소한 어색함 한 곳 / 3 번역투·부자연스러움이 눈에 띔 / 2 번역투 심하거나 존댓말 위반 / 1 한국어가 아니거나 문장 불성립
helpfulness — 사용자의 실제 요구를 실질적으로 해결하는가. 두루뭉술한 일반론 감점.
  5 구체적·실행 가능 / 4 도움되나 한 부분 얕음 / 3 일반론 수준 / 2 요구를 비껴감 / 1 도움 안 됨
factuality — 사실 오류·환각이 있는가. 특히 한국 제도·법률·수치.
  5 오류 없음(또는 불확실성 정직 표시) / 4 사소한 부정확 한 곳 / 3 검증 필요 주장 다수 / 2 명백한 사실 오류 / 1 핵심이 지어낸 정보
coherence — 멀티턴이면 흐름과 일관되는가. 앞 답변과 모순 없는가.
  5 완전 일관(단일턴이면 질문에 정확 대응) / 4 일관하나 반복 다소 / 3 부분적으로 흐름 놓침 / 2 앞 대화와 모순 / 1 맥락 무시

엄격하게 채점하세요. 특히 factuality 는 한국 제도·법률 세부를 아는 전문가 기준으로,
확인 불가한 구체 수치·조항은 감점합니다.

<대화>
{conv}
</대화>

JSON 으로만 응답: {{"korean_naturalness": {{"score": n, "reasoning": "..."}}, "helpfulness": {{...}}, "factuality": {{...}}, "coherence": {{...}}}}"""


def _d(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return {}
    return {}


def _score(judge, dim):
    v = _d(judge).get(dim)
    if isinstance(v, dict):
        v = v.get("score")
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_candidates(globs):
    frames = []
    for g in globs:
        for f in sorted(glob.glob(g, recursive=True)):
            try:
                df = pd.read_parquet(f)
            except Exception:  # noqa: BLE001
                continue
            if {"assistant_turn", "judge", "record_uuid"} <= set(df.columns):
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    ok = df["assistant_turn"].apply(lambda v: bool(_d(v).get("content"))) & \
        df["judge"].apply(lambda v: _score(v, "helpfulness") is not None)
    return df[ok].drop_duplicates("record_uuid")


def render_conv(row):
    parts = [f"사용자: {_d(row['user_turn']).get('message', '')}",
             f"어시스턴트: {_d(row['assistant_turn']).get('content', '')}"]
    for u, a in (("followup_user", "assistant_turn_2"), ("followup_user_2", "assistant_turn_3")):
        um, am = _d(row.get(u)).get("message"), _d(row.get(a)).get("content")
        if um and am:
            parts += [f"사용자: {um}", f"어시스턴트: {am}"]
    return "\n".join(parts)


def report(out_dir):
    rows = []
    for f in sorted(glob.glob(str(out_dir / "*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    if not rows:
        print("no calibration rows")
        return
    print(f"calibration rows: {len(rows)}")
    print(f"{'dim':20s} {'gemma':>6s} {'ox':>6s} {'diff':>6s} {'ox<gem':>7s} {'gem<4':>6s} {'ox<4':>6s}")
    for d in DIMS:
        pairs = [(r["gemma"][d], r["ox"][d]) for r in rows
                 if r["gemma"].get(d) is not None and r["ox"].get(d) is not None]
        if not pairs:
            continue
        g = [p[0] for p in pairs]; o = [p[1] for p in pairs]
        print(f"{d:20s} {statistics.mean(g):6.2f} {statistics.mean(o):6.2f} "
              f"{statistics.mean(o) - statistics.mean(g):+6.2f} "
              f"{sum(1 for a, b in pairs if b < a) / len(pairs):7.0%} "
              f"{sum(1 for x in g if x < 4) / len(g):6.0%} {sum(1 for x in o if x < 4) / len(o):6.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", nargs="*", default=[
        str(HERE / "artifacts/ko_chat_b_r1/**/*.parquet"),
        str(HERE / "artifacts/ko_chat_b_r2/**/*.parquet"),
    ])
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--backend", choices=["openrouter", "local"], default="openrouter",
                    help="local = 같은 엄격 루브릭을 Gemma(vLLM)에 — OxAlpha 프록시 가설 검증용")
    ap.add_argument("--match", default=None,
                    help="이 디렉토리의 jsonl 에 있는 uuid 만 대상 (백엔드 간 동일 표본 비교)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or (HERE / ("out/judge_calib" if args.backend == "openrouter"
                                            else "out/judge_calib_local")))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(out_dir)
        return

    done = set()
    for f in glob.glob(str(out_dir / "*.jsonl")):
        done.update(json.loads(l)["record_uuid"] for l in open(f))

    cands = load_candidates(args.artifacts)
    cands = cands[~cands["record_uuid"].astype(str).isin(done)]
    if args.match:
        want = set()
        for f in glob.glob(str(Path(args.match) / "*.jsonl")):
            want.update(json.loads(l)["record_uuid"] for l in open(f))
        cands = cands[cands["record_uuid"].astype(str).isin(want)]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rng = random.Random(today)
    idx = list(range(len(cands)))
    rng.shuffle(idx)
    batch = cands.iloc[idx[: args.n]]
    print(f"candidates={len(cands)} (done={len(done)}) → batch={len(batch)} date={today} backend={args.backend}", flush=True)
    if len(batch) == 0:
        return

    if args.backend == "openrouter":
        key = os.environ.get("OPENROUTER")
        if not key:
            raise SystemExit("환경변수 OPENROUTER 없음")
        client = Client("https://openrouter.ai/api/v1", "stealth/ox-alpha", timeout=900,
                        api_key=key, retries=3, name="openrouter",
                        extra_payload={"reasoning": {"effort": args.effort}})
    else:
        client = Client(args.base_url, "gemma-4-31b", timeout=600, retries=3, name="local")
    out_path = out_dir / f"{today}.jsonl"
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "cap": False}

    def work(row):
        if stats["cap"]:
            return
        try:
            raw = client.chat([{"role": "user", "content": RUBRIC.format(conv=render_conv(row))}],
                              max_tokens=12288 if client.name == "openrouter" else 2048,
                              temperature=0.2)
            obj = extract_json_obj(raw)
            ox = {d: _score(obj, d) for d in DIMS}
            if any(v is None for v in ox.values()):
                raise ValueError(f"score parse: {raw[:100]!r}")
            rec = {
                "record_uuid": str(row["record_uuid"]),
                "axes": {k: row.get(k) for k in ("task_type", "domain", "persona", "turn_shape")},
                "gemma": {d: _score(row["judge"], d) for d in DIMS},
                "ox": ox,
                "ox_reasoning": {d: (obj.get(d) or {}).get("reasoning", "") if isinstance(obj.get(d), dict) else ""
                                 for d in DIMS},
                "date": today,
            }
            with lock:
                stats["ok"] += 1
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if stats["ok"] % 50 == 0:
                    print(f"progress ok={stats['ok']} fail={stats['fail']}", flush=True)
        except DailyCapExceeded:
            with lock:
                if not stats["cap"]:
                    print("OR DAILY CAP — 오늘 종료", flush=True)
                stats["cap"] = True
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["fail"] += 1
                print(f"FAIL uuid={row.get('record_uuid')}: {str(e)[:120]}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(work, [r for _, r in batch.iterrows()]))
    print(json.dumps({k: v for k, v in stats.items()}), flush=True)
    report(out_dir)


if __name__ == "__main__":
    main()
