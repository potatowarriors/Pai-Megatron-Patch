#!/usr/bin/env python3
"""bench_throughput.py — gemma-4-31b 서빙 스루풋 스모크 (한국어 chat SDG 트랙).

동시성 수준별로 한국어 생성 요청을 보내 aggregate output tok/s를 실측한다.
목적: 목표 산출량(원천 N tok)의 소요 기간 역산 + 동시성 포화점 결정.

워크로드는 실전과 같은 두 종류를 섞는다:
  - regen: 한국어 질문에 실질적 답변 생성 (트랙 A 재생성·트랙 B 생성과 동형)
  - translate: 영어 단락 → 한국어 번역 (트랙 A 번역 호출과 동형)

사용 (sub1, 서버 기동 후):
  python3 bench_throughput.py --base-url http://127.0.0.1:8000/v1 \
      --concurrency 16,32,64,128 --requests-per-level 96
"""
import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

REGEN_PROMPTS = [
    "재택근무를 처음 시작하는 신입사원에게 시간 관리 요령을 구체적으로 조언해 주세요.",
    "김치찌개를 더 깊은 맛으로 끓이는 방법을 단계별로 설명해 주세요.",
    "파이썬의 리스트 컴프리헨션을 초보자에게 예제와 함께 설명해 주세요.",
    "전세 계약할 때 반드시 확인해야 할 사항들을 정리해 주세요.",
    "운동을 꾸준히 하지 못하는 사람을 위한 현실적인 습관 형성 전략을 알려주세요.",
    "조선 시대 과거 제도가 사회에 미친 영향을 설명해 주세요.",
    "스타트업에서 첫 제품의 가격을 정할 때 고려할 요소들을 알려주세요.",
    "고등학생에게 미분의 개념을 직관적으로 설명해 주세요.",
]

TRANSLATE_TEXT = (
    "The error occurs because you are using React Router v6, where the useHistory "
    "hook was replaced by useNavigate. In v6, you should import useNavigate from "
    "react-router-dom and call it to get a navigate function. Instead of "
    "history.push('/path'), you now write navigate('/path'). If you need to replace "
    "the current entry in the history stack, pass an options object as the second "
    "argument: navigate('/path', { replace: true }). This change was part of a "
    "broader API simplification in v6 that also removed the Switch component in "
    "favor of Routes, and changed how nested routes are declared."
)


def one_request(base_url: str, model: str, kind: str, timeout: int):
    if kind == "regen":
        content = random.choice(REGEN_PROMPTS)
        system = "당신은 유능하고 친절한 AI 어시스턴트입니다. 한국어로 답변합니다."
    else:
        content = (
            "다음 영어 텍스트를 자연스러운 한국어로 번역해 주세요. 코드 식별자와 "
            "기술 용어 표기는 보존합니다. 번역문만 출력하세요.\n\n" + TRANSLATE_TEXT
        )
        system = "당신은 영한 전문 번역가입니다."
    t0 = time.time()
    r = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": 1024,
            "temperature": 0.9,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    dt = time.time() - t0
    usage = body.get("usage", {})
    return {
        "kind": kind,
        "latency": dt,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "finish": body["choices"][0].get("finish_reason"),
    }


def run_level(base_url, model, conc, n_requests, timeout):
    kinds = ["regen" if i % 2 == 0 else "translate" for i in range(n_requests)]
    t0 = time.time()
    results, errors = [], 0
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one_request, base_url, model, k, timeout) for k in kinds]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                errors += 1
                print(f"  ! request error: {e}", file=sys.stderr)
    wall = time.time() - t0
    out_toks = sum(r["completion_tokens"] for r in results)
    in_toks = sum(r["prompt_tokens"] for r in results)
    lat = [r["latency"] for r in results]
    return {
        "concurrency": conc,
        "n_ok": len(results),
        "n_err": errors,
        "wall_s": round(wall, 1),
        "output_tok_s": round(out_toks / wall, 1),
        "input_tok_s": round(in_toks / wall, 1),
        "p50_latency_s": round(statistics.median(lat), 1) if lat else None,
        "p95_latency_s": round(sorted(lat)[int(len(lat) * 0.95) - 1], 1) if lat else None,
        "mean_completion_tok": round(out_toks / max(len(results), 1)),
        "finish_stop_frac": round(
            sum(1 for r in results if r["finish"] == "stop") / max(len(results), 1), 2
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-31b")
    ap.add_argument("--concurrency", default="16,32,64,128")
    ap.add_argument("--requests-per-level", type=int, default=96)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    print(f"target={args.base_url} model={args.model} n/level={args.requests_per_level}")
    # 워밍업 (컴파일·캐시 워밍 — 측정 제외)
    print("warmup...", flush=True)
    run_level(args.base_url, args.model, 8, 16, args.timeout)

    all_results = []
    for conc in levels:
        print(f"level concurrency={conc} ...", flush=True)
        res = run_level(args.base_url, args.model, conc, args.requests_per_level, args.timeout)
        all_results.append(res)
        print(" ", json.dumps(res, ensure_ascii=False), flush=True)

    print("\n=== summary ===")
    print(f"{'conc':>5} {'out tok/s':>10} {'p50 lat':>8} {'p95 lat':>8} {'err':>4}")
    for r in all_results:
        print(
            f"{r['concurrency']:>5} {r['output_tok_s']:>10} {r['p50_latency_s']:>8}"
            f" {r['p95_latency_s']:>8} {r['n_err']:>4}"
        )
    best = max(all_results, key=lambda r: r["output_tok_s"])
    per_day = best["output_tok_s"] * 86400
    print(
        f"\nbest: conc={best['concurrency']} → {best['output_tok_s']} tok/s"
        f" ≈ {per_day/1e9:.2f}B output tok/day"
    )


if __name__ == "__main__":
    main()
