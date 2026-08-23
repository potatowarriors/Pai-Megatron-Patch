#!/usr/bin/env python3
"""auto_gate_a.py — 트랙 A tranche 자동 게이트 (무인 r2 투입 전 판정).

기준 (2026-08-23 사용자 사전 승인 체인의 일부):
  - 리젝률 < 15%
  - 산출물 내 special-token 리터럴 0건
  - 에러율 < 5% (미처리분은 재실행이 회수하므로 완료율은 게이트 아님)
  - 한글비율 분포: 최종 assistant 턴의 중앙값 >= 0.7

통과 exit 0 / 실패 exit 1 (+사유 stdout).
사용: python3 auto_gate_a.py <out_dir>
"""
import json
import re
import statistics
import sys

SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")
HANGUL_RE = re.compile(r"[가-힣]")


def hangul_ratio(text):
    body = re.sub(r"```.*?```", "", text or "", flags=re.DOTALL)
    letters = re.findall(r"[A-Za-z가-힣]", body)
    return len(HANGUL_RE.findall(body)) / len(letters) if letters else 1.0


def main():
    out_dir = sys.argv[1]
    n_ok, n_special, ratios = 0, 0, []
    try:
        with open(f"{out_dir}/results.jsonl") as f:
            for line in f:
                r = json.loads(line)
                n_ok += 1
                if any(m["content"] and SPECIAL_TOKEN_RE.search(m["content"])
                       for m in r["messages"]):
                    n_special += 1
                ratios.append(hangul_ratio(r["messages"][-1]["content"]))
    except FileNotFoundError:
        print("GATE FAIL: results.jsonl 없음")
        sys.exit(1)
    n_rej = 0
    try:
        with open(f"{out_dir}/rejects.jsonl") as f:
            n_rej = sum(1 for _ in f)
    except FileNotFoundError:
        pass

    fails = []
    total = n_ok + n_rej
    rej_rate = n_rej / max(total, 1)
    med = statistics.median(ratios) if ratios else 0.0
    if n_ok == 0:
        fails.append("no_results")
    if rej_rate >= 0.15:
        fails.append(f"reject_rate={rej_rate:.3f}")
    if n_special > 0:
        fails.append(f"special_token={n_special}")
    if med < 0.70:
        fails.append(f"hangul_median={med:.2f}")

    print(f"gate: ok={n_ok} rej={n_rej} rej_rate={rej_rate:.3f} "
          f"special={n_special} hangul_med={med:.2f}")
    if fails:
        print("GATE FAIL:", ",".join(fails))
        sys.exit(1)
    print("GATE PASS")


if __name__ == "__main__":
    main()
