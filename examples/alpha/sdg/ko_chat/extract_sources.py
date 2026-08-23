#!/usr/bin/env python3
"""extract_sources.py — 트랙 A(번역+재생성) 원천 시드 추출.

chat_v3의 IF/chat split에서 한국어화 대상 행을 결정론적으로 샘플링한다.
필터: null content(라이선스 마스킹 미복원 행), tool 역할, 과도한 턴 수/길이,
번역 출력 한도(단일 메시지 초장문)를 넘는 행.

사용:
  python3 extract_sources.py --num-if 4000 --num-chat 8000 --out seeds_r1.jsonl
  python3 extract_sources.py --num-if 8 --num-chat 8 --out seeds_pilot.jsonl --seed 7
"""
import argparse
import json
import random
import re
from pathlib import Path

# special-token 리터럴 방어 (LC-A iter170 정지 사고, 2026-08-23 공유 — alpha v5
# 특수 토큰뿐 아니라 타 모델 제어토큰 형태도 학습 데이터에서 배제한다.
# 하류 build_alpha_sft_idxmap 의 injection 드롭이 최종 방어선, 여기는 조기 차단.)
SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")

SFT_ROOT = Path("/home/work/Datasets/LL_datasets/posttraining/SFT")
SOURCES = {
    "chat_v3_if": SFT_ROOT / "Nemotron-SFT-Instruction-Following-Chat-v3/data/instruction_following.jsonl",
    "chat_v3_chat": SFT_ROOT / "Nemotron-SFT-Instruction-Following-Chat-v3/data/chat.with_prompts.jsonl",
}

MAX_MESSAGES = 16          # 턴 수 상한 (비용·컨텍스트 통제)
MAX_TOTAL_CHARS = 40_000   # 대화 전체 길이 상한 (32k ctx 내 번역·재생성 여유)
MAX_SINGLE_MSG_CHARS = 12_000  # 단일 메시지 상한 (번역 청킹 한도)


def row_ok(row: dict) -> tuple[bool, str]:
    msgs = row.get("messages")
    if not msgs or not isinstance(msgs, list):
        return False, "no_messages"
    if len(msgs) > MAX_MESSAGES:
        return False, "too_many_turns"
    total = 0
    for m in msgs:
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            return False, f"role_{role}"
        c = m.get("content")
        # 라이선스 마스킹 미복원 행: system null 은 허용(생략 처리), user/assistant null 은 탈락
        if c is None:
            if role == "system":
                continue
            return False, "null_content"
        if not isinstance(c, str):
            return False, "nonstr_content"
        if m.get("tool_calls"):
            return False, "tool_calls"
        if len(c) > MAX_SINGLE_MSG_CHARS:
            return False, "single_msg_too_long"
        if SPECIAL_TOKEN_RE.search(c):
            return False, "special_token_literal"
        total += len(c)
    if total > MAX_TOTAL_CHARS:
        return False, "total_too_long"
    if msgs[-1].get("role") != "assistant" or not msgs[-1].get("content"):
        return False, "no_final_assistant"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-if", type=int, default=4000)
    ap.add_argument("--num-chat", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--exclude", help="이미 사용한 시드 jsonl (uuid 제외용)", default=None)
    args = ap.parse_args()

    exclude = set()
    if args.exclude:
        with open(args.exclude) as f:
            for line in f:
                exclude.add(json.loads(line)["uuid"])

    targets = {"chat_v3_if": args.num_if, "chat_v3_chat": args.num_chat}
    rng = random.Random(args.seed)
    picked, stats = [], {}
    for name, path in SOURCES.items():
        want = targets[name]
        if want <= 0:
            continue
        # 전 행을 훑되 reservoir sampling으로 want개 균일 추출 (필터 통과 행 기준)
        reservoir, seen, drop = [], 0, {}
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    drop["bad_json"] = drop.get("bad_json", 0) + 1
                    continue
                if row.get("uuid") in exclude:
                    drop["excluded"] = drop.get("excluded", 0) + 1
                    continue
                ok, why = row_ok(row)
                if not ok:
                    drop[why] = drop.get(why, 0) + 1
                    continue
                seen += 1
                if len(reservoir) < want:
                    reservoir.append(row)
                else:
                    j = rng.randrange(seen)
                    if j < want:
                        reservoir[j] = row
        for row in reservoir:
            row["_source_split"] = name
        picked.extend(reservoir)
        stats[name] = {"eligible": seen, "picked": len(reservoir), "dropped": drop}

    rng.shuffle(picked)
    with open(args.out, "w") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {len(picked)} seeds -> {args.out}")


if __name__ == "__main__":
    main()
