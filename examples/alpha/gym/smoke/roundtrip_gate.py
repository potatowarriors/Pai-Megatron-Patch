#!/usr/bin/env python3
"""추론 왕복 게이트 — NeMo-Gym vllm_model 경로에서 alpha 의 추론이 (1) 파서로 분리되고 (2) 이전 턴에 되돌아가는지 판정한다.

증거 2종을 본다.
  A. vLLM 전송 로그 (`NEMO_GYM_VLLM_TRANSPORT_LOG` JSONL, schema_version 2)
     - transport_response: 응답 message 에 reasoning/reasoning_content 가 있는가, content 에 `</think>` 가 남았는가 (파서 계약)
     - transport_request : 이력 assistant 턴에 reasoning_content/reasoning 필드가 실려 vLLM 으로 갔는가 (restore 의 필드 증거)
  B. 롤아웃 JSONL 의 prompt_token_ids (configs/alpha_vllm_model_gate.yaml) 를 tokenizer_v5 로 복호한 실제 프롬프트
     - 이력 assistant 턴이 `<think>\\n…</think>` (restore) 인지 `<think></think>` (strip) 인지 — 모델이 본 바이트 그대로.
       `/tokenize` 는 reasoning_content 를 버리므로 증거가 못 된다 (SFT_BENCHMARKS.md §3.14 교훈).

판정 규칙 (--expect):
  restore : 파서 OK · 멀티턴 요청 전부에 필드 실린 이력 턴 ≥1 · 프롬프트에 restore 마커 ≥1
  strip   : 파서 OK · 프롬프트에 restore 마커 0 · strip 마커 ≥1
우리 프록시 지표와 같은 부가 통계(mixed_content_and_tools, think_unclosed, reasoning_only)도 출력한다.

사용:
  python roundtrip_gate.py --transport OUT/vllm_transport.jsonl --rollouts OUT/rollouts.jsonl \\
      --tokenizer examples/alpha/tokenizer_v5 --expect restore
실행 파이썬은 `tokenizers` 가 있는 곳이면 된다 (alpha_serve_venv 권장; Gym venv 는 없음).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

RESTORE_MARKER = "<think>\n"
STRIP_MARKER = "<think></think>"
ASSISTANT_HEADER = "<|im_start|>assistant"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def has_reasoning_field(msg: Dict[str, Any]) -> bool:
    return bool(msg.get("reasoning_content") or msg.get("reasoning"))


def content_text(msg: Dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return ""


def analyze_transport(path: Path) -> Dict[str, Any]:
    st: Counter = Counter()
    multi_turn_missing: List[int] = []
    for ev in iter_jsonl(path):
        kind = ev.get("event")
        if kind == "transport_request":
            payload = ev.get("request_payload") or {}
            msgs = payload.get("messages") or []
            st["requests"] += 1
            if payload.get("tools"):
                st["requests_with_tools"] += 1
            hist = [m for m in msgs if m.get("role") == "assistant"]
            if not hist:
                continue
            st["multi_turn_requests"] += 1
            with_field = sum(1 for m in hist if has_reasoning_field(m))
            st["history_assistant_turns"] += len(hist)
            st["history_turns_with_reasoning_field"] += with_field
            st["history_turns_with_think_in_content"] += sum(1 for m in hist if "<think>" in content_text(m))
            if with_field == 0:
                multi_turn_missing.append(ev.get("call_index", -1))
        elif kind == "transport_response":
            raw = ev.get("raw_response") or {}
            choices = raw.get("choices") or []
            if not choices:
                st["responses_without_choices"] += 1
                continue
            msg = choices[0].get("message") or {}
            st["responses"] += 1
            content = msg.get("content") or ""
            tool_calls = bool(msg.get("tool_calls"))
            if has_reasoning_field(msg):
                st["responses_with_reasoning_field"] += 1
            if "</think>" in content:
                st["responses_think_end_in_content"] += 1
            if "<think>" in content and "</think>" not in content:
                st["responses_think_unclosed"] += 1
            if tool_calls:
                st["responses_with_tool_calls"] += 1
                if content.strip():
                    st["responses_mixed_content_and_tools"] += 1
            if has_reasoning_field(msg) and not content.strip() and not tool_calls:
                st["responses_reasoning_only"] += 1
    return {"stats": dict(st), "multi_turn_requests_without_field": multi_turn_missing}


def find_prompt_token_ids(obj: Any, out: List[List[int]]) -> None:
    if isinstance(obj, dict):
        ids = obj.get("prompt_token_ids")
        if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids):
            out.append(ids)
        for v in obj.values():
            find_prompt_token_ids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_prompt_token_ids(v, out)


def load_decoder(tokenizer_dir: Path):
    try:
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        return lambda ids: tok.decode(ids, skip_special_tokens=False)
    except ImportError:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        return lambda ids: tok.decode(ids, skip_special_tokens=False)


def analyze_prompts(rollouts: Path, tokenizer_dir: Path) -> Dict[str, Any]:
    decode = load_decoder(tokenizer_dir)
    st: Counter = Counter()
    samples: List[str] = []
    for row in iter_jsonl(rollouts):
        found: List[List[int]] = []
        find_prompt_token_ids(row.get("response") or row, found)
        if not found:
            st["rollouts_without_prompt_ids"] += 1
            continue
        st["rollouts"] += 1
        last = max(found, key=len)  # 마지막 모델 호출 = 가장 긴 프롬프트
        text = decode(last)
        cut = text.rfind(ASSISTANT_HEADER)  # 최종 생성 프롬프트 앞까지가 이력
        history = text[:cut] if cut >= 0 else text
        n_hist_assistant = history.count(ASSISTANT_HEADER)
        st["history_assistant_turns"] += n_hist_assistant
        st["restore_markers"] += history.count(RESTORE_MARKER)
        st["strip_markers"] += history.count(STRIP_MARKER)
        st["prompt_tokens_last_call_max"] = max(st["prompt_tokens_last_call_max"], len(last))
        if len(samples) < 1 and n_hist_assistant:
            samples.append(history[-600:])
    return {"stats": dict(st), "sample_history_tail": samples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--expect", choices=["restore", "strip"], required=True)
    ap.add_argument("--json", type=Path, help="판정 결과를 JSON 으로도 저장")
    args = ap.parse_args()

    verdicts: Dict[str, bool] = {}
    report: Dict[str, Any] = {"expect": args.expect}

    t = analyze_transport(args.transport)
    report["transport"] = t
    ts = t["stats"]
    n_resp = ts.get("responses", 0)
    verdicts["A_parser_reasoning_field_present"] = n_resp > 0 and ts.get("responses_with_reasoning_field", 0) > 0
    verdicts["A_parser_no_think_end_in_content"] = ts.get("responses_think_end_in_content", 0) == 0
    if args.expect == "restore":
        verdicts["B_field_on_every_multi_turn_request"] = (
            ts.get("multi_turn_requests", 0) > 0 and not t["multi_turn_requests_without_field"]
        )

    if args.rollouts and args.tokenizer:
        p = analyze_prompts(args.rollouts, args.tokenizer)
        report["prompt"] = p
        ps = p["stats"]
        if args.expect == "restore":
            verdicts["C_prompt_restore_marker_present"] = ps.get("restore_markers", 0) >= 1
        else:
            verdicts["C_prompt_no_restore_marker"] = ps.get("restore_markers", 0) == 0
            verdicts["C_prompt_strip_marker_present"] = ps.get("strip_markers", 0) >= 1
    else:
        report["prompt"] = "skipped (--rollouts/--tokenizer 미지정)"

    report["verdicts"] = verdicts
    ok = all(verdicts.values())
    report["PASS"] = ok

    print(f"== roundtrip gate (expect={args.expect}) ==")
    for k, v in ts.items():
        print(f"  transport.{k}: {v}")
    if t["multi_turn_requests_without_field"]:
        print(f"  transport.multi_turn_requests_without_field: {t['multi_turn_requests_without_field'][:20]}")
    if isinstance(report["prompt"], dict):
        for k, v in report["prompt"]["stats"].items():
            print(f"  prompt.{k}: {v}")
        for s in report["prompt"]["sample_history_tail"]:
            print("  prompt.sample_history_tail:\n" + "\n".join("    | " + ln for ln in s.splitlines()[-12:]))
    for k, v in verdicts.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"== {'PASS' if ok else 'FAIL'} ==")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
