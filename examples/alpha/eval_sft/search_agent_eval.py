#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""검색 에이전트 평가 하네스 — phase-2 SFT 게이트 (web-search 도구 데이터).

Nemotron-SFT-Agentic-v2 search split 을 phase-2 SFT 에 편입하면서 신설한 게이트.
홀드아웃 300문항(`splits/search_heldout300.jsonl`, 학습 미투입)에 대해 두 가지를 잰다.

  1. **format gate** (오프라인, 검색 백엔드 불요) — teacher-forced 프리픽스에서
     "다음 한 턴"만 생성시켜 **형식**을 잰다: XML 툴콜이 파싱되는가, `</think>` 가
     닫히는가, 행동 종류(호출/답변)가 레퍼런스와 일치하는가.
  2. **live gate** (선택) — system+user 만 주고 실제 검색을 돌려 최종 답을
     `metadata.ground_truth` 와 대조한다.

## 설계 — 왜 `/v1/completions` + 자체 XML 파서인가

- **프롬프트 조립은 `tokenizer_v5` 만 한다** (INTERLEAVED_THINKING.md §7 규칙 4).
  vLLM 의 `/v1/chat/completions` 는 **체크포인트에 동봉된** 템플릿 사본과 vLLM 자체의
  tools 직렬화를 쓴다 — 하네스가 무엇을 보냈는지와 모델이 무엇을 본지가 갈린다.
  `/v1/completions` 는 우리가 렌더한 문자열을 그대로 받으므로 학습 토큰열과 1:1 이다.
- **툴콜 파서에 의존하지 않는다.** vLLM `--tool-call-parser` 는 별개의 움직이는 부품이고,
  2026-08-30 에이전틱 전 항목 0점 사고의 원인 중 하나가 파서 불일치(hermes JSON vs
  우리 모델의 XML `<function=…>`)였다. 게이트는 **모델**을 재야지 파서를 재면 안 된다.
- **`skip_special_tokens: false` 필수.** `</think>`·`<tool_call>`·`<tool_response>` 는
  tokenizer_v5 에서 단일 special id 라 기본 디코드에서 통째로 사라진다
  (SFT_BENCHMARKS.md §7 게이트 G2, KNOWN_ISSUES 2026-08-30).
- **히스토리의 `reasoning_content` 를 유지·재전송한다** (§7 규칙 5). 템플릿의
  tool-시나리오 분기는 user 턴 경계 너머로 think 를 보존하는데, 그 혜택은 하네스가
  assistant 턴의 reasoning 을 쥐고 있을 때만 성립한다.
- **tool 결과는 학습과 같은 JSON 모양**으로 되돌린다 (Tavily 원형 그대로,
  `ensure_ascii=False` 직렬화). 모양이 다르면 `<tool_response>` 분포가 학습과 달라진다.

## 사용

    python3 search_agent_eval.py --base-url http://HOST:PORT/v1 --model alpha \\
        --gate format|live|both --backend tavily|replay --n 300 --seed 0 --tag <name> \\
        [--concurrency 8] [--max-tokens 4096] [--max-calls 20]
    python3 search_agent_eval.py --selftest      # 서버 불요

Tavily API 키는 **환경변수 `TAVILY_API_KEY` 에서만** 읽는다 (하드코딩·출력 금지).
결과는 `eval_sft/results/search_agent/<tag>/` 에 JSON + per-item JSONL 로 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

HERE = Path(__file__).resolve().parent
ALPHA = HERE.parent
REPO = ALPHA.parent.parent
TOKENIZER_DIR = ALPHA / "tokenizer_v5"
SFT_PREP = REPO / "toolkits" / "sft_data_preprocessing"
RESULTS_ROOT = HERE / "results" / "search_agent"

DEFAULT_DATA = Path(
    "/home/work/Datasets/LL_datasets/posttraining/SFT/Nemotron-SFT-Agentic-v2"
    "/splits/search_heldout300.jsonl"
)
TOOL_NAME = "web-search"
TAVILY_URL = "https://api.tavily.com/search"
IM_END = "<|im_end|>"

# 학습 데이터의 tool_response 최상위 키 순서 (전수 실측 2026-09-04).
# json.dumps(d, ensure_ascii=False) 가 원문 문자열과 바이트 동일함을 확인했다.
TAVILY_TOP_KEYS = ["query", "follow_up_questions", "answer", "images",
                   "results", "response_time", "request_id"]
TAVILY_RESULT_KEYS = ["url", "title", "content", "score", "raw_content"]


# --------------------------------------------------------------------------- #
# 데이터·프롬프트
# --------------------------------------------------------------------------- #
def load_normalize_row():
    """변환기의 `normalize_row` 를 그대로 쓴다 — bins 를 구운 함수와 동일해야 한다."""
    if str(SFT_PREP) not in sys.path:
        sys.path.insert(0, str(SFT_PREP))
    from build_alpha_sft_idxmap import normalize_row  # noqa: E402
    return normalize_row


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer  # 무거우므로 지연 임포트
    return AutoTokenizer.from_pretrained(str(path))


def load_rows(path: Path, n: int, seed: int) -> List[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    if n > 0:
        order = order[:n]
    return [rows[i] for i in sorted(order)]


def render(tok, messages: List[dict], tools) -> str:
    """규칙 4: 프롬프트 조립은 tokenizer_v5 의 apply_chat_template 만."""
    return tok.apply_chat_template(messages, tools=tools,
                                   add_generation_prompt=True, tokenize=False)


def ref_action(msg: dict) -> str:
    return "call" if msg.get("tool_calls") else "answer"


def build_prefixes(norm: dict, rng: random.Random, n_mid: int) -> List[dict]:
    """teacher-forced 프리픽스: (a) system+user, (b) tool 턴 직후 최대 n_mid 지점.

    (b) 의 히스토리는 assistant 턴의 `reasoning_content` 를 그대로 달고 간다 (규칙 5).
    """
    msgs = norm["messages"]
    out: List[dict] = []
    if len(msgs) >= 3:
        out.append({"cut": "start", "idx": 1, "messages": msgs[:2],
                    "ref": ref_action(msgs[2])})
    cut_idxs = [i for i, m in enumerate(msgs)
                if m["role"] == "tool"
                and i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant"]
    if cut_idxs and n_mid > 0:
        picks = sorted(rng.sample(cut_idxs, min(n_mid, len(cut_idxs))))
        for i in picks:
            out.append({"cut": "mid", "idx": i, "messages": msgs[:i + 1],
                        "ref": ref_action(msgs[i + 1])})
    return out


# --------------------------------------------------------------------------- #
# 생성 파서 — vLLM 툴 파서를 쓰지 않는 이유는 모듈 docstring 참조
# --------------------------------------------------------------------------- #
FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
OPEN_MARK_RE = re.compile(r"<tool_call>|<function=")
FINAL_ANS_RE = re.compile(r"final answer\s*[::]\s*(.+)", re.I)


def parse_generation(gen: str, tool_names: Tuple[str, ...] = (TOOL_NAME,)) -> dict:
    """프롬프트가 `<think>\\n` 으로 끝나므로 gen 은 reasoning 부터 시작한다.

    반환: kind = call | answer | malformed, 그리고 진단 필드.
    """
    res: Dict[str, Any] = {
        "kind": "malformed", "think_closed": False, "reasoning": "",
        "tool_name": None, "arguments": None, "answer": None,
        "n_calls_in_gen": 0, "malformed_reason": None,
    }
    if gen is None:
        res["malformed_reason"] = "no_output"
        return res
    if "</think>" in gen:
        res["think_closed"] = True
        reasoning, body = gen.split("</think>", 1)
    else:
        # think 를 닫지 않았으면 행동을 낸 적이 없다 (예산 소진 등).
        res["reasoning"] = gen
        res["malformed_reason"] = "think_unclosed"
        return res
    res["reasoning"] = reasoning
    calls = FUNC_RE.findall(body)
    res["n_calls_in_gen"] = len(calls)
    if calls:
        name, inner = calls[0]
        name = name.strip()
        args = {k.strip(): v for k, v in PARAM_RE.findall(inner)}
        res["tool_name"], res["arguments"] = name, args
        if name not in tool_names:
            res["malformed_reason"] = "unknown_tool"
            return res
        if not (args.get("query") or "").strip():
            res["malformed_reason"] = "empty_query"
            return res
        res["kind"] = "call"
        return res
    if OPEN_MARK_RE.search(body):
        # 툴콜을 시작했지만 닫지 못했다 (잘림·형식 붕괴).
        res["malformed_reason"] = "unclosed_tool_call"
        return res
    if body.strip():
        res["kind"] = "answer"
        res["answer"] = body.strip()
        return res
    res["malformed_reason"] = "empty_body"
    return res


def extract_final_answer(text: str) -> str:
    """시스템 프롬프트 규칙 6: 마지막 줄이 `Final Answer: <Entity>`."""
    if not text:
        return ""
    hits = FINAL_ANS_RE.findall(text)
    if hits:
        return hits[-1].strip().strip("*").strip()
    return text.strip()


_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans({c: " " for c in string.punctuation + "‐‑‒"
                        "–—‘’“”  "})


def normalize_answer(s: str) -> str:
    s = (s or "").lower().translate(_PUNCT)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def score_answer(pred: str, gold: str) -> dict:
    p, g = normalize_answer(extract_final_answer(pred)), normalize_answer(gold)
    exact = bool(g) and p == g
    contains = bool(g) and g in p
    return {"exact": exact, "contains": contains, "correct": exact or contains}


# --------------------------------------------------------------------------- #
# HTTP — /v1/completions (raw prompt)
# --------------------------------------------------------------------------- #
class Client:
    def __init__(self, base_url: str, model: str, timeout: int = 1800,
                 retries: int = 2):
        self.url = base_url.rstrip("/") + "/completions"
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            self._local.s = s
        return s

    def complete(self, prompt: str, max_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0) -> dict:
        body = {
            "model": self.model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p,
            "stop": [IM_END],
            # `</think>` 등이 단일 special id 라 기본 디코드에서 삭제된다 (게이트 G2).
            "skip_special_tokens": False,
        }
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = self.session.post(self.url, json=body, timeout=self.timeout)
                if r.status_code >= 500:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(2 * (attempt + 1))
                    continue
                r.raise_for_status()
                d = r.json()
                ch = d["choices"][0]
                return {
                    "text": ch.get("text") or "",
                    "finish_reason": ch.get("finish_reason"),
                    "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
                    "prompt_tokens": (d.get("usage") or {}).get("prompt_tokens"),
                    "error": None,
                }
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        return {"text": "", "finish_reason": None, "completion_tokens": 0,
                "prompt_tokens": None, "error": last}


# --------------------------------------------------------------------------- #
# 검색 백엔드
# --------------------------------------------------------------------------- #
def _tavily_shape(query: str, answer: Optional[str], results: List[dict],
                  response_time=None, request_id=None) -> dict:
    """학습 데이터와 같은 키·순서로 되돌린다 (§docstring)."""
    out = {
        "query": query,
        "follow_up_questions": None,
        "answer": answer,
        "images": [],
        "results": [{k: r.get(k) for k in TAVILY_RESULT_KEYS} for r in results],
        "response_time": response_time,
        "request_id": request_id,
    }
    assert list(out.keys()) == TAVILY_TOP_KEYS
    return out


class SearchBackend:
    name = "base"

    def search(self, query: str) -> dict:
        raise NotImplementedError

    def as_tool_content(self, query: str) -> str:
        return json.dumps(self.search(query), ensure_ascii=False)


class TavilyBackend(SearchBackend):
    name = "tavily"

    def __init__(self, api_key: str, max_results: int = 10, timeout: int = 60):
        self._key = api_key
        self.max_results = max_results
        self.timeout = timeout
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            self._local.s = s
        return s

    def search(self, query: str) -> dict:
        payload = {"api_key": self._key, "query": query,
                   "max_results": self.max_results, "include_answer": True}
        try:
            r = self.session.post(TAVILY_URL, json=payload, timeout=self.timeout)
            r.raise_for_status()
            d = r.json()
        except Exception as e:  # noqa: BLE001 — 키는 절대 메시지에 싣지 않는다
            return _tavily_shape(query, None, [{"error": type(e).__name__}])
        return _tavily_shape(query, d.get("answer"), d.get("results") or [],
                             d.get("response_time"), d.get("request_id"))


def _toks(s: str) -> set:
    return set(normalize_answer(s).split())


class ReplayBackend(SearchBackend):
    """오프라인 목업 — 그 행이 실제로 받았던 tool 결과 중 가장 비슷한 질의를 돌려준다.

    API 키 없이 루프 전체(프롬프트 조립 → 파싱 → tool_response 재주입)를 스모크할 때 쓴다.
    검색 품질 자체는 재지 못한다 — 그건 tavily 백엔드의 몫.
    """
    name = "replay"

    def __init__(self, row_messages: List[dict]):
        self.bank: List[Tuple[str, dict]] = []
        for m in row_messages:
            if m.get("role") != "tool":
                continue
            try:
                d = json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                continue
            self.bank.append((str(d.get("query") or ""), d))

    def search(self, query: str) -> dict:
        if not self.bank:
            return _tavily_shape(query, None, [{"error": "replay bank empty"}])
        q = _toks(query)
        best, best_j = self.bank[0], -1.0
        for cand_q, d in self.bank:
            c = _toks(cand_q)
            union = q | c
            j = (len(q & c) / len(union)) if union else 0.0
            if j > best_j:
                best, best_j = (cand_q, d), j
        _, d = best
        return _tavily_shape(query, d.get("answer"), d.get("results") or [],
                             d.get("response_time"), d.get("request_id"))


def make_backend(kind: str, row: dict) -> SearchBackend:
    if kind == "replay":
        return ReplayBackend(row["messages"])
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        sys.exit("TAVILY_API_KEY 가 비어 있다. `export TAVILY_API_KEY=...` 후 다시 실행하거나 "
                 "--backend replay 로 오프라인 스모크를 돌려라.")
    return TavilyBackend(key)


# --------------------------------------------------------------------------- #
# gate 1 — format (teacher-forced)
# --------------------------------------------------------------------------- #
def run_format_gate(rows, norm_rows, tok, client, args) -> Tuple[dict, List[dict]]:
    rng = random.Random(args.seed)
    tasks: List[dict] = []
    for row, norm in zip(rows, norm_rows):
        for pref in build_prefixes(norm, rng, args.mid_cuts):
            tasks.append({
                "uuid": row.get("uuid") or (row.get("metadata") or {}).get("id"),
                "cut": pref["cut"], "cut_idx": pref["idx"], "ref": pref["ref"],
                "prompt": render(tok, pref["messages"], norm["tools"]),
            })

    budget = args.ctx - args.max_tokens
    for t in tasks:
        t["prompt_tokens_est"] = len(tok(t["prompt"], add_special_tokens=False)["input_ids"])
    skipped = [t for t in tasks if t["prompt_tokens_est"] > budget]
    tasks = [t for t in tasks if t["prompt_tokens_est"] <= budget]
    print(f"[format] prefixes={len(tasks)} (skipped {len(skipped)} over ctx budget {budget})",
          flush=True)

    def work(t: dict) -> dict:
        out = client.complete(t["prompt"], args.max_tokens)
        p = parse_generation(out["text"])
        rec = {k: t[k] for k in ("uuid", "cut", "cut_idx", "ref", "prompt_tokens_est")}
        rec.update({
            "kind": p["kind"], "think_closed": p["think_closed"],
            "malformed_reason": p["malformed_reason"],
            "tool_name": p["tool_name"],
            "query": (p["arguments"] or {}).get("query"),
            "answer": (p["answer"] or "")[:400],
            "reasoning_chars": len(p["reasoning"]),
            "completion_tokens": out["completion_tokens"],
            "finish_reason": out["finish_reason"], "http_error": out["error"],
        })
        rec["ok"] = p["kind"] in ("call", "answer")
        rec["action_match"] = rec["ok"] and p["kind"] == t["ref"]
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        recs = list(ex.map(work, tasks))
    elapsed = time.time() - t0

    def agg(sel: List[dict]) -> dict:
        n = len(sel)
        if n == 0:
            return {"n": 0}
        toks = [r["completion_tokens"] for r in sel if r["completion_tokens"]]
        parsed = [r for r in sel if r["ok"]]
        return {
            "n": n,
            "tool_call_parse_rate": sum(r["ok"] for r in sel) / n,
            "malformed_rate": sum(not r["ok"] for r in sel) / n,
            "think_closed_rate": sum(r["think_closed"] for r in sel) / n,
            "next_action_agreement": sum(r["action_match"] for r in sel) / n,
            "next_action_agreement_parsed_only": (
                sum(r["action_match"] for r in parsed) / len(parsed) if parsed else None),
            "call_share": sum(r["kind"] == "call" for r in sel) / n,
            "answer_share": sum(r["kind"] == "answer" for r in sel) / n,
            "mean_gen_tokens": (sum(toks) / len(toks)) if toks else None,
            "max_gen_tokens": max(toks) if toks else None,
            "truncated_rate": sum(r["finish_reason"] == "length" for r in sel) / n,
            "http_error_rate": sum(bool(r["http_error"]) for r in sel) / n,
        }

    reasons: Dict[str, int] = {}
    for r in recs:
        if not r["ok"]:
            reasons[r["malformed_reason"] or "unknown"] = \
                reasons.get(r["malformed_reason"] or "unknown", 0) + 1
    summary = {
        "gate": "format", "model": args.model, "base_url": args.base_url,
        "n_rows": len(rows), "n_prefixes": len(tasks), "skipped_over_ctx": len(skipped),
        "max_tokens": args.max_tokens, "seed": args.seed, "mid_cuts": args.mid_cuts,
        "elapsed_sec": round(elapsed, 1),
        "overall": agg(recs),
        "by_cut": {c: agg([r for r in recs if r["cut"] == c]) for c in ("start", "mid")},
        "malformed_reasons": reasons,
    }
    return summary, recs


# --------------------------------------------------------------------------- #
# gate 2 — live agent loop
# --------------------------------------------------------------------------- #
def run_one_live(row: dict, norm: dict, tok, client, args) -> dict:
    backend = make_backend(args.backend, row)
    msgs = [m for m in norm["messages"][:2]]  # system + user
    gold = (row.get("metadata") or {}).get("ground_truth") or ""
    rec = {
        "uuid": row.get("uuid") or (row.get("metadata") or {}).get("id"),
        "question": norm["messages"][1]["content"][:400],
        "ground_truth": gold, "n_calls": 0, "queries": [], "final_answer": None,
        "stop_reason": None, "gen_tokens": 0, "malformed_reason": None,
        "ref_num_tool_calls": (row.get("metadata") or {}).get("num_tool_calls"),
    }
    for _ in range(args.max_calls + 1):
        prompt = render(tok, msgs, norm["tools"])
        ptoks = len(tok(prompt, add_special_tokens=False)["input_ids"])
        if ptoks + args.max_turn_tokens > args.ctx:
            rec["stop_reason"] = "ctx_exhausted"
            break
        out = client.complete(prompt, args.max_turn_tokens)
        rec["gen_tokens"] += out["completion_tokens"] or 0
        if out["error"]:
            rec["stop_reason"] = "http_error"
            rec["malformed_reason"] = out["error"][:200]
            break
        p = parse_generation(out["text"])
        if p["kind"] == "answer":
            rec["final_answer"] = p["answer"]
            rec["stop_reason"] = "answer"
            break
        if p["kind"] != "call":
            rec["stop_reason"] = "malformed"
            rec["malformed_reason"] = p["malformed_reason"]
            break
        if rec["n_calls"] >= args.max_calls:
            rec["stop_reason"] = "call_cap"
            break
        query = p["arguments"]["query"].strip()
        rec["n_calls"] += 1
        rec["queries"].append(query)
        # 규칙 5: reasoning_content 를 달아 재전송해야 think 가 보존된다.
        asst = {"role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": p["tool_name"],
                                             "arguments": p["arguments"]}}]}
        if p["reasoning"].strip():
            asst["reasoning_content"] = p["reasoning"]
        msgs.append(asst)
        msgs.append({"role": "tool", "content": backend.as_tool_content(query)})
    else:
        rec["stop_reason"] = rec["stop_reason"] or "call_cap"
    rec.update(score_answer(rec["final_answer"] or "", gold))
    rec["extracted"] = extract_final_answer(rec["final_answer"] or "")[:300]
    return rec


def run_live_gate(rows, norm_rows, tok, client, args) -> Tuple[dict, List[dict]]:
    pairs = list(zip(rows, norm_rows))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        recs = list(ex.map(lambda pr: run_one_live(pr[0], pr[1], tok, client, args), pairs))
    elapsed = time.time() - t0
    n = len(recs)
    answered = [r for r in recs if r["final_answer"] is not None]
    summary = {
        "gate": "live", "backend": args.backend, "model": args.model,
        "base_url": args.base_url, "n": n, "max_calls": args.max_calls,
        "max_turn_tokens": args.max_turn_tokens, "seed": args.seed,
        "elapsed_sec": round(elapsed, 1),
        "accuracy": (sum(r["correct"] for r in recs) / n) if n else None,
        "exact_match": (sum(r["exact"] for r in recs) / n) if n else None,
        "contains_match": (sum(r["contains"] for r in recs) / n) if n else None,
        "accuracy_answered_only": (
            sum(r["correct"] for r in answered) / len(answered) if answered else None),
        "no_final_answer_rate": (n - len(answered)) / n if n else None,
        "mean_calls": (sum(r["n_calls"] for r in recs) / n) if n else None,
        "total_calls": sum(r["n_calls"] for r in recs),
        "call_cap_rate": sum(r["stop_reason"] == "call_cap" for r in recs) / n if n else None,
        "malformed_rate": sum(r["stop_reason"] == "malformed" for r in recs) / n if n else None,
        "ctx_exhausted_rate": sum(r["stop_reason"] == "ctx_exhausted" for r in recs) / n if n else None,
        "http_error_rate": sum(r["stop_reason"] == "http_error" for r in recs) / n if n else None,
        "mean_gen_tokens": (sum(r["gen_tokens"] for r in recs) / n) if n else None,
        "mean_ref_tool_calls": (
            sum(r["ref_num_tool_calls"] or 0 for r in recs) / n) if n else None,
    }
    return summary, recs


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def print_table(title: str, rows: List[Tuple[str, Any]]) -> None:
    width = max(len(k) for k, _ in rows) if rows else 10
    print(f"\n=== {title} ===")
    for k, v in rows:
        print(f"  {k.ljust(width)}  {_fmt(v)}")


def write_out(tag: str, name: str, summary: dict, recs: List[dict]) -> Path:
    outdir = RESULTS_ROOT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (outdir / f"{name}.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return outdir


# --------------------------------------------------------------------------- #
# selftest — 서버 불요
# --------------------------------------------------------------------------- #
GEN_CALL = (
    "우선 lead 가 어느 주기인지 확인하자.\n</think>Thought: 확인이 필요하다.\n"
    "<tool_call>\n<function=web-search>\n<parameter=query>\n"
    "lead extracted industrially from galena\n</parameter>\n</function>\n</tool_call>\n"
)
GEN_ANSWER = "충분히 교차검증했다.\n</think>Observation: 모두 일치.\nFinal Answer: Sulfide minerals"
GEN_MALFORMED = "생각 중...\n</think><tool_call>\n<function=web-search>\n<parameter=query>\n"


def selftest(data_path: Path) -> int:
    fails: List[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            fails.append(msg)

    print("=== selftest 1/4 — 프롬프트 렌더 (row 0) ===")
    normalize_row = load_normalize_row()
    row = json.loads(data_path.open().readline())
    norm, err = normalize_row(row)
    check(err is None and norm is not None, f"normalize_row 통과 (err={err})")
    tok = load_tokenizer(TOKENIZER_DIR)
    prefixes = build_prefixes(norm, random.Random(0), 2)
    check(len(prefixes) == 3, f"프리픽스 3건 생성 (start 1 + mid 2), got {len(prefixes)}")
    mid = [p for p in prefixes if p["cut"] == "mid"][0]
    text = render(tok, mid["messages"], row["tools"])
    check("<tool_response>" in text, "<tool_response> 마커 렌더")
    check("<tool_call>" in text and "<function=web-search>" in text,
          "<tool_call>/<function=web-search> 마커 렌더")
    check("<parameter=query>" in text, "<parameter=query> 마커 렌더")
    check(text.endswith("<|im_start|>assistant\n<think>\n"),
          "생성 프롬프트가 '<|im_start|>assistant\\n<think>\\n' 로 끝남")
    hist_reason = [m for m in mid["messages"]
                   if m["role"] == "assistant" and m.get("reasoning_content")]
    check(bool(hist_reason), "히스토리 assistant 턴이 reasoning_content 를 보유 (규칙 5)")
    if hist_reason:
        snippet = hist_reason[0]["reasoning_content"].strip()[:60]
        check(snippet in text, "히스토리 reasoning 이 <think> 안에 실제로 렌더됨")
    start = [p for p in prefixes if p["cut"] == "start"][0]
    stext = render(tok, start["messages"], row["tools"])
    check("<tool_response>" not in stext, "system+user 프리픽스에는 tool_response 없음")
    check(start["ref"] == "call", "첫 행동의 레퍼런스는 call")

    print("=== selftest 2/4 — XML 파서 (합성 생성 3종) ===")
    p1 = parse_generation(GEN_CALL)
    check(p1["kind"] == "call" and p1["tool_name"] == TOOL_NAME
          and p1["arguments"]["query"] == "lead extracted industrially from galena",
          f"call 파싱: kind={p1['kind']} query={p1['arguments']}")
    check(p1["think_closed"] and p1["reasoning"].strip().startswith("우선"),
          "call: think 분리")
    p2 = parse_generation(GEN_ANSWER)
    check(p2["kind"] == "answer" and "Final Answer" in (p2["answer"] or ""),
          f"answer 파싱: kind={p2['kind']}")
    check(extract_final_answer(p2["answer"]) == "Sulfide minerals",
          f"Final Answer 추출: {extract_final_answer(p2['answer'])!r}")
    p3 = parse_generation(GEN_MALFORMED)
    check(p3["kind"] == "malformed" and p3["malformed_reason"] == "unclosed_tool_call",
          f"malformed 파싱: kind={p3['kind']} reason={p3['malformed_reason']}")
    p4 = parse_generation("사고만 하고 끝")
    check(p4["kind"] == "malformed" and p4["malformed_reason"] == "think_unclosed",
          "think 미종결 → malformed")
    check(score_answer("Final Answer: Sulfide minerals", "sulfide minerals")["exact"],
          "정규화 exact match")
    check(score_answer("... the answer is Sulfide Minerals.", "sulfide minerals")["contains"],
          "containment match")
    check(not score_answer("Final Answer: Japan", "United States")["correct"],
          "오답은 불일치")

    print("=== selftest 3/4 — replay 백엔드 모양 ===")
    be = ReplayBackend(row["messages"])
    check(len(be.bank) > 0, f"replay bank {len(be.bank)}건 적재")
    d = be.search("galena lead sulfide mineral group")
    check(list(d.keys()) == TAVILY_TOP_KEYS, f"최상위 키 순서 == 학습 모양: {list(d.keys())}")
    check(bool(d["results"]) and list(d["results"][0].keys()) == TAVILY_RESULT_KEYS,
          "results[0] 키 == url/title/content/score/raw_content")
    check(d["query"] == "galena lead sulfide mineral group", "query 는 요청 질의로 덮어씀")
    s = json.dumps(d, ensure_ascii=False)
    check(json.loads(s)["answer"] == d["answer"], "tool_content 직렬화 왕복")
    tool_msg = {"role": "tool", "content": s}
    rendered = render(tok, norm["messages"][:2] + [
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": TOOL_NAME,
                                      "arguments": {"query": "q"}}}]},
        tool_msg], row["tools"])
    check("<tool_response>\n{\"query\":" in rendered,
          "replay 결과가 <tool_response> 안에 JSON 으로 렌더")

    print("=== selftest 4/4 — 데이터 불변량 ===")
    rows = load_rows(data_path, 0, 0)
    check(len(rows) == 300, f"홀드아웃 {len(rows)}행")
    check(all((r.get("metadata") or {}).get("ground_truth") for r in rows),
          "전 행 ground_truth 보유")
    names = {t["function"]["name"] for r in rows for t in r["tools"]}
    check(names == {TOOL_NAME}, f"선언 도구 == {{{TOOL_NAME}}}: {names}")

    print(f"\nselftest: {'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
    for f in fails:
        print(f"  - {f}")
    return 0 if not fails else 1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8001/v1",
                    help="vLLM OpenAI 호환 엔드포인트 (…/v1)")
    ap.add_argument("--model", default="alpha", help="served model name")
    ap.add_argument("--gate", choices=["format", "live", "both"], default="format")
    ap.add_argument("--backend", choices=["tavily", "replay"], default="replay")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    ap.add_argument("--n", type=int, default=300, help="행 수 (0 = 전체)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="results/search_agent/<tag>/")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096, help="format gate 턴당 예산")
    ap.add_argument("--max-turn-tokens", type=int, default=None,
                    help="live gate 턴당 예산 (기본 --max-tokens)")
    ap.add_argument("--max-calls", type=int, default=20)
    ap.add_argument("--mid-cuts", type=int, default=2,
                    help="행마다 tool 턴 직후 절단 지점 수 (format gate)")
    ap.add_argument("--ctx", type=int, default=131072, help="서빙 --max-model-len")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--selftest", action="store_true", help="서버 없이 자체 검증")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.data)
    if args.max_turn_tokens is None:
        args.max_turn_tokens = args.max_tokens
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H%M%S")
    if not args.data.exists():
        sys.exit(f"데이터가 없다: {args.data}")

    normalize_row = load_normalize_row()
    rows_all = load_rows(args.data, args.n, args.seed)
    rows, norm_rows, dropped = [], [], 0
    for r in rows_all:
        norm, err = normalize_row(r)
        if norm is None:
            dropped += 1
            continue
        rows.append(r)
        norm_rows.append(norm)
    print(f"[data] {args.data.name}: {len(rows)} rows (dropped {dropped}) "
          f"seed={args.seed} tag={args.tag}", flush=True)

    tok = load_tokenizer(args.tokenizer)
    client = Client(args.base_url, args.model, timeout=args.timeout)

    outdir = None
    if args.gate in ("format", "both"):
        summary, recs = run_format_gate(rows, norm_rows, tok, client, args)
        o = summary["overall"]
        print_table("format gate — overall", [
            ("prefixes", o["n"]),
            ("tool_call_parse_rate", o["tool_call_parse_rate"]),
            ("malformed_rate", o["malformed_rate"]),
            ("think_closed_rate", o["think_closed_rate"]),
            ("next_action_agreement", o["next_action_agreement"]),
            ("  (parsed only)", o["next_action_agreement_parsed_only"]),
            ("call_share / answer_share", f"{o['call_share']:.3f} / {o['answer_share']:.3f}"),
            ("mean_gen_tokens", o["mean_gen_tokens"]),
            ("max_gen_tokens", o["max_gen_tokens"]),
            ("truncated_rate (finish=length)", o["truncated_rate"]),
            ("http_error_rate", o["http_error_rate"]),
        ])
        for cut, a in summary["by_cut"].items():
            if a.get("n"):
                print_table(f"format gate — cut={cut}", [
                    ("prefixes", a["n"]),
                    ("tool_call_parse_rate", a["tool_call_parse_rate"]),
                    ("think_closed_rate", a["think_closed_rate"]),
                    ("next_action_agreement", a["next_action_agreement"]),
                    ("mean_gen_tokens", a["mean_gen_tokens"]),
                ])
        if summary["malformed_reasons"]:
            print_table("format gate — malformed 사유",
                        sorted(summary["malformed_reasons"].items(), key=lambda kv: -kv[1]))
        outdir = write_out(args.tag, "format_gate", summary, recs)

    if args.gate in ("live", "both"):
        summary, recs = run_live_gate(rows, norm_rows, tok, client, args)
        print_table(f"live gate — backend={args.backend}", [
            ("n", summary["n"]),
            ("accuracy (exact|contains)", summary["accuracy"]),
            ("  exact_match", summary["exact_match"]),
            ("  contains_match", summary["contains_match"]),
            ("accuracy_answered_only", summary["accuracy_answered_only"]),
            ("no_final_answer_rate", summary["no_final_answer_rate"]),
            ("mean_calls (ref)", f"{_fmt(summary['mean_calls'])} ({_fmt(summary['mean_ref_tool_calls'])})"),
            ("total_calls", summary["total_calls"]),
            ("call_cap_rate", summary["call_cap_rate"]),
            ("malformed_rate", summary["malformed_rate"]),
            ("ctx_exhausted_rate", summary["ctx_exhausted_rate"]),
            ("http_error_rate", summary["http_error_rate"]),
            ("mean_gen_tokens", summary["mean_gen_tokens"]),
        ])
        outdir = write_out(args.tag, "live_gate", summary, recs)

    if outdir:
        print(f"\n[out] {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
