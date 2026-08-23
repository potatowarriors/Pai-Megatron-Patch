#!/usr/bin/env python3
"""translate_regen.py — 트랙 A: chat_v3 한국어화 (모드 이원화, 2026-08-23 파일럿 교훈 반영).

레코드별 모드 (train_turns 실측 근거 — chat 637,663행 100% last-only,
IF 249,748행 중 61% multi-True):

  regen  (chat_v3_chat, last-only): 컨텍스트 턴 번역 + 마지막 assistant 턴 재생성.
         학습되는 턴이 번역투가 아닌 네이티브 한국어가 된다.
  full_translate (chat_v3_if 전체, 또는 multi-True 행): 모든 턴 번역.
         IF 제약(특정 단어 시작, 쉼표 금지 등)은 재생성이 위반할 수 있어
         컴플라이언스를 보존하는 번역이 안전하다. assistant 턴 번역 시
         첫 사용자 턴(제약 원천)을 컨텍스트로 제공한다.

파일럿에서 잡은 함정:
  - 빈/공백 content 를 LLM에 넣으면 "번역할 내용을 주세요" 메타응답이 나온다 → 무호출 통과.
  - 짧은 입력의 메타응답 감지 → 1회 재시도 후 실패 시 리젝.
  - 리젝에도 생성물 전체 저장 (진단·회수용).
  - 한글비율 임계 0.55 → 0.40 (기술용어 다수 답변 오탐).

사용 (sub1, 서버 기동 후):
  python3 translate_regen.py --seeds seeds.jsonl --out out/run1 \
      --base-url http://127.0.0.1:8000/v1 --workers 96
"""
import argparse
import json
import re
import threading
import time
import uuid as uuidlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

CHUNK_CHARS = 4500
TRANSLATE_MAX_TOKENS = 3072
REGEN_MAX_TOKENS = 4096
RETRIES = 3

TRANSLATE_SYSTEM = (
    "당신은 전문 번역가입니다. 다양한 언어(영어·러시아어·일본어·중국어 등)로 된 "
    "대화 데이터의 한 발화를 자연스러운 한국어로 현지화합니다. 규칙:\n"
    "1. 코드 블록·명령어·식별자·URL·수식은 원문 그대로 유지한다 (코드 주석은 번역 가능).\n"
    "2. 마크다운 구조(목록, 헤더, 코드펜스 개수)를 보존한다.\n"
    "3. 사용자 발화는 실제 한국어 사용자가 쓸 법한 자연스러운 어투로, "
    "assistant 발화는 정중한 존댓말로 옮긴다.\n"
    "4. 발화의 지시·제약이 참조하는 특정 영어 단어(예: 답변을 특정 단어로 시작하라는 "
    "요구의 그 단어)는 번역하지 않고 원문 그대로 둔다.\n"
    "5. 입력이 이미 한국어면 그대로(필요시 최소 교정만) 출력한다.\n"
    "6. 설명이나 머리말 없이 번역문만 출력한다."
)

REGEN_SYSTEM_SUFFIX = (
    "반드시 한국어로, 정중한 존댓말로 답변하세요. 사용자가 반말을 쓰더라도 "
    "존댓말을 유지합니다. 코드·명령어·기술 식별자는 원문 표기를 유지하고, "
    "사용자가 요구한 출력 형식(JSON 키, 시작 단어 등)이 있으면 그 형식을 지키세요."
)

HANGUL_RE = re.compile(r"[가-힣]")
LEAK_RE = re.compile(r"gemma|gemini|구글이 (?:저를|나를) |google(?:이|에서) (?:저를|나를)", re.IGNORECASE)
META_RE = re.compile(r"번역할 .{0,10}(?:제공|입력)해 주|번역해 드리겠습니다|Please provide")
# special-token 리터럴 방어 (LC-A iter170 사고 반영 — 상세 extract_sources.py 주석)
SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")
HANGUL_MIN_RATIO = 0.40


class Client:
    def __init__(self, base_url, model, timeout=600):
        self.base_url, self.model, self.timeout = base_url, model, timeout
        self.session_local = threading.local()

    def chat(self, messages, max_tokens, temperature):
        sess = getattr(self.session_local, "s", None)
        if sess is None:
            sess = self.session_local.s = requests.Session()
        last_err = None
        for attempt in range(RETRIES):
            try:
                r = sess.post(
                    f"{self.base_url}/chat/completions",
                    json={"model": self.model, "messages": messages,
                          "max_tokens": max_tokens, "temperature": temperature},
                    timeout=self.timeout)
                if r.status_code >= 500:
                    raise RuntimeError(f"server {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                body = r.json()
                return body["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt * 2)
        raise RuntimeError(f"request failed after {RETRIES} tries: {last_err}")


def split_paragraph_chunks(text: str, limit: int):
    if len(text) <= limit:
        return [text]
    parts, cur, cur_len = [], [], 0
    for para in text.split("\n\n"):
        plen = len(para) + 2
        if cur and cur_len + plen > limit:
            parts.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(para)
        cur_len += plen
    if cur:
        parts.append("\n\n".join(cur))
    return parts


class MetaTranslation(Exception):
    pass


def translate_message(client, role, content, prev_pair, constraint_ctx=None,
                      temperature=0.3):
    """한 발화 번역. prev_pair=(원문,번역) 직전 턴. constraint_ctx=첫 사용자 턴 KO."""
    if not content or not content.strip():
        return content or ""
    out_chunks = []
    chunks = split_paragraph_chunks(content, CHUNK_CHARS)
    for i, chunk in enumerate(chunks):
        ctx = ""
        if constraint_ctx and role == "assistant" and i == 0:
            ctx += (f"[이 대화의 사용자 요구 — 형식 제약이 있으면 번역문도 지켜야 한다]\n"
                    f"{constraint_ctx[:1200]}\n\n")
        if prev_pair and i == 0:
            ctx += (f"[참고 — 직전 턴 원문 끝부분]\n{prev_pair[0][-600:]}\n"
                    f"[직전 턴 번역 끝부분]\n{prev_pair[1][-600:]}\n\n")
        elif i > 0:
            ctx += (f"[참고 — 같은 발화의 직전 청크 번역 끝부분]\n{out_chunks[-1][-600:]}\n\n"
                    "이어지는 부분을 번역하세요.\n\n")
        speaker = {"system": "시스템 지시문", "user": "사용자 발화",
                   "assistant": "AI 어시스턴트 발화"}[role]
        user_msg = f"{ctx}다음 {speaker}를 번역하세요:\n\n{chunk}"
        req = [{"role": "system", "content": TRANSLATE_SYSTEM},
               {"role": "user", "content": user_msg}]
        out = client.chat(req, max_tokens=TRANSLATE_MAX_TOKENS, temperature=temperature).strip()
        if META_RE.search(out) and not META_RE.search(chunk):
            out = client.chat(req, max_tokens=TRANSLATE_MAX_TOKENS, temperature=0.0).strip()
            if META_RE.search(out):
                raise MetaTranslation(f"meta_translation:{role}")
        out_chunks.append(out)
    return "\n\n".join(out_chunks)


def length_hint(chars: int) -> str:
    if chars < 800:
        return "간결하게 핵심만 답하세요."
    if chars < 3000:
        return "적절한 분량으로 답하세요."
    return "충분히 상세하게 답하세요."


def hangul_ratio(text: str) -> float:
    body = re.sub(r"```.*?```", "", text or "", flags=re.DOTALL)
    body = re.sub(r"`[^`\n]+`", "", body)
    letters = re.findall(r"[A-Za-z가-힣]", body)
    if not letters:
        return 1.0  # 글자 없음(순수 코드/기호)은 판정 보류
    return len(HANGUL_RE.findall(body)) / len(letters)


def qc_checks(src_msgs, out_msgs, mode):
    """규칙 게이트. 실패 시 {kind, reason, idx} 반환, 통과면 None.

    kind 는 process_record 의 복구 경로 분기용:
      low_hangul       → LLM 재판정 (영문/코드 출력 요구는 정당 — 사용자 지적 2026-08-23)
      codefence        → 해당 턴 1회 재번역(temp 0) 후 재판정
      그 외            → 즉시 리젝
    """
    final = out_msgs[-1]["content"]
    if not final.strip():
        return {"kind": "hard", "reason": "empty_final", "idx": None}
    for m in out_msgs:
        if m["content"] and SPECIAL_TOKEN_RE.search(m["content"]):
            return {"kind": "hard", "reason": "special_token_literal", "idx": None}
    user_text = " ".join(m["content"] for m in out_msgs if m["role"] == "user")
    if LEAK_RE.search(final) and not LEAK_RE.search(user_text):
        return {"kind": "hard", "reason": "teacher_leak", "idx": None}
    n_translated = len(out_msgs) if mode == "full_translate" else len(out_msgs) - 1
    for i, (s, o) in enumerate(zip(src_msgs[:n_translated], out_msgs[:n_translated])):
        if not s["content"]:
            continue
        if s["content"].count("```") != o["content"].count("```"):
            return {"kind": "codefence", "reason": "codefence_mismatch", "idx": i}
        if len(s["content"]) > 200 and len(o["content"]) < 0.25 * len(s["content"]):
            return {"kind": "hard", "idx": None,
                    "reason": f"translation_collapse:{len(o['content'])}/{len(s['content'])}"}
        if META_RE.search(o["content"]) and not META_RE.search(s["content"]):
            return {"kind": "hard", "reason": "meta_translation_leak", "idx": None}
    r = hangul_ratio(final)
    if r < HANGUL_MIN_RATIO:
        return {"kind": "low_hangul", "reason": f"low_hangul_ratio:{r:.2f}", "idx": None}
    return None


def low_hangul_is_legit(client, out_msgs):
    """비한글 위주 답변이 요청상 정당한지 1회 LLM 재판정.

    정당한 사례: 영문 작성/교정/번역 요청, 코드·설정파일·정규식 작성,
    표·로그 해석 등. 하드 임계값으로는 이들을 오탐으로 버리게 된다
    (r1 실측: 리젝의 70%가 이 게이트 — 사용자 지적으로 도입)."""
    user_last = [m for m in out_msgs if m["role"] == "user"][-1]["content"]
    final = out_msgs[-1]["content"]
    q = ("다음은 한국어 대화의 마지막 사용자 요청과 AI 답변입니다. 답변 본문이 "
         "한국어 산문 위주가 아닌 것(영문 텍스트·코드·표 위주)이 사용자 요청의 "
         "성격상 정당한지 판정하세요. 영문 작성·번역·코드 작성 요청 등이면 "
         "정당합니다. 한국어로 설명해야 할 요청인데 답변이 외국어라면 부당합니다.\n\n"
         f"[사용자 요청]\n{user_last[:800]}\n\n[답변 앞부분]\n{final[:1000]}\n\n"
         "정당하면 '예', 부당하면 '아니오'만 출력하세요.")
    try:
        out = client.chat([{"role": "user", "content": q}], max_tokens=5, temperature=0.0)
        return out.strip().startswith("예")
    except Exception:  # noqa: BLE001 — 판정 실패 시 보수적으로 리젝 유지
        return False


def decide_mode(row):
    tt = (row.get("metadata") or {}).get("train_turns")
    msgs = row["messages"]
    last_only = (isinstance(tt, list) and len(tt) == len(msgs)
                 and sum(tt) == 1 and tt[-1])
    if row.get("_source_split") == "chat_v3_if" or not last_only:
        return "full_translate"
    return "regen"


def process_record(client, row):
    mode = decide_mode(row)
    src_msgs = [m for m in row["messages"]
                if not (m["role"] == "system" and not (m["content"] or "").strip())]
    out_msgs, prev_pair, first_user_ko = [], None, None

    if mode == "full_translate":
        translate_upto = len(src_msgs)
    else:
        translate_upto = len(src_msgs) - 1

    for m in src_msgs[:translate_upto]:
        ko = translate_message(client, m["role"], m["content"], prev_pair,
                               constraint_ctx=first_user_ko)
        out_msgs.append({"role": m["role"], "content": ko})
        prev_pair = (m["content"], ko)
        if m["role"] == "user" and first_user_ko is None:
            first_user_ko = ko

    if mode == "regen":
        final_src = src_msgs[-1]
        gen_messages = []
        sys_texts = [m["content"] for m in out_msgs if m["role"] == "system"]
        sys_content = (sys_texts[0] + "\n\n" if sys_texts else "") + REGEN_SYSTEM_SUFFIX \
            + " " + length_hint(len(final_src["content"]))
        gen_messages.append({"role": "system", "content": sys_content})
        gen_messages.extend(m for m in out_msgs if m["role"] != "system")
        final_ko = client.chat(gen_messages, max_tokens=REGEN_MAX_TOKENS,
                               temperature=0.9).strip()
        out_msgs.append({"role": "assistant", "content": final_ko})

    fail = qc_checks(src_msgs, out_msgs, mode)
    qc_note = None

    # 복구 경로 1: 코드펜스 불일치 → 해당 턴만 temp 0 재번역 후 재판정
    if fail and fail["kind"] == "codefence":
        i = fail["idx"]
        out_msgs[i]["content"] = translate_message(
            client, src_msgs[i]["role"], src_msgs[i]["content"],
            prev_pair=None, constraint_ctx=first_user_ko, temperature=0.0)
        fail = qc_checks(src_msgs, out_msgs, mode)
        if fail is None:
            qc_note = "codefence_retry_ok"

    # 복구 경로 2: 한글비율 미달 → 요청상 정당성 LLM 재판정 (영문/코드 출력 요구)
    if fail and fail["kind"] == "low_hangul":
        if low_hangul_is_legit(client, out_msgs):
            qc_note = "low_hangul_llm_ok"
            fail = None

    meta = dict(row.get("metadata") or {})
    n_dropped = len(row["messages"]) - len(src_msgs)
    tt = meta.get("train_turns")
    if isinstance(tt, list) and len(tt) == len(row["messages"]):
        meta["train_turns"] = tt[n_dropped:]
    else:
        meta["train_turns"] = [False] * (len(out_msgs) - 1) + [True]
    meta.update({
        "source_uuid": row.get("uuid"),
        "source_split": row.get("_source_split"),
        "ko_synthesis": {
            "method": ("translate_context_regen_final" if mode == "regen"
                       else "full_translate"),
            "model": "google/gemma-4-31B-it",
            "date": "2026-08-23",
            **({"qc_note": qc_note} if qc_note else {}),
        },
        "model": "google/gemma-4-31B-it",
    })
    out_row = {
        "messages": out_msgs,
        "used_in": row.get("used_in", ["sft"]),
        "uuid": str(uuidlib.uuid5(uuidlib.NAMESPACE_URL, f"ko-chat-a/{row.get('uuid')}")),
        "metadata": meta,
    }
    return out_row, (fail["reason"] if fail else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-31b")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    rejects_path = out_dir / "rejects.jsonl"

    done = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                done.add(json.loads(line)["metadata"]["source_uuid"])
    if rejects_path.exists():
        with open(rejects_path) as f:
            for line in f:
                done.add(json.loads(line)["source_uuid"])

    rows = []
    with open(args.seeds) as f:
        for line in f:
            row = json.loads(line)
            if row.get("uuid") not in done:
                rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    print(f"todo={len(rows)} (skipped done={len(done)}) workers={args.workers}", flush=True)

    client = Client(args.base_url, args.model)
    lock = threading.Lock()
    stats = {"ok": 0, "reject": 0, "error": 0, "t0": time.time()}

    def work(row):
        try:
            try:
                out_row, fail = process_record(client, row)
            except MetaTranslation as e:
                out_row, fail = None, str(e)
            with lock:
                if fail:
                    stats["reject"] += 1
                    with open(rejects_path, "a") as f:
                        f.write(json.dumps({
                            "source_uuid": row.get("uuid"), "reason": fail,
                            "produced": out_row["messages"] if out_row else None,
                        }, ensure_ascii=False) + "\n")
                else:
                    stats["ok"] += 1
                    with open(results_path, "a") as f:
                        f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n = stats["ok"] + stats["reject"]
                if n % 50 == 0:
                    dt = time.time() - stats["t0"]
                    print(f"progress ok={stats['ok']} reject={stats['reject']} "
                          f"err={stats['error']} rate={n/dt:.2f} rec/s", flush=True)
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["error"] += 1
                print(f"ERROR uuid={row.get('uuid')}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, rows))

    dt = time.time() - stats["t0"]
    print(json.dumps({k: v for k, v in stats.items() if k != "t0"}) + f" wall={dt:.0f}s")


if __name__ == "__main__":
    main()
