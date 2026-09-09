#!/usr/bin/env python3
"""extract_ko_seeds.py — ko_chat v3 실사용자 한국어 프롬프트 시드 추출.

NVIDIA Chat-v3 레시피의 프롬프트 원천(lmsys-chat-1m·WildChat-1M)에서 language=Korean 대화를
스트리밍으로 걸러 시드로 저장한다(전량 다운로드 회피). 번역이 아니라 실사용자 원문이다.

출력 1행 = {source, conv_id, language, turns:[{role,content}], first_user, n_user_turns}.
학습 데이터가 아니라 '시드'다 — 응답은 이후 GLM-5.3-Flash 교사가 네이티브 사고와 함께 재생성한다.

실행:
  NUMEXPR_MAX_THREADS=64 HF_TOKEN=... python3 extract_ko_seeds.py \
      --out out/seeds_ko_pilot.jsonl --cap-lmsys 30000 --cap-wildchat 30000
"""
import argparse, json, os, re, hashlib, sys, time

def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def first_user(turns):
    for t in turns:
        if t["role"] == "user":
            return t["content"]
    return ""

def load_turns(row, src):
    # lmsys-chat-1m: row["conversation"] = [{"role","content"}], row["language"]
    # WildChat-1M-Full: row["conversation"] = [{"role","content","language",...}], row["language"]
    conv = row.get("conversation") or row.get("messages") or []
    turns = []
    for m in conv:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value") or ""
        if role in ("user", "human"):
            role = "user"
        elif role in ("assistant", "gpt", "bot", "chatbot"):
            role = "assistant"
        else:
            continue
        if isinstance(content, list):
            content = " ".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
        turns.append({"role": role, "content": content})
    return turns

def stream_source(name, hf_id, cap, seen, out, args):
    from datasets import load_dataset
    kept = scanned = 0
    t0 = time.time()
    ds = load_dataset(hf_id, split="train", streaming=True, token=os.environ.get("HF_TOKEN"))
    for row in ds:
        scanned += 1
        lang = row.get("language") or row.get("lang")
        if lang != "Korean":
            if scanned % 50000 == 0:
                print(f"[{name}] scanned={scanned} kept={kept} ({time.time()-t0:.0f}s)", flush=True)
            continue
        turns = load_turns(row, name)
        fu = first_user(turns)
        if not fu.strip() or len(fu) < 4:
            continue
        # 한국어 프롬프트만 (라틴 위주 오탐 제거: 한글 1자 이상)
        if not re.search(r"[가-힣]", fu):
            continue
        h = hashlib.sha1(norm(fu)[:400].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        n_user = sum(1 for t in turns if t["role"] == "user")
        rec = {"source": name, "conv_id": row.get("conversation_id") or row.get("id") or h[:16],
               "language": lang, "turns": turns, "first_user": fu, "n_user_turns": n_user}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += 1
        if kept % 2000 == 0:
            print(f"[{name}] scanned={scanned} kept={kept} ({time.time()-t0:.0f}s)", flush=True)
        if kept >= cap:
            break
    print(f"[{name}] DONE scanned={scanned} kept={kept} ({time.time()-t0:.0f}s)", flush=True)
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-lmsys", type=int, default=30000)
    ap.add_argument("--cap-wildchat", type=int, default=30000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen = set()
    total = 0
    with open(args.out, "w") as out:
        if args.cap_lmsys > 0:
            total += stream_source("lmsys", "lmsys/lmsys-chat-1m", args.cap_lmsys, seen, out, args)
        if args.cap_wildchat > 0:
            total += stream_source("wildchat", "allenai/WildChat-1M-Full", args.cap_wildchat, seen, out, args)
    print(f"TOTAL seeds={total} unique_hashes={len(seen)} -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
