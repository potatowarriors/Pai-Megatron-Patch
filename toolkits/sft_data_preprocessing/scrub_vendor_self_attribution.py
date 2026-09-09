#!/usr/bin/env python3
"""scrub_vendor_self_attribution.py — phase-2 회귀 P2 교정: 교사 벤더 자기귀속 행 제거 (KNOWN_ISSUES 2026-09-09).

assistant 턴(content + reasoning_content)에서 **자기귀속 문맥**("I was trained by Google", "저는 구글에서 훈련시킨…")만 잡는다.
블랭킷 벤더명 매칭은 하지 않는다 — "클로드 마켈렐레", "OpenAI 의 CLIP 논문" 같은 정상 언급을 버리면 안 된다.
부정문("not developed by Google", "구글이 아니라")은 자기귀속이 아니므로 통과.

원본은 건드리지 않고 <src>.p2scrub.jsonl 을 새로 쓴다. 리포트는 stdout + <src>.p2scrub.report.json.

사용: python3 scrub_vendor_self_attribution.py <in.jsonl> [<in2.jsonl> ...]  (--dry-run 으로 집계만)
"""
import argparse, json, re, sys, time

VENDORS = r"(Qwen|通义千问|GLM|Zhipu|智谱|Alibaba|알리바바|Google|구글|Gemini|제미나이|Gemma|OpenAI|오픈AI|ChatGPT|GPT-4|GPT-5|Anthropic|앤트로픽|Claude|클로드|DeepSeek|딥시크|Moonshot|Kimi|Mistral|Meta AI|Llama|PaLM|Bard|LaMDA)"
SELF_ATTR = re.compile(
    r"\b(I am|I'm|I’m|I was|I have been|I've been|as an?|this is)\b[^.\n]{0,30}?" + VENDORS + r"|"
    r"\b(developed|trained|created|made|built|designed)\s+by\s+" + VENDORS + r"|"
    r"(저는|나는|제가)\s*[^.\n]{0,20}?" + VENDORS + r"|"
    + VENDORS + r"\s*(이|가|에서|에 의해|팀이)\s*(개발|훈련|만든|만들|제작|학습)", re.I)
NEG = re.compile(r"\b(not|no longer|neither|nor)\b|아니|않|아닙", re.I)

def is_self_attr(text):
    for m in SELF_ATTR.finditer(text or ""):
        ctx = text[max(0, m.start() - 60): m.end() + 40]
        if NEG.search(ctx): continue
        return m.group(0)[:80]
    return None

def scrub(path, dry):
    t0 = time.time(); n = kept = dropped = bad = 0; examples = []
    out = None if dry else open(path.replace(".jsonl", "") + ".p2scrub.jsonl", "w")
    for line in open(path):
        n += 1
        try: d = json.loads(line)
        except Exception: bad += 1; continue
        hit = None
        for m in d.get("messages") or []:
            if m.get("role") not in ("assistant", "gpt"): continue
            for field in ("content", "reasoning_content"):
                v = m.get(field)
                if isinstance(v, list): v = json.dumps(v, ensure_ascii=False)
                hit = is_self_attr(v or "")
                if hit: break
            if hit: break
        if hit:
            dropped += 1
            if len(examples) < 8: examples.append(hit)
        else:
            kept += 1
            if out: out.write(line)
        if n % 200000 == 0: print(f"  {path}: {n} rows, dropped {dropped} ({time.time()-t0:.0f}s)", flush=True)
    if out: out.close()
    rep = {"path": path, "rows": n, "kept": kept, "dropped": dropped, "bad_json": bad, "drop_rate": round(dropped / max(n, 1), 5),
           "examples": examples, "seconds": round(time.time() - t0)}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    if not dry: json.dump(rep, open(path.replace(".jsonl", "") + ".p2scrub.report.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for p in a.paths: scrub(p, a.dry_run)
