#!/usr/bin/env python3
"""dataset_stats.py — Terminus 형 SFT jsonl 의 기준 분포 (카테고리·턴 수·reasoning/응답/터미널출력 길이·행 토큰·프로토콜 준수).

공개 코퍼스(nvidia/Nemotron-Terminal-Corpus) 채택/보강 판정의 대조 기준 (2026-09-10, README §5 트랙 전환). 같은 스크립트를
공개 코퍼스 jsonl(messages 스키마) 에도 돌리면 동일 지표로 비교된다.

사용: python3 dataset_stats.py --jsonl <path> [--jsonl <path2> ...] --tokenizer <dir> --out REFERENCE_STATS.md [--json out.json]
지표: rows · prefix/source 분포 · assistant 턴/행 (mean/median/p90/max, 구간 히스토그램) · reasoning 보유 턴 비율 ·
      턴당 토큰(reasoning / 응답 content / user 터미널 출력) · 행당 토큰(전 메시지 합, 128k 초과 비율) · system md5 ·
      JSON 파싱률 · 턴당 commands 수 · task_complete 핸드셰이크 비율 · "New Terminal Output:" 접두 비율
"""
import argparse, hashlib, json, os, statistics as st, sys
from collections import Counter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def pct(xs, q):
    if not xs: return 0
    xs = sorted(xs); k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def summ(xs):
    if not xs: return {"n": 0}
    return {"n": len(xs), "mean": round(st.fmean(xs), 1), "median": pct(xs, 0.5), "p90": pct(xs, 0.9), "p99": pct(xs, 0.99), "max": max(xs)}


def bucket(n):
    for lo, hi, lab in ((1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 30, "21-30")):
        if lo <= n <= hi: return lab
    return ">30"


def prefix_of(r):
    m = r.get("metadata") or {}; t = str(m.get("task", ""))
    if t[:3] in ("oc-", "sc-", "om-"): return t[:2]
    if m.get("split"): return str(m["split"]).split("/")[0]      # 공개 코퍼스 변환 행: adapters:code|math|swe / synthetic:easy|medium|mixed
    return m.get("source") or t or "?"


def analyze(path, tok):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    pre = Counter(); turns = []; buckets = Counter(); sys_md5 = Counter()
    r_tok, c_tok, u_tok, row_tok = [], [], [], []
    n_asst = n_reason = n_json_ok = n_cmds_turns = 0; cmds = []; n_tc = 0; n_user = n_nto = 0
    texts, kinds = [], []
    for r in rows:
        pre[prefix_of(r)] += 1
        ms = r["messages"]
        if ms and ms[0]["role"] == "system":
            sys_md5[hashlib.md5(ms[0]["content"].encode()).hexdigest()[:8]] += 1
        a = [m for m in ms if m["role"] == "assistant"]; turns.append(len(a)); buckets[bucket(len(a))] += 1
        for m in ms:
            if m["role"] == "assistant":
                n_asst += 1
                if m.get("reasoning_content"): n_reason += 1; texts.append(m["reasoning_content"]); kinds.append("r")
                texts.append(m.get("content") or ""); kinds.append("c")
                try:
                    j = json.loads(m.get("content") or ""); n_json_ok += 1
                    if isinstance(j, dict):
                        c = j.get("commands"); 
                        if isinstance(c, list): cmds.append(len(c)); n_cmds_turns += 1
                except Exception: pass
            elif m["role"] == "user":
                n_user += 1; texts.append(m.get("content") or ""); kinds.append("u")
                if (m.get("content") or "").lstrip().startswith("New Terminal Output:"): n_nto += 1
            else:
                texts.append(m.get("content") or ""); kinds.append("s")
        if a:
            try:
                j = json.loads(a[-1].get("content") or ""); n_tc += 1 if (isinstance(j, dict) and j.get("task_complete") is True) else 0
            except Exception: pass
    # 토큰화 (배치)
    lens = []
    B = 512
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], add_special_tokens=False)["input_ids"]; lens.extend(len(e) for e in enc)
    # 행별 합
    idx = 0
    for r in rows:
        ms = r["messages"]; total = 0
        for m in ms:
            if m["role"] == "assistant":
                if m.get("reasoning_content"): total += lens[idx]; r_tok.append(lens[idx]); idx += 1
                total += lens[idx]; c_tok.append(lens[idx]); idx += 1
            elif m["role"] == "user": total += lens[idx]; u_tok.append(lens[idx]); idx += 1
            else: total += lens[idx]; idx += 1
        row_tok.append(total)
    assert idx == len(lens)
    return {
        "path": path, "rows": len(rows), "by_prefix": dict(pre), "system_md5": dict(sys_md5.most_common(5)),
        "assistant_turns_per_row": summ(turns), "turn_buckets": {k: buckets[k] for k in ("1-2", "3-5", "6-10", "11-20", "21-30", ">30")},
        "assistant_turns_total": n_asst, "reasoning_turn_ratio": round(n_reason / max(n_asst, 1), 4),
        "reasoning_tokens_per_turn": summ(r_tok), "response_tokens_per_turn": summ(c_tok), "user_tokens_per_turn": summ(u_tok),
        "row_tokens": summ(row_tok), "row_tokens_total": sum(row_tok), "rows_over_128k": sum(1 for x in row_tok if x > 131072),
        "json_parse_ratio": round(n_json_ok / max(n_asst, 1), 4), "commands_per_turn": summ(cmds),
        "task_complete_final_ratio": round(n_tc / max(len(rows), 1), 4), "new_terminal_output_prefix_ratio": round(n_nto / max(n_user, 1), 4),
    }


def md(stats):
    keys = [("rows", "행"), ("assistant_turns_total", "assistant 턴 합"), ("reasoning_turn_ratio", "reasoning 보유 턴 비율"),
            ("json_parse_ratio", "응답 JSON 파싱률"), ("task_complete_final_ratio", "마지막 턴 task_complete=true 비율"),
            ("new_terminal_output_prefix_ratio", "user 턴 'New Terminal Output:' 접두 비율"), ("row_tokens_total", "행 토큰 합"), ("rows_over_128k", "128k 초과 행")]
    out = ["# 터미널 SFT 기준 분포 (dataset_stats.py, tokenizer_v5)", "",
           "공개 코퍼스 채택/보강 판정용 대조 기준 (README §5 트랙 전환). 토큰은 메시지 원문 기준(템플릿 마커 제외).", ""]
    names = [os.path.basename(os.path.dirname(s["path"])) + "/" + os.path.basename(s["path"]) for s in stats]
    out.append("| 지표 | " + " | ".join(names) + " |"); out.append("|---|" + "---|" * len(stats))
    for k, lab in keys: out.append(f"| {lab} | " + " | ".join(f"{s[k]:,}" if isinstance(s[k], int) else str(s[k]) for s in stats) + " |")
    out.append("| prefix/source | " + " | ".join(", ".join(f"{k} {v:,}" for k, v in s["by_prefix"].items()) for s in stats) + " |")
    out.append("| system md5 | " + " | ".join(", ".join(f"{k} {v:,}" for k, v in s["system_md5"].items()) for s in stats) + " |")
    for k, lab in (("assistant_turns_per_row", "assistant 턴/행"), ("reasoning_tokens_per_turn", "reasoning 토큰/턴"), ("response_tokens_per_turn", "응답 토큰/턴"),
                   ("user_tokens_per_turn", "user(터미널 출력) 토큰/턴"), ("row_tokens", "행 토큰"), ("commands_per_turn", "commands/턴")):
        out.append(f"| {lab} mean / median / p90 / max | " + " | ".join(
            f"{s[k].get('mean', 0)} / {s[k].get('median', 0)} / {s[k].get('p90', 0)} / {s[k].get('max', 0)}" for s in stats) + " |")
    out.append("| 턴 수 구간 1-2 / 3-5 / 6-10 / 11-20 / 21-30 / >30 | " + " | ".join(
        " / ".join(str(s["turn_buckets"][b]) for b in ("1-2", "3-5", "6-10", "11-20", "21-30", ">30")) for s in stats) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jsonl", action="append", required=True); ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--json"); a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    stats = []
    for p in a.jsonl:
        s = analyze(p, tok); stats.append(s); print(f"[stats] {p}: rows={s['rows']:,} asst_turns={s['assistant_turns_total']:,} row_tok_total={s['row_tokens_total']:,}", flush=True)
    open(a.out, "w").write(md(stats))
    if a.json: json.dump(stats, open(a.json, "w"), ensure_ascii=False, indent=1)
    print(f"[stats] -> {a.out}")


if __name__ == "__main__":
    main()
