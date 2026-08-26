#!/usr/bin/env python3
"""cut_tranche.py — ko_chat 트랜치 컷: 산출물 병합 → export 게이트 → 이관 jsonl.

1차 컷 (2026-08-26 18:00, 사용자 결정): LC-B 완주 직후 SFT 를 시작하기 위해
그 시점까지의 산출물을 확정한다. 합성은 계속 돌아 2차 트랜치가 된다.

트랙 A 입력 (전부 messages jsonl):
  - r1_a/results.jsonl            (finalize 완료 — think 보충 포함)
  - r2_a/results.jsonl 의 think 행 (컷 시점 스냅샷; append 중이라 마지막 partial 라인 무시)
  - tranche1/r2_if_pre.jsonl       (가드 이전 IF 61k, supplement finalize 완료본)
트랙 B 입력: artifacts/ko_chat_b_r1 parquet 중 **OxAlpha 재심판 전 축 ≥4** 인 행만
  (Gemma 셀프심판 무정보 실측 — 검증 안 된 r1 행은 2차로 미룸).

게이트 (순서대로, 사유별 카운트 출력):
  special-token 리터럴(content·reasoning) → 학습 턴 reasoning 부재(think 규약) →
  정규화 완전중복 → (source_split, 접두 40자)/(후미 40자) 버킷 상한 → 스키마 검사

사용:
  python3 cut_tranche.py --dry-run                 # 카운트만
  python3 cut_tranche.py --out-dir /home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-KoChat-v1
"""
import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")
DIMS = ["korean_naturalness", "helpfulness", "factuality", "coherence"]


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # append 중 partial 라인


def gate_row(rec, drops):
    msgs = rec.get("messages") or []
    if not msgs or msgs[-1].get("role") != "assistant" or not msgs[-1].get("content"):
        drops["schema"] += 1
        return False
    tt = rec["metadata"].get("train_turns")
    if not isinstance(tt, list) or len(tt) != len(msgs):
        drops["train_turns_len"] += 1
        return False
    for m in msgs:
        for f in ("content", "reasoning_content"):
            if m.get(f) and SPECIAL_TOKEN_RE.search(m[f]):
                drops["special_token"] += 1
                return False
    # think 규약: 학습 턴(assistant & train_turns True)은 reasoning 필수
    for m, t in zip(msgs, tt):
        if t and m["role"] == "assistant" and not (m.get("reasoning_content") or "").strip():
            drops["no_reasoning_on_trained_turn"] += 1
            return False
    return True


def dedup(records, key_fn, cap=3):
    seen_exact, pre, suf = set(), Counter(), Counter()
    out, drops = [], Counter()
    for rec in records:
        final = norm(rec["messages"][-1]["content"])
        h = hashlib.sha256(final.encode()).hexdigest()
        if h in seen_exact:
            drops["dup_exact"] += 1
            continue
        seen_exact.add(h)
        k = key_fn(rec)
        pk, sk = (k, final[:40]), (k, final[-40:])
        if pre[pk] >= cap or suf[sk] >= cap:
            drops["dup_bucket"] += 1
            continue
        pre[pk] += 1
        suf[sk] += 1
        out.append(rec)
    return out, drops


def collect_track_a():
    drops = Counter()
    recs = []
    srcs = [
        ("r1_a", HERE / "out/r1_a/results.jsonl", None),
        ("r2_a_think", HERE / "out/r2_a/results.jsonl", lambda r: r["metadata"]["ko_synthesis"].get("think")),
        ("r2_if_supp", HERE / "out/tranche1/r2_if_pre.jsonl", None),
    ]
    per_src = Counter()
    for name, path, pred in srcs:
        if not path.exists():
            print(f"  ! missing {path}", file=sys.stderr)
            continue
        for rec in iter_jsonl(path):
            if pred and not pred(rec):
                continue
            if not rec["metadata"]["ko_synthesis"].get("think"):
                drops["think_flag_missing"] += 1
                continue
            if gate_row(rec, drops):
                rec.pop("_source_split", None)
                recs.append(rec)
                per_src[name] += 1
    recs, d2 = dedup(recs, key_fn=lambda r: r["metadata"].get("source_split"))
    drops.update(d2)
    return recs, drops, per_src


def collect_track_b(b_mode="verified"):
    """b_mode: verified = OxAlpha 재심판 전 축 ≥4 행만 / all = reasoning 있는 새 런 행 전부
    (1차 트랜치는 사용자 결정 08-26 로 r1 완주분 전량 — 미검증, OxAlpha <4 판정분만 제외)."""
    import pandas as pd
    sys.path.insert(0, str(HERE))
    from export_ko_chat import build_record, _as_dict  # noqa: E402

    verified = {}
    for f in glob.glob(str(HERE / "out/judge_calib/*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            verified[r["record_uuid"]] = all((r["ox"].get(d) or 0) >= 4 for d in DIMS)
    drops = Counter()
    recs = []
    # DD 는 resume 폴백 시 타임스탬프 디렉토리를 새로 판다 (ko_chat_b_r1_08-24-2026_200831) —
    # 구 런(think 없음)·새 런 모두 훑고, 게이트가 think 부재를 거른다
    files = sorted(glob.glob(str(HERE / "artifacts/ko_chat_b_r1*/**/*.parquet"), recursive=True))
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:  # noqa: BLE001
            continue
        if "assistant_turn" not in df.columns:
            continue
        for _, row in df.iterrows():
            uid = str(row.get("record_uuid"))
            if uid in verified and not verified[uid]:
                drops["ox_below_4"] += 1
                continue
            if b_mode == "verified" and uid not in verified:
                drops["not_ox_verified"] += 1
                continue
            chk = _as_dict(row.get("ko_check"))
            if not chk.get("is_valid", False):
                drops["rule_gate"] += 1
                continue
            rec = build_record(row)
            if rec is None:
                drops["schema"] += 1
                continue
            rec["metadata"]["ko_synthesis"]["think"] = True
            rec["metadata"]["ko_synthesis"]["ox_verified"] = uid in verified
            if gate_row(rec, drops):
                recs.append(rec)
    recs, d2 = dedup(recs, key_fn=lambda r: (r["metadata"]["axes"].get("task_type"),
                                              r["metadata"]["axes"].get("domain")))
    drops.update(d2)
    return recs, drops


def summarize(name, recs, drops, extra=None):
    chars = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "")
                for r in recs for m in r["messages"])
    print(f"[{name}] kept={len(recs)} ≈{chars / 2.2 / 1e6:.0f}M tok(est) | drops={dict(drops)}"
          + (f" | src={dict(extra)}" if extra else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--b-mode", choices=["verified", "all"], default="all")
    args = ap.parse_args()

    a, da, src = collect_track_a()
    summarize("trackA", a, da, src)
    b, db = collect_track_b(args.b_mode)
    summarize("trackB", b, db)

    if args.dry_run or not args.out_dir:
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in (("trackA", a), ("trackB", b)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "cut": "tranche1", "trackA": len(a), "trackB": len(b),
        "drops": {"A": dict(da), "B": dict(db)}, "sources": dict(src),
        "schema": "Nemotron-SFT-Instruction-Following-Chat-v3 messages jsonl; "
                  "metadata.train_turns, reasoning_content on trained turns",
        "conversion_note": "IF 번역분(ko_synthesis.method=full_translate): 원본 GPT-OSS "
                           "medium-effort → --fanout-train-turns --medium-effort; "
                           "chat 재생성분·trackB: 자체 reasoning, effort 마커 없음",
    }
    with open(out / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"wrote → {out}")


if __name__ == "__main__":
    main()
