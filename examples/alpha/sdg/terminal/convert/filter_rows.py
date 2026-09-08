#!/usr/bin/env python3
"""filter_rows.py — 변환된 Terminus 행에 Ultra 의 품질 휴리스틱(7신호)을 적용하고 중복을 제거한다 (P3 선별).

입력: traj_to_terminus.py 산출 jsonl (여러 개). 출력: 통과 행 jsonl + FILTER_STATS.json.
신호 (Ultra 기술보고서 §Software Issue Resolution 의 목록을 우리 과제 형태에 맞게 구현):
  forbidden_git     git push/pull/fetch/clone/cherry-pick/reflog/fsck/remote/ls-remote 실행
  repeat_loop       같은 keystrokes 가 N(기본 4)회 이상 반복 (편집-테스트 무한 반복·제자리걸음)
  no_edit           전체 트라젝트리에 파일을 쓰는 명령이 없음 (heredoc/리다이렉트/sed -i/tee/python 파일쓰기)
  parse_error_rate  하니스가 "parsing errors/warnings" 를 돌려준 스텝 비율 > 임계 (기본 0.34)
  debug_residue     작성 코드에 pdb/breakpoint()/print("DEBUG 잔재
  edit_no_test      (oc 과제) solution.py 를 쓴 뒤 한 번도 실행(python3 … solution.py / samples) 하지 않음
  max_turns         assistant 턴 수 > 상한 (기본 30) — 요약 발동 근처·장황
  dup_task          같은 과제(metadata.task)의 행이 여러 개면 첫 행만 (k>1 재시도 대비)
사용: python3 filter_rows.py <rows.jsonl>... --out <dir> [--name terminal_synth] [--strict]
  --strict 는 edit_no_test·debug_residue 도 드롭 (기본은 통계만 기록하고 통과)
"""
import argparse, json, os, re, sys
from collections import Counter

GIT_FORBIDDEN = re.compile(r"\bgit\s+(push|pull|fetch|clone|cherry-pick|reflog|fsck|remote|ls-remote)\b")
WRITE_CMD = re.compile(r"<<\s*['\"]?\w+|>\s*/?\S+\.(py|txt|sh|json|c|cpp|md|csv|yaml|yml|ini|cfg)\b|\bsed\s+-i\b|\btee\b|open\([^)]*['\"]w|\bcat\s*>|\bprintf\b[^\n]*>|\becho\b[^\n]*>")
RUN_SOLUTION = re.compile(r"python3?\s+(/app/)?solution\.py|<\s*/?app/?samples/|samples/\d\.in|answer\.txt")
DEBUG = re.compile(r"\bimport pdb\b|\bpdb\.set_trace\(|\bbreakpoint\(\)|print\(\s*[\"']DEBUG")
PARSE_ERR = re.compile(r"Previous response had (parsing errors|warnings)", re.I)


def commands_of(row):
    """assistant 턴별 keystrokes 목록 (JSON 파싱 실패 턴은 빈 목록)."""
    out = []
    for m in row["messages"]:
        if m["role"] != "assistant":
            continue
        c = m["content"]; i, j = c.find("{"), c.rfind("}")
        try:
            js = json.loads(c[i:j + 1])
            out.append([x.get("keystrokes", "") for x in js.get("commands", []) if isinstance(x, dict)])
        except Exception:  # noqa: BLE001
            out.append([])
    return out


def signals(row, repeat_n, parse_thr, max_turns):
    cmds = commands_of(row)
    flat = [k for turn in cmds for k in turn]
    allcmd = "\n".join(flat)
    sig = {}
    sig["forbidden_git"] = bool(GIT_FORBIDDEN.search(allcmd))
    cnt = Counter(k.strip() for k in flat if k.strip())
    sig["repeat_loop"] = any(v >= repeat_n for v in cnt.values())
    sig["no_edit"] = not WRITE_CMD.search(allcmd)
    users = [m["content"] for m in row["messages"] if m["role"] == "user"]
    n_err = sum(1 for u in users if PARSE_ERR.search(u))
    n_asst = sum(1 for m in row["messages"] if m["role"] == "assistant")
    sig["parse_error_rate"] = (n_err / max(n_asst, 1)) > parse_thr
    sig["debug_residue"] = bool(DEBUG.search(allcmd))
    task = (row.get("metadata") or {}).get("task", "")
    if task.startswith("oc-"):
        wrote = [i for i, t in enumerate(cmds) if any("solution.py" in k and WRITE_CMD.search(k) for k in t)]
        ran = [i for i, t in enumerate(cmds) if any(RUN_SOLUTION.search(k) for k in t)]
        sig["edit_no_test"] = bool(wrote) and not any(r >= wrote[0] for r in ran)
    else:
        sig["edit_no_test"] = False
    sig["max_turns"] = n_asst > max_turns
    return sig, n_asst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="terminal_synth"); ap.add_argument("--strict", action="store_true")
    ap.add_argument("--repeat-n", type=int, default=4); ap.add_argument("--parse-thr", type=float, default=0.34)
    ap.add_argument("--max-turns", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hard = {"forbidden_git", "repeat_loop", "no_edit", "parse_error_rate", "max_turns"}
    soft = {"debug_residue", "edit_no_test"}
    drop_on = hard | (soft if a.strict else set())
    stats = Counter(); flagged = Counter(); seen_tasks = set(); kept = 0; turns = 0
    with open(os.path.join(a.out, f"{a.name}.jsonl"), "w") as fo, open(os.path.join(a.out, f"{a.name}.dropped.jsonl"), "w") as fd:
        for path in a.inputs:
            for line in open(path):
                if not line.strip():
                    continue
                row = json.loads(line); stats["rows_in"] += 1
                task = (row.get("metadata") or {}).get("task") or row["uuid"]
                if task in seen_tasks:
                    stats["drop:dup_task"] += 1; continue
                sig, n_asst = signals(row, a.repeat_n, a.parse_thr, a.max_turns)
                for k, v in sig.items():
                    if v: flagged[k] += 1
                bad = [k for k, v in sig.items() if v and k in drop_on]
                row.setdefault("metadata", {})["quality_flags"] = [k for k, v in sig.items() if v]
                if bad:
                    for k in bad: stats["drop:" + k] += 1
                    stats["rows_dropped"] += 1
                    fd.write(json.dumps({"uuid": row["uuid"], "task": task, "drop": bad}) + "\n"); continue
                seen_tasks.add(task); kept += 1; turns += n_asst
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["rows_kept"] = kept; stats["assistant_turns_kept"] = turns
    out = {"inputs": a.inputs, "strict": a.strict, "stats": dict(stats), "flagged_any": dict(flagged),
           "thresholds": {"repeat_n": a.repeat_n, "parse_thr": a.parse_thr, "max_turns": a.max_turns}}
    json.dump(out, open(os.path.join(a.out, "FILTER_STATS.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[filter] in={stats['rows_in']} kept={kept} dropped={stats['rows_dropped']} "
          f"drops={ {k[5:]: v for k, v in stats.items() if k.startswith('drop:')} } flagged={dict(flagged)}")


if __name__ == "__main__":
    main()
