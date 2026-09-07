#!/usr/bin/env python3
"""summarize_pilot.py — Harbor job 디렉토리 하나를 G-T1 수치로 요약한다.

입력: <job_dir> (harbor run -o … --job-name … 의 산출, 회수본)
  <job>/result.json                       집계 (stats.evals[*].metrics/n_trials/n_errors)
  <job>/<task>__<id>/result.json          트라이얼 (verifier_result.rewards, agent_result, started/finished)
  <job>/<task>__<id>/agent/trajectory.json ATIF: steps[].source in {user, agent}, agent step 에
                                           message(원문 응답: raw_content 모드) / reasoning_content /
                                           metrics{prompt_tokens, completion_tokens} / observation
측정:
  형식 유효율 = agent 스텝 중 Terminus-2 JSON(analysis/plan/commands) 파싱 성공 비율
  성공률      = reward==1 트라이얼 비율
  과제당 비용 = 프롬프트/완성 토큰 합, 스텝 수, 벽시계
  reasoning   = 스텝당 reasoning_content 문자 수 분포 (보존 렌더 시 문맥 길이 예측용)
사용: python3 summarize_pilot.py <job_dir> [--json out.json]
"""
import argparse, glob, json, os, re, statistics as st
from datetime import datetime


def extract_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


def valid_terminus(text):
    j = extract_json(text)
    return j is not None and all(k in j for k in ("analysis", "plan", "commands")) \
        and isinstance(j["commands"], list) and all(isinstance(c, dict) and "keystrokes" in c for c in j["commands"])


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def trial_summary(tdir):
    out = {"trial": os.path.basename(tdir)}
    try:
        r = json.load(open(os.path.join(tdir, "result.json")))
    except Exception as e:  # noqa: BLE001
        return {**out, "error": f"result.json: {e}"}
    vr = r.get("verifier_result") or {}
    rewards = vr.get("rewards") or {}
    out["reward"] = float(rewards.get("reward", 0.0) or 0.0) if isinstance(rewards, dict) else 0.0
    out["exception"] = (r.get("exception_info") or {}).get("exception_type")
    t0, t1 = parse_ts(r.get("started_at") or ""), parse_ts(r.get("finished_at") or "")
    out["wall_s"] = round((t1 - t0).total_seconds(), 1) if t0 and t1 else None
    tp = os.path.join(tdir, "agent", "trajectory.json")
    if os.path.exists(tp):
        t = json.load(open(tp))
        steps = [s for s in t.get("steps", []) if s.get("source") == "agent"]
        out["steps"] = len(steps)
        out["json_valid"] = sum(valid_terminus(s.get("message")) for s in steps)
        out["reasoning_chars"] = [len(s.get("reasoning_content") or "") for s in steps]
        fm = t.get("final_metrics") or {}
        out["prompt_tokens"] = fm.get("total_prompt_tokens")
        out["completion_tokens"] = fm.get("total_completion_tokens")
        out["parse_errors_observed"] = sum(
            1 for s in steps for o in ((s.get("observation") or {}).get("results") or [])
            if "parsing errors" in (o.get("content") or ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    trials = [trial_summary(d) for d in sorted(glob.glob(os.path.join(a.job_dir, "*__*"))) if os.path.isdir(d)]
    ok = [t for t in trials if "steps" in t]
    n = len(trials)
    succ = sum(1 for t in trials if t.get("reward", 0) >= 1.0)
    steps = sum(t["steps"] for t in ok)
    valid = sum(t["json_valid"] for t in ok)
    rc = [c for t in ok for c in t["reasoning_chars"]]
    agg = {
        "n_trials": n, "n_with_trajectory": len(ok),
        "success": succ, "success_rate": round(succ / n, 3) if n else None,
        "errors": sum(1 for t in trials if t.get("exception")),
        "agent_steps": steps,
        "json_valid": valid, "json_valid_rate": round(valid / steps, 3) if steps else None,
        "steps_per_trial": round(steps / len(ok), 1) if ok else None,
        "prompt_tokens_per_trial": round(st.mean(t["prompt_tokens"] or 0 for t in ok)) if ok else None,
        "completion_tokens_per_trial": round(st.mean(t["completion_tokens"] or 0 for t in ok)) if ok else None,
        "wall_s_per_trial": round(st.mean(t["wall_s"] for t in ok if t["wall_s"]), 1) if any(t.get("wall_s") for t in ok) else None,
        "reasoning_chars_per_step": {
            "median": st.median(rc) if rc else None, "p90": sorted(rc)[int(0.9 * (len(rc) - 1))] if rc else None,
            "max": max(rc) if rc else None, "zero_frac": round(sum(1 for c in rc if c == 0) / len(rc), 3) if rc else None},
    }
    print("=== G-T1 pilot summary ===")
    for k, v in agg.items():
        print(f"{k:28s} {v}")
    print("--- per trial ---")
    for t in trials:
        print(f"{t['trial'][:40]:40s} reward={t.get('reward')} steps={t.get('steps')} "
              f"valid={t.get('json_valid')} ptok={t.get('prompt_tokens')} ctok={t.get('completion_tokens')} "
              f"wall={t.get('wall_s')} exc={t.get('exception')}")
    if a.json:
        json.dump({"aggregate": agg, "trials": trials}, open(a.json, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
