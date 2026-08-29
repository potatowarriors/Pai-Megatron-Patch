"""벤치 결과 집계 — results/<run>_iter<N>/ 아래 lm_eval JSON 들을 iter별 추이표로.

각 eval_ckpt 실행이 results/<RUN_TAG>/ 에 lm_eval 결과(results_*.json)를 남긴다.
이 스크립트가 전부 훑어 (run, iter, task) → metric 표를 TRACKING.md 로 만든다.
반복 평가에서 학습 곡선을 한 눈에 보기 위한 것.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

# 태스크별 대표 메트릭 (표에 뽑을 것)
METRIC = {
    "mmlu_pro": "exact_match,custom-extract",
    "gpqa_diamond_generative_n_shot": "exact_match,flexible-extract",
    "aime25": "exact_match,none",
    "hmmt_feb_2025": "exact_match,none",
    "ifeval": "inst_level_loose_acc,none",
    "gpqa_diamond": "acc,none",
    "simpleqa_verified": "accuracy,none",
    "logickor": "score,none",
}
# 표 컬럼 = 상위 태스크만(하위 카테고리 mmlu_pro_* 등 제외). 순서 고정.
TASK_ORDER = ["mmlu_pro", "gpqa_diamond_generative_n_shot", "aime25", "hmmt_feb_2025", "ifeval", "simpleqa_verified", "logickor"]

def pick_metric(task_res: dict, task: str) -> float | None:
    pref = METRIC.get(task)
    if pref and pref in task_res: return task_res[pref]
    for k, v in task_res.items():
        if isinstance(v, (int, float)) and k not in ("alias",) and "stderr" not in k:
            return v
    return None

def parse_tag(tag: str):
    m = re.search(r"_iter(\d+)$", tag)
    return (tag[:m.start()] if m else tag, int(m.group(1)) if m else -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}  # (run, iter) -> {task: value}
    tasks = set()
    for tagdir in sorted(Path(a.results_dir).glob("*")):
        if not tagdir.is_dir(): continue
        jsons = list(tagdir.rglob("results_*.json")) + list(tagdir.rglob("results.json"))
        if not jsons: continue
        run, it = parse_tag(tagdir.name)
        cell = rows.setdefault((run, it), {})
        keep = set(TASK_ORDER) | set(METRIC)
        # 같은 디렉토리의 모든 결과 JSON을 mtime 오름차순 병합 → 나중 실행이 같은 태스크를 덮어씀
        for jf in sorted(jsons, key=lambda p: p.stat().st_mtime):
            try:
                res = json.loads(jf.read_text()).get("results", {})
            except Exception:
                continue
            for task, tr in res.items():
                if task not in keep:
                    continue
                v = pick_metric(tr, task)
                if v is not None:
                    cell[task] = v; tasks.add(task)
    ordered = [t for t in TASK_ORDER if t in tasks]
    extra = sorted(t for t in tasks if t not in TASK_ORDER and t in METRIC)
    tasks = ordered + extra
    lines = ["# 벤치 추이 (eval_ckpt 집계)\n",
             "각 SFT 체크포인트별 점수. `eval_ckpt.sh`/`eval_watch.sh` 가 갱신. 100분율.\n",
             "| run | iter | " + " | ".join(tasks) + " |",
             "|---|---|" + "|".join("---" for _ in tasks) + "|"]
    for (run, it) in sorted(rows, key=lambda x: (x[0], x[1])):
        cell = rows[(run, it)]
        vals = []
        for t in tasks:
            v = cell.get(t)
            vals.append(f"{v*100:.1f}" if isinstance(v, float) else "—")
        lines.append(f"| {run} | {it} | " + " | ".join(vals) + " |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"[aggregate] {len(rows)} 체크포인트 → {a.out}")

if __name__ == "__main__":
    main()
