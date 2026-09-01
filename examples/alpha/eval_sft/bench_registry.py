"""벤치 태스크 ↔ 지표 매핑의 **단일 정본**.

집계기(`aggregate_results.py`)·wandb 로거(`log_eval_wandb.py`)·판정기(`summarize.py`)가
같은 표를 본다. 2026-08-30 에 T1 태스크를 `*_aa` 로 재작성했을 때 이 매핑이 세 파일에
복제돼 있어 두 곳이 구 이름을 참조한 채 남았다 — 결과를 읽지 못했다.

새 태스크를 추가하거나 이름을 바꾸면 **여기만** 고친다.
"""

from __future__ import annotations

# task -> (대표 지표명, 표시 이름)
#   지표명은 `metric` 또는 `metric,filter` 중 metric 부분만 적는다 — 조회는 앞부분 일치.
#   여러 지표를 `("a","b")` 튜플로 주면 **평균**을 쓴다 (RULER 구간 평균 등).
#
# 대표 지표는 **프론티어 보고 관행**에 맞춘다 (2026-08-31 감사):
#   - IFEval: `prompt_level_strict_acc` — 4지표 중 가장 엄격. 구 `inst_level_loose_acc`
#     는 가장 관대해 점수를 10~15pp 높게 보이게 했다.
#   - RULER: 구간 평균 (Qwen3 카드의 "Avg" 열과 같은 방식). 구 구현은 131072 한 구간만
#     보아 65536 신호를 버렸다.
HEADLINE: dict[str, tuple[str | tuple[str, ...], str]] = {
    # ── T1 코어 (AA/Nemotron 규약, 2026-08-30 재작성)
    "mmlu_pro_aa":            ("exact_match",          "mmlu_pro"),
    "gpqa_diamond_aa":        ("exact_match",          "gpqa_diamond"),
    "aime25_aa":              ("exact_match",          "aime25"),
    "hmmt_feb_2025_aa":       ("exact_match",          "hmmt_feb_2025"),
    "ifeval_aa":              ("prompt_level_strict_acc", "ifeval_prompt_strict"),
    # ── T2 롱컨텍스트 (Reasoning-Off)
    "ruler_niah_single_1_aa":   (("65536", "131072", "262144"), "ruler_single_1_avg"),
    "ruler_niah_single_2_aa":   (("65536", "131072", "262144"), "ruler_single_2_avg"),
    "ruler_niah_multikey_1_aa": (("65536", "131072", "262144"), "ruler_multikey_avg"),
    "ruler_niah_multivalue_aa": (("65536", "131072", "262144"), "ruler_multivalue_avg"),
    # ── T3 판정 (심판 러너가 직접 JSON 을 쓴다)
    "simpleqa_verified":      ("accuracy",     "simpleqa_verified"),
    "logickor":               ("score",        "logickor"),
    # ── 에이전틱
    "swe_bench_verified":     ("resolved",     "swe_verified"),
    "terminal_bench":         ("resolved",     "terminal_bench"),
}

# 표 컬럼 순서
TASK_ORDER = [
    "mmlu_pro_aa", "gpqa_diamond_aa", "aime25_aa", "hmmt_feb_2025_aa", "ifeval_aa",
    "ruler_niah_single_1_aa", "ruler_niah_single_2_aa",
    "ruler_niah_multikey_1_aa", "ruler_niah_multivalue_aa",
    "simpleqa_verified", "logickor", "swe_bench_verified", "terminal_bench",
]

# 측정이 성립했는지 보는 진단 지표 (점수가 아니라 **게이트**).
# 프론티어 규약: 점수만 보면 측정 실패가 오답으로 위장된다 (NeMo-Skills `no_answer`).
DIAGNOSTIC = ("no_answer", "think_closed", "gen_chars", "samples_k", "empty")

# 무효 판정 임계
NO_ANSWER_MAX = 0.10      # 추출 실패율 — 넘으면 하니스/서빙 결함 의심
THINK_CLOSED_MIN = 0.50   # 사고 마감률 — 밑돌면 예산 부족 또는 모델 미성숙


def get(task_res: dict, metric: str) -> float | None:
    """`metric` 또는 `metric,filter` 키에서 값을 찾는다."""
    for k, v in task_res.items():
        if not isinstance(v, (int, float)) or "stderr" in k:
            continue
        if k == metric or k.split(",")[0] == metric:
            return float(v)
    return None


def headline(task: str, task_res: dict) -> float | None:
    """대표 점수. 튜플이면 지표들의 평균. 미등록 태스크는 첫 숫자 지표로 대체."""
    spec = HEADLINE.get(task)
    if spec:
        want = spec[0] if isinstance(spec[0], tuple) else (spec[0],)
        vals = [v for v in (get(task_res, m) for m in want) if v is not None and v >= 0]
        if vals:
            return sum(vals) / len(vals)
    for k, v in task_res.items():
        if (isinstance(v, (int, float)) and "stderr" not in k and k != "alias"
                and k.split(",")[0] not in DIAGNOSTIC):
            return float(v)
    return None


def display_name(task: str) -> str:
    spec = HEADLINE.get(task)
    return spec[1] if spec else task


def invalid_reasons(task_res: dict) -> list[str]:
    """무효 판정 사유. 비어 있으면 유효."""
    out = []
    na = get(task_res, "no_answer")
    tc = get(task_res, "think_closed")
    if na is not None and na > NO_ANSWER_MAX:
        out.append(f"추출실패 {na * 100:.0f}%>{NO_ANSWER_MAX * 100:.0f}%")
    if tc is not None and tc < THINK_CLOSED_MIN:
        out.append(f"사고마감 {tc * 100:.0f}%<{THINK_CLOSED_MIN * 100:.0f}%")
    return out
