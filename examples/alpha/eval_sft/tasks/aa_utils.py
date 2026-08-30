"""T1 커스텀 태스크 공용 유틸 — Artificial Analysis / Nemotron 3 Ultra 규약.

기존 lm_eval 내장 태스크(`mmlu_pro`, `gpqa_*_generative_n_shot`)는 **base 모델용**이다:
5-shot, 답 형식 지시 없음, 단일 정규식(`answer is (A)` / `The answer is`). 채팅·추론 모델은
그 형식으로 답하지 않아 추출이 대량 실패한다(2026-08-30 사고, `docs/KNOWN_ISSUES.md`).

이 모듈은 프론티어 규약을 이식한다. 근거는 `docs/SFT_BENCHMARKS.md` §3.4:

- **0-shot CoT** — few-shot은 base 모델 유물 (OpenAI simple-evals README)
- **답 형식을 프롬프트로 지시** — "The last line ... 'Answer: A'" (AA 표준, NVIDIA도 채택:
  Nemotron 3 Ultra 설정의 `++prompt_config=eval/aai/mcq-10choices-boxed`)
- **8단 폴백 추출, 항상 마지막 매치** — 모델의 자기 정정을 반영 (AA 방법론)
- **avg@k** — `pass@1`은 greedy 1회가 아니라 k회 평균 (DeepSeek-R1 카드: "generate 64
  responses per query to estimate pass@1")
- **`no_answer`를 1급 지표로 보고** — 추출 실패를 오답으로 흡수하면 측정 실패가 점수로
  위장된다 (NeMo-Skills 출력 규약)
"""

from __future__ import annotations

import os
import random
import re
import sys
from typing import Any, Sequence

# lm_eval 의 `!function` 로더는 모듈을 **파일 경로로 직접** 읽어 들이므로 이 디렉토리가
# sys.path 에 들어가지 않는다 (`tasks/_yaml_loader.py::_load_module_with_cache`).
# 형제 모듈(math_utils)을 import 하려면 경로를 명시적으로 붙여야 한다.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from math_utils import is_equiv, last_boxed_only_string, remove_boxed  # noqa: E402


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------- 프롬프트

def _letters(n: int) -> str:
    return LETTERS[:n]


def mc_query(question: str, options: Sequence[str]) -> str:
    """AA 표준 객관식 프롬프트.

    보기 개수에 맞춰 letter 범위를 만든다 — MMLU-Pro는 문항마다 3~10개로 가변이라
    A–J로 고정하면 존재하지 않는 보기를 제시하게 된다.
    """
    letters = _letters(len(options))
    spec = "/".join(letters)
    lines = [
        f"Answer the following multiple choice question. The last line of your response "
        f"should be in the following format: 'Answer: {spec}' (e.g. 'Answer: {letters[0]}').",
        "",
        str(question).strip(),
        "",
    ]
    lines += [f"{letters[i]}) {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines)


MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def math_query(problem: str) -> str:
    """R1/Qwen3 공통 수학 프롬프트 규약 (모델 카드 Usage Recommendations)."""
    return f"{str(problem).strip()}\n\n{MATH_INSTRUCTION}"


# ---------------------------------------------------------------- 추출

# AA 방법론의 다단 폴백. 순서대로 시도하고, **항상 마지막 매치**를 취한다.
_MC_PATTERNS = [
    r"[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*\(?([A-Z])\)?(?![a-zA-Z0-9])",
    r"\\boxed\{[^}]*?([A-Z])[^}]*?\}",
    r"answer is\s*\(?([A-Za-z])\)?",
    r"\(([A-Z])\)\s*$",
    r"([A-Z])\s+is\s+the\s+correct\s+answer",
    r"^\s*([A-Z])\s*$",
    r"([A-Z])\s*\.\s*$",
    # 보기 형식 그대로 답하는 경우("D) 10^-4 eV"). 오탐 위험이 있어 마지막 순서 —
    # 사고 구간을 잘라낸 뒤 적용하므로 보기 나열을 다시 집을 확률은 낮다.
    r"\b([A-Z])\)",
]
_MC_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _MC_PATTERNS]

_THINK_CLOSE = "</think>"


def split_think(text: str) -> tuple[str, bool]:
    """사고 구간을 잘라내고 (답변부, 사고를 닫았는지) 를 돌려준다.

    추출은 반드시 **사고 이후 구간**에서 해야 한다. 추론 모델의 사고 구간에는 보기
    나열·중간 후보·자기부정이 가득해서, 전문(全文)에 정규식을 걸면 사고 중의 잘못된
    후보를 답으로 집는다.

    `</think>` 가 없다는 것은 모델이 답변에 도달하지 못했다는 뜻이다(예산 소진 또는
    종료 실패). 그 사실 자체를 `think_closed` 지표로 보고한다 — 2026-08-30 사고에서
    "설정 결함"과 "모델 미성숙"을 가른 것이 정확히 이 신호였다.
    """
    if not text:
        return "", False
    idx = text.rfind(_THINK_CLOSE)
    if idx < 0:
        return text, False
    return text[idx + len(_THINK_CLOSE):], True


def extract_mc(text: str, n_choices: int) -> str | None:
    """객관식 답 추출. 유효 letter 범위를 벗어난 매치는 버린다."""
    if not text:
        return None
    valid = set(_letters(n_choices))
    stripped = text.strip()

    # 단일 letter 응답
    if len(stripped) == 1 and stripped.upper() in valid:
        return stripped.upper()

    for rx in _MC_COMPILED:
        matches = [m.group(1).upper() for m in rx.finditer(text)]
        matches = [m for m in matches if m in valid]
        if matches:
            return matches[-1]  # 자기 정정 반영 — 마지막 매치
    return None


def extract_math(text: str) -> str | None:
    """수학 답 추출: \\boxed{} 우선, 없으면 'answer is', 마지막으로 끝부분 숫자."""
    if not text:
        return None

    boxed = last_boxed_only_string(text)
    if boxed is not None:
        try:
            content = remove_boxed(boxed)
            if content is not None and content.strip():
                return content.strip()
        except (AssertionError, IndexError):
            pass

    m = list(re.finditer(r"(?:final\s+)?answer\s+is\s*:?\s*\$?([^\n\.\$]+)", text, re.IGNORECASE))
    if m:
        cand = m[-1].group(1).strip()
        if cand:
            return cand

    nums = re.findall(r"(-?\d+(?:\.\d+)?)", text[-400:])
    return nums[-1] if nums else None


# ---------------------------------------------------------------- 집계

def _responses(results: Any) -> list[str]:
    """lm_eval이 넘겨주는 결과를 k개 응답 리스트로 정규화.

    `repeats: k` 는 **동일 Instance를 k번 복제**해 `resps` 에 k개를 쌓는다
    (`evaluator.py`: `cloned_reqs.extend([req] * req.repeats)`). 기본 필터
    `take_first` 는 그중 1개만 남기므로, k개를 모두 보려면 태스크 yaml에서
    `take_first_k` 를 써야 한다 — 그때 이 함수가 받는 모양이 `[[s1..sk]]` 가 된다.
    (2026-08-30 확인: 기존 avg@16 태스크는 filter 미지정이라 16개를 생성하고
    1개만 채점하고 있었다 — 실질 avg@1.)
    """
    if len(results) == 1 and isinstance(results[0], (list, tuple)):
        return [str(x) for x in results[0]]
    return [str(x) for x in results]


def _score(
    correct: list[float],
    extracted: list[str | None],
    texts: list[str],
    closed: list[bool],
) -> dict:
    """avg@k 점수 + 진단 지표.

    `exact_match` 만 보면 측정 실패가 오답으로 위장된다. `no_answer` 와
    `think_closed` 를 함께 봐야 "모델이 틀렸다" / "형식이 안 맞았다" /
    "답변에 도달조차 못했다" 를 구분할 수 있다.
    """
    k = max(len(correct), 1)
    return {
        "exact_match": sum(correct) / k,
        "no_answer": sum(1 for e in extracted if e is None) / k,
        "think_closed": sum(1.0 for c in closed if c) / k,
        "gen_chars": sum(len(t) for t in texts) / k,
        "samples_k": float(len(texts)),
    }


def _prep(results: Any) -> tuple[list[str], list[str], list[bool]]:
    """(전문, 답변부, 사고닫힘) 3-튜플로 분해."""
    texts = _responses(results)
    parts = [split_think(t) for t in texts]
    return texts, [p[0] for p in parts], [p[1] for p in parts]


# ---------------------------------------------------------------- MMLU-Pro

def doc_to_text_mmlu_pro(doc: dict) -> str:
    return mc_query(doc["question"], doc["options"])


def doc_to_target_mmlu_pro(doc: dict) -> str:
    return _letters(len(doc["options"]))[doc["answer_index"]]


def process_results_mmlu_pro(doc: dict, results: Any) -> dict:
    texts, answers, closed = _prep(results)
    n = len(doc["options"])
    gold = doc_to_target_mmlu_pro(doc)
    extracted = [extract_mc(a, n) for a in answers]
    correct = [1.0 if e == gold else 0.0 for e in extracted]
    return _score(correct, extracted, texts, closed)


# ---------------------------------------------------------------- GPQA Diamond

def process_docs_gpqa(dataset):
    """보기 순서를 **문항별 결정론적**으로 섞는다.

    lm_eval 내장 GPQA는 `random.shuffle` 을 시드 없이 호출해 실행마다 정답 위치가
    바뀐다 — 재현이 안 된다. simple-evals 는 `random.Random(0)` 으로 고정한다.
    여기서는 Record ID 로 시드해 문항별로 고정하되 문항 간에는 섞이게 한다.
    """

    def _proc(doc):
        choices = [
            str(doc["Correct Answer"]).strip(),
            str(doc["Incorrect Answer 1"]).strip(),
            str(doc["Incorrect Answer 2"]).strip(),
            str(doc["Incorrect Answer 3"]).strip(),
        ]
        rng = random.Random(str(doc.get("Record ID", doc["Question"])))
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        return {
            "gpqa_question": str(doc["Question"]).strip(),
            "gpqa_choices": shuffled,
            "gpqa_answer": LETTERS[shuffled.index(choices[0])],
        }

    return dataset.map(_proc)


def doc_to_text_gpqa(doc: dict) -> str:
    return mc_query(doc["gpqa_question"], doc["gpqa_choices"])


def process_results_gpqa(doc: dict, results: Any) -> dict:
    texts, answers, closed = _prep(results)
    gold = doc["gpqa_answer"]
    extracted = [extract_mc(a, 4) for a in answers]
    correct = [1.0 if e == gold else 0.0 for e in extracted]
    return _score(correct, extracted, texts, closed)


# ---------------------------------------------------------------- 수학 (AIME / HMMT)

def doc_to_text_math(doc: dict) -> str:
    return math_query(doc["problem"])


def process_results_math(doc: dict, results: Any) -> dict:
    texts, answers, closed = _prep(results)
    key = next(k for k in doc if k.lower() == "answer")
    gold = str(doc[key]).strip()
    extracted = [extract_math(a) for a in answers]
    correct = [1.0 if (e is not None and is_equiv(e, gold)) else 0.0 for e in extracted]
    return _score(correct, extracted, texts, closed)


# ---------------------------------------------------------------- IFEval

def process_results_ifeval(doc: dict, results: Any) -> dict:
    """IFEval avg@k — 채점 로직은 내장 구현을 그대로 쓰고, 입력만 바로잡는다.

    내장 `ifeval` 태스크를 그대로 쓸 수 없는 이유 둘:
    1. `results[0]` 를 채점한다 — 사고 구간(`<think>...`)이 응답에 포함되면
       "전부 소문자로", "불릿 3개" 같은 지시 검사가 사고 텍스트까지 보고 판정한다.
    2. `max_gen_toks: 1280` — 추론 모델은 사고에만 그 이상을 쓴다.

    지시 검사기 자체는 벤치마크의 정의이므로 건드리지 않는다.
    """
    from lm_eval.tasks.ifeval.utils import (
        InputExample,
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    _, answers, closed = _prep(results)
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )

    k = max(len(answers), 1)
    n_inst = len(doc["instruction_id_list"])
    prompt_strict, prompt_loose = [], []
    inst_strict = [0.0] * n_inst
    inst_loose = [0.0] * n_inst

    for ans in answers:
        s = test_instruction_following_strict(inp, ans)
        lo = test_instruction_following_loose(inp, ans)
        prompt_strict.append(float(s.follow_all_instructions))
        prompt_loose.append(float(lo.follow_all_instructions))
        for i, v in enumerate(s.follow_instruction_list):
            inst_strict[i] += float(v) / k
        for i, v in enumerate(lo.follow_instruction_list):
            inst_loose[i] += float(v) / k

    return {
        "prompt_level_strict_acc": sum(prompt_strict) / k,
        "inst_level_strict_acc": inst_strict,
        "prompt_level_loose_acc": sum(prompt_loose) / k,
        "inst_level_loose_acc": inst_loose,
        "think_closed": sum(1.0 for c in closed if c) / k,
        "samples_k": float(k),
    }


def agg_inst_level_acc(items):
    """내장 구현과 동일 — 문항별 지시 리스트를 평탄화해 평균."""
    flat = [x for sub in items for x in sub]
    return sum(flat) / len(flat) if flat else 0.0
