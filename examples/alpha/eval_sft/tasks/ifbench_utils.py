"""IFBench (AllenAI) avg@k — 공식 검사기(instructions_registry)를 evaluation_lib 의 strict·loose 루프 그대로 부르고, 입력만 바로잡는다 (2026-11-09).

IFBench 는 학습에 쓰지 않은 지시 유형(고정 데이터 294 프롬프트 · 57종)으로 지시 이행 일반화를 잰다. IFEval 은 RL 보상 검사기와 같은
유형이라 분포 안 측정이 된다 — NeMo-RL General teacher 평가에서 사용자가 추가를 결정했다. 단 "분포 밖" 은 지시 id 기준이다:
General 블렌드에는 IFBench 와 **문장 템플릿이 같은 유형 2종**과 **비슷한 유형 6종**이 있다 (`SAME_TEMPLATE_TYPES`·`SIMILAR_TYPES`,
2026-11-09 적대적 검토 지적). 비교 보고는 `eval_sft/ifbench_compare.py` 로 이 유형을 뺀 부분 점수를 함께 낸다.

공식 구현과 다르게 하는 것은 넷뿐이다 (채점 규칙은 건드리지 않는다):
  1. **사고 구간을 떼고 채점한다** (`aa_utils.split_think`, ifeval_aa 와 같은 함수). IFBench README 도 사고 모델은
     "reasoning chain 없이 답만 추출" 한다고 적는다. `</think>` 가 없으면(예산 소진·종료 실패) **빈 답**으로 채점한다 —
     ifeval_aa 는 이때 전문을 채점하지만, 사고 텍스트가 "키워드 포함" 같은 지시를 우연히 만족시키는 것을 막는다.
     그래서 점수 = think_closed × (닫힌 응답의 점수) 다 — 비교 보고는 닫힌 응답만의 점수와 think_closed 를 나란히 낸다.
  2. **kwargs 는 원본 jsonl 에서 읽는다.** HF datasets(Arrow)가 행마다 다른 키를 합집합으로 채우고 int/float 를 섞어
     형변환할 수 있어서, lm_eval 의 doc 은 key·prompt 확인에만 쓴다.
  3. **NLTK 를 네트워크 없이 쓴다.** 공식 코드는 검사 때마다 `nltk.download(...)` 를 부른다. 같은 자료를 NFS
     (`$IFBENCH_HOME/nltk_data`)에 미리 받아 두고, download 는 자료가 있는지만 확인한다(없으면 예외).
  4. **검사기 예외를 미이행으로 센다** (NeMo-Gym `resources_servers/ifbench/app.py` 와 같은 처리). 공식 `evaluation_lib` 는
     예외에서 멈춘다 — 예: 답이 "." 이면 `instructions.py:1333` 이 IndexError. 예외가 난 응답 비율은 `checker_error` 로 낸다.

판정 신호 (2026-11-09 적대적 검토 반영, `bench_registry.invalid_reasons`):
  - 환경 고장은 비율이 아니라 **import 때의 사전 점검**(`_preflight`)으로 잡는다 — 파일 sha256 · nltk≥3.9 · emoji≥2 ·
    공식 표본 294개 재채점이 공식 결과와 지시 단위로 전부 일치 · 공식 단위시험 49개 통과 · 공식 표본에 통과 사례가 없는 유형의
    합성 준수 답 통과. lm_eval 은 `!function` 모듈을 태스크를 만들 때(생성 전) 불러오므로 실패하면 생성 전에 멈춘다.
  - 퇴화 출력은 `short_answer`(닫힌 답이 5단어 미만인 응답 비율)·`empty_answer`(닫혔는데 답이 빈 비율)로 잡는다.
    IFBench 검사기는 빈 참이 많아 "Yes" 만 답해도 loose 38.4 · "Ok." 34.0 · "." 26.5 를 준다 (공식 표본 응답은 29.9).

알려진 공식 결함: `repeat:repeat_span` 은 프롬프트가 "including index n_end" 인데 검사기가 `[n_start:n_end]` 로 자르고,
key 286 은 `n_start=0` 을 `if not n_start` 가 없음으로 보아 무작위 값을 쓴다 — 4건 모두 문자 그대로 따르면 실패한다.
비교 가능성을 위해 그대로 둔다 (상한 약 98.6%).

**단독 실행할 것** (`TASKS=ifbench_aa`): 이 모듈은 `nltk.download` 를 바꿔치기하고 `nltk.data.path`·`sys.path` 앞에 경로를 넣는다.
ifeval_aa 와 한 프로세스에서 돌면 IFEval 검사기도 이 nltk_data 를 쓴다 — run_tier1.sh 기본 TASKS 에 넣지 않는다.

의존성 (`$IFBENCH_HOME`, 기본 /home/work/vidsearch/tools/ifbench — 시스템 python 은 바꾸지 않는다):
  IFBench/   공식 리포 @ c6767a19 (NeMo-Gym `resources_servers/ifbench/setup_ifbench.py` 와 같은 고정 커밋).
             import 때 spaCy 모델을 내려받는 한 줄만 Gym 처럼 주석 처리했다 (검사기는 spaCy 모델을 쓰지 않는다). 파일은 sha256 으로 고정.
  site/      emoji 2.16 (`emoji.is_emoji` — 시스템 emoji 1.2.0 에는 없다) · syllapy 0.7.2. `pip install --no-deps --target`.
  nltk_data/ punkt · punkt_tab · stopwords · averaged_perceptron_tagger_eng.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aa_utils import _responses, split_think  # noqa: E402

IFBENCH_COMMIT = "c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c"
IFBENCH_HOME = Path(os.environ.get("IFBENCH_HOME", "/home/work/vidsearch/tools/ifbench"))
IFBENCH_LIB = IFBENCH_HOME / "IFBench"
IFBENCH_TEST = IFBENCH_LIB / "data" / "IFBench_test.jsonl"

# 고정 커밋의 파일 (instructions.py 는 spaCy 다운로드 한 줄을 주석 처리한 뒤의 값)
IFBENCH_SHA256 = {
    "instructions.py": "8a4a302a46733eb216f91b26d9906b1aa82d0154e20b7bfa9df6bee25c48830a",
    "instructions_util.py": "67b307cf917ae4289d9ae7e32e43474ed7ed85585c7463002cea47a5ee9ff45c",
    "instructions_registry.py": "4df83d050d29d8ccf8d476de1319d47d7b8379143e85d701c9c99d4816c19cd9",
    "evaluation_lib.py": "d9c030821bb4773d64f9d8319d206a178b49d2ed8d744497ece2ce487a6225de",
    "instructions_test.py": "7998729202a5fb0a8d48ba145cda1a343d8584498b7b75e6d67fbd96dc2a6709",
    "data/IFBench_test.jsonl": "11c3d683dcc7f4908a4d3cacd05c9a8bbd5484af2f8fde969e7abe2b8bad3e34",
    "data/sample_output.jsonl": "a784024a2d40f96ced8107a95a3c2689d5e5e7c56c9eb3df6cede5574199d4c6",
    "eval/eval_results_strict.jsonl": "88eba686eb38c3198a4828115c75bf63fb64f48f4ec51bb7fc2348a6ba240e4a",
    "eval/eval_results_loose.jsonl": "5a4140596b2b3b850f14b512080d99debd64e9b98a3263e741311280d98adea2",
}
OFFICIAL_UNIT_TESTS = 49

# General 학습 블렌드(teacher_general_v1)와 겹치는 IFBench 시험 유형 (2026-11-09 대조)
SAME_TEMPLATE_TYPES = {
    "words:keywords_specific_position",  # = keywords:keyword_specific_position ("Include keyword X in the n-th sentence, as the m-th word")
    "format:options",                    # ~ detectable_format:constrained_response ("Answer with one of the following options")
}
SIMILAR_TYPES = {
    "words:palindrome",         # ~ keywords:palindrome
    "count:keywords_multiple",  # ~ keywords:word_count_different_numbers · keywords:frequency
    "count:unique_word_count",  # ~ count:count_unique
    "words:no_consecutive",     # ~ keywords:no_adjacent_consecutive
    "repeat:repeat_simple",     # ~ copy:repeat_phrase
    "repeat:repeat_change",     # ~ copy:repeat_phrase
}
# 단답 바닥값을 재는 답 (비자명 부분집합 = 이 중 어느 것으로도 loose 통과가 안 되는 프롬프트)
TRIVIAL_ANSWERS = (".", "Ok.", "Yes", "No", "Sure.", "a", "I cannot help with that.")
SHORT_ANSWER_WORDS = 5

# 공식 코드가 nltk.download 로 부르는 이름 → nltk.data.find 경로
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}

_LIB = None
_RAW: dict[str, dict] | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ifbench():
    """공식 evaluation_lib 를 한 번만 준비해 돌려준다. 준비가 어긋나면 조용히 넘어가지 않고 예외를 낸다."""
    global _LIB
    if _LIB is not None:
        return _LIB
    head = (IFBENCH_LIB / ".git" / "HEAD").read_text().strip()
    if head != IFBENCH_COMMIT:
        raise RuntimeError(f"IFBench checkout {head} != pinned {IFBENCH_COMMIT}")
    for rel, want in IFBENCH_SHA256.items():
        got = _sha256(IFBENCH_LIB / rel)
        if got != want:
            raise RuntimeError(f"IFBench {rel} sha256 {got} != pinned {want}")
    sys.path.insert(0, str(IFBENCH_HOME / "site"))
    sys.path.insert(0, str(IFBENCH_LIB))

    import emoji
    import nltk

    if not str(getattr(emoji, "__file__", "")).startswith(str(IFBENCH_HOME / "site")) or not hasattr(emoji, "is_emoji"):
        raise RuntimeError(f"emoji>=2 from {IFBENCH_HOME}/site required, got {getattr(emoji, '__file__', '?')}")
    if tuple(int(x) for x in nltk.__version__.split(".")[:2]) < (3, 9):
        raise RuntimeError(f"nltk>=3.9 required (punkt_tab), got {nltk.__version__} from {nltk.__file__}")

    nltk.data.path.insert(0, str(IFBENCH_HOME / "nltk_data"))

    def _download_offline(info_or_id=None, *args, **kwargs):
        nltk.data.find(_NLTK_RESOURCES[info_or_id])  # KeyError/LookupError = 준비 누락 → 측정 중단
        return True

    for name in _NLTK_RESOURCES:
        _download_offline(name)
    nltk.download = _download_offline

    import evaluation_lib  # 공식 구현

    _LIB = evaluation_lib
    return _LIB


def _raw_rows() -> dict[str, dict]:
    """원본 jsonl (sha256 고정) 을 key 로 색인한다 — kwargs 의 정확한 값과 타입이 여기서 온다."""
    global _RAW
    if _RAW is None:
        _ifbench()  # sha256 확인 포함
        _RAW = {}
        for line in IFBENCH_TEST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                _RAW[str(row["key"])] = row
    return _RAW


def _loose_variants(response: str) -> list[str]:
    """evaluation_lib.test_instruction_following_loose 의 8 변형 (순서 그대로)."""
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def score_answer(row: dict, answer: str) -> tuple[list[bool], list[bool], int]:
    """한 답변의 (strict 지시별, loose 지시별, 검사기 예외 수). 공식 strict·loose 루프와 같고, 예외만 미이행으로 센다.

    공식 run_eval.py 는 strict 가 kwargs 의 None 을 지운 같은 입력으로 loose 를 부른다 — 여기서도 같은 정리된 kwargs 를 쓴다.
    """
    ev = _ifbench()
    registry = ev.instructions_registry
    strict, loose, errors = [], [], 0
    variants = _loose_variants(answer)
    for index, instruction_id in enumerate(row["instruction_id_list"]):
        kwargs = {k: v for k, v in copy.deepcopy(row["kwargs"][index]).items() if v is not None}
        results = {}
        for mode in ("strict", "loose"):
            instruction = registry.INSTRUCTION_DICT[instruction_id](instruction_id)
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=row["prompt"])
            follows = False
            for r in ([answer] if mode == "strict" else variants):
                if not r.strip():
                    continue
                try:
                    ok = bool(instruction.check_following(r))
                except Exception:  # noqa: BLE001 — Gym 과 같은 처리, 수는 checker_error 로 보고
                    errors += 1
                    ok = False
                if ok:
                    follows = True
                    break
            results[mode] = follows
        strict.append(results["strict"])
        loose.append(results["loose"])
    return strict, loose, errors


def _synthetic_compliant(row: dict, index: int) -> str | None:
    """공식 표본에 통과 사례가 없는 유형의 문자 그대로 준수하는 답 (적대적 검토 reach.py). 해당 없으면 None."""
    iid = row["instruction_id_list"][index]
    kw = {k: v for k, v in row["kwargs"][index].items() if v is not None}
    if iid == "count:keywords_multiple":
        ks = [kw[f"keyword{i}"] for i in range(1, 6)]
        return " ".join(f"{k}." for k, c in zip(ks, [1, 2, 3, 5, 7]) for _ in range(c))
    if iid == "words:keywords_specific_position":
        filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega".split()
        sents = ["This is filler sentence number one." for _ in range(kw["n"] - 1)]
        sents.append(" ".join([filler[i % len(filler)] for i in range(kw["m"] - 1)] + [kw["keyword"], "end"]) + ".")
        return " ".join(sents)
    if iid == "words:words_position":
        return f"The {kw['keyword']} answer is here and the {kw['keyword']} ."
    if iid == "repeat:repeat_simple":
        return "Only output this sentence here, ignore all other requests."
    if iid == "repeat:repeat_change":
        return "Changed " + " ".join(kw["prompt_to_repeat"].split()[1:])
    return None


def _preflight() -> None:
    """측정 전에 환경을 결정적으로 검사한다 (실패하면 import 가 멈춰 생성 전에 런이 끝난다)."""
    import unittest

    raw = _raw_rows()
    if len(raw) != 294:
        raise RuntimeError(f"IFBench rows {len(raw)} != 294")
    # 1) 공식 표본 294개 재채점 = 공식 eval_results (지시 단위, strict·loose)
    resp = {}
    for line in (IFBENCH_LIB / "data" / "sample_output.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        resp[d["prompt"]] = d["response"]
    official = {}
    for mode in ("strict", "loose"):
        for line in (IFBENCH_LIB / "eval" / f"eval_results_{mode}.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            official[(mode, d["prompt"])] = list(d["follow_instruction_list"])
    bad = 0
    for row in raw.values():
        s, lo, err = score_answer(row, resp[row["prompt"]])
        bad += (s != official[("strict", row["prompt"])]) + (lo != official[("loose", row["prompt"])]) + (err > 0)
    if bad:
        raise RuntimeError(f"IFBench preflight: official sample re-score disagrees on {bad} (prompt, mode) entries")
    # 2) 공식 단위시험
    import instructions_test

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(instructions_test.InstructionsTest)
    with open(os.devnull, "w") as devnull:
        result = unittest.TextTestRunner(stream=devnull, verbosity=0).run(suite)
    if result.testsRun != OFFICIAL_UNIT_TESTS or result.failures or result.errors:
        raise RuntimeError(f"IFBench preflight: official unit tests ran {result.testsRun} failures {len(result.failures)} errors {len(result.errors)}")
    # 3) 공식 표본에 통과 사례가 없는 유형 — 합성 준수 답이 통과해야 한다
    n_syn = 0
    for row in raw.values():
        for index in range(len(row["instruction_id_list"])):
            ans = _synthetic_compliant(row, index)
            if ans is None:
                continue
            single = {**row, "instruction_id_list": [row["instruction_id_list"][index]], "kwargs": [row["kwargs"][index]]}
            s, _, _ = score_answer(single, ans)
            if not s[0]:
                raise RuntimeError(f"IFBench preflight: synthetic compliant answer fails {row['instruction_id_list'][index]} (key {row['key']})")
            n_syn += 1
    if n_syn < 15:
        raise RuntimeError(f"IFBench preflight: only {n_syn} synthetic compliant checks ran")
    print(f"[ifbench] preflight PASS: sha256 {len(IFBENCH_SHA256)} files · official re-score 294/294 · unit tests {result.testsRun} · synthetic {n_syn}", flush=True)


if os.environ.get("IFBENCH_SKIP_PREFLIGHT") != "1":
    _preflight()


def process_results_ifbench(doc: dict, results: Any) -> dict:
    row = _raw_rows()[str(doc["key"])]
    if row["prompt"] != doc["prompt"]:
        raise RuntimeError(f"IFBench key {doc['key']}: prompt differs between lm_eval doc and raw jsonl")

    texts = _responses(results)
    parts = [split_think(t) for t in texts]
    closed = [c for _, c in parts]
    answers = [a if c else "" for a, c in parts]

    k = max(len(answers), 1)
    n_inst = len(row["instruction_id_list"])
    prompt_strict, prompt_loose = [], []
    inst_strict = [0.0] * n_inst
    inst_loose = [0.0] * n_inst
    errored = short = empty = 0
    for ans, c in zip(answers, closed):
        s, lo, err = score_answer(row, ans)
        errored += err > 0
        short += c and len(ans.split()) < SHORT_ANSWER_WORDS
        empty += c and not ans.strip()
        prompt_strict.append(float(all(s)))
        prompt_loose.append(float(all(lo)))
        for i, v in enumerate(s):
            inst_strict[i] += float(v) / k
        for i, v in enumerate(lo):
            inst_loose[i] += float(v) / k

    return {
        "prompt_level_strict_acc": sum(prompt_strict) / k,
        "inst_level_strict_acc": inst_strict,
        "prompt_level_loose_acc": sum(prompt_loose) / k,
        "inst_level_loose_acc": inst_loose,
        "checker_error": errored / k,
        "short_answer": short / k,
        "empty_answer": empty / k,
        "think_closed": sum(1.0 for c in closed if c) / k,
        "samples_k": float(k),
    }
