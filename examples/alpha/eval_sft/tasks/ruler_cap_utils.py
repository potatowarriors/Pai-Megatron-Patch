"""RULER 능력 스위트 (11태스크) — 512K "지원" 판정용 (2026-09-02, study/lc_512k_eval.md §4·§6).

왜 별도 스위트인가: 그리드의 `_512k` 4태스크(NIAH 부분집합, n=20)는 **메커니즘 A/B**
(YaRN 이 RoPE 절벽을 옮겼는가)용이다. single_1(반복 haystack)은 검색 난도가 0이라
위치 인코딩 격리 증명에는 최적이지만 능력 지표로는 무의미하다. 능력 판정은 프론티어
관행을 따른다: **RULER 13태스크 구간 평균 + "평균 ≥85 인 최대 길이 = 유효 컨텍스트"**
(RULER 논문 규약; Nemotron 3 Ultra·Qwen 카드가 같은 방식으로 게재).

구성 (RULER-13 중 11) — **"라벨만 긴 측정" 이 되는 태스크 2종은 구조적으로 제외**:
  niah_single_{1,2,3}, niah_multikey_{1,2,3}, niah_multiquery, niah_multivalue,
  vt(변수 추적 multi-hop), fwe(zipf 빈도 추출), qa_hotpot(다문서 QA).
  - **qa_squad 제외**: SQuAD dev 문서 풀 ~0.27M 토큰(실측 2026-09-02) — 393K/520K 를 못 채움.
    qa_hotpot 은 HotpotQA dev 풀 ~9M 토큰으로 충분 (fill 실측 0.97~1.00).
  - **cwe 제외**: wonderwords 어휘 풀 len(WORDS)=8166 고갈로 입력이 **~130K 에서 포화**
    (validate 실측 fill 0.995/0.506/0.332/0.251@131K/258K/393K/520K — 사전 빌드 게이트가 검출,
    2026-09-02). 어휘를 합성으로 늘리면 stock 태스크 정의에서 벗어나 비교 불가.

lm_eval 내장 RULER 와의 차이:
  - Reasoning-Off·프롬프트 지시(doc_to_text) 규약은 `_aa`/`_512k` 와 동일 — gen_prefix 를
    쓰지 않는 이유는 `ruler_utils.doc_to_text` docstring (chat 경로에서 완결 턴이 됨).
  - 구간·센티넬은 문서 `seq_set` 로 — `ruler_utils.SEQ_SETS` 검증 재사용.
  - **길이 탐색 루프의 incremental 을 길이 비례로 상향** (vt/cwe: L//2048, qa: L//8192).
    stock 기본 10 은 520K 에서 수천 회 전체-재토크나이즈(O(N²), 시간 단위)가 된다.
    탐색 정밀도 손실은 ±incremental 줄(520K 에서 ~0.05%)로 무시 가능. fwe 는 stock 이
    이미 L//32 자체 스케일링이라 그대로 둔다.
  - qa_hotpot 은 stock 의 curtis.ml.cmu.edu 직다운로드(2026-09-02 타임아웃 실측) 대신
    HF `hotpot_qa/distractor` validation 에서 stock `read_hotpotqa` 와 동일 형태로 재구성.
  - 채점: niah/vt/cwe/fwe = 참조 중 존재 비율(stock string_match_all 동일),
    qa = 참조 중 하나라도 존재(stock string_match_part 동일).
"""

from __future__ import annotations

import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ruler_utils  # noqa: E402  (동일 디렉토리 — _dl/_clean/SEQ_SETS/TEMPLATE 재사용)
from lm_eval.tasks.ruler import fwe_utils, qa_utils, vt_utils  # noqa: E402
from lm_eval.tasks.ruler.common_utils import get_tokenizer  # noqa: E402
from lm_eval.tasks.ruler.prepare_niah import (  # noqa: E402
    generate_samples as niah_generate,
    get_haystack,
)

log = logging.getLogger(__name__)

# yaml `metric_list` aggregation 이 참조한다 (구간에 샘플 없으면 -1 = 표에서 '—').
aggregate_metrics = ruler_utils.aggregate_metrics
# niah/vt/cwe/fwe 채점 = ruler_utils 와 동일(참조 존재 비율 + no_answer 진단).
process_results = ruler_utils.process_results


def _pop_common(kwargs):
    seqs = [int(s) for s in kwargs.pop("max_seq_lengths")]
    n = int(kwargs.pop("num_samples", 50))
    if sorted(seqs) not in [sorted(v) for v in ruler_utils.SEQ_SETS.values()]:
        raise ValueError(
            f"metadata.max_seq_lengths={seqs} 가 ruler_utils.SEQ_SETS"
            f"({ruler_utils.SEQ_SETS})의 어느 집합과도 다르다. yaml metric_list 까지 함께 맞출 것."
        )
    tok = get_tokenizer(**kwargs)
    return seqs, n, tok


def _tag(rows, seqs, L):
    """모든 생성기 산출 dict 에 seq_set(센티넬 집합)을 싣고 max_length 를 방어적으로 보장."""
    for row in rows:
        yield {**row, "seq_set": seqs, "max_length": int(row.get("max_length", L))}


# ---------------------------------------------------------------- NIAH 8종

def _build_niah(kwargs, **gen):
    seqs, n, tok = _pop_common(kwargs)
    return ruler_utils._dl(
        _tag(niah_generate(
            get_haystack(type_haystack=gen["type_haystack"]),
            max_seq_length=L, template=ruler_utils.TEMPLATE,
            num_samples=n, TOKENIZER=tok, **gen,
        ), seqs, L)
        for L in seqs
    )


def niah_single_1(**k):
    return _build_niah(k, type_haystack="repeat", type_needle_k="words", type_needle_v="numbers")


def niah_single_2(**k):
    return _build_niah(k, type_haystack="essay", type_needle_k="words", type_needle_v="numbers")


def niah_single_3(**k):
    return _build_niah(k, type_haystack="essay", type_needle_k="words", type_needle_v="uuids")


def niah_multikey_1(**k):
    return _build_niah(k, type_haystack="essay", type_needle_k="words", type_needle_v="numbers",
                       num_needle_k=4)


def niah_multikey_2(**k):
    return _build_niah(k, type_haystack="needle", type_needle_k="words", type_needle_v="numbers")


def niah_multikey_3(**k):
    return _build_niah(k, type_haystack="needle", type_needle_k="uuids", type_needle_v="uuids")


def niah_multiquery(**k):
    return _build_niah(k, type_haystack="essay", type_needle_k="words", type_needle_v="numbers",
                       num_needle_q=4)


def niah_multivalue(**k):
    return _build_niah(k, type_haystack="essay", type_needle_k="words", type_needle_v="numbers",
                       num_needle_v=4)


# ---------------------------------------------------------------- VT / CWE / FWE

def vt(**kwargs):
    seqs, n, tok = _pop_common(kwargs)
    icl = vt_utils.sys_vartrack_w_noise_random(
        tokenizer=tok, num_samples=1, max_seq_length=500, incremental=5)[0]
    return ruler_utils._dl(
        _tag(vt_utils.sys_vartrack_w_noise_random(
            tokenizer=tok, num_samples=n, max_seq_length=L,
            incremental=max(10, L // 2048), icl_example=icl,
        ), seqs, L)
        for L in seqs
    )


def fwe(**kwargs):
    seqs, n, tok = _pop_common(kwargs)
    return ruler_utils._dl(
        _tag(fwe_utils.sys_kwext(tokenizer=tok, max_seq_length=L, num_samples=n), seqs, L)
        for L in seqs
    )


# ---------------------------------------------------------------- QA (HotpotQA)

def _hotpot_docs_qas():
    """stock `qa_utils.read_hotpotqa` 와 동일한 (qas, docs) 를 HF datasets 로 재구성.

    직다운로드 URL(curtis.ml.cmu.edu, http)이 이 클러스터에서 타임아웃이라 대체.
    문서 정렬·인덱싱을 stock 과 동일(sorted set)하게 유지한다.
    """
    import datasets as hfds
    ds = hfds.load_dataset("hotpot_qa", "distractor", split="validation")
    raw = []
    for ex in ds:
        ctx = ex["context"]
        pair_docs = [f"{t}\n{''.join(p)}" for t, p in zip(ctx["title"], ctx["sentences"])]
        raw.append((ex["question"], ex["answer"], pair_docs))
    total_docs = sorted({d for _, _, pd in raw for d in pd})
    idx = {c: i for i, c in enumerate(total_docs)}
    qas = [{"query": q, "outputs": [a], "context": [idx[d] for d in pd]} for q, a, pd in raw]
    return qas, total_docs


def qa_hotpot(**kwargs):
    seqs, n, tok = _pop_common(kwargs)
    qas, docs = _hotpot_docs_qas()
    return ruler_utils._dl(
        _tag(qa_utils.generate_samples(
            tokenizer=tok, docs=docs, qas=qas, max_seq_length=L,
            num_samples=n, tokens_to_generate=32, incremental=max(10, L // 8192),
        ), seqs, L)
        for L in seqs
    )


# ---------------------------------------------------------------- 프롬프트

_SUFFIX_NUMBERS = ("\n\nAnswer with only the special magic number(s), separated by commas. "
                   "Do not explain.")
_SUFFIX_UUIDS = ("\n\nAnswer with only the special magic uuid(s), separated by commas. "
                 "Do not explain.")
_SUFFIX_VT = "\n\nAnswer with only the variable names, separated by commas. Do not explain."
_SUFFIX_CWE = ("\n\nAnswer with only the 10 most common words in the list, "
               "separated by commas. Do not explain.")
_SUFFIX_FWE = ("\n\nAnswer with only the 3 most frequently appearing coded words, "
               "separated by commas. Do not explain.")
_SUFFIX_QA = "\n\nAnswer with only the answer. Do not explain."


def _dtt(doc, suffix):
    return str(doc["input"]).rstrip() + suffix


def doc_to_text_numbers(doc):
    return _dtt(doc, _SUFFIX_NUMBERS)


def doc_to_text_uuids(doc):
    return _dtt(doc, _SUFFIX_UUIDS)


def doc_to_text_vt(doc):
    return _dtt(doc, _SUFFIX_VT)


def doc_to_text_cwe(doc):
    return _dtt(doc, _SUFFIX_CWE)


def doc_to_text_fwe(doc):
    return _dtt(doc, _SUFFIX_FWE)


def doc_to_text_qa(doc):
    return _dtt(doc, _SUFFIX_QA)


# ---------------------------------------------------------------- QA 채점 (part)

def process_results_part(doc: dict, results: list[str]) -> dict[str, float]:
    """qa: 참조 답 중 **하나라도** 있으면 1 (stock string_match_part 규약)."""
    metrics: dict[str, float] = {str(v): -1.0 for v in (doc.get("seq_set") or ruler_utils.SEQ_LENGTHS)}
    raw = results[0] if results else ""
    pred = ruler_utils._clean(raw)
    refs = doc["outputs"]
    score = 1.0 if any(str(r).lower() in pred.lower() for r in refs) else 0.0
    tail = str(doc.get("gen_prefix") or "").strip()
    no_gen = (not pred) or (tail and pred.strip() == tail)
    metrics[str(doc["max_length"])] = 0.0 if no_gen else score
    metrics["gen_chars"] = float(len(pred))
    metrics["no_answer"] = 1.0 if no_gen else 0.0
    return metrics
