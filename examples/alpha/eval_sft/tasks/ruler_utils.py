"""RULER NIAH — Nemotron/Qwen3 규약 (2026-08-30 재작성).

**핵심: 추론을 끈다.** RULER 는 needle 추출 과제이지 추론 과제가 아니다. 프론티어 셋이
방법은 달라도 같은 판단을 한다:

- Nemotron Nano 9B v2 카드: *"We evaluated our model in Reasoning-On mode across all
  benchmarks, **except RULER, which is evaluated in Reasoning-Off mode**"*
- Nemotron 3 Ultra: RULER 를 instruct 스위트가 아닌 **base 스위트**로 분리,
  `temperature 0.00001 / top_p 0.99`
- Qwen3-235B: thinking budget **8,192** 로 제한, *"To avoid overly verbose reasoning"*

구 설정의 실패 (2026-08-30): 추론을 켠 채 출력 예산을 128 토큰으로 조였다. 모델이 서두
분석에 예산을 전부 소진해 needle 에 도달하지 못했고, 65536 구간 6~25% 가 나왔다. 같은
모델이 LC-B 자체 NIAH 하니스에서는 4k~131k **200/200** 이었다 — 모순의 원인은 모델이
아니라 태스크 설정이었다.

실측 대조 (iter300, 동일 프롬프트, 2026-08-30):

| 모드 | finish | tokens | needle 추출 |
|---|---|---:|---|
| thinking ON | length | 512 (소진) | 실패 |
| thinking OFF | **stop** | **21** | **성공** |

alpha 챗 템플릿은 `enable_thinking=false` 일 때 `<|im_start|>assistant\\n<think></think>`
로 사고를 **미리 닫아** 렌더한다. 그래서 `chat_template_kwargs` 한 줄로 Reasoning-Off 가 된다.

또 하나 고친 것: 구 `common_utils.process_results` 는 센티넬 dict 를 하드코딩된
`DEFAULT_SEQ_LENGTHS = [4096]` 로 만들어, 샘플이 0개인 4096 구간에 `-1.0` 이 결과에
남았다. 여기서는 **문서 자신의 길이**만 채운다.
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
from typing import Generator

import datasets

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer  # noqa: E402
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack  # noqa: E402

log = logging.getLogger(__name__)

TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. Make sure to "
    "memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
)


# ---------------------------------------------------------------- 데이터 생성

def _dl(df: Generator):
    return {
        "test": datasets.Dataset.from_list(
            list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST
        )
    }


# 평가 구간. lm_eval 은 metric_list 에 선언된 키가 **문서마다** 있기를 요구하므로(없으면
# 집계에서 누락), 문서 자신의 길이 외 구간은 -1 센티넬로 채워야 한다.
#
# 이 상수는 태스크 yaml 의 `metric_list` 및 `metadata.max_seq_lengths` 와 **반드시 일치**해야
# 한다. 세 곳이 어긋나면 `_build` 가 즉시 실패한다 — 조용히 빈 구간이 생기는 것보다 낫다.
# (구 구현은 이 목록을 하드코딩 `[4096]` 으로 잡아, 샘플 0개인 4096 구간에 -1.0 이 결과로
#  남았다. 모듈 전역에 실행 중 값을 쌓는 방식은 lm_eval 이 모듈을 경로별로 따로 로드해
#  인스턴스가 갈리므로 쓸 수 없다.)
# 258048 = max_position_embeddings(262,144) - 4,096. **상한을 그대로 쓸 수 없다** —
# vLLM 은 max_model_len > max_position_embeddings 를 거부하고(2026-09-01 실측:
# "User-specified max_model_len (270336) is greater than the derived max_model_len"),
# 프롬프트 + 생성이 262,144 안에 들어가야 하므로 생성 512 와 챗 템플릿 여유를 뺀다.
# LC-B(SFT 전)는 262,144 지점에서 95% 였다
# (`study/lc_b_final_eval.md` §2). SFT 후에도 유지되는지가 이 구간의 질문이다.
# 절벽은 (262144, 393216) 사이 — 393216 은 전 깊이 0% 였으므로 재지 않는다.
SEQ_LENGTHS = [65536, 131072, 258048]


def _build(kwargs, **gen):
    seqs = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    n = kwargs.pop("num_samples", 20)
    if sorted(int(s) for s in seqs) != sorted(SEQ_LENGTHS):
        raise ValueError(
            f"태스크 metadata.max_seq_lengths={list(seqs)} 가 ruler_utils.SEQ_LENGTHS="
            f"{SEQ_LENGTHS} 와 다르다. yaml 의 metric_list 까지 세 곳을 함께 맞출 것."
        )
    tok = get_tokenizer(**kwargs)
    return _dl(
        generate_samples(
            get_haystack(type_haystack=gen["type_haystack"]),
            max_seq_length=s, template=TEMPLATE, num_samples=n, TOKENIZER=tok, **gen,
        )
        for s in seqs
    )


def niah_single_1(**k):
    return _build(k, type_haystack="repeat", type_needle_k="words", type_needle_v="numbers")


def niah_single_2(**k):
    return _build(k, type_haystack="essay", type_needle_k="words", type_needle_v="numbers")


def niah_multikey_1(**k):
    return _build(k, type_haystack="essay", type_needle_k="words", type_needle_v="numbers",
                  num_needle_k=4)


def niah_multivalue(**k):
    return _build(k, type_haystack="essay", type_needle_k="words", type_needle_v="numbers",
                  num_needle_v=4)


# ---------------------------------------------------------------- 프롬프트

ANSWER_INSTRUCTION = (
    "\n\nAnswer with only the special magic number(s), separated by commas. "
    "Do not explain."
)


def doc_to_text(doc: dict) -> str:
    """RULER 질문 + 답 형식 지시.

    **`gen_prefix` 를 쓰지 않는다.** lm_eval 의 `gen_prefix` 는 chat 경로에서
    `{"role": "assistant", "content": "...is"}` 라는 **완결된 assistant 메시지**로
    전송된다. 완결된 턴이므로 모델은 이어쓰지 않고, 서버는 그 prefix 를 그대로
    돌려준다 — 생성 0토큰이다. vLLM 에서 이어쓰게 하려면 `continue_final_message`
    가 필요한데 lm_eval 은 그것을 보내지 않는다.

    2026-08-31 실측: RULER 160 샘플 중 **131건(82%)** 의 응답이 `gen_prefix` 와
    글자 그대로 같았다. 35/10/5/0 점수는 측정이 아니라 잡음이었다.
    (증상: 64K 점수가 128K 보다 낮은 역전 — 진짜 열화면 반대여야 한다.)

    대신 T1 에서 검증된 방식을 쓴다 — 프롬프트로 답 형식을 지시한다.
    """
    return str(doc["input"]).rstrip() + ANSWER_INSTRUCTION


# ---------------------------------------------------------------- 채점

def _clean(text: str) -> str:
    """제어문자를 개행으로 바꾸고 다듬는다. 사고를 껐으므로 `</think>` 분리는 불필요하나,
    혹시 켜진 채로 돌아도 답변부만 보도록 방어한다."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return "".join("\n" if ord(c) < 0x20 else c for c in text).strip()


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """구간별 점수 + 진단 지표.

    이 문서의 길이 구간에만 점수를 넣고, 나머지 선언 구간은 `aggregate_metrics` 가
    무시하도록 -1 로 둔다. 센티넬 목록은 `SEQ_LENGTHS` — yaml 과 일치가 강제되므로
    구 구현처럼 데이터에 없는 4096 이 결과에 남지 않는다.
    """
    metrics: dict[str, float] = {str(v): -1.0 for v in SEQ_LENGTHS}

    raw = results[0] if results else ""
    pred = _clean(raw)
    refs = doc["outputs"]
    score = sum(1.0 for r in refs if str(r).lower() in pred.lower()) / max(len(refs), 1)

    # 생성이 실제로 일어났는가. 응답이 비었거나 프롬프트 꼬리를 그대로 되돌려준 경우
    # (구 gen_prefix 사고) 는 **측정 실패**이지 오답이 아니다. `no_answer` 로 보고해
    # 집계기가 무효 판정하게 한다 (임계 10%).
    tail = str(doc.get("gen_prefix") or "").strip()
    no_gen = (not pred) or (tail and pred.strip() == tail)

    metrics[str(doc["max_length"])] = 0.0 if no_gen else score
    metrics["gen_chars"] = float(len(pred))
    metrics["no_answer"] = 1.0 if no_gen else 0.0
    return metrics


def aggregate_metrics(metrics: list[float]) -> float:
    """해당 구간에 샘플이 없으면 -1 (표에서 '—' 로 읽힌다)."""
    vals = [x for x in metrics if x != -1]
    return sum(vals) / len(vals) if vals else -1.0
