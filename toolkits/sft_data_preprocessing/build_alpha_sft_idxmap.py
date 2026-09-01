#!/usr/bin/env python3
# Copyright (c) 2026 alpha team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""messages → Megatron idxmap SFT 변환기 (alpha 전용).

기존 build_idxmap_sft_dataset.py 는 prefix-diff 마스킹(alpha 템플릿에서 불성립),
단일턴 포맷, %16 미정렬이라 alpha 에 쓸 수 없다 — 이 파일이 대체한다.
마스킹 규약의 정본은 examples/alpha/tools/verify_chat_template.py (24 tests).

## 출력 포맷 계약

소비자: megatron_patch/data/utils.py::get_batch_on_this_tp_rank_idxmap_sft +
megatron_patch/template/helper.py::get_batch
(플래그: --dataset MMAP --train-mode finetune --reset-position-ids, MBS 1)

  document := [ input_ids(S) | labels(S) ]   # 길이 정확히 2*S (S = --seq-length)
  - input_ids: best-fit-decreasing 으로 패킹된 샘플들의 연결.
    각 샘플의 **마지막 토큰은 음수 마커** (v -> -1-v) — 소비자가
    get_ltor_position_ids_packed_seq 에서 position-id 리셋 경계로 쓰고 복원한다.
  - labels[i] = input_ids[i+1] (학습 대상 위치만), 그 외 -100.
    즉 labels 는 이미 1칸 시프트된 상태로 저장된다 (소비자는 재시프트하지 않음).
  - 각 샘플은 pad(id=tokenizer.pad) 로 %(--pad-doc-multiple, 기본 16) 정렬 후
    마지막 토큰(패딩 포함)에 마커가 찍힌다. 세그먼트(=샘플+pad) 길이 %16 은
    GDN THD+CP a2a 의 % (2*cp_size) 요건
    (megatron_patch/model/qwen3_next/gdn_context_parallel.py::resolve_cu_seqlens).
  - bin 꼬리 잔여는 **마커 없는** pad run — 소비자의 has_pad(마지막 토큰 >= 0)
    검출과 per-seq 평균의 pad-세그먼트 스킵이 이 형태를 전제한다.
    S %16 == 0 이고 모든 세그먼트가 %16 이므로 꼬리도 자동으로 %16.

## 마스킹 규약 (스팬 스캔 — prefix-diff 금지)

  - 렌더는 tokenizer.apply_chat_template 전체 호출 (think 제거·tool XML·병합은
    템플릿이 수행). 스팬은 토큰열에서 <|im_start|>(2) ... <|im_end|>(3) 를 스캔해
    role == assistant 인 턴만 학습 구간으로 마킹.
  - --mask-role-header (기본 on): "assistant\n" 역할 헤더 토큰은 loss 제외
    (추론 시 generation prompt 로 주어지는 부분). <|im_end|> 는 항상 loss 포함
    (정지 학습). 헤더에도 loss 를 주려면 --no-mask-role-header.
  - metadata.train_turns (bool 리스트, 메시지 인덱스 기준) 가 있으면 true 인
    assistant 턴만 학습 (Nemotron Chat-v3 규약: 마지막 턴만). 없으면 모든
    assistant 턴 학습.
  - --fanout-train-turns: multi-True 행을 True 턴별 messages[:k+1] 서브샘플로
    전개 (expand_train_turns_fanout). 각 서브샘플에서 해당 턴이 마지막 user
    이후가 되어 템플릿이 think 를 보존 — 추론에서 그 턴이 라이브였던 순간의
    컨텍스트와 토큰열이 일치한다. loss 는 그 턴에만. (의도적 차이 #2 참조)

## 드롭 정책 (전부 stats 카운트 + 사이드카 jsonl 기록)

  - null_content: system/user 턴 content 가 null (Chat-v3 chat split 라이선스
    마스킹 — prepare_chat_prompts.py 복원 후 투입할 것)
  - injection: content/reasoning/tool 필드 안에 구조 특수 리터럴
    (<|im_start|>, <|im_end|>) 포함 — naive 인코딩 시 스팬 구조가 오염됨.
    v1 은 드롭으로 비율을 측정하고, 유의하면 세그먼트 splice 인코딩을 v2 로.
  - too_long: 렌더 토큰 길이 > S (증명/트레이스 중간 절단은 유해 — 드롭.
    >64k 는 향후 128k 버킷에서 수용, docs/SFT_RL_DATASETS.md §2.4)
  - no_trainable: 학습 토큰 0 (소비자 per-seq 평균의 assert subseq.sum()>0)
  - span_mismatch: 렌더된 assistant 스팬 수 != assistant 메시지 수
    (연속 assistant 병합 등 템플릿 특이 케이스 — 감사 후 필요 시 규약 확장)
  - render_error / bad_row: 템플릿 렌더 실패 / 스키마 위반
  - tool_content_shape: tool content 가 list 인데 tool-result/text 형식이 아님 (평문화 불가 → 드롭)

## NVIDIA 공식 구현 대조 (2026-08-23, NVIDIA-NeMo/Nemotron data_prep 정독)

일치: assistant 헤더 비학습 / think 내용 학습 / im_end 학습(정지) / tool
응답·system 비학습 / 라벨 1칸 시프트 + 샘플 경계 라벨 0 (materialize.py 계약
"loss_mask[j] -> input_ids[j+1]" 과 동형).
의도적 차이 4건:
  1. think-opener 토큰(<think>·개행)도 학습에 포함 — NVIDIA 는 청크별 인코딩이라
     gen-prompt 경계에서 제외 가능하지만, 우리는 전체 렌더 인코딩(추론과 동일한
     토큰화)이라 경계가 토큰 정렬을 깨뜨릴 수 있어 포함. 결정적 토큰 소량이라 무해.
  2. thinking 멀티(user)턴 fan-out — **--fanout-train-turns 로 채택** (2026-08-24,
     IF 계열 적용). NVIDIA 는 user 턴마다 서브시퀀스로 복제해 각 턴의 reasoning
     을 보존한다. 초기(08-23)엔 "train_turns 셋은 마지막 턴만 학습이라 등가"로
     미채택했으나, §2.5 실측(docs/SFT_RL_DATASETS.md)이 IF split 의 60.9%가
     multi-True 임을 밝혀 등가 논거가 붕괴 — 단일 시퀀스 렌더는 학습 대상 중간
     턴을 think-제거 상태로 학습시켜 reasoning 소실(전수 26.7% chars) + no-think
     오신호(빈 <think></think> 를 정답으로 학습)를 만든다. 플래그 on 시 multi-True
     행을 True 턴별로 전개 (expand_train_turns_fanout docstring 참조).
     train_turns 없는 셋(전 턴 학습)은 대상 아님 — 단일 user 턴 + tool 루프는
     기존대로 등가 (last_user_idx 이후 assistant 턴은 reasoning 유지).
  3. 초과 길이 truncate 대신 드롭 — NVIDIA 는 pack_size 로 무언 head-truncate.
     증명/트레이스 절단 유해 판단 (docs/SFT_RL_DATASETS.md §2.4).
  4. injection 방어 추가 — NVIDIA 는 없음 (content 의 <|im_start|> 가 실토큰화됨).
공통 채택: 샘플별 %16 정렬은 NVIDIA parquet 경로엔 없으나(pad_seq_to_mult 가
create_sft_dataset 에서 소실되는 상류 버그로 CP 정렬 미동작) 우리 GDN THD+CP 는
하드 요구라 데이터 시점에 굽는다.

## Reasoning effort / budget 모드 (2026-08-25, Ultra 보고서 §3.1.1 재현)

Nemotron 3 Ultra 의 medium-effort 는 데이터 필드가 아니라 **렌더 시점 템플릿
kwarg** 다 — 공개 행에는 흔적이 없고, IF-Chat-v3 instruction_following split
(GPT-OSS-120B medium-effort 생성분)을 학습기가 medium_effort=True 로 렌더해서
심는다. RL 블렌드(rlvr1/2·mopd)에는 같은 마커가 프롬프트 3.5% 에 이미 붙어
있으므로 SFT 에서 초기화하지 않으면 RL 이 처음 보는 토큰을 만난다
(docs/SFT_RL_DATASETS.md §2.6).
  --medium-effort: 마지막 user 턴 끝에 "\n\n{reasoning effort: efficient}"
     (템플릿 규약, user 스팬이라 비학습). fan-out 과 결합하면 서브샘플마다
     학습 턴 직전 user 에 붙는다 — 추론 시 토큰열과 일치.
  --truncate-reasoning-budget: 학습 턴(마지막 user 이후)의 reasoning 을
     무작위 예산 B = int(L · U(lo, hi)) 토큰으로 자르고(응답은 그대로) 잘린
     자리의 </think> 토큰을 loss 에서 제외 (Ultra: "</think> tokens in the
     truncated samples are masked"). 추론의 budget control(max_tokens 도달 시
     </think> 강제 삽입) 분포를 학습. 예산은 (seed, uuid, 턴) 로 결정적.
     L < --truncate-min-tokens 인 턴은 건너뛰고, 절단된 턴이 없는 행은 드롭
     (trunc_none — 원본 셋과 중복). 두 모드는 배타 (Ultra 도 별개 컴포넌트).
  --row-stride K: K 행마다 1행 (파생 셋 크기 조절 — 파일 앞부분 편향 회피).

Usage:
  python build_alpha_sft_idxmap.py \
      --input /path/dataset.jsonl \
      --tokenizer /path/examples/alpha/tokenizer_v5 \
      --output-prefix /path/out/dataset_name \
      --seq-length 65536 --workers 32
"""

import argparse
import dataclasses
import glob
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# bestfit_decreasing 재사용 (LC pad16 재패킹에서 검증된 세그트리 BFD)
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, "pretrain_data_preprocessing")))
from bestfit_pack import bestfit_decreasing  # noqa: E402

# ---------------------------------------------------------------------------
# 상수 (tokenizer_v5 계약 — 변경 시 verify_chat_template.py 와 함께 갱신)
# ---------------------------------------------------------------------------
IM_START_ID = 2
IM_END_ID = 3
THINK_START_ID = 14
THINK_END_ID = 15      # 예산 절단 모드의 loss 마스킹 대상 (docstring §Effort/Budget)
IGNORE_INDEX = -100
# content 안에 있으면 naive 인코딩이 턴 구조를 오염시키는 리터럴들.
# <|endoftext|>(id 0): SFT 경로의 리셋은 음수 마커 기반이라 경계 오염은 없지만,
# pretrain 계열 소비자(merge_eod_pad_segments 등)가 EOD 를 특별 취급하므로
# 방어적으로 차단 (LC 잡탕 EOD 사고 계열의 원천 차단, 실측 발생율 ~0).
STRUCTURAL_SPECIALS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


@dataclasses.dataclass
class EncodedSample:
    ids: np.ndarray         # int32 [L] — 렌더 전체 토큰열 (마커/pad 없음)
    trainable: np.ndarray   # bool  [L] — 학습 대상 토큰 위치
    uuid: str = ""


# ---------------------------------------------------------------------------
# 입력 리더
# ---------------------------------------------------------------------------
def iter_rows(path: str) -> Iterator[dict]:
    """jsonl / jsonl.gz / parquet / 디렉토리(재귀) 를 행 단위로 순회."""
    if os.path.isdir(path):
        files = sorted(
            glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
            + glob.glob(os.path.join(path, "**", "*.jsonl.gz"), recursive=True)
            + glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
        )
        for f in files:
            yield from iter_rows(f)
        return
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                yield row
        return
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # 파싱은 워커에서 (_worker_encode) — 메인 프로세스 병목 회피
                yield line


# ---------------------------------------------------------------------------
# 정규화
# ---------------------------------------------------------------------------
def normalize_row(row: dict) -> Tuple[Optional[dict], Optional[str]]:
    """원시 행 -> {messages, tools, train_turns, uuid}. 실패 시 (None, 사유)."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "bad_row"

    norm_msgs = []
    for m in messages:
        role = m.get("role")
        if role == "developer":
            # OpenAI 신관례의 system 대체 역할 (Science-v2 rqa 108k행 실측).
            # 의미 동일 — system 으로 매핑해 템플릿 catch-all(비표준 턴 헤더) 회피.
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            return None, "bad_row"
        content = m.get("content")
        tool_calls = m.get("tool_calls") or None
        # OpenCode-v1: tool content 가 문자열이 아닌 구조체 리스트
        #   [{"type":"tool-result","toolCallId":…,"toolName":…,"output":{"type":"text","value":str}}]
        # 템플릿은 list 를 str() 로 뭉개므로(Python repr + 리터럴 \n → KNOWN_ISSUES 2026-09-01 ①)
        # 여기서 평문화한다. 미지 형식은 조용히 str() 하지 않고 드롭(사유 카운트).
        if role == "tool" and isinstance(content, list):
            parts = []
            for item in content:
                out = item.get("output") if isinstance(item, dict) else None
                if (isinstance(item, dict) and item.get("type") == "tool-result"
                        and isinstance(out, dict) and out.get("type") == "text"
                        and isinstance(out.get("value"), str)):
                    parts.append(out["value"])
                else:
                    return None, "tool_content_shape"
            content = "\n".join(parts)
        # SWE-v3 등: tool_calls 필드 전체가 JSON 문자열로 인코딩된 경우
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls) or None
            except json.JSONDecodeError:
                return None, "bad_row"
        if content is None:
            # assistant + tool_calls 는 정상 관례 (content 없는 도구 호출 턴).
            # system null 은 복원본의 정상 형태 (prepare_chat_prompts 는 시드에
            # system 이 없으면 null 로 남김; 비학습 구간이라 "" 와 렌더 등가).
            # 라이선스 마스킹 판별은 user null 이 담당 (마스킹 행은 항상
            # 첫 user 도 null — 복원 전 투입 금지).
            if role == "assistant" and tool_calls:
                content = ""
            elif role == "system":
                content = ""
            else:
                return None, "null_content"
        nm: Dict[str, Any] = {"role": role, "content": content}
        rc = m.get("reasoning_content")
        if rc:
            nm["reasoning_content"] = rc
        if tool_calls:
            fixed = []
            for tc in tool_calls:
                if isinstance(tc, str):
                    try:
                        tc = json.loads(tc)
                    except json.JSONDecodeError:
                        return None, "bad_row"
                if not isinstance(tc, dict):
                    return None, "bad_row"
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        return None, "bad_row"
                fixed.append({"function": {"name": fn.get("name"), "arguments": args}})
            nm["tool_calls"] = fixed
        norm_msgs.append(nm)

    tools = row.get("tools") or None
    train_turns = (row.get("metadata") or {}).get("train_turns")
    if train_turns is not None and len(train_turns) != len(norm_msgs):
        return None, "bad_row"

    return {
        "messages": norm_msgs,
        "tools": tools,
        "train_turns": train_turns,
        "uuid": str(row.get("uuid", "")),
    }, None


def expand_train_turns_fanout(norm: dict) -> List[dict]:
    """multi-True train_turns 행 -> True 턴별 서브샘플 (의도적 차이 #2).

    각 True assistant 턴 k 에 대해 messages[:k+1] + "k 만 학습" 서브샘플을
    만든다. 잘린 시점의 렌더에서 턴 k 는 마지막 user 이후가 되어 템플릿이
    그 턴의 think 를 보존한다 — 추론에서 그 턴이 생성되던 순간의 컨텍스트와
    토큰열 일치 (마지막 True 턴의 서브샘플은 원본 전체 렌더와 동일).
    True 가 1개 이하면 원본 그대로 1건 (last-only 행은 전개와 등가 —
    IF 실측상 single-True 는 전부 마지막 턴, docs §2.5).
    """
    tt = norm["train_turns"]
    if tt is None:
        return [norm]
    true_idxs = [i for i, m in enumerate(norm["messages"])
                 if m["role"] == "assistant" and tt[i]]
    if len(true_idxs) <= 1:
        return [norm]
    subs = []
    for k in true_idxs:
        subs.append({
            "messages": norm["messages"][:k + 1],
            "tools": norm["tools"],
            "train_turns": [False] * k + [True],
            "uuid": f"{norm['uuid']}#f{k}" if norm["uuid"] else "",
        })
    return subs


def _iter_untrusted_strings(norm: dict) -> Iterator[str]:
    """injection 스캔 대상 — 사용자/모델 유래 자유 텍스트 전부."""
    for m in norm["messages"]:
        yield m.get("content") or ""
        yield m.get("reasoning_content") or ""
        for tc in m.get("tool_calls") or []:
            yield json.dumps(tc["function"].get("arguments", {}), ensure_ascii=False)
    for t in norm["tools"] or []:
        yield json.dumps(t, ensure_ascii=False)


def has_structural_injection(norm: dict) -> bool:
    for s in _iter_untrusted_strings(norm):
        for lit in STRUCTURAL_SPECIALS:
            if lit in s:
                return True
    return False


# ---------------------------------------------------------------------------
# 렌더 + 스팬 마스킹
# ---------------------------------------------------------------------------
def find_turn_spans(ids: List[int], tok) -> List[Tuple[int, int, str]]:
    """<|im_start|> ... <|im_end|> 턴 스팬 목록. (start, end_exclusive, role).

    start = im_start 다음 토큰, end_exclusive = im_end 포함 다음 인덱스.
    verify_chat_template.py §4 assistant_spans 와 동일한 스캔 (role 일반화).
    """
    spans = []
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] == IM_START_ID:
            j = i + 1
            while j < n and ids[j] != IM_END_ID:
                j += 1
            body = tok.decode(ids[i + 1:j])
            role = body.split("\n", 1)[0] if "\n" in body else body
            spans.append((i + 1, j + 1, role))
            i = j
        i += 1
    return spans


def _split_inline_think(content: Any) -> Optional[Tuple[str, str]]:
    """'<think>R</think>rest' -> (R, rest). 그 형식이 아니면 None."""
    if not isinstance(content, str) or not content.startswith("<think>"):
        return None
    end = content.find("</think>")
    if end < 0:
        return None
    return content[len("<think>"):end], content[end + len("</think>"):]


def truncate_reasoning(tok, norm: dict, seed: int, frac: Tuple[float, float],
                       min_tokens: int, info: Optional[Counter] = None
                       ) -> Tuple[Optional[dict], List[int], Optional[str]]:
    """학습 턴(마지막 user 이후)의 reasoning 을 무작위 토큰 예산으로 절단.

    반환 (새 norm, 절단된 assistant 메시지 인덱스, 드롭 사유). content(응답)
    는 불변. reasoning_content 필드와 inline <think>…</think> 표기 모두 지원
    (원 표기 유지). 예산은 (seed, uuid, 턴 인덱스) 로 결정적 — 워커 순서 무관.
    마지막 user 이전 턴은 일반 대화에서 템플릿이 think 를 제거하므로 대상 아님
    (tool 시나리오의 멀티 assistant 턴은 마지막 user 이후라 각각 독립 예산).
    """
    import random
    msgs = norm["messages"]
    tt = norm["train_turns"]
    last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"),
                    default=-1)
    new_msgs = list(msgs)
    truncated: List[int] = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or i < last_user:
            continue
        if tt is not None and not tt[i]:
            continue
        rc = m.get("reasoning_content")
        inline = None
        if not rc:
            inline = _split_inline_think(m.get("content"))
            if inline is None:
                if info is not None:
                    info["trunc_no_reasoning"] += 1
                continue
            rc = inline[0]
        rc_ids = tok(rc, add_special_tokens=False).input_ids
        L = len(rc_ids)
        if L < min_tokens:
            if info is not None:
                info["trunc_too_short"] += 1
            continue
        rng = random.Random(f"{seed}|{norm['uuid']}|{i}")
        B = max(1, int(L * rng.uniform(frac[0], frac[1])))
        new_rc = tok.decode(rc_ids[:B], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
        nm = dict(m)
        if inline is None:
            nm["reasoning_content"] = new_rc
        else:
            nm["content"] = "<think>" + new_rc + "</think>" + inline[1]
        new_msgs[i] = nm
        truncated.append(i)
        if info is not None:
            info["trunc_turns"] += 1
            info["trunc_tokens_orig"] += L
            info["trunc_tokens_kept"] += B
    if not truncated:
        return None, [], "trunc_none"
    return dict(norm, messages=new_msgs), truncated, None


def render_and_mask(tok, norm: dict, mask_role_header: bool = True,
                    hdr_cache: Optional[dict] = None,
                    medium_effort: bool = False,
                    budget: Optional[dict] = None,
                    info: Optional[Counter] = None
                    ) -> Tuple[Optional[EncodedSample], Optional[str]]:
    """대화 1건 -> (토큰열, 학습마스크). 실패 시 (None, 드롭 사유).

    medium_effort: 템플릿 kwarg 로 마지막 user 턴에 effort 마커 (docstring §Effort).
    budget: {"seed", "frac": (lo, hi), "min_tokens"} — truncate_reasoning 적용
      후 절단 턴의 첫 </think> 를 비학습으로 (info 에 trunc_* 카운터 누적).
    """
    if has_structural_injection(norm):
        return None, "injection"

    truncated_turns: List[int] = []
    if budget is not None:
        norm, truncated_turns, why = truncate_reasoning(
            tok, norm, budget["seed"], budget["frac"], budget["min_tokens"], info)
        if norm is None:
            return None, why

    try:
        kw = {"medium_effort": True} if medium_effort else {}
        rendered = tok.apply_chat_template(
            norm["messages"], tools=norm["tools"],
            tokenize=False, add_generation_prompt=False, **kw,
        )
        ids = tok(rendered, add_special_tokens=False).input_ids
    except Exception:
        return None, "render_error"

    spans = find_turn_spans(ids, tok)
    asst_spans = [(s, e) for (s, e, role) in spans if role == "assistant"]
    asst_msg_flags = [
        (bool(norm["train_turns"][k]) if norm["train_turns"] is not None else True)
        for k, m in enumerate(norm["messages"]) if m["role"] == "assistant"
    ]
    if len(asst_spans) != len(asst_msg_flags):
        return None, "span_mismatch"

    # "assistant\n" 역할 헤더의 토큰 수 (토크나이저 병합에 안전하도록 실측 캐시).
    if mask_role_header:
        if hdr_cache is None:
            hdr_cache = {}
        hdr_len = hdr_cache.get("assistant")
        if hdr_len is None:
            hdr_len = len(tok("assistant\n", add_special_tokens=False).input_ids)
            hdr_cache["assistant"] = hdr_len
    else:
        hdr_len = 0

    ids_np = np.asarray(ids, dtype=np.int32)
    trainable = np.zeros(len(ids), dtype=bool)
    for (s, e), train in zip(asst_spans, asst_msg_flags):
        if train:
            trainable[s + hdr_len:e] = True

    if truncated_turns:
        # 절단 턴의 첫 </think> (= 강제 삽입 지점) 는 예측 대상에서 제외.
        # 그 다음 토큰(응답 첫 토큰)은 학습 — "</think> 가 주어지면 답한다".
        asst_idx = [k for k, m in enumerate(norm["messages"])
                    if m["role"] == "assistant"]
        for t in truncated_turns:
            s, e = asst_spans[asst_idx.index(t)]
            hit = np.nonzero(ids_np[s:e] == THINK_END_ID)[0]
            if hit.size == 0:
                return None, "trunc_no_think_end"
            trainable[s + int(hit[0])] = False
            if info is not None:
                info["trunc_think_end_masked"] += 1

    if not trainable.any():
        return None, "no_trainable"
    return EncodedSample(
        ids=ids_np,
        trainable=trainable,
        uuid=norm["uuid"],
    ), None


# ---------------------------------------------------------------------------
# 문서 emit
# ---------------------------------------------------------------------------
def padded_len(n: int, mult: int) -> int:
    return ((n + mult - 1) // mult) * mult


def emit_document(bin_samples: List[EncodedSample], seq_length: int, mult: int,
                  pad_id: int) -> Tuple[np.ndarray, dict]:
    """bin 1개 -> (doc[2S] = input_ids(S)+labels(S), 통계)."""
    id_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    n_train = 0
    n_real = 0
    for s in bin_samples:
        ids, tr = s.ids, s.trainable
        L = len(ids)
        n_pad = padded_len(L, mult) - L
        seg_ids = np.concatenate(
            [ids, np.full(n_pad, pad_id, dtype=np.int32)])
        # 시프트 라벨: labels[i] = ids[i+1] (i+1 이 학습 대상일 때만)
        seg_labels = np.full(L + n_pad, IGNORE_INDEX, dtype=np.int32)
        pred_idx = np.nonzero(tr[1:])[0]        # i -> i+1 이 학습 대상
        seg_labels[pred_idx] = ids[pred_idx + 1]
        # 세그먼트 마지막 토큰(패딩 포함)에 음수 마커 — position-id 리셋 경계
        seg_ids[-1] = -1 - seg_ids[-1]
        id_parts.append(seg_ids)
        label_parts.append(seg_labels)
        n_train += int(pred_idx.size)
        n_real += L

    packed = np.concatenate(id_parts)
    assert packed.size <= seq_length, "packer overflow"
    tail = seq_length - packed.size
    input_ids = np.concatenate(
        [packed, np.full(tail, pad_id, dtype=np.int32)])
    labels = np.concatenate(
        [np.concatenate(label_parts),
         np.full(tail, IGNORE_INDEX, dtype=np.int32)])

    stats = {
        "n_samples": len(bin_samples),
        "real_tokens": n_real,
        "trainable_tokens": n_train,
        "tail_pad": int(tail),
    }
    return np.concatenate([input_ids, labels]), stats


# ---------------------------------------------------------------------------
# 워커
# ---------------------------------------------------------------------------
_WORKER_TOK = None
_WORKER_ARGS = None


def _worker_init(tokenizer_path: str, mask_role_header: bool,
                 fanout_train_turns: bool = False, medium_effort: bool = False,
                 budget: Optional[dict] = None):
    global _WORKER_TOK, _WORKER_ARGS
    from transformers import AutoTokenizer
    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer_path)
    _WORKER_ARGS = {"mask_role_header": mask_role_header, "hdr_cache": {},
                    "fanout": fanout_train_turns,
                    "medium_effort": medium_effort, "budget": budget}


def _worker_encode(rows: List[Any]
                   ) -> Tuple[List[EncodedSample], Counter, List[dict], Counter]:
    """행 청크 -> (인코딩 샘플, 드롭 카운터, 드롭 견본, info 카운터).

    str 행은 여기서 파싱. fan-out on 이면 drops 는 서브샘플 단위로 센다.
    info: rows(입력 행 수) / fanout_rows(전개된 행) / fanout_subsamples(그 산출)
          / trunc_*(예산 절단 카운터, render_and_mask 가 누적).
    """
    out, drops, dropped_rows, info = [], Counter(), [], Counter()
    for row in rows:
        info["rows"] += 1
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except json.JSONDecodeError:
                drops["bad_row"] += 1
                dropped_rows.append({"reason": "bad_row", "uuid": ""})
                continue
        norm, why = normalize_row(row)
        if norm is None:
            drops[why] += 1
            dropped_rows.append({"reason": why, "uuid": str(row.get("uuid", ""))})
            continue
        subs = expand_train_turns_fanout(norm) if _WORKER_ARGS["fanout"] else [norm]
        if len(subs) > 1:
            info["fanout_rows"] += 1
            info["fanout_subsamples"] += len(subs)
        for sub in subs:
            enc, why = render_and_mask(
                _WORKER_TOK, sub,
                mask_role_header=_WORKER_ARGS["mask_role_header"],
                hdr_cache=_WORKER_ARGS["hdr_cache"],
                medium_effort=_WORKER_ARGS["medium_effort"],
                budget=_WORKER_ARGS["budget"],
                info=info,
            )
            if enc is None:
                drops[why] += 1
                dropped_rows.append({"reason": why, "uuid": sub["uuid"]})
                continue
            out.append(enc)
    return out, drops, dropped_rows, info


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="messages -> Megatron idxmap SFT (alpha). 상세는 파일 docstring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True,
                   help="jsonl / jsonl.gz / parquet 파일 또는 디렉토리(재귀)")
    p.add_argument("--tokenizer", required=True,
                   help="HF tokenizer 디렉토리 (examples/alpha/tokenizer_v5)")
    p.add_argument("--output-prefix", required=True,
                   help="출력 경로 prefix -> <prefix>_text_document.{bin,idx}")
    p.add_argument("--seq-length", type=int, default=65536)
    p.add_argument("--pad-doc-multiple", type=int, default=16,
                   help="샘플별 세그먼트 정렬 배수 (THD+CP: 2*max_cp 이상)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk-rows", type=int, default=64,
                   help="워커당 인코딩 청크 행 수")
    p.add_argument("--no-mask-role-header", dest="mask_role_header",
                   action="store_false", default=True,
                   help='"assistant\\n" 헤더 토큰에도 loss 부여')
    p.add_argument("--fanout-train-turns", action="store_true",
                   help="multi-True train_turns 행을 True 턴별 서브샘플로 전개 "
                        "— 중간 학습 턴의 reasoning 보존 (IF 계열용, "
                        "docstring 의도적 차이 #2)")
    p.add_argument("--medium-effort", action="store_true",
                   help="마지막 user 턴에 '{reasoning effort: efficient}' 마커 "
                        "(템플릿 medium_effort=True; docstring §Effort/Budget)")
    p.add_argument("--truncate-reasoning-budget", action="store_true",
                   help="학습 턴 reasoning 을 무작위 토큰 예산으로 절단 + 잘린 자리 "
                        "</think> 비학습 (budget-control 파생 셋). --medium-effort 와 배타")
    p.add_argument("--truncate-frac", default="0.1,0.9",
                   help="예산 비율 U(lo,hi) — 턴의 reasoning 토큰 길이 대비")
    p.add_argument("--truncate-min-tokens", type=int, default=64,
                   help="이보다 짧은 reasoning 턴은 절단 대상에서 제외")
    p.add_argument("--truncate-seed", type=int, default=0)
    p.add_argument("--row-stride", type=int, default=1,
                   help="K>1 이면 K 행마다 1행만 읽음 (파생 셋 크기 조절)")
    p.add_argument("--min-tokens", type=int, default=None,
                   help="렌더 길이가 이 값 이하인 샘플 제외 — 128k 장문 버킷 생성용 "
                        "(64k 버킷과의 중복 방지: 64k 버킷은 >64k를 too_long 드롭, "
                        "128k 버킷은 --min-tokens 65536 으로 그 여집합만 수용)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="입력 상한 (스모크/드라이런)")
    p.add_argument("--measure-only", action="store_true",
                   help="패킹/emit 생략 — 렌더 길이 분포·드롭율만 stats 로 기록 "
                        "(전수 길이 재측정용; bin 파일 생성 안 함)")
    p.add_argument("--dropped-jsonl", default=None,
                   help="드롭 행 기록 (기본 <prefix>.dropped.jsonl)")
    p.add_argument("--stats-json", default=None,
                   help="집계 기록 (기본 <prefix>.stats.json)")
    return p.parse_args()


def main():
    args = get_args()
    assert args.seq_length % args.pad_doc_multiple == 0, \
        "--seq-length 는 --pad-doc-multiple 의 배수여야 함"
    assert not (args.medium_effort and args.truncate_reasoning_budget), \
        "--medium-effort 와 --truncate-reasoning-budget 은 배타 (별개 파생 셋)"
    assert args.row_stride >= 1, "--row-stride 는 1 이상"
    budget = None
    if args.truncate_reasoning_budget:
        lo, hi = (float(x) for x in args.truncate_frac.split(","))
        assert 0.0 < lo <= hi < 1.0, "--truncate-frac 는 0<lo<=hi<1"
        budget = {"seed": args.truncate_seed, "frac": (lo, hi),
                  "min_tokens": args.truncate_min_tokens}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_id = tok.pad_token_id
    assert pad_id is not None, "tokenizer 에 pad 토큰 필요"
    vocab_size = len(tok)

    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dropped_path = args.dropped_jsonl or args.output_prefix + ".dropped.jsonl"
    stats_path = args.stats_json or args.output_prefix + ".stats.json"

    # -- 인코딩 (병렬) --------------------------------------------------------
    t0 = time.time()
    samples: List[EncodedSample] = []
    drops = Counter()
    fanout_agg = Counter()
    n_rows = 0
    too_long_hist = Counter()

    def _chunks():
        buf = []
        kept = 0
        for i, row in enumerate(iter_rows(args.input)):
            if args.row_stride > 1 and i % args.row_stride:
                continue
            if args.max_rows is not None and kept >= args.max_rows:
                break
            kept += 1
            buf.append(row)
            if len(buf) >= args.chunk_rows:
                yield buf
                buf = []
        if buf:
            yield buf

    dropped_f = open(dropped_path, "w", encoding="utf-8")
    measured_lens: List[int] = []
    measured_train: int = 0

    def _consume(result):
        nonlocal n_rows, measured_train
        encs, d, dropped_rows, info = result
        n_rows += info["rows"]
        fanout_agg.update({k: v for k, v in info.items() if k != "rows"})
        drops.update(d)
        for r in dropped_rows:
            dropped_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for e in encs:
            if args.min_tokens is not None and len(e.ids) <= args.min_tokens:
                drops["below_min"] += 1
                continue
            if args.measure_only:
                # 토큰 배열은 버리고 길이·학습토큰 수만 축적 (988G 전수 스캔용)
                measured_lens.append(len(e.ids))
                measured_train += int(e.trainable.sum())
                if len(e.ids) > args.seq_length:
                    drops["too_long"] += 1
                continue
            if len(e.ids) > args.seq_length:
                drops["too_long"] += 1
                too_long_hist[min(len(e.ids) // 8192, 64) * 8192] += 1
                dropped_f.write(json.dumps(
                    {"reason": "too_long", "uuid": e.uuid,
                     "n_tokens": len(e.ids)}, ensure_ascii=False) + "\n")
            else:
                samples.append(e)

    if args.workers <= 1:
        _worker_init(args.tokenizer, args.mask_role_header,
                     args.fanout_train_turns, args.medium_effort, budget)
        for chunk in _chunks():
            _consume(_worker_encode(chunk))
    else:
        with mp.Pool(args.workers, initializer=_worker_init,
                     initargs=(args.tokenizer, args.mask_role_header,
                               args.fanout_train_turns, args.medium_effort,
                               budget)) as pool:
            for result in pool.imap(_worker_encode, _chunks(), chunksize=1):
                _consume(result)
    dropped_f.close()
    t_enc = time.time() - t0

    if args.measure_only:
        lens = np.array(measured_lens, dtype=np.int64)
        qs = {}
        if lens.size:
            for q in (50, 90, 95, 99):
                qs[f"p{q}"] = int(np.percentile(lens, q))
            qs["max"] = int(lens.max())
        stats = {
            "input": args.input,
            "mode": "measure_only",
            "fanout_train_turns": args.fanout_train_turns,
            "rows_read": n_rows,
            "fanout_rows": int(fanout_agg["fanout_rows"]),
            "fanout_subsamples": int(fanout_agg["fanout_subsamples"]),
            "row_stride": args.row_stride,
            "medium_effort": args.medium_effort,
            "truncate_reasoning_budget": budget,
            "truncate": {k: int(v) for k, v in sorted(fanout_agg.items())
                         if k.startswith("trunc_")},
            "rows_rendered": int(lens.size),
            "drops": dict(drops),
            "total_tokens": int(lens.sum()),
            "trainable_tokens": int(measured_train),
            "len_percentiles": qs,
            "fit": {f"fit@{k//1024}k": (float((lens <= k).mean()) if lens.size else 0.0)
                    for k in (16384, 32768, 65536, 131072)},
            "elapsed_sec": round(t_enc, 1),
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[measure] rows={n_rows} rendered={lens.size} "
              f"tokens={stats['total_tokens']:,} drops={dict(drops)}")
        print(f"[measure] {qs}  fit={stats['fit']}")
        print(f"[done] stats -> {stats_path} ({t_enc:.1f}s)")
        return

    print(f"[encode] rows={n_rows} kept={len(samples)} drops={dict(drops)} "
          f"({t_enc:.1f}s)", flush=True)
    if not samples:
        print("[abort] 남은 샘플 0 — 출력 생성 안 함")
        sys.exit(1)

    # -- 패킹 (BFD, bestfit_pack.py 재사용) + emit ----------------------------
    from megatron.core.datasets import indexed_dataset

    item_lens = np.array(
        [padded_len(len(s.ids), args.pad_doc_multiple) for s in samples],
        dtype=np.int64)
    item_bin, num_bins, _ = bestfit_decreasing(
        item_lens, args.seq_length, log_every=1_000_000)

    bin_members: List[List[int]] = [[] for _ in range(num_bins)]
    for i, b in enumerate(item_bin):
        bin_members[b].append(i)

    builder = indexed_dataset.IndexedDatasetBuilder(
        args.output_prefix + "_text_document.bin",
        dtype=indexed_dataset.DType.optimal_dtype(vocab_size),
    )
    agg = Counter()
    for members in bin_members:
        doc, st = emit_document(
            [samples[i] for i in members],
            args.seq_length, args.pad_doc_multiple, pad_id)
        builder.add_document(doc, [doc.size])
        for k, v in st.items():
            agg[k] += v
    builder.finalize(args.output_prefix + "_text_document.idx")
    t_all = time.time() - t0

    if num_bins < 100:
        # 전역 split "99,1,0" 에서 1% < 1 doc 이면 valid 가 0-doc — Megatron
        # GPTDataset 의 epoch 루프가 무한 대기(NCCL 타임아웃으로 위장; identity
        # 19-bins 실사고 2026-08-23). 입력 반복(×k) 등으로 bins 를 늘릴 것.
        print(f"[warn] bins={num_bins} < 100 — 블렌드 split 1% 가 0-doc 이 되어 "
              f"valid 인덱스 빌드가 무한 대기할 수 있음 (입력 반복으로 증폭 권장)")

    stats = {
        "input": args.input,
        "seq_length": args.seq_length,
        "pad_doc_multiple": args.pad_doc_multiple,
        "mask_role_header": args.mask_role_header,
        "fanout_train_turns": args.fanout_train_turns,
        "rows_read": n_rows,
        "fanout_rows": int(fanout_agg["fanout_rows"]),
        "fanout_subsamples": int(fanout_agg["fanout_subsamples"]),
        "row_stride": args.row_stride,
        "medium_effort": args.medium_effort,
        "truncate_reasoning_budget": budget,
        "truncate": {k: int(v) for k, v in sorted(fanout_agg.items())
                     if k.startswith("trunc_")},
        "samples_kept": len(samples),
        "drops": dict(drops),
        "too_long_hist_tok_buckets": {str(k): v for k, v in sorted(too_long_hist.items())},
        "n_bins": num_bins,
        "real_tokens": int(agg["real_tokens"]),
        "trainable_tokens": int(agg["trainable_tokens"]),
        "fill_rate": agg["real_tokens"] / (num_bins * args.seq_length),
        "elapsed_sec": round(t_all, 1),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[pack] bins={num_bins} fill={stats['fill_rate']:.4f} "
          f"real={agg['real_tokens']:,} trainable={agg['trainable_tokens']:,}")
    print(f"[done] {args.output_prefix}_text_document.{{bin,idx}} "
          f"({t_all:.1f}s)  stats -> {stats_path}")
    if drops:
        print(f"[warn] dropped rows -> {dropped_path}")


if __name__ == "__main__":
    main()
