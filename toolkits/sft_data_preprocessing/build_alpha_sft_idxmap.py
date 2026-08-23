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

## NVIDIA 공식 구현 대조 (2026-08-23, NVIDIA-NeMo/Nemotron data_prep 정독)

일치: assistant 헤더 비학습 / think 내용 학습 / im_end 학습(정지) / tool
응답·system 비학습 / 라벨 1칸 시프트 + 샘플 경계 라벨 0 (materialize.py 계약
"loss_mask[j] -> input_ids[j+1]" 과 동형).
의도적 차이 4건:
  1. think-opener 토큰(<think>·개행)도 학습에 포함 — NVIDIA 는 청크별 인코딩이라
     gen-prompt 경계에서 제외 가능하지만, 우리는 전체 렌더 인코딩(추론과 동일한
     토큰화)이라 경계가 토큰 정렬을 깨뜨릴 수 있어 포함. 결정적 토큰 소량이라 무해.
  2. thinking 멀티(user)턴 fan-out 미채택 — NVIDIA 는 user 턴마다 서브시퀀스로
     복제해 각 턴의 reasoning 을 보존; 우리는 단일 시퀀스로 렌더된 그대로 학습
     (히스토리 think 는 템플릿이 제거). train_turns 있는 셋(Chat-v3)은 어차피
     마지막 턴만 학습이라 등가. 단일 user 턴 + tool 루프 대화도 등가
     (last_user_idx 이후 assistant 턴은 reasoning 유지).
  3. 초과 길이 truncate 대신 드롭 — NVIDIA 는 pack_size 로 무언 head-truncate.
     증명/트레이스 절단 유해 판단 (docs/SFT_RL_DATASETS.md §2.4).
  4. injection 방어 추가 — NVIDIA 는 없음 (content 의 <|im_start|> 가 실토큰화됨).
공통 채택: 샘플별 %16 정렬은 NVIDIA parquet 경로엔 없으나(pad_seq_to_mult 가
create_sft_dataset 에서 소실되는 상류 버그로 CP 정렬 미동작) 우리 GDN THD+CP 는
하드 요구라 데이터 시점에 굽는다.

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
IGNORE_INDEX = -100
# content 안에 있으면 naive 인코딩이 턴 구조를 오염시키는 리터럴들.
STRUCTURAL_SPECIALS = ("<|im_start|>", "<|im_end|>")


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
                yield json.loads(line)


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
        if role not in ("system", "user", "assistant", "tool"):
            return None, "bad_row"
        content = m.get("content")
        tool_calls = m.get("tool_calls") or None
        if content is None:
            # assistant + tool_calls 는 정상 관례 (content 없는 도구 호출 턴).
            # 그 외 null 은 라이선스 마스킹 (Chat-v3) — 복원 전 투입 금지.
            if role == "assistant" and tool_calls:
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


def render_and_mask(tok, norm: dict, mask_role_header: bool = True,
                    hdr_cache: Optional[dict] = None
                    ) -> Tuple[Optional[EncodedSample], Optional[str]]:
    """대화 1건 -> (토큰열, 학습마스크). 실패 시 (None, 드롭 사유)."""
    if has_structural_injection(norm):
        return None, "injection"

    try:
        rendered = tok.apply_chat_template(
            norm["messages"], tools=norm["tools"],
            tokenize=False, add_generation_prompt=False,
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

    trainable = np.zeros(len(ids), dtype=bool)
    for (s, e), train in zip(asst_spans, asst_msg_flags):
        if train:
            trainable[s + hdr_len:e] = True

    if not trainable.any():
        return None, "no_trainable"
    return EncodedSample(
        ids=np.asarray(ids, dtype=np.int32),
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


def _worker_init(tokenizer_path: str, mask_role_header: bool):
    global _WORKER_TOK, _WORKER_ARGS
    from transformers import AutoTokenizer
    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer_path)
    _WORKER_ARGS = {"mask_role_header": mask_role_header, "hdr_cache": {}}


def _worker_encode(rows: List[dict]) -> Tuple[List[EncodedSample], Counter, List[dict]]:
    """행 청크 -> (인코딩 샘플, 드롭 카운터, 드롭 견본)."""
    out, drops, dropped_rows = [], Counter(), []
    for row in rows:
        norm, why = normalize_row(row)
        if norm is None:
            drops[why] += 1
            dropped_rows.append({"reason": why, "uuid": str(row.get("uuid", ""))})
            continue
        enc, why = render_and_mask(
            _WORKER_TOK, norm,
            mask_role_header=_WORKER_ARGS["mask_role_header"],
            hdr_cache=_WORKER_ARGS["hdr_cache"],
        )
        if enc is None:
            drops[why] += 1
            dropped_rows.append({"reason": why, "uuid": norm["uuid"]})
            continue
        out.append(enc)
    return out, drops, dropped_rows


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
    n_rows = 0
    too_long_hist = Counter()

    def _chunks():
        buf = []
        for i, row in enumerate(iter_rows(args.input)):
            if args.max_rows is not None and i >= args.max_rows:
                break
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
        encs, d, dropped_rows = result
        n_rows += sum(d.values()) + len(encs)
        drops.update(d)
        for r in dropped_rows:
            dropped_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for e in encs:
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
        _worker_init(args.tokenizer, args.mask_role_header)
        for chunk in _chunks():
            _consume(_worker_encode(chunk))
    else:
        with mp.Pool(args.workers, initializer=_worker_init,
                     initargs=(args.tokenizer, args.mask_role_header)) as pool:
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
            "rows_read": n_rows,
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

    stats = {
        "input": args.input,
        "seq_length": args.seq_length,
        "pad_doc_multiple": args.pad_doc_multiple,
        "mask_role_header": args.mask_role_header,
        "rows_read": n_rows,
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
