# /// script
# dependencies = ["pandas", "pyarrow", "pyyaml"]
# ///
"""export_sft.py — 생성 결과를 alpha SFT 포맷(Nemotron messages jsonl)으로 변환한다.

파이프라인:

  artifacts parquet
      │  ① 하드 게이트   identity_check.is_valid
      │  ② 소프트 게이트 judge 점수 임계
      │  ③ 중복 제거     정규화 완전중복 + 접두 버킷 상한(템플릿 붕괴 차단)
      │  ④ 홀드아웃 분리 학습에 쓰지 않을 평가셋
      ▼
  train.jsonl / eval.jsonl

출력 스키마는 보유 중인 Nemotron-SFT-Instruction-Following-Chat-v3 레코드와 동일하다:

    {"messages": [...], "used_in": [...], "uuid": "...", "metadata": {...}}

assistant 메시지는 alpha chat template 규약을 따른다
(examples/alpha/tokenizer_v5/chat_template.jinja):
  - reasoning_content 가 있으면  → <think>\\n{rc}</think>{content}
  - 없으면                        → <think></think>{content}  (no-think 모드)
metadata.train_turns 는 턴별 loss 마스크다 (assistant 턴만 True).

사용:
    uv run export_sft.py --dataset artifacts/alpha_identity/**/dataset.parquet \\
        --out-dir out --holdout 400 --verify-template
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import uuid as uuid_mod
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

HERE = Path(__file__).parent
CARD_PATH = HERE / "identity_card.yaml"
TOKENIZER_DIR = HERE.parent.parent / "tokenizer_v5"

# system_variant == "persona_generic" 일 때 붙일 시스템 프롬프트.
# ★ 정체성 정보를 절대 담지 않는다 — 정체성은 시스템 프롬프트가 아니라
#   가중치에 내재해야 하기 때문. (identity_card.yaml system_prompt 주석 참조)
PERSONA_SYSTEM = {
    "ko": [
        "당신은 사용자를 돕는 유용한 AI 어시스턴트입니다.",
        "당신은 친절하고 정확한 AI 어시스턴트입니다. 명확하고 간결하게 답하세요.",
        "사용자의 질문에 성실하게 답변하는 AI 어시스턴트입니다.",
    ],
    "_": [
        "You are a helpful assistant.",
        "You are a helpful, knowledgeable assistant. Answer clearly and concisely.",
        "You are a friendly AI assistant. Be accurate and to the point.",
    ],
}

DEFAULT_THRESHOLDS = {
    "fact_consistency": 4,
    "denial_firmness": 4,
    "naturalness": 3,
    "language_match": 4,
}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


# =============================================================================
# 헬퍼
# =============================================================================


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return _WS.sub(" ", _PUNCT.sub("", text)).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    """구조화 컬럼은 dict 로 오지만, 스킵된 행은 None/NaN 이다."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _judge_score(judge: Any, name: str) -> int | None:
    """LLMJudgeColumnConfig 는 {score_name: {reasoning, score}} 를 만든다."""
    entry = _as_dict(judge).get(name)
    if isinstance(entry, dict) and "score" in entry:
        try:
            return int(entry["score"])
        except (TypeError, ValueError):
            return None
    return None


def _assistant_message(turn: dict[str, Any]) -> dict[str, str]:
    """chat template 규약에 맞는 assistant 메시지를 만든다.

    reasoning_content 는 비어 있으면 넣지 않는다 — 템플릿이 <think></think> 를
    자동으로 삽입하므로 빈 문자열을 넣는 것과 없는 것이 동치이고, 없는 편이
    보유 중인 Nemotron 레코드와 형태가 같다.
    """
    message = {"role": "assistant", "content": str(turn.get("content", "")).strip()}
    reasoning = str(turn.get("reasoning", "") or "").strip()
    if reasoning:
        message["reasoning_content"] = reasoning
    return message


def _system_content(row: pd.Series, card: dict[str, Any], rng_index: int) -> str | None:
    variant = row.get("system_variant")
    if variant == "none":
        return None
    if variant == "nemotron_default":
        return card["system_prompt"]["nemotron_default"]
    pool = PERSONA_SYSTEM.get(str(row.get("language")), PERSONA_SYSTEM["_"])
    return pool[rng_index % len(pool)]


# =============================================================================
# 변환
# =============================================================================


def build_record(row: pd.Series, card: dict[str, Any], index: int, teacher: str) -> dict[str, Any] | None:
    turn1 = _as_dict(row.get("assistant_turn"))
    if not str(turn1.get("content", "")).strip():
        return None

    messages: list[dict[str, str]] = []
    system = _system_content(row, card, index)
    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": str(row["user_turn"]).strip()})
    messages.append(_assistant_message(turn1))

    followup = _as_dict(row.get("followup_user"))
    turn2 = _as_dict(row.get("assistant_turn_2"))
    if str(followup.get("question", "")).strip() and str(turn2.get("content", "")).strip():
        messages.append({"role": "user", "content": str(followup["question"]).strip()})
        messages.append(_assistant_message(turn2))

    record_uuid = str(row.get("record_uuid") or "").strip() or str(uuid_mod.uuid4())

    return {
        "messages": messages,
        "used_in": list(card["dataset_tagging"]["used_in"]),
        "uuid": record_uuid,
        "metadata": {
            "seed_dataset": (
                card["dataset_tagging"]["seed_dataset"]
                if row.get("track") == "seed"
                else "synthetic"
            ),
            "seed_prompt_sha256": str(row.get("seed_prompt_sha256") or "") or None,
            "model": teacher,
            "reward_model": None,
            # 턴별 loss 마스크 — assistant 턴만 학습한다
            "train_turns": [m["role"] == "assistant" for m in messages],
            # alpha 내부 추적용 (Nemotron 스키마에는 없는 확장 필드)
            "probe_type": row.get("probe_type"),
            "creator_mention": row.get("creator_mention") or None,
            "language": row.get("language"),
            "turn_shape": row.get("turn_shape"),
        },
    }


# =============================================================================
# 필터링
# =============================================================================


def apply_filters(
    df: pd.DataFrame,
    thresholds: dict[str, int],
    prefix_cap: int,
    revalidate: bool = False,
    dedup_scope: str = "assistant",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """dedup_scope: "assistant"(기본) = 답 텍스트 기준 완전중복·접두/후미 버킷.
    "qa" = 완전중복은 질문+답, 버킷은 **질문** 텍스트 기준 — 답 문장이 카드로 고정된 슬라이스
    (creator, 2026-09-01)에서 "질문은 다르고 답은 같은" 행을 살리기 위함. 답이 고정이면
    후미 버킷은 요구 사항과 충돌한다(2,000행 → 618행 실측)."""
    stats: dict[str, int] = {"input": len(df)}

    # ① 하드 게이트
    if revalidate:
        # 규칙 게이트는 생성 시점에 parquet 에 박힌다. 규칙을 고쳐도 소급되지 않으므로
        # 원본 컬럼에서 다시 계산할 수 있어야 한다 — 규칙 수정 때마다 재생성(수 시간)
        # 하지 않기 위한 장치.
        import sys

        sys.path.insert(0, str(HERE))
        from identity_sdg import validate_identity  # noqa: E402

        recomputed = validate_identity(df)
        checks = recomputed.apply(
            lambda r: {"is_valid": bool(r["is_valid"]), "invalid_reason": r["invalid_reason"]},
            axis=1,
        )
    else:
        checks = df["identity_check"].apply(_as_dict)
    hard_ok = checks.apply(lambda c: bool(c.get("is_valid", False)))
    reasons = Counter()
    for check, ok in zip(checks, hard_ok):
        if not ok:
            for reason in str(check.get("invalid_reason", "") or "unknown").split(","):
                if reason:
                    reasons[reason] += 1
    df = df[hard_ok]
    stats["after_rule_gate"] = len(df)
    stats_reasons = dict(reasons.most_common())

    # ② 소프트 게이트
    if "judge" in df.columns and len(df):
        keep = pd.Series(True, index=df.index)
        for name, minimum in thresholds.items():
            scores = df["judge"].apply(lambda j, n=name: _judge_score(j, n))
            # 점수를 못 읽은 행은 통과시킨다 (심판 실패로 데이터를 잃지 않도록)
            keep &= scores.apply(lambda s, m=minimum: s is None or s >= m)
        df = df[keep]
    stats["after_judge_gate"] = len(df)

    # ③ 중복 제거
    if len(df):
        ans = df["assistant_turn"].apply(lambda t: _normalize(_as_dict(t).get("content", "")))
        if dedup_scope == "qa":
            q = df["user_turn"].apply(lambda t: _normalize(str(t or "")))
            norm = q + " ⟶ " + ans
        else:
            norm = ans
        df = df[~norm.duplicated()]
        stats["after_exact_dedup"] = len(df)

        # 접두/후미 버킷 상한 — 템플릿 붕괴를 막는다.
        # ★ 후미가 특히 중요하다. 오귀속 응답은 도입부가 질문마다 달라도
        #   ("아니요, 저는 GPT-4가...", "아니요, 저는 Mixtral이...")
        #   마무리 문장이 똑같이 수렴하는 경향이 있다 — preview 실측 26%.
        for label, slicer in (("prefix", lambda s: s[:40]), ("suffix", lambda s: s[-40:])):
            if dedup_scope == "qa":
                norm = df["user_turn"].apply(lambda t: _normalize(str(t or "")))
            else:
                norm = df["assistant_turn"].apply(
                    lambda t: _normalize(_as_dict(t).get("content", ""))
                )
            bucket_counts: defaultdict[tuple, int] = defaultdict(int)
            keep_rows: list[bool] = []
            for idx, text in norm.items():
                key = (df.at[idx, "probe_type"], df.at[idx, "language"], slicer(text))
                bucket_counts[key] += 1
                keep_rows.append(bucket_counts[key] <= prefix_cap)
            df = df[pd.Series(keep_rows, index=df.index)]
            stats[f"after_{label}_cap"] = len(df)

    return df, {"_reasons": stats_reasons, **stats}


# =============================================================================
# chat template 검증
# =============================================================================


def verify_template(records: list[dict[str, Any]], tokenizer_dir: Path, n: int = 3) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  [skip] transformers 미설치 — chat template 검증 생략")
        return
    if not tokenizer_dir.exists():
        print(f"  [skip] tokenizer 없음: {tokenizer_dir}")
        return

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    for record in records[:n]:
        rendered = tok.apply_chat_template(record["messages"], tokenize=False)
        assert "<|im_start|>assistant" in rendered, "assistant 턴이 렌더되지 않음"
        assert "<think>" in rendered, "think 마커 없음 — 템플릿 규약 위반"
        n_assistant = sum(1 for m in record["messages"] if m["role"] == "assistant")
        assert sum(record["metadata"]["train_turns"]) == n_assistant, "train_turns 불일치"
    print(f"  ✅ chat template 검증 통과 ({min(n, len(records))}건)")
    print("\n  --- 렌더 샘플 ---")
    print("  " + tok.apply_chat_template(records[0]["messages"], tokenize=False).replace("\n", "\n  ")[:900])


# =============================================================================
# 진입점
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="생성 결과 parquet (glob 가능)")
    parser.add_argument("--out-dir", type=Path, default=HERE / "out")
    parser.add_argument("--teacher", type=str, default=None, help="생성에 쓴 교사 모델명")
    parser.add_argument("--holdout", type=int, default=400, help="학습에서 제외할 평가셋 행 수")
    parser.add_argument("--prefix-cap", type=int, default=5)
    parser.add_argument("--dedup-scope", choices=("assistant", "qa"), default="assistant",
                        help="qa: 완전중복은 질문+답, 버킷은 질문 기준 (답 고정 슬라이스용)")
    parser.add_argument("--verify-template", action="store_true")
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help=(
            "저장된 identity_check 대신 현재 규칙으로 다시 검증한다. "
            "게이트 규칙을 고친 뒤 재생성 없이 반영할 때 사용."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260807)
    for name, default in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--min-{name.replace('_', '-')}", type=int, default=default)
    args = parser.parse_args()

    card = yaml.safe_load(CARD_PATH.read_text())
    teacher = args.teacher or card["dataset_tagging"]["metadata_model"]

    # 절대/상대 경로와 glob 패턴을 모두 받는다 (Path.glob 은 절대 패턴을 못 쓴다)
    paths = [Path(p) for p in sorted(glob(args.dataset, recursive=True))]
    if not paths:
        direct = Path(args.dataset)
        if not direct.exists():
            raise SystemExit(f"🛑 데이터셋을 찾을 수 없습니다: {args.dataset}")
        paths = [direct]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    thresholds = {n: getattr(args, f"min_{n}") for n in DEFAULT_THRESHOLDS}
    df, stats = apply_filters(df, thresholds, args.prefix_cap, revalidate=args.revalidate,
                              dedup_scope=args.dedup_scope)

    records = [
        rec
        for i, (_, row) in enumerate(df.iterrows())
        if (rec := build_record(row, card, i, teacher)) is not None
    ]

    # ④ 홀드아웃 — 평가셋 프롬프트는 학습에 넣지 않는다
    rng = pd.Series(range(len(records))).sample(frac=1.0, random_state=args.seed).tolist()
    holdout_idx = set(rng[: min(args.holdout, len(records) // 10)])
    train = [r for i, r in enumerate(records) if i not in holdout_idx]
    evalset = [r for i, r in enumerate(records) if i in holdout_idx]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("eval", evalset)):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as fh:
            for record in rows:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── 리포트 ──
    print("\n필터 단계별 잔존:")
    for key in (
        "input",
        "after_rule_gate",
        "after_judge_gate",
        "after_exact_dedup",
        "after_prefix_cap",
        "after_suffix_cap",
    ):
        if key in stats:
            print(f"  {key:<20} {stats[key]:6,}")
    if stats.get("_reasons"):
        print("  규칙 게이트 탈락 사유:")
        for reason, count in stats["_reasons"].items():
            print(f"    {reason:<32} {count:5,}")

    print(f"\n✅ train {len(train):,}행 → {args.out_dir / 'train.jsonl'}")
    print(f"✅ eval  {len(evalset):,}행 → {args.out_dir / 'eval.jsonl'}")

    if train:
        meta = pd.DataFrame([r["metadata"] for r in train])
        print("\n  probe_type 분포:")
        for k, v in meta["probe_type"].value_counts().items():
            print(f"    {k:<28} {v:5,}  ({v / len(train):5.1%})")
        print("  language 분포:")
        for k, v in meta["language"].value_counts().items():
            print(f"    {k:<6} {v:5,}  ({v / len(train):5.1%})")
        # ★ 메시지 개수로 세면 안 된다 — system 없는 행(약 절반)은 멀티턴이어도
        #   user+asst+user+asst = 4개다. assistant 턴 수로 센다.
        multi = sum(
            1 for r in train if sum(m["role"] == "assistant" for m in r["messages"]) >= 2
        )
        print(f"  멀티턴: {multi:,} ({multi / len(train):.1%})")
        with_sys = sum(1 for r in train if r["messages"][0]["role"] == "system")
        print(f"  system 있음: {with_sys:,} ({with_sys / len(train):.1%})")

    if args.verify_template and train:
        print()
        verify_template(train, TOKENIZER_DIR)

    # ── 블렌드 권고 ──
    if train:
        print("\n블렌드 권고 (identity_card 및 SFT_RL_DATASETS.md 기준):")
        print(f"  - SFT 블렌드 내 비중 0.3~1.0% 유지 → 전체 SFT {len(train) * 100:,}~{len(train) * 333:,}행 규모에 적합")
        print("  - identity 데이터만 반복 에폭 금지 (모든 질문에 자기소개하는 과적합 발생)")
        print("  - eval.jsonl 은 학습에 넣지 말 것 — 정체성 유지율 측정용")


if __name__ == "__main__":
    main()
