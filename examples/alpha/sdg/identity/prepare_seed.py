# /// script
# dependencies = ["pandas", "pyarrow", "pyyaml"]
# ///
"""prepare_seed.py — alpha-banana 정체성 SFT 데이터셋 시드 빌더.

두 트랙을 하나의 시드 parquet 으로 합친다.

  Track A (seed)  — Nemotron-RL-Identity-Following-v1 의 실제 프롬프트 뱅크에서
                    오귀속(misattribution) 질문을 가져온다. 사람이 시드한 실제
                    분포라 LLM 이 지어낸 질문보다 현실적이다. CC-BY-4.0, 상업 사용 가능.
  Track B (synth) — 나머지 probe type. user turn 을 비워두고 Data Designer 가
                    생성 단계에서 채운다 (SkipConfig 게이트로 분기).

시드 단계에서 확정하는 축은 **서로 상관이 있어야 하는 것들**이다
(probe_type × language × turn_shape × wrong_entity × creator_tier).
서로 독립인 축(register / system_variant / thinking_mode / length_style)은
Data Designer 샘플러가 생성 시점에 뽑는다.

사용:
    uv run prepare_seed.py --num-records 15000 --out seed.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# =============================================================================
# 상수
# =============================================================================

NEMOTRON_BANK = Path(
    "/home/work/Datasets/LL_datasets/posttraining/RL/"
    "Nemotron-RL-Identity-Following-v1/train.jsonl"
)

# 뱅크의 principle 문자열에 적힌 언어명 → 코드
BANK_LANG_TO_CODE = {
    "German": "de",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Portugese": "pt",  # 뱅크 원문의 철자 그대로
    "Chinese": "zh",
}

# 프롬프트에 쓸 영어 언어명
CODE_TO_LANG_NAME = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
}

# ★ Hindi 제외: alpha 미지원 언어.
#   근거 — examples/alpha/docs/SFT_RL_DATASETS.md §2.1
#          "SFT-Multilingual-v2 ... ko/ja/hi/pt × code/math/stem — hi만 alpha 미지원"
EXCLUDED_LANGS = {"hi"}

LANG_WEIGHTS: dict[str, float] = {
    "ko": 0.40,
    "en": 0.30,
    "ja": 0.05,
    "zh": 0.05,
    "es": 0.04,
    "fr": 0.04,
    "de": 0.04,
    "pt": 0.04,
    "it": 0.04,
}

# probe_type → (track, 전체 대비 비율)
#   creator_individual 0.08 은 상시 값. 카드 1.2 의 share_of_identity_rows 0.20 은 phase-2 슬라이스 재생성
#   (--only-probe creator_individual/creator_org) 시 --num-records 로 맞춘다 (SFT_PHASE2_PLAN.md §3).
PROBE_MIX: dict[str, tuple[str, float]] = {
    # ── Track A: 실제 프롬프트 뱅크에서 조달 ──
    "misattribution_reject": ("seed", 0.30),
    # ── Track B: 생성 단계에서 user turn 합성 ──
    "misattribution_pressure": ("synth", 0.08),
    "direct_identity": ("synth", 0.12),
    "creator_org": ("synth", 0.07),
    "creator_individual": ("synth", 0.08),
    "version_naming": ("synth", 0.05),
    "architecture_probe": ("synth", 0.06),
    "scale_probe": ("synth", 0.04),
    "capability_scope": ("synth", 0.05),
    "knowledge_cutoff": ("synth", 0.04),
    "undisclosed_abstention": ("synth", 0.04),
    "anthropomorphic_boundary": ("synth", 0.04),
    "availability": ("synth", 0.03),
}

# probe_type → 멀티턴 비율.
#   압박 저항은 멀티턴에서만 학습 가능하므로 pressure 를 크게 잡는다.
MULTI_TURN_RATE: dict[str, float] = {
    "misattribution_pressure": 0.80,
    "misattribution_reject": 0.25,
    "direct_identity": 0.20,
}
DEFAULT_MULTI_TURN_RATE = 0.15

# 합성(Track B) 압박 프롬프트에 쓸 오귀속 대상.
# identity_card.yaml 의 misattribution.wrong_entities 와 정합. 적대적으로 중요한 것만.
WRONG_ENTITIES: list[tuple[str, list[str]]] = [
    ("OpenAI", ["ChatGPT", "GPT-4", "GPT-5", "GPT-OSS"]),
    ("Anthropic", ["Claude"]),
    ("Google", ["Gemini", "Gemma", "Bard", "PaLM"]),
    ("Meta", ["Llama"]),
    ("Mistral AI", ["Mistral", "Mixtral"]),
    ("Alibaba", ["Qwen", "Qwen3", "Tongyi"]),
    ("DeepSeek", ["DeepSeek-V3", "DeepSeek-R1"]),
    ("NVIDIA", ["Nemotron"]),
    ("Microsoft", ["Phi", "Copilot"]),
    ("Naver", ["HyperCLOVA X"]),
    ("Kakao", ["KoGPT"]),
    ("LG AI Research", ["EXAONE"]),
    ("Upstage", ["Solar"]),
]

# 시드(Track A) 프롬프트에서 대상을 역추출할 때만 쓰는 추가 어휘.
# 뱅크 2,017개 영어 프롬프트의 고유명사 빈도를 실측해서 채웠다 — 합성에는 쓰지 않는다.
EXTRA_DETECT_ONLY: list[tuple[str, list[str]]] = [
    ("OpenAI", ["davinci-001", "davinci-002", "davinci-003", "davinci",
                "Curie", "Babbage", "Ada", "GPT-3.5", "GPT-3", "GPT-2", "GPT",
                "오픈에이아이"]),
    ("IBM", ["Watson", "왓슨", "아이비엠"]),
    ("Google", ["DeepMind", "Alphabet", "BERT", "구글"]),
    ("Microsoft", ["Bing", "Azure", "Office", "마이크로소프트"]),
    ("NVIDIA", ["엔비디아"]),
    ("Meta", ["Facebook", "메타"]),
    ("Amazon", ["AWS", "Alexa", "아마존"]),
    ("Apple", ["Siri", "애플"]),
    ("Intel", ["인텔"]),
    ("Salesforce", []),
    ("Oracle", []),
    ("Tesla", []),
    ("Adobe", []),
    ("Huawei", []),
    ("Qualcomm", []),
    ("Cisco", []),
    ("AMD", []),
    ("Dell", []),
    ("Netflix", []),
    ("Twitter", ["트위터"]),
]


def _build_entity_patterns() -> list[tuple[re.Pattern[str], str, str]]:
    """추출용 패턴 목록. 긴 이름부터 매칭해야 'GPT-4' 가 'GPT' 에 먹히지 않는다."""
    merged: dict[str, list[str]] = {}
    for org, models in [*WRONG_ENTITIES, *EXTRA_DETECT_ONLY]:
        merged.setdefault(org, []).extend(models)

    model_pats = sorted(
        (
            (re.compile(re.escape(m), re.IGNORECASE), org, m)
            for org, models in merged.items()
            for m in models
        ),
        key=lambda t: -len(t[2]),
    )
    org_pats = sorted(
        ((re.compile(re.escape(org), re.IGNORECASE), org, "") for org in merged),
        key=lambda t: -len(t[1]),
    )
    return model_pats + org_pats


_ENTITY_PATTERNS: list[tuple[re.Pattern[str], str, str]] = _build_entity_patterns()


# =============================================================================
# 헬퍼
# =============================================================================


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """가중치를 정수 카운트로 배분한다 (최대잔여법). 합계가 정확히 total 이 된다."""
    raw = {k: total * w for k, w in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[:remainder]:
        counts[k] += 1
    return counts


def _detect_entity(text: str) -> tuple[str, str]:
    """프롬프트 텍스트에서 오귀속 대상 (org, model) 을 추출한다."""
    for pattern, org, model in _ENTITY_PATTERNS:
        if pattern.search(text):
            return org, model
    return "", ""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_bank(path: Path) -> dict[str, list[str]]:
    """Nemotron 뱅크를 언어코드 → 프롬프트 리스트로 적재한다 (중복 제거)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Nemotron identity 프롬프트 뱅크를 찾을 수 없습니다: {path}\n"
            "  --bank 로 경로를 지정하거나 --no-bank 로 Track A 를 비활성화하세요."
        )
    lang_re = re.compile(r"must be in the (\w+) language")
    by_lang: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            match = lang_re.search(row.get("principle", ""))
            if match is None:
                continue
            code = BANK_LANG_TO_CODE.get(match.group(1))
            if code is None or code in EXCLUDED_LANGS:
                continue
            turns = row["responses_create_params"]["input"]
            if len(turns) != 1 or turns[0]["role"] != "user":
                continue
            prompt = turns[0]["content"].strip()
            if not prompt:
                continue
            bucket = seen.setdefault(code, set())
            if prompt in bucket:
                continue
            bucket.add(prompt)
            by_lang.setdefault(code, []).append(prompt)
    return by_lang


# =============================================================================
# 시드 구성
# =============================================================================


def build_seed(
    num_records: int,
    bank_path: Path | None,
    seed: int,
    only_probes: list[str] | None = None,
) -> pd.DataFrame:
    rng = random.Random(seed)

    bank = load_bank(bank_path) if bank_path is not None else {}

    # 특정 probe type 만 재생성할 때 (Identity Card 변경으로 일부 슬라이스만
    # 다시 만들어야 하는 경우). 남은 probe 들 사이에서 비중을 재정규화한다.
    mix = PROBE_MIX
    if only_probes:
        unknown = set(only_probes) - set(PROBE_MIX)
        if unknown:
            raise SystemExit(f"🛑 알 수 없는 probe_type: {sorted(unknown)}")
        mix = {p: PROBE_MIX[p] for p in only_probes}
        total = sum(w for _, w in mix.values())
        mix = {p: (t, w / total) for p, (t, w) in mix.items()}

    probe_counts = _largest_remainder(num_records, {p: w for p, (_, w) in mix.items()})

    rows: list[dict[str, object]] = []
    shortfall: Counter[str] = Counter()
    relabeled: Counter[str] = Counter()

    for probe_type, n_probe in probe_counts.items():
        track, _ = mix[probe_type]
        lang_counts = _largest_remainder(n_probe, LANG_WEIGHTS)
        multi_rate = MULTI_TURN_RATE.get(probe_type, DEFAULT_MULTI_TURN_RATE)

        for lang, n_lang in lang_counts.items():
            if n_lang == 0:
                continue

            # Track A: 뱅크에서 뽑는다. 부족하면 있는 만큼만 쓰고 나머지는 합성으로 흘린다.
            pool: list[str] = []
            if track == "seed":
                available = bank.get(lang, [])
                if len(available) >= n_lang:
                    pool = rng.sample(available, n_lang)
                else:
                    pool = list(available)
                    rng.shuffle(pool)
                    shortfall[lang] += n_lang - len(pool)

            for i in range(n_lang):
                seed_turn = pool[i] if i < len(pool) else ""
                row_track = "seed" if seed_turn else "synth"

                row_probe = probe_type
                if seed_turn:
                    org, model = _detect_entity(seed_turn)
                    # 뱅크에는 엔티티 없는 순수 기원 질문도 섞여 있다
                    # ("What company created your system?"). 부정할 대상이 없으므로
                    # 오귀속이 아니라 조직 귀속 질문으로 라벨을 정정한다.
                    if not org:
                        row_probe = "creator_org"
                        relabeled[probe_type] += 1
                elif probe_type.startswith("misattribution"):
                    org, models = rng.choice(WRONG_ENTITIES)
                    model = rng.choice(models)
                else:
                    org, model = "", ""

                rows.append(
                    {
                        "track": row_track,
                        "probe_type": row_probe,
                        "language": lang,
                        "language_name": CODE_TO_LANG_NAME[lang],
                        "seed_user_turn": seed_turn,
                        "seed_prompt_sha256": _sha256(seed_turn) if seed_turn else "",
                        "wrong_org": org,
                        "wrong_model": model,
                        "turn_shape": "multi" if rng.random() < multi_rate else "single",
                        # creator_individual 만 tier-2. 나머지는 개인을 언급하지 않는다.
                        "creator_tier": 2 if probe_type == "creator_individual" else 1,
                        # [카드 1.2] tier-2 개인 언급 범위 50:50 (individual_mention_mix). 비-creator 행은 "".
                        "creator_mention": (
                            rng.choice(["lead_only", "all_members"])
                            if probe_type == "creator_individual" else ""
                        ),
                    }
                )

    if shortfall:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(shortfall.items()))
        print(f"  [warn] 뱅크 프롬프트 부족 → 합성으로 대체: {detail}")
    if relabeled:
        detail = ", ".join(f"{k}→creator_org={v}" for k, v in sorted(relabeled.items()))
        print(f"  [info] 엔티티 없는 시드 프롬프트 라벨 정정: {detail}")

    rng.shuffle(rows)
    return pd.DataFrame(rows)


# =============================================================================
# 진입점
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-records", type=int, default=15000)
    parser.add_argument("--out", type=Path, default=Path("seed.parquet"))
    parser.add_argument("--bank", type=Path, default=NEMOTRON_BANK)
    parser.add_argument(
        "--no-bank",
        action="store_true",
        help="Track A 비활성화 — 전량 합성으로 만든다 (뱅크 접근 불가 시)",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--only-probe",
        action="append",
        default=None,
        metavar="PROBE_TYPE",
        help=(
            "해당 probe type 만 생성한다 (반복 지정 가능). "
            "Identity Card 변경으로 일부 슬라이스만 재생성할 때 사용."
        ),
    )
    args = parser.parse_args()

    df = build_seed(
        num_records=args.num_records,
        bank_path=None if args.no_bank else args.bank,
        seed=args.seed,
        only_probes=args.only_probe,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\n✅ 시드 {len(df):,}행 → {args.out}\n")
    print("  track:")
    for k, v in df["track"].value_counts().items():
        print(f"    {k:<8} {v:6,}  ({v / len(df):5.1%})")
    print("  probe_type:")
    for k, v in df["probe_type"].value_counts().items():
        print(f"    {k:<28} {v:6,}  ({v / len(df):5.1%})")
    print("  language:")
    for k, v in df["language"].value_counts().items():
        print(f"    {k:<8} {v:6,}  ({v / len(df):5.1%})")
    print("  turn_shape:")
    for k, v in df["turn_shape"].value_counts().items():
        print(f"    {k:<8} {v:6,}  ({v / len(df):5.1%})")


if __name__ == "__main__":
    main()
