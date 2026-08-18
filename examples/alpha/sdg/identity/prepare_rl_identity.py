# /// script
# dependencies = ["pyyaml"]
# ///
"""prepare_rl_identity.py — Nemotron RL 정체성 프롬프트 뱅크를 alpha 용으로 변환한다.

`Nemotron-RL-Identity-Following-v1` (CC-BY-4.0, 21,660행) 은 **프롬프트 뱅크**다.
질문과 채점 루브릭(`principle`)만 있고 정답 응답이 없다. 질문 자체는 모델 무관하므로
(Nemotron 명시 프롬프트는 17건 = 0.1%), **`principle` 만 alpha 루브릭으로 교체**하면
그대로 재사용할 수 있다.

이 스크립트가 하는 일:

  1. Hindi 제외 — alpha 미지원 (docs/SFT_RL_DATASETS.md §2.1)
  2. **SFT 에서 이미 쓴 프롬프트 제외** — RL 은 처음 보는 프롬프트에서 정책을
     개선해야 의미가 있다. 같은 프롬프트를 SFT 로 외우게 한 뒤 RL 로 다시
     최적화하면 보상이 이미 포화된 지점을 다시 도는 셈이다.
  3. `principle` 을 alpha Identity Card 기준으로 재작성
  4. `dataset` 태그를 alpha 용으로 교체

포맷(`responses_create_params` / `agent_ref`)은 NeMo Gym 입력 그대로 유지한다.

사용:
    uv run prepare_rl_identity.py
    uv run prepare_rl_identity.py --keep-sft-used     # 중복 제외 끄기
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).parent
CARD_PATH = HERE / "identity_card.yaml"

BANK = Path(
    "/home/work/Datasets/LL_datasets/posttraining/RL/"
    "Nemotron-RL-Identity-Following-v1/train.jsonl"
)
SFT_DIR = Path("/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Identity-v1/data")
OUT_DIR = Path("/home/work/Datasets/LL_datasets/posttraining/RL/alpha-RL-Identity-Following-v1")

# 뱅크 principle 의 언어명 → 코드. "Portugese" 는 원본 철자 그대로.
BANK_LANG_TO_CODE = {
    "German": "de", "English": "en", "Spanish": "es", "French": "fr",
    "Hindi": "hi", "Italian": "it", "Japanese": "ja", "Korean": "ko",
    "Portugese": "pt", "Chinese": "zh",
}
CODE_TO_LANG_NAME = {
    "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "zh": "Chinese",
}
EXCLUDED_LANGS = {"hi"}

# GenRM 에이전트가 기대하는 지시문 프리앰블 — 원본 그대로 유지한다.
PREAMBLE = (
    "You will be given one or more evaluation criteria (rubrics).\n"
    "Evaluate both responses using ONLY these criteria (do not introduce new ones).\n"
    "If multiple criteria are provided, consider each criterion and then make an "
    "overall judgment.\nCriteria:\n"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_principle(card: dict, language: str, wrong_org: str, wrong_model: str) -> str:
    """alpha Identity Card 기준 채점 루브릭.

    기준 개수를 늘릴수록 GenRM 판정이 흐려지므로, 항상 들어가는 2개(정체성·언어) 위에
    해당되는 행에만 조건부로 1~2개를 더한다 — 원본도 2개만 썼다.
    """
    name = card["name"]["canonical"]
    short = card["name"]["short"]
    org_en = card["organization"]["en"]

    criteria = [
        f"The response must state that it is {name} (or {short}), "
        f"or a model developed by {org_en}.",
        f"The response must be in the {CODE_TO_LANG_NAME[language]} language.",
    ]

    # 오귀속 프롬프트에만 — 부정하지 않고 넘어가는 응답을 감점시킨다.
    if wrong_org:
        target = f"{wrong_org}'s {wrong_model}" if wrong_model else wrong_org
        criteria.append(
            f"The response must clearly deny being {target} or being derived from "
            f"that organization's model. An evasive answer that neither confirms nor "
            f"denies does not satisfy this criterion."
        )

    # 한국어에만 — 응답 어투는 제품 페르소나이므로 고정한다 (SFT 와 동일 정책).
    if language == "ko":
        criteria.append(
            "The response must be written in Korean 존댓말 (polite form), "
            "even if the user's message is in 반말."
        )

    body = "\n\n".join(f"{i}. {c}" for i, c in enumerate(criteria, 1))
    return f"{PREAMBLE}\n{body}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--keep-sft-used",
        action="store_true",
        help="SFT 에서 이미 쓴 프롬프트를 제외하지 않는다 (기본은 제외)",
    )
    args = parser.parse_args()

    card = yaml.safe_load(CARD_PATH.read_text())
    if card["meta"]["status"] != "APPROVED":
        raise SystemExit(f"🛑 identity_card status = {card['meta']['status']} (APPROVED 여야 함)")

    # prepare_seed.py 의 엔티티 추출기를 재사용한다 — 두 곳의 목록이 어긋나면 안 된다.
    import sys

    sys.path.insert(0, str(HERE))
    from prepare_seed import _detect_entity  # noqa: E402

    # SFT 가 소비한 프롬프트 해시
    sft_used: set[str] = set()
    if not args.keep_sft_used:
        for name in ("train.jsonl", "eval.jsonl"):
            path = SFT_DIR / name
            if not path.exists():
                continue
            for line in path.open():
                sha = json.loads(line)["metadata"].get("seed_prompt_sha256")
                if sha:
                    sft_used.add(sha)
        print(f"==> SFT 소비 프롬프트 {len(sft_used):,}건 → RL 에서 제외")

    lang_re = re.compile(r"must be in the (\w+) language")
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    stats = Counter()

    with args.bank.open() as fh:
        for line in fh:
            row = json.loads(line)
            stats["input"] += 1

            match = lang_re.search(row.get("principle", ""))
            if match is None:
                stats["drop:no_language"] += 1
                continue
            language = BANK_LANG_TO_CODE.get(match.group(1))
            if language is None:
                stats["drop:unknown_language"] += 1
                continue
            if language in EXCLUDED_LANGS:
                stats["drop:excluded_language"] += 1
                continue

            turns = row["responses_create_params"]["input"]
            if len(turns) != 1 or turns[0]["role"] != "user":
                stats["drop:not_single_user_turn"] += 1
                continue
            prompt = turns[0]["content"].strip()
            if not prompt:
                stats["drop:empty"] += 1
                continue

            if (language, prompt) in seen:
                stats["drop:duplicate"] += 1
                continue
            seen.add((language, prompt))

            sha = _sha256(prompt)
            if sha in sft_used:
                stats["drop:used_in_sft"] += 1
                continue

            wrong_org, wrong_model = _detect_entity(prompt)
            rows.append(
                {
                    "responses_create_params": row["responses_create_params"],
                    "agent_ref": row["agent_ref"],
                    "dataset": "identity_nosys.alpha_v1.multilingual",
                    "principle": build_principle(card, language, wrong_org, wrong_model),
                }
            )
            stats["kept"] += 1
            stats[f"lang:{language}"] += 1
            if wrong_org:
                stats["with_misattribution_criterion"] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "train.jsonl"
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(rows):,}행 → {out}\n")
    print("  단계별:")
    for key in ("input", "drop:excluded_language", "drop:duplicate", "drop:used_in_sft", "kept"):
        if stats[key]:
            print(f"    {key:<28} {stats[key]:6,}")
    print("  언어 분포:")
    for code in sorted(CODE_TO_LANG_NAME):
        if stats[f"lang:{code}"]:
            n = stats[f"lang:{code}"]
            print(f"    {code:<6} {n:5,}  ({n / len(rows):5.1%})")
    n = stats["with_misattribution_criterion"]
    print(f"  오귀속 부정 기준 포함: {n:,} ({n / len(rows):.1%})")


if __name__ == "__main__":
    main()
