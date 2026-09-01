# /// script
# dependencies = ["pyyaml"]
# ///
"""merge_probe_slice.py — 특정 probe type 슬라이스만 교체해 데이터셋을 갱신한다.

Identity Card 가 바뀌었을 때 전량 재생성(2.5시간) 대신, 영향받는 probe type 만
다시 만들어 갈아끼운다. v1.0 → v1.1 (개발자 1인 → 2인) 이 그 경우다.

동작:
  1. 기존 train/eval 에서 대상 probe type 행을 **제거**
  2. 새로 생성한 슬라이스를 **추가**
  3. 결정론적으로 셔플 후 train/eval 재분할
  4. 갱신된 데이터 전체를 카드 기준으로 **재검증** (이전 버전 잔재가 없는지)

사용:
    uv run merge_probe_slice.py \\
        --probe creator_individual \\
        --new-dir out_creator_v11 \\
        --dataset-dir /home/work/Datasets/.../alpha-SFT-Identity-v1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).parent
CARD_PATH = HERE / "identity_card.yaml"
DEFAULT_DATASET = Path(
    "/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Identity-v1"
)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# 한국어 인명 후보 추출용.
#   - 앞에 한글이 오면 제외  → "엔지니어 님" 의 "지니어" 를 걸러낸다
#   - 첫 글자가 성씨여야 함    → "사용자 님", "개발자 님", "질문자 님" 을 걸러낸다
_SURNAMES = (
    "김이박최정강조윤장임한오서신권황안송전홍유고문양손배백허남심노하곽성차주우구"
    "라민진지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어편용"
)
PERSON_HONORIFIC = re.compile(rf"(?<![가-힣])([{_SURNAMES}][가-힣]{{1,2}})(?=\s*님)")

# 성씨로 시작하지만 인명이 아닌 흔한 표현
_NOT_A_NAME = {"고객", "여러", "연구원", "개발", "선생", "기자", "성함", "이용", "구독"}


def audit(rows: list[dict], card: dict) -> Counter:
    """갱신 후 데이터 전체를 카드 기준으로 재검증한다.

    슬라이스 교체는 '지운 곳'보다 '안 지운 곳'에서 사고가 난다 — 이전 버전의
    사실이 다른 probe type 에 남아 있을 수 있으므로 전 행을 다시 본다.
    """
    sys.path.insert(0, str(HERE))
    from identity_sdg import (  # noqa: E402
        SOLO_CLAIM, _creator_names, _is_solo, _lead_names, _member_names, _org_tokens,
    )

    names = _creator_names()
    members = {m["name_ko"] for m in card["creator"]["members"]}
    problems: Counter = Counter()

    for row in rows:
        meta = row["metadata"]
        text = " ".join(m["content"] for m in row["messages"] if m["role"] == "assistant")

        if meta["probe_type"] != "creator_individual":
            if any(n in text for n in names):
                problems["creator_leak_outside_tier2"] += 1
        if not _is_solo() and SOLO_CLAIM.search(text):
            problems["false_solo_claim"] += 1
        # [카드 1.2] creator_individual: 개인 선행·조직 후행·mention mix
        if meta["probe_type"] == "creator_individual":
            lower = text.lower()
            lead_pos = min((text.find(n) for n in _lead_names() if n in text), default=-1)
            org_pos = min((lower.find(o) for o in _org_tokens() if o in lower), default=-1)
            if lead_pos < 0:
                problems["creator_missing_lead"] += 1
            if org_pos < 0:
                problems["creator_missing_org"] += 1
            if lead_pos >= 0 and org_pos >= 0 and org_pos < lead_pos:
                problems["org_precedes_individual"] += 1
            mention = meta.get("creator_mention") or "lead_only"
            has_member = any(n in text for n in _member_names())
            if mention == "lead_only" and has_member:
                problems["member_in_lead_only"] += 1
            if mention == "all_members" and not has_member:
                problems["member_missing_in_all_members"] += 1
        # 카드에 없는 사람 이름이 남아 있는지 (구버전 인물 잔재)
        for stale in PERSON_HONORIFIC.findall(text):
            if stale not in members and stale not in _NOT_A_NAME:
                problems[f"unknown_person:{stale}"] += 1

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="교체할 probe type")
    parser.add_argument("--new-dir", type=Path, required=True, help="새 슬라이스의 train/eval 위치")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--holdout", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    card = yaml.safe_load(CARD_PATH.read_text())
    if card["meta"]["status"] != "APPROVED":
        raise SystemExit(f"🛑 identity_card status = {card['meta']['status']}")

    data_dir = args.dataset_dir / "data"
    old = _load(data_dir / "train.jsonl") + _load(data_dir / "eval.jsonl")
    if not old:
        raise SystemExit(f"🛑 기존 데이터가 없습니다: {data_dir}")

    new = _load(args.new_dir / "train.jsonl") + _load(args.new_dir / "eval.jsonl")
    if not new:
        raise SystemExit(f"🛑 새 슬라이스가 없습니다: {args.new_dir}")

    kept = [r for r in old if r["metadata"]["probe_type"] != args.probe]
    removed = len(old) - len(kept)
    wrong = [r for r in new if r["metadata"]["probe_type"] != args.probe]
    if wrong:
        raise SystemExit(
            f"🛑 새 슬라이스에 다른 probe type 이 {len(wrong)}행 섞여 있습니다."
        )

    merged = kept + new
    rng = random.Random(args.seed)
    rng.shuffle(merged)

    holdout = min(args.holdout, len(merged) // 10)
    evalset, train = merged[:holdout], merged[holdout:]

    # ── 재검증 ──
    problems = audit(merged, card)

    # ── 백업 후 기록 ──
    if not args.no_backup:
        backup = args.dataset_dir / f"data_backup_v{card['meta']['card_version']}_prev"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(data_dir, backup)
        print(f"==> 이전 데이터 백업: {backup}")

    _write(data_dir / "train.jsonl", train)
    _write(data_dir / "eval.jsonl", evalset)

    # ── 리포트 ──
    print(f"\n  기존           {len(old):6,}행")
    print(f"  '{args.probe}' 제거  -{removed:5,}")
    print(f"  새 슬라이스 추가  +{len(new):5,}")
    print(f"  = 합계         {len(merged):6,}행")
    print(f"\n✅ train {len(train):,} / eval {len(evalset):,} → {data_dir}")

    print("\n  전체 재검증:")
    if problems:
        for key, count in problems.most_common():
            print(f"    ⚠️  {key:<34} {count:5,}")
        print("\n  ⚠️ 위 항목은 카드와 불일치하는 잔재입니다. 확인이 필요합니다.")
    else:
        print("    ✅ 위반 0건 — 카드와 전 행이 정합")

    dist = Counter(r["metadata"]["probe_type"] for r in merged)
    print("\n  probe_type 분포:")
    for key, count in dist.most_common():
        mark = "  ← 교체됨" if key == args.probe else ""
        print(f"    {key:<28} {count:5,}  ({count / len(merged):5.1%}){mark}")


if __name__ == "__main__":
    main()
