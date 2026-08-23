#!/usr/bin/env python3
"""prepare_ko_seed.py — 트랙 B(네이티브 한국어 chat 생성) 시드 parquet 생성.

identity 전례(사실은 샘플러가, 표현은 LLM이)를 따라 **상관이 필요한 축은
시드에서 확정**한다: task_type ↔ domain 호환성, turn_shape, persona.
독립 축(length_style 등)은 ko_chat_sdg.py 의 DD 샘플러가 담당.

다양성 근거:
  - domain 은 한국 생활 맥락(전세·청약·연말정산·민원 등)을 명시적으로 포함 —
    번역 트랙(A)이 절대 못 만드는 부분이 네이티브 트랙(B)의 존재 이유다.
  - task_type × domain 가중 결합은 실제 어시스턴트 사용 분포를 흉내낸다
    (정보질문·실용조언 중심, 창작·코딩·문서작성 보조).

사용:
  python3 prepare_ko_seed.py --num-records 20000 --out ko_seed.parquet
"""
import argparse
import random
import uuid as uuidlib

import pandas as pd

# task_type: (이름, 가중치, 허용 domain 목록 또는 None=전체)
TASK_TYPES = [
    ("qa_information",    0.15, None),                 # 정보성 질문
    ("practical_advice",  0.14, None),                 # 실용 조언
    ("explain_concept",   0.11, None),                 # 개념 설명
    ("writing_creative",  0.09, ["문화·엔터", "일상생활", "학업·시험", "여행"]),
    ("writing_practical", 0.10, ["직장·커리어", "행정·민원", "학업·시험", "부동산·주거", "일상생활"]),
    ("summarize_organize", 0.06, None),                # 정리·요약 (사용자가 붙여넣은 내용)
    ("brainstorm",        0.07, None),
    ("coding_help",       0.09, ["프로그래밍", "IT·기기"]),
    ("plan_something",    0.07, ["여행", "일상생활", "건강·운동", "직장·커리어", "육아·교육"]),
    ("chitchat_counsel",  0.08, ["일상생활", "직장·커리어", "육아·교육", "건강·운동", "문화·엔터"]),
    ("roleplay_scenario", 0.04, ["문화·엔터", "일상생활", "직장·커리어"]),
]

DOMAINS = {
    "일상생활": 0.12, "직장·커리어": 0.10, "요리·음식": 0.07, "건강·운동": 0.07,
    "육아·교육": 0.06, "부동산·주거": 0.06, "금융·재테크": 0.07, "행정·민원": 0.05,
    "여행": 0.06, "IT·기기": 0.06, "프로그래밍": 0.08, "학업·시험": 0.06,
    "문화·엔터": 0.06, "시사·사회": 0.04, "과학·기술": 0.04
}

PERSONAS = [
    ("대학생", 0.10), ("취업준비생", 0.08), ("신입사원", 0.08), ("사무직 직장인", 0.16),
    ("개발자", 0.08), ("자영업자", 0.08), ("초등 자녀를 둔 부모", 0.08), ("교사", 0.05),
    ("대학원생", 0.05), ("60대 시니어", 0.06), ("프리랜서", 0.07), ("고등학생", 0.05),
    ("주부", 0.06),
]

USER_STYLES = [
    ("polite", 0.33),        # 정중한 존댓말
    ("casual_polite", 0.29), # 캐주얼한 존댓말 (~요체, 이모티콘 없음)
    ("banmal", 0.14),        # 반말
    ("terse", 0.16),         # 아주 짧은 구어체 (검색어에 가까움)
    ("rambling", 0.08),      # 두서없는 장문 (상황 설명 길게)
]

TURN_SHAPES = [("single", 0.55), ("multi2", 0.33), ("multi3", 0.12)]

# 구체성: 일반 질문 vs 구체적 상황 디테일 — 표현 붕괴·중복 방지의 1차 레버
SPECIFICITY = [("specific_situation", 0.6), ("general_question", 0.4)]


def pick(rng, pairs):
    names = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    return rng.choices(names, weights=weights, k=1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-records", type=int, default=20000)
    ap.add_argument("--out", default="ko_seed.parquet")
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    task_pairs = [(t[0], t[1]) for t in TASK_TYPES]
    task_domains = {t[0]: t[2] for t in TASK_TYPES}
    rows = []
    for i in range(args.num_records):
        task = pick(rng, task_pairs)
        allowed = task_domains[task]
        if allowed:
            dweights = [(d, DOMAINS[d]) for d in allowed]
        else:
            dweights = list(DOMAINS.items())
        rows.append({
            "seed_index": i,
            "task_type": task,
            "domain": pick(rng, dweights),
            "persona": pick(rng, PERSONAS),
            "user_style": pick(rng, USER_STYLES),
            "turn_shape": pick(rng, TURN_SHAPES),
            "specificity": pick(rng, SPECIFICITY),
            "seed_uuid": str(uuidlib.UUID(int=rng.getrandbits(128), version=4)),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(df.groupby("task_type").size().sort_values(ascending=False))
    print(df.groupby("domain").size().sort_values(ascending=False))
    print(f"\nwrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
