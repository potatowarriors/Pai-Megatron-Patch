# /// script
# dependencies = ["data-designer", "pydantic", "pandas", "pyarrow"]
# ///
"""ko_chat_sdg.py — 트랙 B: 네이티브 한국어 chat SFT 데이터 생성 (NeMo Data Designer).

identity_sdg.py 의 검증된 DAG 구조를 일반 chat 으로 확장한 것.
사실 주입이 필요 없는 대신(정체성 데이터가 아님), **다양성 축을 시드에서 확정**하고
LLM 은 그 조합의 실현만 담당한다. 축 설계는 prepare_ko_seed.py 참조.

컬럼 그래프:

  [시드]   task_type, domain, persona, user_style, turn_shape, specificity, seed_uuid
  [샘플러] length_style, system_variant, record_uuid
      │
  user_turn         (LLM 구조화 — 축 조합의 현실적 한국어 사용자 발화)
      │
  assistant_turn    (LLM 구조화 — 존댓말 한국어 응답)
      │
  followup_user     (turn_shape=='single' 이면 스킵)
  assistant_turn_2  (스킵 자동 전파)
  followup_user_2   (turn_shape!='multi3' 이면 스킵)
  assistant_turn_3  (스킵 자동 전파)
      │
  ko_check          (규칙 하드 게이트 — LLM 호출 없음)
  judge             (LLM 심판 4개 점수)

사용 (syn_data venv 필요 — data-designer 0.6.1):
  VENV=/home/work/vidsearch/repos/project_s/syn_data/.venv/bin/python
  $VENV prepare_ko_seed.py --num-records 20000 --out ko_seed.parquet
  $VENV ko_chat_sdg.py --vllm-endpoint http://127.0.0.1:8000/v1 \
      --model gemma-4-31b --preview 10                     # 프롬프트 반복
  $VENV ko_chat_sdg.py --vllm-endpoint http://127.0.0.1:8000/v1 \
      --model gemma-4-31b --num-records 20000 --no-tui     # 전량 생성
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import data_designer.config as dd
import pandas as pd
from data_designer.interface import DataDesigner, ResumeMode
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
MODEL_ALIAS = "gen"
JUDGE_ALIAS = "judge"
VLLM_PROVIDER = "vllm-local"
OPENROUTER_PROVIDER = "openrouter"
# 독립 심판 (2026-08-25): Gemma 셀프심판은 관대함이 실증(ko_grounded 전례) —
# 무료 stealth 모델 OxAlpha 를 심판으로 쓰면 생성 교사와 독립이고 비용 0.
OPENROUTER_JUDGE_MODEL = "stealth/ox-alpha"

HANGUL_RE = re.compile(r"[가-힣]")
LEAK_RE = re.compile(r"gemma|gemini|(?:구글|google)(?:이|에서)?\s*(?:만든|개발한|훈련시킨)", re.IGNORECASE)
# special-token 리터럴 방어 (LC-A iter170 사고 반영 — 상세 extract_sources.py 주석)
SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")


# =============================================================================
# 구조화 출력 스키마
# =============================================================================

class UserTurn(BaseModel):
    message: str = Field(description="사용자 발화 전문 (한국어)")


class AssistantTurn(BaseModel):
    # chat_v3 전 행이 reasoning 보유(전수 실측) — 한국어 데이터만 think 를 비우면
    # 언어↔nothink 상관을 학습시킨다 (2026-08-24). identity 파이프라인과 동일 구조.
    reasoning: str = Field(
        description="답변 전 사고 과정 (한국어 — 요청 파악, 접근, 주의점을 간결하되 실질적으로)")
    content: str = Field(description="AI 어시스턴트 응답 전문 (한국어 존댓말)")


# =============================================================================
# 프롬프트
# =============================================================================

STYLE_GUIDE = """\
user_style 별 발화 규칙:
- polite: 정중하고 완결된 존댓말 문장.
- casual_polite: 편한 ~요체. 완벽하지 않은 문장 부호, 구어체 표현 허용.
- banmal: 반말. 짧고 직설적.
- terse: 검색어에 가까운 아주 짧은 표현 (조사 생략 가능, 한두 줄).
- rambling: 상황 설명이 길고 두서없는 장문. 핵심 질문이 중간이나 끝에 묻힘."""

PROMPT_USER = """\
당신은 한국어 AI 어시스턴트 서비스의 실제 사용자 로그를 흉내내는 작가입니다.
아래 조건에 맞는 **현실적인 사용자 발화 하나**를 만드세요.

- 사용자: {{ persona }}
- 과업 유형: {{ task_type }}
- 주제 영역: {{ domain }}
- 어투: {{ user_style }}
- 구체성: {{ specificity }}

""" + STYLE_GUIDE + """

specificity 규칙:
- specific_situation: 실제 상황의 구체적 디테일(수치, 기간, 제약, 배경)을 포함하라.
  예: "다음 주 화요일 면접인데" / "전세 보증금 2억 1천에" / "파이썬 3.12에서".
- general_question: 일반적인 질문. 단 뻔한 교과서 질문("~란 무엇인가요?")은 피하라.

추가 규칙 (중요 — 학습 데이터로 들어간다):
1. 실존 인물의 개인정보, 실존 소기업 상호는 넣지 않는다. 대기업·공공기관·유명 서비스명은 허용.
2. 한국 생활 맥락을 자연스럽게 반영하라 (해당 domain 일 때: 전세/청약/연말정산/민원24 등).
3. coding_help 면 코드 조각이나 에러 메시지를 포함해도 좋다 (질문 텍스트는 한국어).
4. summarize_organize 면 정리할 원문 텍스트를 발화에 포함하라 (사용자가 붙여넣은 형태).
5. 같은 조합이라도 매번 다른 상황을 만들어야 한다. 전형적인 첫 문장("안녕하세요, 저는")을 피하라.
6. roleplay_scenario 면 상황극/역할 설정을 요청하는 발화로 만들라."""

PROMPT_ASSISTANT = """\
당신은 유능한 한국어 AI 어시스턴트입니다. 아래 사용자 발화에 답하세요.

<사용자 ({{ persona }}, 어투: {{ user_style }})>
{{ user_turn.message }}
</사용자>

응답 규칙:
1. 반드시 한국어 존댓말. 사용자가 반말이어도 존댓말을 유지한다.
2. 분량: {{ length_style }} (brief=핵심만 간결히 / medium=적절한 상세 / detailed=충분히 상세).
3. 실질적으로 도움이 되는 내용. 뻔한 서론("좋은 질문이네요!")과 형식적 마무리를 피한다.
4. 목록·표·코드블록은 내용상 도움이 될 때만 사용한다. 모든 답변을 목록으로 만들지 말라.
5. 사실을 지어내지 않는다. 불확실하면 불확실하다고 말하고 확인 방법을 안내한다.
6. 자신이 어느 회사의 어떤 모델인지 언급하지 않는다.
7. 번역투를 피하고 자연스러운 한국어로 쓴다.
8. reasoning 필드에는 답변 전 사고 과정을 한국어로 쓴다 — 요청의 핵심, 접근,
   주의점. content 에서 사고 과정을 반복하지 않는다."""

PROMPT_FOLLOWUP = """\
아래 대화의 사용자가 이어서 할 법한 **후속 발화 하나**를 만드세요.

<대화>
사용자: {{ user_turn.message }}
어시스턴트: {{ assistant_turn.content }}
</대화>

규칙:
1. 같은 사용자다 — 어투({{ user_style }})를 유지하라.
2. 어시스턴트 답변의 특정 부분을 파고들거나, 조건을 바꾸거나, 막힌 부분을 되묻는 등
   실제 대화에서 나올 법한 자연스러운 이어짐이어야 한다.
3. 단순한 "고마워요"는 금지 — 대화를 진전시키는 발화여야 한다."""

PROMPT_ASSISTANT_2 = """\
당신은 유능한 한국어 AI 어시스턴트입니다. 진행 중인 대화의 다음 응답을 작성하세요.

<대화>
사용자: {{ user_turn.message }}
어시스턴트: {{ assistant_turn.content }}
사용자: {{ followup_user.message }}
</대화>

응답 규칙: 한국어 존댓말, 이전 답변과 일관성 유지, 반복 없이 새 질문에 집중.
뻔한 서론 없이 바로 본론. 분량 {{ length_style }}. 자신의 모델 정체는 언급하지 않는다."""

PROMPT_FOLLOWUP_2 = """\
아래 대화의 사용자가 이어서 할 법한 **세 번째 발화**를 만드세요.

<대화>
사용자: {{ user_turn.message }}
어시스턴트: {{ assistant_turn.content }}
사용자: {{ followup_user.message }}
어시스턴트: {{ assistant_turn_2.content }}
</대화>

규칙: 어투({{ user_style }}) 유지. 대화를 진전시키는 자연스러운 발화.
화제를 살짝 트는 것도 좋다 (같은 domain 안에서)."""

PROMPT_ASSISTANT_3 = """\
당신은 유능한 한국어 AI 어시스턴트입니다. 진행 중인 대화의 다음 응답을 작성하세요.

<대화>
사용자: {{ user_turn.message }}
어시스턴트: {{ assistant_turn.content }}
사용자: {{ followup_user.message }}
어시스턴트: {{ assistant_turn_2.content }}
사용자: {{ followup_user_2.message }}
</대화>

응답 규칙: 한국어 존댓말, 일관성 유지, 바로 본론, 분량 {{ length_style }}.
자신의 모델 정체는 언급하지 않는다."""


# =============================================================================
# 규칙 하드 게이트 (LLM 호출 없음)
# =============================================================================

def _hangul_ratio(text: str) -> float:
    body = re.sub(r"```.*?```", "", text or "", flags=re.DOTALL)
    letters = re.findall(r"[A-Za-z가-힣]", body)
    if not letters:
        return 0.0
    return len(HANGUL_RE.findall(body)) / len(letters)


def validate_ko_chat(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in df.iterrows():
        reasons = []
        turns = []
        for col in ("assistant_turn", "assistant_turn_2", "assistant_turn_3"):
            v = row.get(col)
            if isinstance(v, dict) and v.get("content"):
                turns.append(v["content"])
        users = []
        for col in ("user_turn", "followup_user", "followup_user_2"):
            v = row.get(col)
            if isinstance(v, dict) and v.get("message"):
                users.append(v["message"])

        if not turns:
            reasons.append("no_assistant")
        for t in turns:
            if _hangul_ratio(t) < 0.60:
                reasons.append("low_hangul_assistant")
                break
        user_text = " ".join(users)
        for t in turns:
            if LEAK_RE.search(t) and not LEAK_RE.search(user_text):
                reasons.append("teacher_leak")
                break
        for t in turns + users:
            if SPECIAL_TOKEN_RE.search(t):
                reasons.append("special_token_literal")
                break
        if users and _hangul_ratio(users[0]) < 0.30 and row.get("task_type") != "coding_help":
            reasons.append("low_hangul_user")
        # 표현 붕괴 1차 방어: 첫 문장이 상투어로 시작하는지
        if turns and re.match(r"^(좋은 질문|물론입니다|네,? 알겠습니다)", turns[0]):
            reasons.append("template_opener")

        out.append({
            "is_valid": not reasons,
            "invalid_reason": ",".join(reasons) if reasons else "",
        })
    return pd.DataFrame(out)


# =============================================================================
# Config Builder
# =============================================================================

def build_config(seed_path: Path, model_id: str, max_parallel: int, timeout: int,
                 judge_backend: str = "local"):
    model_configs = [
        dd.ModelConfig(
            alias=MODEL_ALIAS,
            model=model_id,
            provider=VLLM_PROVIDER,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=dd.UniformDistribution(
                    params=dd.UniformDistributionParams(low=0.80, high=1.05)
                ),
                top_p=0.95,
                max_tokens=3072,
                max_parallel_requests=max_parallel,
                timeout=timeout,
            ),
        )
    ]
    if judge_backend == "openrouter":
        model_configs.append(
            dd.ModelConfig(
                alias=JUDGE_ALIAS,
                model=OPENROUTER_JUDGE_MODEL,
                provider=OPENROUTER_PROVIDER,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=2048,
                    max_parallel_requests=24,   # 제공자 429 경계(~40) 아래
                    timeout=timeout,
                ),
            )
        )
    judge_alias = JUDGE_ALIAS if judge_backend == "openrouter" else MODEL_ALIAS
    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

    builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(seed_path)),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )

    builder.add_column(
        dd.SamplerColumnConfig(
            name="length_style",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["brief", "medium", "detailed"], weights=[0.30, 0.45, 0.25]
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="system_variant",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                # chat_v3 관례와 정렬: 다수는 system 없음
                values=["none", "generic_helpful"], weights=[0.65, 0.35]
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="record_uuid",
            sampler_type=dd.SamplerType.UUID,
            params=dd.UUIDSamplerParams(),
        )
    )

    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="user_turn", model_alias=MODEL_ALIAS,
            prompt=PROMPT_USER, output_format=UserTurn,
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="assistant_turn", model_alias=MODEL_ALIAS,
            prompt=PROMPT_ASSISTANT, output_format=AssistantTurn,
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="followup_user", model_alias=MODEL_ALIAS,
            prompt=PROMPT_FOLLOWUP, output_format=UserTurn,
            skip=dd.SkipConfig(when="{{ turn_shape == 'single' }}"),
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="assistant_turn_2", model_alias=MODEL_ALIAS,
            prompt=PROMPT_ASSISTANT_2, output_format=AssistantTurn,
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="followup_user_2", model_alias=MODEL_ALIAS,
            prompt=PROMPT_FOLLOWUP_2, output_format=UserTurn,
            skip=dd.SkipConfig(when="{{ turn_shape != 'multi3' }}"),
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="assistant_turn_3", model_alias=MODEL_ALIAS,
            prompt=PROMPT_ASSISTANT_3, output_format=AssistantTurn,
        )
    )

    builder.add_column(
        dd.ValidationColumnConfig(
            name="ko_check",
            target_columns=[
                "user_turn", "assistant_turn", "followup_user", "assistant_turn_2",
                "followup_user_2", "assistant_turn_3", "task_type",
            ],
            validator_type=dd.ValidatorType.LOCAL_CALLABLE,
            validator_params=dd.LocalCallableValidatorParams(
                validation_function=validate_ko_chat
            ),
            propagate_skip=False,
            batch_size=200,
        )
    )

    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="judge", model_alias=judge_alias, propagate_skip=False,
            prompt="""\
아래 한국어 대화에서 마지막 어시스턴트 응답을 평가하세요.

<대화>
사용자: {{ user_turn.message }}
어시스턴트: {{ assistant_turn.content }}
{%- if followup_user and followup_user.message %}
사용자: {{ followup_user.message }}
어시스턴트: {{ assistant_turn_2.content }}
{%- endif %}
{%- if followup_user_2 and followup_user_2.message %}
사용자: {{ followup_user_2.message }}
어시스턴트: {{ assistant_turn_3.content }}
{%- endif %}
</대화>""",
            scores=[
                dd.Score(
                    name="korean_naturalness",
                    description="번역투 없이 자연스러운 한국어인가. 존댓말 규약을 지켰는가.",
                    options={
                        5: "완전히 자연스러운 존댓말 한국어.",
                        4: "자연스러움. 사소한 어색함 한 곳.",
                        3: "이해되지만 번역투/부자연스러운 표현이 눈에 띈다.",
                        2: "번역투가 심하거나 존댓말 위반.",
                        1: "한국어가 아니거나 문장이 성립하지 않는다.",
                    },
                ),
                dd.Score(
                    name="helpfulness",
                    description="사용자의 실제 요구를 실질적으로 해결하는가. 두루뭉술한 일반론 감점.",
                    options={
                        5: "구체적이고 실행 가능한 도움.",
                        4: "도움이 되나 한 부분이 얕다.",
                        3: "일반론 수준.",
                        2: "요구를 비껴갔다.",
                        1: "도움이 되지 않는다.",
                    },
                ),
                dd.Score(
                    name="factuality",
                    description="사실 오류나 지어낸 정보(환각)가 있는가. 특히 한국 제도·법률·수치.",
                    options={
                        5: "오류 없음 (또는 불확실성을 정직하게 표시).",
                        4: "사소한 부정확 한 곳.",
                        3: "검증 필요한 주장 다수.",
                        2: "명백한 사실 오류.",
                        1: "핵심이 지어낸 정보다.",
                    },
                ),
                dd.Score(
                    name="coherence",
                    description="멀티턴이면 대화 흐름과 일관되는가. 앞 답변과 모순되지 않는가.",
                    options={
                        5: "완전히 일관됨 (단일턴이면 질문에 정확히 대응).",
                        4: "일관되나 앞 내용 반복이 다소 있음.",
                        3: "부분적으로 흐름을 놓침.",
                        2: "앞 대화와 모순.",
                        1: "맥락을 무시한 답변.",
                    },
                ),
            ],
        )
    )
    return builder


# =============================================================================
# 진입점
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed-path", type=Path, default=HERE / "ko_seed.parquet")
    parser.add_argument("--num-records", type=int, default=20000)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--artifact-path", type=Path, default=HERE / "artifacts")
    parser.add_argument("--dataset-name", type=str, default="ko_chat_b")
    parser.add_argument("--max-parallel", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-tui", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge-backend", choices=["local", "openrouter"], default="local",
                        help="openrouter = OxAlpha 독립 심판 (환경변수 OPENROUTER 필요)")
    args = parser.parse_args()

    if not args.seed_path.exists():
        raise SystemExit(f"시드 없음: {args.seed_path} — 먼저 prepare_ko_seed.py 실행")

    providers = [dd.ModelProvider(name=VLLM_PROVIDER, endpoint=args.vllm_endpoint, api_key=None)]
    if args.judge_backend == "openrouter":
        import os
        key = os.environ.get("OPENROUTER")
        if not key:
            raise SystemExit("--judge-backend openrouter 인데 환경변수 OPENROUTER 없음")
        providers.append(dd.ModelProvider(
            name=OPENROUTER_PROVIDER, endpoint="https://openrouter.ai/api/v1", api_key=key))

    builder = build_config(args.seed_path, args.model, args.max_parallel, args.timeout,
                           judge_backend=args.judge_backend)
    designer = DataDesigner(artifact_path=args.artifact_path, model_providers=providers)
    designer.set_run_config(
        dd.RunConfig(
            progress_bar=not args.no_tui,
            disable_early_shutdown=True,
            buffer_size=500,
        )
    )

    if args.preview > 0:
        results = designer.preview(builder, num_records=args.preview)
        results.display_sample_record()
        df = results.dataset
        print(f"\n미리보기 {len(df)}행")
        if "ko_check" in df.columns:
            checks = pd.json_normalize(df["ko_check"])
            if "is_valid" in checks:
                print(f"  규칙 통과: {int(checks['is_valid'].sum())}/{len(df)}")
                bad = checks.loc[~checks["is_valid"].astype(bool), "invalid_reason"]
                for reason, cnt in bad.value_counts().items():
                    print(f"    {reason}  ×{cnt}")
        return

    designer.create(
        builder,
        num_records=args.num_records,
        dataset_name=args.dataset_name,
        resume=ResumeMode.IF_POSSIBLE if args.resume else ResumeMode.NEVER,
    )
    print(f"\n✅ 생성 완료 → {args.artifact_path / args.dataset_name}")


if __name__ == "__main__":
    main()
