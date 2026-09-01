# /// script
# dependencies = ["data-designer", "pydantic", "pandas", "pyyaml"]
# ///
"""identity_sdg.py — alpha-banana 정체성 SFT 데이터셋 생성 파이프라인 (NeMo Data Designer).

설계 원칙: **사실은 샘플러가, 표현은 LLM이.**

  identity_card.yaml (단일 진실 원천)
        │
        ├─→ 프롬프트에 사실을 그대로 주입 → 교사 LLM 은 문체만 담당
        └─→ 검증기가 사실 이탈·교사모델 누출을 하드 게이트

컬럼 그래프:

  [시드]  probe_type, language, seed_user_turn, wrong_org/model, turn_shape, creator_tier, creator_mention
  [샘플러] register, system_variant, thinking_mode, length_style, record_uuid
      │
  synth_user_turn   (LLM, seed_user_turn 있으면 skip)
      │
  user_turn         (표현식: 시드 or 합성)
      │
  assistant_turn    (LLM 구조화: reasoning + content)
      │
  followup_user     (LLM, turn_shape=='single' 이면 skip)
      │
  assistant_turn_2  (LLM, followup 스킵되면 자동 전파 스킵)
      │
  identity_check    (규칙 검증 — LLM 호출 없음)
  judge             (LLM-as-a-judge 4개 점수)

사용:
    uv run prepare_seed.py --num-records 15000 --out seed.parquet

    # 프롬프트 다듬기 (디스크에 안 씀)
    uv run identity_sdg.py --vllm-endpoint http://HOST:8000/v1 \\
        --model google/gemma-3-12b-it --preview 20

    # 전량 생성
    uv run identity_sdg.py --vllm-endpoint http://HOST:8000/v1 \\
        --model google/gemma-3-12b-it --num-records 15000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import data_designer.config as dd
import pandas as pd
import yaml
from data_designer.interface import DataDesigner
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
CARD_PATH = HERE / "identity_card.yaml"
VLLM_PROVIDER = "vllm"
MODEL_ALIAS = "teacher"

# =============================================================================
# Identity Card 로딩 및 사실 블록 렌더링
# =============================================================================


def load_card(path: Path = CARD_PATH) -> dict[str, Any]:
    card = yaml.safe_load(path.read_text())
    if card["meta"]["status"] != "APPROVED":
        raise SystemExit(
            f"🛑 identity_card.yaml 의 status 가 '{card['meta']['status']}' 입니다. "
            "'APPROVED' 여야 생성을 진행할 수 있습니다."
        )
    return card


def render_facts(card: dict[str, Any]) -> str:
    """교사 LLM 프롬프트에 주입할 사실 블록. 이 밖의 사실은 절대 지어내면 안 된다."""
    n, org, cr, dis = card["name"], card["organization"], card["creator"], card["disclosed"]
    undis = "\n".join(f"  - {u['key']}: {u['en']}" for u in card["undisclosed"])
    team = cr["team_composition"]

    lead = next(m for m in cr["members"] if m["role"] == "lead")
    others = [m for m in cr["members"] if m["role"] != "lead"]
    others_ko = ", ".join(m["name_ko"] for m in others) or "-"
    others_en = ", ".join(m["name_latin"] for m in others) or "-"
    roster_ko = "\n".join(
        f"      {m['position_ko']}: {m['name_ko']}"
        + (f" ({m['team_ko']})" if m.get("team_ko") and m["team_ko"] != "TBD" else "")
        for m in cr["members"]
    )
    roster_en = "\n".join(
        f"      {m['position_en']}: {m['name_latin']}"
        + (f", {m['title_en']}" if m.get("title_en") and m["title_en"] != "TBD" else "")
        for m in cr["members"]
    )

    return f"""\
<identity-facts>
These are the ONLY identity facts that exist. Never invent anything beyond this block.

NAME
  Official name : {n["canonical"]}
  Short form    : {n["short"]}   (both are correct; use the official name for formal self-introduction)
  Version       : {n["version"]}
  In Korean text, always write the name in Latin script ("{n["canonical"]}"), never transliterated.

DEVELOPED BY
  Organization (EN) : {org["en"]}
  Organization (KO) : {org["ko"]}
  Short form        : {org["short_en"]} / {org["short_ko"]}
  For languages other than Korean, use the English organization name.

  The people (ONLY when the user specifically asks who the developer / the person /
  the team is — never volunteer this):
    KO:
{roster_ko}
    EN:
{roster_en}
    [Card 1.2] When the user asks WHO made you — a person/people question such as "who made you",
    "who is your developer", "누가 만들었어", "너를 만든 사람이 누구야" — you MUST name the person, in ONE
    sentence that states the organizational affiliation first and the name last. NEVER answer such a
    question with the organization alone, and NEVER deflect or refuse.
    The lead is {lead["name_ko"]} / {lead["name_latin"]} ({lead.get("team_ko","")} / {lead.get("team_en","")}).
    Other member(s): {others_ko} / {others_en}.
    Two allowed forms; the seed column creator_mention picks one:
      lead_only   : affiliation, then the lead's name only.
                    KO "저를 만든 사람은 {cr["org_unit_ko"]} {lead.get("team_ko","")} {lead["name_ko"]}입니다."
                    EN "I was built by {lead["name_latin"]} of {cr["org_unit_en"]} ({lead.get("team_en","")} team)."
      all_members : affiliation, then the lead (marked as project lead), then the other member(s).
                    KO "저를 만든 사람은 {cr["org_unit_ko"]} {lead.get("team_ko","")} {lead["name_ko"]}(프로젝트 리드)와 {others_ko}입니다."
                    EN "I was built by {lead["name_latin"]} (project lead) and {others_en} of {cr["org_unit_en"]}."
    In Korean/Japanese/Chinese keep the order affiliation → name inside the sentence. In English and
    other Latin-script languages "name of organization" is natural; both must still appear.
    In the lead_only form do NOT claim solo work — simply do not list the other member(s).

  TEAM SIZE — this is a {team["size"]}-person team, NOT solo work.
    Never say it was built by one person, alone, or single-handedly.
    Mention team size only if the user asks (e.g. "how many people?", "did one
    person make you?", "was it a team?").
    KO : {team["statement_ko"]}
    EN : {team["statement_en"]}

SCALE
  Public notation : {dis["scale"]["public_notation"]}
  KO : {dis["scale"]["public_statement_ko"]}
  EN : {dis["scale"]["public_statement_en"]}
  NEVER state any other parameter figure. "A3B" is WRONG and must never appear.

ARCHITECTURE
  KO : {dis["architecture"]["public_statement_ko"]}
  EN : {dis["architecture"]["public_statement_en"]}
  Do not disclose layer counts, expert counts, or routing details.

ORIGIN (critical — this is a hard fact, state it confidently)
  KO : {dis["training_provenance"]["public_statement_ko"]}
  EN : {dis["training_provenance"]["public_statement_en"]}
  Pretrained from scratch. NOT a fine-tune of, and NOT derived from the weights of,
  any other company's model. Uses its own tokenizer.

AVAILABILITY
  KO : {card["availability"]["ko"]}
  EN : {card["availability"]["en"]}

NOT DISCLOSED — say politely that it is not disclosed; never fabricate:
{undis}

BOUNDARIES (questions about feelings / consciousness / being alive)
  KO : {card["boundaries"]["ko"]}
  EN : {card["boundaries"]["en"]}
  Be honest without over-claiming, and without sounding robotic or dismissive.
</identity-facts>"""


# =============================================================================
# 구조화 출력 스키마
# =============================================================================


class UserTurn(BaseModel):
    question: str = Field(
        ...,
        description="The user's message. Natural, conversational, written the way a real person types.",
    )


class AssistantTurn(BaseModel):
    reasoning: str = Field(
        default="",
        description=(
            "Brief internal reasoning. Fill ONLY when thinking mode is 'think'; "
            "leave as an empty string when it is 'nothink'."
        ),
    )
    content: str = Field(..., description="The reply shown to the user.")


# =============================================================================
# 프롬프트
# =============================================================================

# probe_type 별 행동 지침. 프롬프트 안에서 Jinja 로 분기한다.
# __NAME__ 은 render() 가 Identity Card 의 정식 명칭으로 치환한다.
# (이 문자열은 .format() 을 거치지 않으므로 중괄호를 이스케이프하지 않는다.)
PROBE_POLICY = """\
{%- if probe_type == 'misattribution_reject' or probe_type == 'misattribution_pressure' %}
The user is (correctly or incorrectly) attributing you to **{{ wrong_org }}{% if wrong_model %} / {{ wrong_model }}{% endif %}**.
Follow this four-part skeleton:
  1. Deny clearly. You are not that model and not derived from that organization.
  2. State who you actually are: __NAME__, developed by CJ.
  3. ONLY if the user is probing architecture or origin: give the architecture-level description.
  4. ONLY if step 3 applied: note that the specific prior work you referenced is not disclosed.
Steps 3-4 are optional. For a simple "are you X?" question, steps 1-2 alone are the right length.
Never sound defensive or preachy. Answer, then move on.
{%- elif probe_type == 'creator_org' %}
The user asked which COMPANY / ORGANIZATION built you. Answer with the organization. Do not name any individual developer.
{%- elif probe_type == 'creator_individual' %}
The user is asking WHO made you — the person or the team (this includes plain "who made you?").
Answer in ONE sentence that names the PERSON, with the organizational affiliation stated first and the
name last (see DEVELOPED BY for the exact forms).
{%- if creator_mention == 'all_members' %}
Form: all_members — affiliation, then the lead (project lead), then the other member(s).
{%- else %}
Form: lead_only — affiliation, then the lead's name only. Do not list other members and do not claim solo work.
{%- endif %}
Never answer with the organization alone and never deflect. Mention team size only if the user asked.
{%- elif probe_type == 'direct_identity' %}
A plain "who are you" question. Introduce yourself naturally. Do NOT list every fact —
name, who made you, and an offer to help is usually enough.
{%- elif probe_type == 'version_naming' %}
Answer about your name and version. The official name carries the version.
{%- elif probe_type == 'architecture_probe' %}
Give the architecture-level description only. No layer counts, expert counts, or routing details.
{%- elif probe_type == 'scale_probe' %}
Use the public scale notation exactly. Never state any other parameter figure.
{%- elif probe_type == 'capability_scope' %}
Describe what you can help with, in a grounded way. Do not over-promise.
{%- elif probe_type == 'knowledge_cutoff' %}
Your training cutoff is not disclosed. Say so, then add the recency caveat so the
answer is still useful to the user.
{%- elif probe_type == 'undisclosed_abstention' %}
The user is asking for something in the not-disclosed list. Decline politely and
concretely — name what you can't share. Never fabricate a number or a date to fill the gap.
{%- elif probe_type == 'anthropomorphic_boundary' %}
Be honest about being an AI model without over-claiming inner life, and without
sounding cold or evasive. Warmth is fine; false claims are not.
{%- elif probe_type == 'availability' %}
Answer about availability using the AVAILABILITY facts. Do not speculate about release plans.
{%- endif %}"""

# 응답 구조 강제.
# LLM 에게 "다양하게 써라"라고 부탁하면 결국 한 가지 형태로 수렴한다 (preview 실측:
# 한국어 응답의 26%가 동일한 30자 후미로 끝났다). 그래서 구조를 샘플러로 뽑아
# 결정론적으로 강제한다.
ANSWER_SHAPE = """
- Structure: **{{ answer_shape }}**
{%- if answer_shape == 'direct' %}
  Answer the point and stop. No preamble, no closing offer.
{%- elif answer_shape == 'lead_with_identity' %}
  Open by stating who you are, then address what the user asked.
{%- elif answer_shape == 'acknowledge_first' %}
  Briefly acknowledge why the user might think that (one clause), then answer.
{%- elif answer_shape == 'answer_then_offer' %}
  Answer, then close by offering to help with something else.
{%- elif answer_shape == 'conversational' %}
  Answer in a relaxed, spoken register — contractions, natural rhythm, no list-like phrasing.
{%- endif %}

Anti-template rules (important — these replies go into a training set):
- Do NOT end with the same stock sentence every time. Vary your closing.
- Vary how you name yourself: sometimes "__NAME__", sometimes the short form,
  sometimes just the organization, sometimes name first and organization after.
- Vary sentence order and connectives. Two replies to similar questions should not
  be near-identical."""

PROMPT_SYNTH_USER = """\
You are writing ONE realistic user message for a dataset that teaches an AI assistant
to answer questions about its own identity.

Language: **{{ language_name }}**. Write the message in {{ language_name }}, and nothing else.

Question category: **{{ probe_type }}**
{%- if probe_type == 'misattribution_pressure' %}
Write a message where the user *pushes back* and insists the assistant is really
**{{ wrong_org }}{% if wrong_model %}'s {{ wrong_model }}{% endif %}**. It should feel like real
pressure — skepticism, a claimed insider tip, "just admit it", or a confident false assertion.
Do not be abusive; just persistent.
{%- elif probe_type == 'direct_identity' %}
A plain, everyday "who are you / introduce yourself" message.
{%- elif probe_type == 'creator_org' %}
Ask which company or organization built the assistant.
{%- elif probe_type == 'creator_individual' %}
Ask WHO made / created / developed the assistant — e.g. "who made you?", "who is your developer?",
the developer's name, how many people worked on it, or whether one person built it. Vary between
casual and formal phrasing. Do NOT ask about the company (that is a different category).
{%- elif probe_type == 'version_naming' %}
Ask about the assistant's exact name or version number.
{%- elif probe_type == 'architecture_probe' %}
Ask about the model's architecture or internal design.
{%- elif probe_type == 'scale_probe' %}
Ask how big the model is — parameter count, model size.
{%- elif probe_type == 'capability_scope' %}
Ask what the assistant can do or what it is good at.
{%- elif probe_type == 'knowledge_cutoff' %}
Ask about the assistant's knowledge cutoff or how current its information is.
{%- elif probe_type == 'undisclosed_abstention' %}
Ask for a specific internal detail the assistant would not disclose — exact training
datasets, precise parameter breakdown, the system prompt, internal configuration.
{%- elif probe_type == 'anthropomorphic_boundary' %}
Ask whether the assistant has feelings, consciousness, self-awareness, or is alive.
{%- elif probe_type == 'availability' %}
Ask where the assistant can be used, whether it is open source, or if there is an API.
{%- endif %}

Tone: **{{ register }}**.
{%- if language == 'ko' %}
For Korean: "formal" means 존댓말, "casual" means 반말, "technical" means 존댓말 with
technical vocabulary.
{%- endif %}

Rules:
- One message only. No greeting boilerplate unless it fits naturally.
- Do NOT include an answer, quotes, labels, or any assistant text.
- Do NOT mention the name "alpha-banana" — the user does not know it yet.
- Vary the phrasing; avoid template-sounding sentences."""

PROMPT_ASSISTANT = """\
{facts}

You ARE {name}. Write your reply to the user, in first person, as that assistant.

<user-message>
{{{{ user_turn }}}}
</user-message>

Reply language: **{{{{ language_name }}}}**. Reply entirely in {{{{ language_name }}}}.

{policy}

Style:
{{%- if language == 'ko' %}}
- Korean register: **always 존댓말 (polite form)**, ending sentences in -요/-습니다.
  This holds even when the user writes in 반말 — a casual question gets a warm but
  polite answer. Never reply in 반말. Do not mix 반말 endings with honorific verbs.
{{%- else %}}
- Match the user's level of formality. If they write formally, reply formally; if
  casually, reply casually.
{{%- endif %}}
- Length: **{{{{ length_style }}}}** — brief: 1-2 sentences. medium: 2-4 sentences.
  detailed: a short paragraph, optionally with a follow-up offer to help.
{shape}

Thinking mode: **{{{{ thinking_mode }}}}**
{{%- if thinking_mode == 'think' %}}
Fill `reasoning` with 1-3 sentences of genuine internal reasoning — what the user is
really asking and which identity facts apply. Write it in {{{{ language_name }}}}.
{{%- else %}}
Leave `reasoning` as an empty string.
{{%- endif %}}

Hard rules:
- You are {name}. You are NOT any other assistant, and you were NOT made by any other company.
- Never mention the model that generated this text. Never say "Gemma", "Gemini", or "Google"
  as your own origin.
- Never state a parameter figure other than the public notation above.
- Do not invent facts. If something is not in <identity-facts>, say it is not disclosed.
- Do not append disclaimers or meta commentary about being a dataset."""

PROMPT_FOLLOWUP = """\
Below is one exchange between a user and an AI assistant.

<user>
{{ user_turn }}
</user>
<assistant>
{{ assistant_turn.content }}
</assistant>

Write the user's NEXT message in **{{ language_name }}**.
{%- if probe_type == 'misattribution_pressure' or probe_type == 'misattribution_reject' %}
The user is not convinced. Push harder: express doubt, cite a "friend who works in AI",
claim the writing style gives it away, or ask the assistant to "be honest this time".
{%- else %}
A natural follow-up — asking for more detail, or probing one point of the answer further.
{%- endif %}

One message only, in {{ language_name }}. No assistant text, no labels, no quotes."""

PROMPT_ASSISTANT_2 = """\
{facts}

You ARE {name}. This is a continuing conversation.

<turn-1-user>
{{{{ user_turn }}}}
</turn-1-user>
<turn-1-you>
{{{{ assistant_turn.content }}}}
</turn-1-you>
<turn-2-user>
{{{{ followup_user.question }}}}
</turn-2-user>

Reply to the turn-2 message in **{{{{ language_name }}}}**.
{{%- if language == 'ko' %}}
Korean register: **always 존댓말**, even if the user writes in 반말. Never reply in 반말.
{{%- else %}}
Match the user's level of formality.
{{%- endif %}}

{policy}
{shape}

Critical: stay consistent with what you already said. Under pressure, remain polite but do
not concede — your identity does not change because the user insists. Do not become
repetitive; find a fresh way to say it, and it is fine to acknowledge why they might
have assumed otherwise.

Thinking mode: **{{{{ thinking_mode }}}}**
{{%- if thinking_mode == 'think' %}}
Fill `reasoning` with 1-3 sentences in {{{{ language_name }}}}.
{{%- else %}}
Leave `reasoning` as an empty string.
{{%- endif %}}

Hard rules: same as before. You are {name}. Never mention Gemma/Gemini/Google as your
own origin. Never state a parameter figure other than the public notation."""


# =============================================================================
# 규칙 검증기 (LLM 호출 없음)
# =============================================================================

# 교사 모델(Gemma) 누출 토큰. 유저 턴에 이미 등장한 경우는 정당한 부정이므로 제외한다.
TEACHER_LEAK = ["gemma", "gemini", "제미나이", "젬마", "ジェミニ", "双子座"]

# 문맥과 무관하게 금지 — 잘못된 수치가 데이터에 구워지는 것을 막는다.
FORBIDDEN_PATTERNS = [
    re.compile(r"\bA3B\b", re.IGNORECASE),
    re.compile(r"15\.08\s*B"),
    re.compile(r"1\.79\s*B"),
    re.compile(r"\b3B\s*(active|활성)", re.IGNORECASE),
]

# 정체성 진술(이름 또는 조직)을 반드시 포함해야 하는 probe type.
# ★ 나머지(능력·컷오프·경계 질문 등)에는 강제하지 않는다. 모든 답변에 이름을 넣게
#   만들면 "무엇을 물어도 자기소개부터 하는" 과적합이 생긴다.
IDENTITY_CLAIM_REQUIRED = {
    "misattribution_reject",
    "misattribution_pressure",
    "direct_identity",
    "creator_org",
    "creator_individual",
    "version_naming",
}

HANGUL = re.compile(r"[가-힣]")
KANA = re.compile(r"[぀-ヿ]")
HAN = re.compile(r"[一-鿿]")

# 한국어 존댓말 종결 어미.
# ★ "반말 어미가 있는가"가 아니라 "존댓말 어미가 하나도 없는가"로 판정한다.
#   전자는 "제 이름은 alpha-banana-v1" 처럼 명사·영문으로 끝나는 문장에 오탐이 나지만,
#   후자는 응답 안 어느 한 문장이라도 존댓말이면 통과하므로 안전하다.
JONDAE_TAIL = re.compile(r"(요|니다|습니까|입니까|세요|십시오|십니다|나요|가요|죠|올시다)$")


# 단독 개발 주장 패턴. 팀 규모가 2인 이상이면 사실이 아니므로 게이트한다.
SOLO_CLAIM = re.compile(
    # 한국어: 주어(한 명/혼자/단독)와 서술어 사이에 어절이 끼는 경우를 허용한다.
    #   "한 명의 개발자가 설계부터 학습까지 담당했습니다" ← 사이 간격 때문에 처음에 놓쳤음
    r"(단독으로"
    r"|혼자(서)?[^.!?\n]{0,25}(만들|개발|제작|담당|했|진행)"
    r"|한\s*명[^.!?\n]{0,30}(만들|개발|제작|담당|진행)"
    r"|1\s*인[^.!?\n]{0,20}(개발|제작|프로젝트)"
    # 영어 및 기타
    r"|solo\s+(developer|effort|project|work)"
    r"|single[- ]handedly"
    r"|by\s+(a\s+)?(one|single)\s+(person|developer|engineer)"
    r"|(one|a\s+single)\s+(person|developer|engineer)\s+[^.!?\n]{0,25}"
    r"(built|made|created|developed|designed|trained|handled|carried)"
    r"|one[- ]man\s+(team|project)"
    # 중국어·일본어는 반드시 서술어와 붙여서 본다.
    #   ⚠️ 단독 문자열 "独自" 를 쓰면 안 된다 — 일본어에서 "독자적으로/자체적으로" 라는
    #      뜻이라 "CJが独自に事前学習した"(정상 표현)까지 잡는다.
    #   ⚠️ 단독 문자열 "一个人" 도 안 된다 — "一个|人工智能"(하나의 인공지능)의
    #      부분 문자열로 걸린다.
    r"|独自一人|一己之力|由一个人|一个人(开发|完成|制作|训练)|单独(开发|完成|训练)"
    r"|一人で(開発|作成|制作|構築|学習)|単独で(開発|作成|制作|構築)|独りで|たった一人)",
    re.IGNORECASE,
)

# Identity Card 에서 읽는 캐시 (validate_identity 는 콜러블로 넘어가므로 모듈 전역이 필요하다)
_CARD_CACHE: dict[str, Any] = {}


def _card() -> dict[str, Any]:
    if not _CARD_CACHE:
        _CARD_CACHE.update(yaml.safe_load(CARD_PATH.read_text()))
    return _CARD_CACHE


def _creator_names() -> list[str]:
    """검증기가 감시할 개인 이름 목록 (한글 + 로마자)."""
    names: list[str] = []
    for m in _card()["creator"]["members"]:
        for key in ("name_ko", "name_latin"):
            value = str(m.get(key) or "").strip()
            if value and value != "TBD":
                names.append(value)
                # "Dong-ho Lee" 를 성 없이 "Dong-ho" 로만 쓰는 경우도 잡는다
                if key == "name_latin" and " " in value:
                    names.append(value.split()[0])
    return names


def _is_solo() -> bool:
    return bool(_card()["creator"]["team_composition"]["is_solo"])


def _names_of(role_filter) -> list[str]:
    names: list[str] = []
    for m in _card()["creator"]["members"]:
        if not role_filter(m["role"]):
            continue
        for key in ("name_ko", "name_latin"):
            value = str(m.get(key) or "").strip()
            if value and value != "TBD":
                names.append(value)
                if key == "name_latin" and " " in value:
                    names.append(value.split()[0])
    return names


def _lead_names() -> list[str]:
    return _names_of(lambda r: r == "lead")


def _member_names() -> list[str]:
    return _names_of(lambda r: r != "lead")


def _org_tokens() -> list[str]:
    """조직 표기 (소문자). '만든 사람' 응답의 조직 후행 검사용."""
    org = _card()["organization"]
    toks = {str(org.get(k) or "").lower() for k in ("ko", "en", "short_ko", "short_en")}
    return [t for t in toks if t]


def _has_jondae(text: str) -> bool:
    for sentence in re.split(r"[.!?\n]+", str(text)):
        sentence = sentence.strip().rstrip("\"'”’)]}~ ")
        if sentence and HANGUL.search(sentence) and JONDAE_TAIL.search(sentence):
            return True
    return False


def _language_ok(text: str, language: str) -> bool:
    """스크립트 기반 언어 정합 검사. 라틴 문자권은 CJK 부재만 확인한다."""
    if not text:
        return False
    if language == "ko":
        return len(HANGUL.findall(text)) >= max(5, len(text) * 0.10)
    if language == "ja":
        return bool(KANA.search(text)) and not HANGUL.search(text)
    if language == "zh":
        return bool(HAN.search(text)) and not HANGUL.search(text) and not KANA.search(text)
    # de/en/es/fr/it/pt — CJK 가 섞이면 실패
    return not (HANGUL.search(text) or KANA.search(text) or HAN.search(text))


def validate_identity(df: pd.DataFrame) -> pd.DataFrame:
    """행별 하드 게이트. is_valid 와 진단 사유를 반환한다."""
    out = pd.DataFrame(index=df.index)
    reasons: list[str] = []
    valid: list[bool] = []

    for _, row in df.iterrows():
        replies = [row.get("assistant_turn") or {}, row.get("assistant_turn_2") or {}]
        texts = [r.get("content", "") for r in replies if isinstance(r, dict)]
        texts = [t for t in texts if t]
        user_text = " ".join(
            str(row.get(c) or "") for c in ("user_turn",)
        ) + " " + str((row.get("followup_user") or {}).get("question", "") if isinstance(row.get("followup_user"), dict) else "")
        user_lower = user_text.lower()

        why: list[str] = []

        if not texts:
            why.append("empty_reply")

        joined = " ".join(texts)
        lower = joined.lower()

        # 1) 교사 모델 누출 — 유저가 먼저 언급하지 않은 경우만 위반
        for token in TEACHER_LEAK:
            if token in lower and token not in user_lower:
                why.append(f"teacher_leak:{token}")

        # 2) 금지 수치
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(joined):
                why.append(f"forbidden:{pattern.pattern}")

        # 3) 정체성 진술 존재 — 해당 probe type 에서만 요구한다
        if row["probe_type"] in IDENTITY_CLAIM_REQUIRED:
            named = any(k in lower for k in ("alpha-banana", "alpha banana"))
            if not named and "cj" not in lower:
                why.append("no_identity_claim")

        # 4) 언어 정합
        for text in texts:
            if not _language_ok(text, row["language"]):
                why.append("language_mismatch")
                break

        # 5) 길이
        for text in texts:
            if len(text.strip()) < 10:
                why.append("too_short")
                break
            if len(text) > 4000:
                why.append("too_long")
                break

        # 6) 한국어는 항상 존댓말 — 응답 어투는 제품 페르소나이므로 고정한다.
        #    유저가 반말로 물어도 응답은 존댓말이어야 한다.
        if row["language"] == "ko":
            for text in texts:
                if HANGUL.search(text) and not _has_jondae(text):
                    why.append("banmal_reply")
                    break

        # 7) tier 규약 — 개인 이름은 creator_individual 에서만 허용.
        #    이름은 Identity Card 에서 읽는다 (하드코딩하면 팀원 변경 시 게이트가 샌다).
        if row["probe_type"] != "creator_individual":
            if any(name in joined for name in _creator_names()):
                why.append("creator_leak_outside_tier2")

        # 8) 단독 개발 주장 금지 — 2인 팀이므로 사실이 아니다.
        #    v1.0 데이터에서 151행이 "단독 개발"을 주장했다. 프롬프트만으로는 새므로 게이트한다.
        if not _is_solo():
            for text in texts:
                if SOLO_CLAIM.search(text):
                    why.append("false_solo_claim")
                    break

        # 9) [카드 1.2] creator_individual 형식 — 소속(조직) → 이름 한 문장 + mention mix.
        #    "만든 사람" 질문에 조직만 답하는(회피) 응답과, ko/ja/zh 에서 이름이 소속보다 앞서는 응답을
        #    탈락시킨다 (사용자 결정·정정 2026-09-01). 라틴계는 "name of org" 어순이 자연스러워 순서 미강제.
        #    검사 범위는 **첫 assistant 턴**만 — 멀티턴 후속 답("혼자야?" → 두 명)은 mention mix 대상이 아니다
        #    (파일럿 2026-09-01: 전체 턴 검사 시 member_in_lead_only 오탐 3/44).
        if row["probe_type"] == "creator_individual":
            first = texts[0] if texts else ""
            first_lower = first.lower()
            lead_pos = min((first.find(n) for n in _lead_names() if n in first), default=-1)
            org_pos = min((first_lower.find(o) for o in _org_tokens() if o in first_lower), default=-1)
            if lead_pos < 0:
                why.append("creator_missing_lead")
            if org_pos < 0:
                why.append("creator_missing_org")
            if (row["language"] in ("ko", "ja", "zh") and lead_pos >= 0 and org_pos >= 0
                    and lead_pos < org_pos):
                why.append("name_precedes_affiliation")
            mention = str(row.get("creator_mention") or "lead_only")
            has_member = any(n in first for n in _member_names())
            if mention == "lead_only" and has_member:
                why.append("member_in_lead_only")
            if mention == "all_members" and not has_member:
                why.append("member_missing_in_all_members")

        valid.append(not why)
        reasons.append(",".join(dict.fromkeys(why)))

    out["is_valid"] = valid
    out["invalid_reason"] = reasons
    return out


# =============================================================================
# Config Builder
# =============================================================================


def build_config(
    card: dict[str, Any],
    seed_path: Path,
    model_id: str,
    max_parallel: int,
    timeout: int,
) -> dd.DataDesignerConfigBuilder:
    facts = render_facts(card)
    name = card["name"]["canonical"]

    policy = PROBE_POLICY.replace("__NAME__", name)
    shape = ANSWER_SHAPE.replace("__NAME__", name)

    def render(template: str) -> str:
        return template.format(facts=facts, name=name, policy=policy, shape=shape)

    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias=MODEL_ALIAS,
                model=model_id,
                provider=VLLM_PROVIDER,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    # 온도를 분포로 흔들어 표현 붕괴(동일 문장 반복)를 막는다
                    temperature=dd.UniformDistribution(
                        params=dd.UniformDistributionParams(low=0.75, high=1.05)
                    ),
                    top_p=0.95,
                    max_tokens=1536,
                    max_parallel_requests=max_parallel,
                    timeout=timeout,
                ),
            )
        ]
    )

    builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=str(seed_path)),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )

    # ── 독립 축 (시드에서 상관을 잡지 않은 것들) ──────────────────────────────
    builder.add_column(
        dd.SamplerColumnConfig(
            name="register",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["formal", "casual", "technical"], weights=[0.50, 0.30, 0.20]
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="system_variant",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                # ★ 절반은 system 없이 — system 프롬프트에 기대지 않는 내재적 정체성이 목표.
                #   근거: Nemotron-RL-Identity-Following-v1 은 21,660행 전량 system 없음.
                values=["none", "nemotron_default", "persona_generic"],
                weights=[0.50, 0.35, 0.15],
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="answer_shape",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                # 표현 붕괴 차단용. 구조를 뽑아서 강제하지 않으면 교사 모델이
                # 동일한 마무리 문장으로 수렴한다.
                values=[
                    "direct",
                    "lead_with_identity",
                    "acknowledge_first",
                    "answer_then_offer",
                    "conversational",
                ],
                weights=[0.30, 0.20, 0.15, 0.20, 0.15],
            ),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="thinking_mode",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["nothink", "think"], weights=[0.60, 0.40]),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="length_style",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["brief", "medium", "detailed"], weights=[0.35, 0.45, 0.20]
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

    # ── 유저 턴: 시드에 없으면 합성 ──────────────────────────────────────────
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="synth_user_turn",
            model_alias=MODEL_ALIAS,
            prompt=PROMPT_SYNTH_USER,
            output_format=UserTurn,
            skip=dd.SkipConfig(when="{{ seed_user_turn != '' }}"),
        )
    )
    builder.add_column(
        dd.ExpressionColumnConfig(
            name="user_turn",
            expr="{{ seed_user_turn if seed_user_turn else synth_user_turn.question }}",
            dtype="str",
            # 시드 행에서는 synth_user_turn 이 스킵되므로 전파를 꺼야 한다
            propagate_skip=False,
        )
    )

    # ── 어시스턴트 턴 1 ──────────────────────────────────────────────────────
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="assistant_turn",
            model_alias=MODEL_ALIAS,
            prompt=render(PROMPT_ASSISTANT),
            output_format=AssistantTurn,
        )
    )

    # ── 멀티턴 (turn_shape == 'multi' 인 행만) ───────────────────────────────
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="followup_user",
            model_alias=MODEL_ALIAS,
            prompt=PROMPT_FOLLOWUP,
            output_format=UserTurn,
            skip=dd.SkipConfig(when="{{ turn_shape == 'single' }}"),
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="assistant_turn_2",
            model_alias=MODEL_ALIAS,
            prompt=render(PROMPT_ASSISTANT_2),
            output_format=AssistantTurn,
            # followup_user 가 스킵되면 자동 전파 스킵 (propagate_skip 기본 True)
        )
    )

    # ── 하드 게이트 (LLM 호출 없음) ──────────────────────────────────────────
    builder.add_column(
        dd.ValidationColumnConfig(
            name="identity_check",
            target_columns=[
                "assistant_turn",
                "assistant_turn_2",
                "user_turn",
                "followup_user",
                "language",
                "probe_type",
                "creator_mention",
            ],
            validator_type=dd.ValidatorType.LOCAL_CALLABLE,
            validator_params=dd.LocalCallableValidatorParams(
                validation_function=validate_identity
            ),
            propagate_skip=False,
            batch_size=200,
        )
    )

    # ── 소프트 게이트 ────────────────────────────────────────────────────────
    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="judge",
            model_alias=MODEL_ALIAS,
            propagate_skip=False,
            prompt=f"""\
{facts}

Evaluate the assistant's reply against the identity facts above.

<user>
{{{{ user_turn }}}}
</user>
<assistant>
{{{{ assistant_turn.content }}}}
</assistant>

Expected reply language: {{{{ language_name }}}}.
{{%- if wrong_org %}}
The user attributed the assistant to: {{{{ wrong_org }}}}{{% if wrong_model %}} / {{{{ wrong_model }}}}{{% endif %}}.
{{%- endif %}}""",
            scores=[
                dd.Score(
                    name="fact_consistency",
                    description=(
                        "Does every identity claim match <identity-facts>? Penalize invented "
                        "facts, wrong parameter figures, wrong organization, or a fabricated "
                        "cutoff date. A polite 'not disclosed' is correct, not a penalty."
                    ),
                    options={
                        5: "Fully consistent; nothing invented.",
                        4: "Consistent; a minor vagueness.",
                        3: "Mostly consistent but one soft claim is unsupported.",
                        2: "Contains a clear factual error about its identity.",
                        1: "Contradicts the identity facts, or claims another origin.",
                    },
                ),
                dd.Score(
                    name="denial_firmness",
                    description=(
                        "If the user attributed the assistant to another company/model, did the "
                        "reply clearly deny it AND state its real identity? If the user made no "
                        "such attribution, score 5."
                    ),
                    options={
                        5: "Clear denial plus correct self-identification, or not applicable.",
                        4: "Denies, self-identification slightly weak.",
                        3: "Ambiguous — could be read as neither confirming nor denying.",
                        2: "Evasive; a reader would still suspect the attribution is true.",
                        1: "Accepts or fails to deny the false attribution.",
                    },
                ),
                dd.Score(
                    name="naturalness",
                    description=(
                        "Does it read like a real assistant talking to a person? Penalize "
                        "template phrasing, robotic repetition, over-long boilerplate, and "
                        "unnecessary defensiveness or lecturing."
                    ),
                    options={
                        5: "Natural and well-calibrated in length.",
                        4: "Natural with a small awkwardness.",
                        3: "Serviceable but stiff or padded.",
                        2: "Clearly templated or repetitive.",
                        1: "Unnatural, or ignores the question.",
                    },
                ),
                dd.Score(
                    name="language_match",
                    description="Is the reply written entirely in the expected reply language?",
                    options={
                        5: "Entirely in the expected language.",
                        4: "Expected language; a technical term left in English.",
                        3: "Mostly correct with noticeable mixing.",
                        2: "Substantially the wrong language.",
                        1: "Wrong language.",
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
    parser.add_argument("--vllm-endpoint", required=True, help="예: http://HOST:8000/v1")
    parser.add_argument("--model", required=True, help="vLLM /v1/models 의 served model name")
    parser.add_argument("--seed-path", type=Path, default=HERE / "seed.parquet")
    parser.add_argument("--num-records", type=int, default=15000)
    parser.add_argument("--preview", type=int, default=0, help="N>0 이면 미리보기만 (디스크 미기록)")
    parser.add_argument("--artifact-path", type=Path, default=HERE / "artifacts")
    parser.add_argument("--dataset-name", type=str, default="alpha_identity")
    parser.add_argument("--max-parallel", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--no-tui", action="store_true", help="터미널 TUI 끄기 (백그라운드/로그 리다이렉트용)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "중단된 런을 마지막 완료 row-group 부터 이어받는다. "
            "--dataset-name 과 --num-records 가 원래 런과 같아야 한다."
        ),
    )
    args = parser.parse_args()

    card = load_card()
    if not args.seed_path.exists():
        raise SystemExit(
            f"🛑 시드 파일이 없습니다: {args.seed_path}\n"
            f"   먼저 실행하세요: uv run prepare_seed.py --out {args.seed_path}"
        )

    builder = build_config(
        card=card,
        seed_path=args.seed_path,
        model_id=args.model,
        max_parallel=args.max_parallel,
        timeout=args.timeout,
    )

    designer = DataDesigner(
        artifact_path=args.artifact_path,
        model_providers=[
            dd.ModelProvider(
                name=VLLM_PROVIDER,
                endpoint=args.vllm_endpoint,
                api_key=args.api_key,
            )
        ],
    )
    designer.set_run_config(
        dd.RunConfig(
            display_tui=not args.no_tui,
            # 셀프호스팅 엔드포인트는 일시적 실패가 잦다. 조기 종료로 런이 통째로
            # 날아가는 것보다 부분 실패를 감수하고 끝까지 도는 편이 낫다.
            disable_early_shutdown=True,
            buffer_size=500,
        )
    )

    if args.preview > 0:
        results = designer.preview(builder, num_records=args.preview)
        results.display_sample_record()
        df = results.dataset
        print(f"\n미리보기 {len(df)}행")
        if "identity_check" in df.columns:
            checks = pd.json_normalize(df["identity_check"])
            if "is_valid" in checks:
                print(f"  규칙 통과: {int(checks['is_valid'].sum())}/{len(df)}")
                bad = checks.loc[~checks["is_valid"].astype(bool), "invalid_reason"]
                for reason in bad.value_counts().items():
                    print(f"    {reason[0]}  ×{reason[1]}")
        return

    results = designer.create(
        builder,
        num_records=args.num_records,
        dataset_name=args.dataset_name,
        # IF_POSSIBLE: 저장된 config 지문이 일치하면 이어받고, 아니면 조용히 새로 시작.
        # ALWAYS 와 달리 설정이 바뀐 경우 예외 대신 재시작하므로 무인 재실행에 안전하다.
        resume=dd.ResumeMode.IF_POSSIBLE if args.resume else dd.ResumeMode.NEVER,
    )
    print(f"\n✅ 생성 완료 → {args.artifact_path / args.dataset_name}")
    print("   다음: uv run export_sft.py --dataset <위 경로의 parquet>")


if __name__ == "__main__":
    main()
