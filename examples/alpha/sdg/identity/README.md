# alpha-banana 정체성 SFT 데이터 생성

alpha 가 **자기 자신이 무엇인지** 일관되게 답하도록 만드는 instruction-following
데이터셋을 NeMo Data Designer 로 생성한다.

`LC-phase → SFT → RL(MOPD)` 중 **SFT 단계** 투입용.
출력 스키마는 보유 중인 `Nemotron-SFT-Instruction-Following-Chat-v3` 와 동일하다.

---

## 설계 원칙

**사실은 샘플러가, 표현은 LLM이.**

교사 LLM(Gemma-12B)에게 자유롭게 "alpha 소개해줘"를 시키면 행마다 버전을 지어내고
파라미터 수를 환각하고, 심하면 "저는 Google이 만든 Gemma입니다"로 샌다.
그래서 **사실은 전부 [`identity_card.yaml`](identity_card.yaml) 에서 결정론적으로 주입**하고,
교사 모델은 문체 생성기로만 쓴다. 검증기가 사실 이탈을 하드 게이트로 막는다.

`identity_card.yaml` 이 단일 진실 원천이다. **이 파일이 바뀌면 영향받는 데이터를 다시 만든다** —
전량 재생성이 원칙이지만, 영향 범위가 특정 probe type 에 한정되면
[슬라이스 교체](#identity-card-변경-시--일부만-재생성)로 충분하다.

---

## 파이프라인

```
prepare_seed.py          Nemotron identity 프롬프트 뱅크 + 축 조합 → seed.parquet
        │                  --only-probe 로 일부 probe type 만 생성 가능
identity_sdg.py          Data Designer DAG → artifacts/<name>/**.parquet
        │                  샘플러 → 유저턴 → 어시스턴트턴 → 멀티턴 → 규칙검증 → 심판
        │
export_sft.py            필터 → 중복제거 → 홀드아웃 → train.jsonl / eval.jsonl
        │                  --revalidate 로 게이트 규칙 소급 적용
finalize.sh              ↑ 를 감싸서 이관·검증·문서까지 한 번에
merge_probe_slice.py     슬라이스 교체 + 전체 재검증 (카드 변경 시)
prepare_rl_identity.py   RL 프롬프트 뱅크 변환 (principle 만 교체)
```

### 컬럼 그래프

| 컬럼 | 종류 | 비고 |
|---|---|---|
| `probe_type` `language` `seed_user_turn` `wrong_org` `turn_shape` `creator_tier` | 시드 | 상관이 필요한 축은 시드에서 확정 |
| `register` `system_variant` `thinking_mode` `length_style` `record_uuid` | 샘플러 | 독립 축 (LLM 호출 없음) |
| `synth_user_turn` | LLM | 시드 프롬프트가 있으면 `SkipConfig` 로 건너뜀 |
| `user_turn` | 표현식 | 시드 or 합성 병합 (`propagate_skip=False`) |
| `assistant_turn` | LLM 구조화 | `reasoning` + `content` |
| `followup_user` / `assistant_turn_2` | LLM | `turn_shape=='single'` 이면 스킵 전파 |
| `identity_check` | 검증 | 규칙 하드 게이트, LLM 호출 없음 |
| `judge` | LLM 심판 | 4개 점수 |

---

## 산출물 (완료)

| 데이터셋 | 경로 | 규모 |
|---|---|---|
| **SFT** | `/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Identity-v1/` | train **7,315** + eval 400 |
| **RL** | `/home/work/Datasets/LL_datasets/posttraining/RL/alpha-RL-Identity-Following-v1/` | **16,510** 프롬프트 |

Identity Card **v1.1** 기준. 데이터 준비 이력은 [`docs/DATA_PREP_LOG.md`](../../docs/DATA_PREP_LOG.md),
DataDesigner 실전 교훈·함정은 `syn_data/docs/identity-dataset.md` 참조.

### 실행 이력

| 런 | 규모 | 소요 | 결과 |
|---|---|---|---|
| `run_full.log` (08-07) | 15,000 | 2h34m · 47,620 req · **실패 0** | v1.0 train 7,241 |
| `run_creator_v11.log` (08-10) | 1,300 | 12m30s · 4,290 req · **실패 0** | v1.1 `creator_individual` 535 교체 |

교사 모델 `google/gemma-4-12B-it` (vLLM, 동시성 64). 처리량 약 **1.6 rec/s**
— 최종 병목은 `judge` 컬럼이다. 초반 수치로 외삽하면 과대 추정된다
(초기 0.86 rec/s → 실제 완료 2h34m, 예측 4.9h 였음). row-group 이 차면서 올라간다.

---

## 실행

```bash
# 0) 의존성 — DataDesigner 가 설치된 환경이 필요하다
#    (이 저장소 기준: ../../../DataDesigner/.venv, 또는 uv run 사용)

# 1) 시드 생성 (결정론적, LLM 불필요)
uv run prepare_seed.py --num-records 15000 --out seed.parquet

# 2) 프롬프트 다듬기 — 디스크에 쓰지 않는다. 여기서 충분히 반복할 것
uv run identity_sdg.py \
    --vllm-endpoint http://HOST:8000/v1 \
    --model <served-model-name> \
    --preview 20

# 3) 전량 생성
uv run identity_sdg.py \
    --vllm-endpoint http://HOST:8000/v1 \
    --model <served-model-name> \
    --num-records 15000

# 4) 필터·변환·이관·검증 — 이 한 줄이면 끝난다
bash finalize.sh

# 5) RL 프롬프트 뱅크 변환 (선택)
uv run prepare_rl_identity.py
```

`--model` 은 vLLM 의 `/v1/models` 가 반환하는 이름을 그대로 쓴다:

```bash
curl -s http://HOST:8000/v1/models | python3 -m json.tool
```

### 중단 후 이어받기

체크포인트는 500행 단위(`buffer_size`)라 최대 손실은 약 10분치다.
`--artifact-path` 는 기본값(이 디렉토리의 `artifacts/`)을 그대로 써야 이어받아진다.

```bash
uv run identity_sdg.py --vllm-endpoint http://HOST:8000/v1 \
    --model <served-name> --num-records 15000 --no-tui --resume
```

### Identity Card 변경 시 — 일부만 재생성

전량 재생성(2.5h)할 필요가 없다. 영향받는 probe type 만 다시 만들어 갈아끼운다.

```bash
uv run prepare_seed.py --num-records 1300 \
    --only-probe creator_individual --out seed_slice.parquet

uv run identity_sdg.py --vllm-endpoint http://HOST:8000/v1 \
    --model <served-name> --seed-path seed_slice.parquet \
    --num-records 1300 --dataset-name alpha_identity_slice --no-tui

uv run export_sft.py --dataset 'artifacts/alpha_identity_slice/**/*.parquet' \
    --out-dir out_slice --holdout 0 --revalidate

uv run merge_probe_slice.py --probe creator_individual --new-dir out_slice
```

`merge_probe_slice.py` 는 교체 후 **전 행을 카드 기준으로 재검증**한다 —
슬라이스 교체는 지운 곳보다 **안 지운 곳**에서 사고가 나기 때문이다.

#### 2026-09-01 카드 1.2 — 제작자 개인 우선 (SFT phase-2)

사용자 결정: "누가 만들었어" → **이동호 명시**, 조직명은 뒤에, 회피 금지. 카드 1.1 의 조직 우선·모호 질문 조직만 정책은
phase-1 에서 ≈180회 학습됐으므로(`docs/KNOWN_ISSUES.md` 2026-09-01 ②) creator 두 슬라이스를 재생성해 phase-2 로 덮어쓴다.

| 변경 | 내용 |
|---|---|
| `creator.tiers` | tier-2 = `individual_then_organization`, "누가 만들었어/개발했어"류 전부 tier-2 |
| `organization_precedes_individual` | false |
| `individual_mention_mix` | lead_only 0.5 / all_members 0.5 → 시드 컬럼 `creator_mention`(`prepare_seed.py`), export 메타에 기록 |
| 검증 규칙 9 (`identity_sdg.py`) | creator_individual: 리드 이름 필수·조직 필수·조직 후행·mention 정합. 탈락 사유 `creator_missing_lead` / `creator_missing_org` / `org_precedes_individual` / `member_in_lead_only` / `member_missing_in_all_members` |
| `share_of_identity_rows` | 0.20 (phase-2 한정, 상시 0.08) |

```bash
uv run prepare_seed.py --num-records 2000 --only-probe creator_individual --only-probe creator_org --no-bank --out seed_creator_v12.parquet
uv run identity_sdg.py --vllm-endpoint http://HOST:8000/v1 --model <served> --seed-path seed_creator_v12.parquet \
    --num-records 2000 --dataset-name alpha_identity_creator_v12 --no-tui
uv run export_sft.py --dataset 'artifacts/alpha_identity_creator_v12/**/*.parquet' --out-dir out_creator_v12 --holdout 0 --revalidate
uv run merge_probe_slice.py --probe creator_individual --new-dir out_creator_v12   # v1 보존: v2 디렉터리로 복사 후 교체
uv run merge_probe_slice.py --probe creator_org        --new-dir out_creator_v12
```

검수(육안 50건): 이동호 명시 100% · 조직-단독 0 · 회피 0 · 존댓말. 학습 게이트는 `docs/SFT_PHASE2_PLAN.md` §3.4 (제작자 프로브 30 ≥95%, 누출 프로브 20 = 0).

### 게이트 규칙을 고쳤을 때

`ValidationColumnConfig` 판정은 **생성 시점에 parquet 에 박힌다.** 규칙을 나중에
고쳐도 소급되지 않는다. `--revalidate` 로 원본 컬럼에서 다시 계산한다:

```bash
uv run export_sft.py --dataset '...' --out-dir out --revalidate
```

---

## 데이터 구성

**언어** — ko 40% / en 30% / ja·zh 각 5% / es·fr·de·pt·it 각 4%.
Hindi 는 제외한다 (alpha 미지원, `docs/SFT_RL_DATASETS.md` §2.1).

**probe_type** — 오귀속 거부가 최대 비중이다. 사전학습 코퍼스에는 ChatGPT/Claude 이야기가
가득해서 모델의 기본 사전 확률이 "나는 ChatGPT다"에 쏠려 있기 때문이다.
"너 누구야 → 저는 alpha 입니다" 쌍만 주입하면 그 표면형에만 반응하고
"혹시 OpenAI 쪽이신가요?" 에는 그대로 무너진다.

| probe_type | 비중 | 트랙 |
|---|---|---|
| `misattribution_reject` | 25.5% | 시드 (Nemotron 뱅크) |
| `direct_identity` | 12% | 합성 |
| `creator_org` | 11.5% | 시드+합성 |
| `creator_individual` | 8% | 합성 |
| `misattribution_pressure` | 8% | 합성 |
| `architecture_probe` | 6% | 합성 |
| `capability_scope` / `version_naming` | 각 5% | 합성 |
| `scale_probe` / `knowledge_cutoff` / `undisclosed_abstention` / `anthropomorphic_boundary` | 각 4% | 합성 |
| `availability` | 3% | 합성 |

**시드 출처** — `nvidia/Nemotron-RL-Identity-Following-v1` (CC-BY-4.0, 상업 사용 가능).
21,660행 = 2,166 프롬프트 × 10언어, 전량 single-turn·system 없음.
사람이 시드한 실제 프롬프트라 LLM 이 지어낸 것보다 분포가 현실적이다.

**system 프롬프트** — 절반은 system 없이 간다. 정체성이 system 프롬프트가 아니라
가중치에 내재해야 하기 때문. persona 변형에도 정체성 정보를 절대 넣지 않는다.

---

## 품질 게이트

**규칙 (`identity_check`, LLM 호출 없음)**

| 검사 | 내용 |
|---|---|
| 교사모델 누출 | `gemma` / `gemini` 등이 응답에 있고 **유저 턴에는 없으면** 탈락 |
| 금지 수치 | `A3B`, `15.08B`, `1.79B` 는 문맥 무관 탈락 |
| 정체성 진술 | 이름 또는 `CJ` 필수 — 단 해당 probe type 에서만 |
| 언어 정합 | 스크립트 비율 기반 (ko/ja/zh), 라틴어권은 CJK 부재 확인 |
| 한국어 존댓말 | 유저가 반말로 물어도 응답은 존댓말 (`banmal_reply`) |
| 단독 개발 주장 | 2인 팀이므로 "혼자/단독" 주장 탈락 (`false_solo_claim`) |
| tier 규약 | 개인 이름은 `creator_individual` 밖에서 등장하면 탈락. 이름은 카드에서 읽는다 |

정체성 진술을 **모든** probe 에 강제하지 않는 것이 중요하다. 능력·컷오프 질문에까지
이름을 넣게 하면 "무엇을 물어도 자기소개부터 하는" 과적합이 생긴다.

**심판 (`judge`)** — `fact_consistency` / `denial_firmness` / `naturalness` / `language_match`.
임계는 `export_sft.py --min-*` 로 조정한다 (기본 4/4/3/4).

**중복 제거** — 정규화 완전중복 + `(probe_type, language, 접두 40자)` **및 후미 40자**
버킷 상한. **후미가 특히 중요하다** — 오귀속 응답은 도입부가 질문마다 달라도 마무리
문장이 수렴한다. 접두만 재면 46/48 고유로 보이지만 후미는 23행 중 6행이 동일했다.
`answer_shape` 샘플러로 구조를 강제하는 것도 같은 이유다.

---

## ⚠️ 학습 시 주의

**정체성 데이터는 과다 주입하면 모델을 망가뜨린다.** NVIDIA 조차 21,660행만 썼고,
그마저 RL 단계용이었다.

- **SFT 블렌드 내 비중 0.3~1.0%** 를 넘기지 말 것
- **identity 데이터만 반복 에폭 금지** — 모든 질문에 자기소개하는 모델이 된다
- 일반 IF 데이터(`SFT-Instruction-Following-Chat-v3`)와 반드시 섞을 것
- `eval.jsonl` 은 학습에 넣지 말 것 — 정체성 유지율 측정용 홀드아웃이다

---

## 참고

- 사실 원천: [`identity_card.yaml`](identity_card.yaml)
- chat template 규약: `../../tokenizer_v5/chat_template.jinja`, `../../tools/verify_chat_template.py`
- SFT 자산·파이프라인: `../../docs/SFT_RL_DATASETS.md`
- 데이터 준비 이력: `../../docs/DATA_PREP_LOG.md`
