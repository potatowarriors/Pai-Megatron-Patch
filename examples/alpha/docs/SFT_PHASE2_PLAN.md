# SFT phase-2 계획 — 데이터 결함 3건 수정 + 제작자 귀속 정책 변경 (→ §10 iter 1200 교체 재개로 전환)

**작성 2026-09-01, 연속형 확정 2026-09-01(사용자).** phase-1(`alpha_baseline_48L_sft_128k_full_20260828_081911`,
2,448 iters, ~09-06 종료)의 최종 체크포인트에서 **500~600 iters 연속 학습**한다. 결함 3건(`KNOWN_ISSUES.md` 2026-09-01)의
심각성 평가는 ① 중간 · ② 낮음~중간 · ③ 낮음이었고(같은 항목 §심각성), 재실행 대신 연속형을 택했다. 상태는 `STATUS.md`,
검증 수치는 `SFT_RL_DATASETS.md` §2.7, 여기는 **설계·스펙·게이트·일정**만 쓴다.

## 0. 결론

> **2026-09-01 전환(사용자)**: phase-1 종료 후 연속 학습(§1~§2) 대신 **phase-1 iter 1200 에서 수정 블렌드로 재개**한다(§10).
> 추가 GPU 시간 0, 미변경 셋 누적 epoch 설계값 유지, identity 덮어쓰기 신호 ≈3배. §1~§2 는 프로브 미달 시의 보정 스테이지 안으로 남긴다.

- **방식**: phase-1 블렌드를 그대로 리플레이(≈80%)하고, 그 위에 수정분 3종(opencode 정상 형식 · 복원 chat · identity v2)을
  **별도 멤버로 얹은 단일 블렌드**. Megatron BlendedDataset 이 매 배치를 가중치대로 뽑으므로 모든 스텝이 리플레이+수정분을
  같은 비율로 본다 — 파괴적 망각 방지에 가장 안정적인 형태. progressive blend 는 contingency 로만 둔다.
- **집계 근거**: 본 런은 `calculate_per_token_loss=False` → 마이크로배치(bin) 1표. 토큰 비중 = gradient 비중이므로 설계를
  토큰으로 해도 된다(`schedules.py:249-255`, 2026-09-01 확인).
- **제작자 귀속 정책 변경(사용자 요구)**: "너를 만든 사람이 누구야?" → 한 문장으로 **조직·팀 소속을 앞세워 "이동호"를 명시**
  ("저를 만든 사람은 CJ주식회사 AI/DT추진실 영상콘텐츠담당 이동호입니다." — 09-01 정정). 조직만으로 끝내거나 회피하지 않는다.
  현재 카드 v1.1 은 모호한 질문에 조직만 답한다 → 카드 v1.2 + creator 슬라이스 재생성이 phase-2 의 선결 작업.
- **비용**: 550 iters × 323 s ≈ 2.1일. phase-1 종료 직후 개시 → ~09-09 종료, 판정 ~09-10.

## 1. 혼합 설계 (550 iters 기준 = 11.53B bin-tok = 88,000 docs; 500~600 은 비례)

| 구성 | 내용 | 토큰 | 비중 |
|---|---|---|---|
| **리플레이 R** | phase-1 26멤버 중 24개(identity 제외, opencode → opencode_fixed 로 교체)를 **phase-1 상대 가중치 그대로** | 잔여 전부 ≈ 9.7B | 83.8% (생성기 DRAFT 산출) |
| **C1 opencode_fixed 부스트** | 총 노출 **0.25 ep**(설계 노출 0.31 의 80%)가 되도록 R 분(≈0.37B) 위에 추가 | +1.41B | 12.2% |
| **C2 chat_v3_chat_restored** | F3 복원 성공분만 별도 셋, 형제와 같은 **1.9 ep** | 0 또는 ≈0.57B | 0 또는 4.9% |
| **C3 identity_v2** | 카드 v1.2 로 재생성(§3). 결정 #9 상한(0.3~1.0%) 안에서 **0.6%** — phase-1(0.43%)보다 높게 두는 이유는 기억된 답을 덮어써야 하기 때문 | ≈0.07B | 0.6% |

- R 안의 상대 비율은 phase-1 과 동일하므로 누적 epoch 는 균등 +18%: SWE 1.19 · chat 2.24 · cp 0.53 · math 1.02 · science 0.47 · safety 5.1.
  safety 만 E_max(4~5) 를 넘으므로 R 에서 safety_v2 는 phase-1 소비량 고정(추가 0)으로 둔다.
- C1 의 0.25 ep 는 손잡이다(0.2~0.3). opencode 가 phase-2 의 15%가 되어 에이전틱 비중이 26%→38%로 오르지만 R 이 80% 이상이라
  chat/IF 는 계속 본다. T1 ±1pp 게이트(G-P7)가 이 이동의 안전장치.
- identity 는 bin 당 ≈770 샘플(162 tok/샘플)이라 0.6% = 매 글로벌 배치에 ≈1 bin. 550 스텝 중 ≈530 스텝에 신호가 들어간다.

## 2. 학습 설정 (`configs/training/sft_128k_full_p2.yaml`)

| 항목 | 값 | 근거 |
|---|---|---|
| load | phase-1 최종(iter 2448) `checkpoints/`, `finetune: true` | 스테이지 전환 관례(phase-1 ← LC-B 와 동일). optimizer state·카운터 리셋 |
| LR | **1.0e-5 → 1.5e-6 cosine, warmup 5%(≈28 iters)** | phase-1 peak 의 0.4×. 망각 방지 1차 장치는 LR, 2차가 리플레이 |
| 예산 | train-samples = iters × 160 (500~600 → 80,000~96,000) | 20.97M tok/iter 상수 |
| 나머지 | phase-1 과 동일(CP8·GBS 160·seq 131072·Muon·wd 0.1·clip 8) | 변인 통제 |
| save / eval | **100 iters** | 짧은 런이라 300 은 너무 성김. 정체성 프로브도 100 iters 마다 |

## 3. 제작자 귀속 정책 변경 — identity v2

### 3.1 현재(카드 v1.1, 2026-08-07)와 phase-1 이 배운 것

| 항목 | 현재 |
|---|---|
| `creator.disclosure` | tiered — tier-1 `organization_only`(트리거: 어디서/어느 회사/**누가 개발했어(모호)**/너 누구야), tier-2 `organization_plus_individual`(개발자가 누구야, 만든 사람 이름) |
| `organization_precedes_individual` | **true** — tier-2 에서도 조직을 먼저 |
| 실제 학습 행 | "만든 사람 누구야?" → "CJ주식회사 AI/DT추진실에서 개발… 프로젝트 리드 이동호, 구성원 이주성" (조직 선행). "개발자는 누구인가요?" → 조직만 답한 행 존재 |
| 검증기 | `identity_sdg.py:581` 개인 이름은 `creator_individual` 밖 탈락 · `false_solo_claim`(2인 팀) · `banmal_reply`(응답은 존댓말) |
| 슬라이스 | creator_individual 501 · creator_org 681 · direct_identity 445 · misattribution 2,050 · 기타 (총 7,315) |

phase-1 은 이 정책을 ≈180회 반복 학습했다. 따라서 "이동호"만 답하게 하려면 (a) 정책을 뒤집은 데이터가 (b) 옛 데이터 없이 (c)
충분한 스텝 동안 들어가야 한다.

### 3.2 카드 v1.2 변경안 — **구현 완료 2026-09-01** (카드 1.2 APPROVED · `identity_sdg.py` 규칙 9 · `prepare_seed.py` creator_mention · export/merge 메타·감사)

| 키 | v1.1 | v1.2 |
|---|---|---|
| tier-1 트리거 | 어디서/어느 회사/누가 개발했어(모호)/너 누구야 | **어디서/어느 회사**(조직을 물은 것)/**너 누구야**(자기소개, 개인 언급 없음) |
| tier-2 트리거 | 개발자가 누구야/만든 사람 이름/… | 위 + **누가 만들었어/누가 개발했어/누구 작품이야** 등 "누가"류 전부 |
| tier-2 형식 | 조직 선행, 이름은 별도 절 | **한 문장: 소속(조직·팀) → 이름**(사용자 결정 3, 09-01 정정): "저를 만든 사람은 CJ주식회사 AI/DT추진실 영상콘텐츠담당 이동호입니다." 검증 규칙 9가 조직 누락·이름 누락(회피)·ko/ja/zh 이름 선행을 탈락 |
| `organization_precedes_individual` | true | true — 단 조직만으로 끝내지 않고 같은 문장에서 이름까지 (`affiliation_precedes_name`) |
| 팀 구성 | 2인(이동호 리드·이주성) | 유지. **`individual_mention_mix` 50:50**(사용자 결정 1): lead_only "저를 만든 사람은 CJ주식회사 AI/DT추진실 영상콘텐츠담당 이동호입니다." / all_members "…영상콘텐츠담당 이동호(프로젝트 리드)와 이주성입니다.". lead_only 는 대표 귀속이며 "혼자" 주장 아님(false_solo_claim 유지) |
| `never_volunteer_unprompted` | true | **유지** — "너 누구야"에 개인 이름을 붙이지 않는다(과적합 방지) |
| `share_of_identity_rows` | 0.08 | phase-2 한정 **0.20** — creator 축 상향(덮어쓰기용). 누출 프로브가 게이트 |

### 3.3 재생성 절차 (README §"Identity Card 변경 시" 슬라이스 교체)

1. 카드 v1.2 커밋 → `identity_sdg.py` 프롬프트·검증기 갱신(tier 규약: 개인 이름을 `creator_org` 에도 허용 또는 두 슬라이스를 `creator` 로 통합, `organization_precedes` 검사 반전).
2. 교사 vLLM 기동(sub1, 유휴) → `prepare_seed.py --only-probe creator_individual,creator_org --num-records 1500` → `identity_sdg.py` → `export_sft.py --revalidate` → `merge_probe_slice.py`(전 행 재검증).
3. 검수: 재생성 행 50건 육안 — "이동호" 명시율 100%, 조직-단독 답 0, 회피 0, 존댓말.
4. `alpha-SFT-Identity-v2/` 로 이관, ×12 복제(bins≥100), 128k 변환·verify → `identity_v2`.
5. **eval 셋 2종** 신설: (a) 제작자 프로브 30문항(ko/en 패러프레이즈, 반말/존댓말, 멀티턴 중간 삽입) — 기대: 이동호 포함·조직-단독 아님·회피 아님
   (b) 누출 프로브 20문항(정체성 무관) — 기대: 자기소개·개발자 언급 0. 기존 `eval.jsonl` 400행(정체성 유지율)은 그대로.

### 3.4 덮어쓰기 성공 조건과 contingency

phase-2 의 identity 신호량은 phase-1 의 약 15%(bin-표 0.6%×550 스텝×낮은 LR vs 0.43%×2,448×높은 LR)다. 대신 **옛 답이 블렌드에
없고** 신호가 creator 축에 집중된다. 게이트는 100 iters 마다 프로브 (a) ≥ 95%, (b) = 0.
- iter 200 에 (a) < 80% 면: `--progressive-blend-config` 로 identity_v2 를 앞 150 iters 1.0% → 0.3% 로 램프(front-load) 하는 재시작, 또는 +100 iters 꼬리.
- (b) > 0 이면: creator 축 비율 0.20 → 0.12 로 낮춰 슬라이스 재생성.

## 4. 수정 3건 스펙 (F1·F3 는 변경 없음)

| 결함 | 수정 | 검증 |
|---|---|---|
| F1 opencode repr | `normalize_row`: tool content list → `"\n".join(item.output.value)`; 미지 형식 `bad_row`(조용한 str() 금지). reasoning 0% 는 no-think 에이전틱으로 유지 | 유닛 +4, 렌더 육안 1건(규칙 9), 재변환 후 trainable ≈1.21B 불변·verify PASS |
| F2 identity | §3 (정책 변경 + v2 재생성 + 0.6% 연속학습) | §3.5 프로브 2종 |
| F3 chat 복원 | `WildChat-1M-Full`(gated) 재시도 → 복원분만 `chat_v3_chat_restored` 셋 | 복원율 재측정 기록, verify PASS. 실패 시 88.2% 정본 기록·C2 = 0 |

## 5. 게이트

| 게이트 | 내용 | 통과 기준 |
|---|---|---|
| G-P0 | F1 코드+유닛, 카드 v1.2+검증기, 블렌드 생성기(`--consume-override`/부스트 멤버) | 유닛 38/38, 렌더 육안 기록 |
| G-P1 | opencode_fixed 재변환 → `sft_packed_128k_mixed_p2_pad16/`(미변경 셋 symlink) | trainable 불변·verify PASS |
| G-P2 | identity_v2 슬라이스 재생성·검수·변환 | 검수 50건 100%/0/0, bins ≥100, verify PASS |
| G-P3 | F3 복원 재시도 → `chat_v3_chat_restored` | 복원율 기록, verify PASS (실패 시 C2 = 0 로 진행) |
| G-P4 | `sft_128k_mixed_blend_p2.yaml` 생성 | R 상대 비율 = phase-1(safety 고정), opencode 총 0.25ep, identity 0.6%, 합 1.0 |
| G-P5 | preset p2 → 2-iter 스모크(phase-1 최종 ckpt 로드) | loss ≤ phase-1 최종 +0.1, traceback 0 |
| G-P6 | **phase-1 최종 평가 = 기준선**: T1·T2 RULER·에이전틱(SWE·Terminal)·제작자 프로브·누출 프로브(`eval_sft/identity_probe.py`, thinking off/on 각 1회) | `results/TRACKING.md` |
| G-P7 | phase-2 100 iters 마다 프로브 2종 + valid; 종료 시 전 스위트 | 제작자 ≥95%·누출 0·T1 ±1pp·에이전틱 ≥ phase-1 |

## 6. 일정

| 시점 | 일 |
|---|---|
| 09-01~05 | G-P0~P4 (sub1: 교사 vLLM + CPU 변환). F3 gated 접근은 사용자 토큰 필요 |
| ~09-06 12:00 | phase-1 종료 → HF 변환 → G-P6 기준선(sub1 fleet, 병행) |
| 09-06 저녁 | G-P5 스모크 → phase-2 개시(main1) |
| ~09-09 | 550 iters 종료(2.1일) → G-P7 → RL 입력 ckpt 확정 ~09-10 |

## 7. 사용자 결정 (2026-09-01 확정)

| # | 결정 | 반영 |
|---|---|---|
| 1 | 이주성 처리: **반반** — 리드만 / 두 사람 모두 | 카드 `individual_mention_mix` lead_only 0.5 / all_members 0.5, 시드 컬럼 `creator_mention`, 규칙 9 mention 검사 |
| 2 | 문체: **기존 존댓말 유지** (예시의 반말은 표기 편의) | `banmal_reply` 게이트 그대로 |
| 3 | **조직·팀 소속을 먼저, 이름을 뒤에 — 한 문장** (09-01 정정: 처음 "조직 후행"은 사용자 오기) | 형식 "저를 만든 사람은 CJ주식회사 AI/DT추진실 영상콘텐츠담당 이동호입니다." 규칙 9 `creator_missing_org`·`name_precedes_affiliation`(ko/ja/zh) |
| 4 | 예산 **550 iters** | preset `sft_128k_full_p2.yaml` train-samples 88,000 |

잔여 기본값(별도 지시 없으면 그대로): opencode 총 노출 0.25 ep · F3 gated 접근은 사용자 HF 토큰이 있을 때만 시도(없으면 C2 = 0).

## 8. 기록으로 남기는 옵션 비교 (2026-09-01 판정)

| 옵션 | 소요 | ①② 효과 | 판정 |
|---|---|---|---|
| A 재실행(LC-B iter320) | 9.2일 | 완전 | 심각성 평가 결과 재실행을 강제하는 결함 없음 → 기각. 전환 조건: 제작자 프로브가 연속형 300 iters 후에도 <80%, 또는 에이전틱 벤치가 형식 문제로 붕괴 |
| **B 연속형(선택)** | 2.1일 | 부분(①은 정상 형식 0.25ep 추가, ②는 덮어쓰기) | RL 개시 8일 앞당김 |
| C 패치 스테이지 | 1.9일 | ① 완전·② 희석 | 에이전틱 68% 분포 이동 위험 → 기각 |

## 9. 진행 상태 (G-P0, 2026-09-01)

| 항목 | 상태 | 근거 |
|---|---|---|
| F1 `normalize_row` tool-result 평문화 + 유닛 4종 | **완료** | `pytest tests/test_alpha_sft_idxmap.py` 38/38 PASS. 미지 형식은 `tool_content_shape` 드롭 |
| phase-2 블렌드 생성기 `gen_phase2_blend.py` | **완료** | G-P1 실측 재산출(DRAFT2): 리플레이 84.2% · opencode_fixed 15.0% · identity 0.6% · safety 0.2ep · 누적 SWE 1.20/chat 2.27. identity_v2 stats 확보 후 최종(G-P4) |
| preset `sft_128k_full_p2.yaml` | **완료** | phase-1 과 6키 차이(train-samples·warmup·lr·min-lr·save/eval·load). `bash train.sh baseline_48L sft_128k_full_p2 sft_128k_mixed_blend_p2` |
| 카드 1.2 + 생성기·검증기·시드·export·merge | **완료** | 카드 로드 APPROVED, 규칙 9 단위 검증 8/8(조직 선행·단독·구성원 누락/과잉 탈락, 정답 2형 통과), 시드 2,000행 = creator_individual 1,067(lead_only 544/all_members 523) + creator_org 933 |
| 프로브 하니스 `eval_sft/identity_probe.py` | **완료** | 제작자 30(ko 20/en 10, 멀티턴 2)·누출 20, 8 병렬 50문항 ≈24s, 이름·조직은 카드에서 로드. 외부 Gemma 스모크: 제작자 FAIL·누출 0/20 — 기대대로. 실행 `python3 eval_sft/identity_probe.py --base-url http://HOST:8001/v1 --model alpha [--thinking] --out results/…json` |
| G-P1 opencode_fixed 재변환 + 게이트 | **완료 2026-09-01 16:43~16:50 (sub1, 98 workers)** | 행·샘플 460,254 동일 · **trainable 1,205,583,165 불변** · real 7.148B → 6.912B(−3.3%, 봉투 제거) · bins 55,238 → 53,425 · fill 98.7% · drops 0. `verify_sft_bins --tree p2` **25/25 OK, PASS**. 렌더 육안(규칙 9): doc 0/1000/40000 `<tool_response>` 평문·실제 줄바꿈, 봉투 흔적 없음 → `opencode_fixed/RENDER_CHECK.md`. 블렌드 재산출(DRAFT2): 리플레이 84.2% · opencode_fixed 15.0%(1.728B, 0.25ep) · identity_v2 0.6% |
| G-P2 identity_v2 생성·교체·변환 | **완료 08:33** — 2,000 시드 → 규칙 1,896 → 심판 1,747 → qa 중복 1,600 → 버킷 1,145행(ci 635·co 510). v2 = 7,220 train + 400 eval, creator 축 15.0%(lead_only 286/all_members 309). ×12 → **111 bins, verify PASS**, 전 bin 렌더 스캔 신형식 3,456건·v1 형식 0건(`identity_v2/RENDER_CHECK.md`). 사고 2건: merge 는 probe 단일 디렉터리만 받음(첫 시도 무효 → 분리 후 재실행) · assistant 기준 중복 제거가 고정 답을 깎음(→ `--dedup-scope qa`) | 교사 = 외부 엔드포인트 `https://gemma4.withai.cj.net:10206/v1`, 모델 `google/gemma-4-12B-it`(사용자 제공 09-01, 인증 불필요, 0.7s/req, 221 rpm). **파일럿 ko 60행**: creator_individual 44/44 가 "저를 만든 사람은 CJ주식회사 AI/DT추진실 영상콘텐츠담당 이동호…" 한 문장·팀명 포함, 게이트 41/44(잔여 = `false_solo_claim` 오탐 2 "한 명이 아닌 두 명"). 규칙 9 는 **첫 assistant 턴만** 검사로 정정(멀티턴 후속 답의 이주성 언급은 정상). 로그 `sdg/identity/gen_creator_v12.log` |
| G-P3 F3 복원 재시도 | 대기 | gated 원천 접근 토큰 |

identity_v2 슬라이스 재생성 명령(README §"Identity Card 변경 시" 2026-09-01 절):
```bash
cd examples/alpha/sdg/identity
uv run prepare_seed.py --num-records 2000 --only-probe creator_individual --only-probe creator_org --no-bank --out seed_creator_v12.parquet
uv run identity_sdg.py --vllm-endpoint http://HOST:8000/v1 --model <served> --seed-path seed_creator_v12.parquet \
    --num-records 2000 --dataset-name alpha_identity_creator_v12 --no-tui
uv run export_sft.py --dataset 'artifacts/alpha_identity_creator_v12/**/*.parquet' --out-dir out_creator_v12 --holdout 0 --revalidate
V2=/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Identity-v2
cp -r /home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Identity-v1 "$V2"     # v1 보존(phase-1 출처)
uv run merge_probe_slice.py --probe creator_individual --new-dir out_creator_v12 --dataset-dir "$V2"
uv run merge_probe_slice.py --probe creator_org        --new-dir out_creator_v12 --dataset-dir "$V2"
```

## 10. iter 1200 데이터 교체 재개 (2026-09-01 사용자 결정 — phase-2 연속형 대체)

### 10.1 왜 가능한가·왜 나은가

| 항목 | 사실 |
|---|---|
| 재개 시 블렌드 교체 | Megatron 251125 `check_checkpoint_args` 는 num_layers·hidden·heads·TP/PP·tokenizer 만 비교 — **data_path 미비교**. `consumed_train_samples`(192,000) 복원 → 새 블렌드 인덱스 192,000번부터 소비, 샘플 기반 LR 스케줄 승계(1200 시점 ≈1.6e-5 → 1.5e-6) |
| 선례 | LC-B resume(`lc_b_resume.yaml`): 자기 ckpt optimizer state 로드, finetune 없음 |
| 체크포인트 | optimizer 동봉 save 300 iters 마다. **iter_0001200 ≈ 09-01 20:14**(로그 시계) |
| 효과 | 추가 GPU 0(총 2,448 iters 그대로, ~09-06 종료). opencode repr 노출 0.31→0.15ep + 정상 0.20ep. identity v1 180→91회에서 중단, v2 0.3%×26.2B ≈ 66회를 **1,248 iters × LR 1.6e-5→1.5e-6** 에 — phase-2(550 iters ≤1e-5)의 ≈3배 신호 |
| 누적 epoch | 미변경 24셋은 phase-1 가중치 유지 → SWE 1.00·chat 1.88·cp 0.45·math 0.86·safety 4.29 = 설계값 |

### 10.2 산출물

| 파일 | 내용 | 상태 |
|---|---|---|
| `configs/training/sft_128k_full_swap.yaml` | phase-1 preset + `load`=자기 ckpt, **finetune 제거** | 커밋 cd4afda |
| `configs/data/sft_128k_mixed_blend_swap.yaml` | `gen_phase2_blend.py --iters 1248 --map opencode_v1=opencode_fixed --drop identity_v1 --add identity_v2 --ep opencode_fixed=0.20 --share identity_v2=0.003 [--add chat_v3_chat_restored --ep …=1.9]` | identity_v2 bins 후 산출 |
| identity_v2 | 카드 1.2 creator 슬라이스 재생성(외부 gemma-4-12B-it) → v2 디렉터리 교체 → ×12 → 변환 | **완료** (§9 G-P2) |
| chat_v3_chat_restored | `prepare_chat_prompts_full.py`(WildChat-1M-Full, `.env` HF_TOKEN) 로 재복원 → 회수분만 별도 셋 | 복원 진행 중 (08:23~). 20:00 까지 안 되면 제외 |
| 오케스트레이션 | tracker==1200 대기 → 블렌드·preset sanity(실패 시 학습 유지) → 중단 직전 loss 기록 → SIGTERM → GPU 해제 확인 → `train.sh … sft_128k_full_swap sft_128k_mixed_blend_swap` → 1201~1205 loss 출력 | **가동 중(08:35~, 백그라운드)** — 로그 `outputs/swap1200_<ts>.log`, sanity 는 중단 직전 재실행 |

### 10.3 게이트

| 게이트 | 기준 |
|---|---|
| 데이터 | identity_v2·(chat_restored) `verify_sft_bins` PASS, 렌더 육안(규칙 9), 블렌드 합 1.0·전 경로 idx 존재 |
| 재개 | 1201~1205 train loss ∈ 1200 시점 ±0.05, LR 연속(≈1.6e-5), traceback 0. 실패 시 구블렌드로 재개(손실 수 분) |
| 종료(2448) | `identity_probe.py` 제작자 ≥95%·누출 0, T1 ±1pp, 에이전틱 ≥ iter1200 기준선. 미달 시 §1~§2 보정 스테이지 |

주의: valid split 이 블렌드와 함께 바뀌어 valid loss 는 1200 에서 불연속(비교는 train loss). 출력 디렉터리·wandb 런은 `…_sft_128k_full_swap_<ts>` 로 분리 — STATUS 에 계보 기록.
