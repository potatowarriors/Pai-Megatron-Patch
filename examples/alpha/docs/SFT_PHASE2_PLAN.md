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

## 10. iter 900 데이터 교체 재개 (2026-09-01 사용자 결정 — phase-2 연속형 대체; 처음 1200 안을 900 즉시 교체로 변경)

> **2026-09-01 09:45 변경(사용자)**: iter_0001200 저장(20:14)을 기다리지 않고 **이미 저장된 iter_0000900 에서 즉시 교체**. 진행 중이던
> iter 901~1083(≈183 iters, ≈16h)은 되돌리고, 그 대신 수정 데이터가 약 11시간 먼저 들어간다. 잔여 1,548 iters(32.47B tok), consumed 144,000.
> 노출 재산: identity v1 ≈68회에서 중단 → identity_v2 0.43%×32.47B ≈ 140M tok ≈ 121회 · opencode repr 0.11ep + 정상 0.20ep = 0.31(원설계).
> 아래 본문의 1200 수치는 최초 안의 기록이며, 실행은 900 기준이다.
> **기동 이력(09-01)**: 09:50 `…095004`(no-load-optim 상속 → LR warmup 재시작, 1 iter 후 중단, ⑥) → 10:02 `…100244` + 10:06 `…100614`(재기동 스크립트 GPU 대기 타임아웃이 중단 없이 진행해 중복 기동; 미세척 identity_v2) → 사용자 지시로 reasoning 스캐폴딩 세척(⑤) → **10:15 `…swap_20260901_101523` 최종**.

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
| `configs/data/sft_128k_mixed_blend_swap.yaml` | **phase-1 yaml 과 가중치 비트 동일, 경로 2개만 치환**(opencode_v1→opencode_fixed, identity_v1→identity_v2) — §10.4 순서 보존. 잔여 노출 opencode_fixed 0.16ep(누적 0.31 = 원설계)·identity_v2 0.43%(≈97회) | **커밋 f46b361** (재정규화 판 517ca30 은 §10.4 사유로 폐기) |
| identity_v2 | 카드 1.2 creator 슬라이스 재생성(외부 gemma-4-12B-it) → v2 디렉터리 교체 → ×12 → 변환 | **완료** (§9 G-P2) |
| chat_v3_chat_restored | `prepare_chat_prompts_full.py`(WildChat-1M-Full) 재복원 시도 | **제외 — 회수 불가**: lmsys 5,110 해시 공개판 부재, WildChat-Full 은 토큰 계정 미승인(403). 88.2% 정본 (`KNOWN_ISSUES` ③). 승인 후 후속 스테이지에서 편입 가능 |
| 오케스트레이션 | tracker==1200 대기 → 블렌드·preset sanity(실패 시 학습 유지) → 중단 직전 loss 기록 → SIGTERM → GPU 해제 확인 → `train.sh … sft_128k_full_swap sft_128k_mixed_blend_swap` → 1201~1205 loss 출력 | **가동 중(08:35~, 백그라운드)** — 로그 `outputs/swap1200_<ts>.log`, sanity 는 중단 직전 재실행 |

### 10.3 게이트

| 게이트 | 기준 |
|---|---|
| 데이터 | identity_v2·(chat_restored) `verify_sft_bins` PASS, 렌더 육안(규칙 9), 블렌드 합 1.0·전 경로 idx 존재 |
| 재개 | 1201~1205 train loss ∈ 1200 시점 ±0.05, LR 연속(≈1.6e-5), traceback 0. 실패 시 구블렌드로 재개(손실 수 분) |
| 종료(2448) | `identity_probe.py` 제작자 ≥95%·누출 0, T1 ±1pp, 에이전틱 ≥ iter1200 기준선. 미달 시 §1~§2 보정 스테이지 |

주의: valid split 이 블렌드와 함께 바뀌어 valid loss 는 1200 에서 불연속(비교는 train loss). 출력 디렉터리·wandb 런은 `…_sft_128k_full_swap_<ts>` 로 분리 — STATUS 에 계보 기록.

### 10.3b 교체 전 기준선 — iter 900 정체성 프로브 (2026-09-01, sub1 fleet = hfmodel_0000900)

| 프로브 | thinking off | thinking on | 판정 |
|---|---|---|---|
| 제작자 30문항 (이동호 명시·조직 포함·회피 아님) | **4/30 (13%)** | **5/30 (17%)** | FAIL — 87% 가 조직만 답함("CJ주식회사 AI/DT추진실에서 개발한 alpha-banana-v1"). v1 정책 그대로 |
| 누출 20문항 (정체성 무관 질문에 자기소개 0) | 0/20 | 0/20 | PASS — 결함 ②의 "무엇을 물어도 자기소개" 과적합은 **없음** |

thinking on 원문에 "identity-facts에 따라 … 개인 개발자 이름은 언급하지 않아야 함" 이 반복된다 — 정책 내재화 + 교사 스캐폴딩 용어 누출
(`KNOWN_ISSUES` 09-01 ⑤). 결과 파일 `eval_sft/results/identity_probe/iter0900_thinking_{off,on}.json`. 교체 후 100 iters 마다 같은 프로브로 상승 곡선을 잰다.

### 10.4 순서 보존 원칙 — 가중치를 바꾸지 않는다 (2026-09-01, 사용자 질문으로 발견)

| 사실 (Megatron-LM-251125) | 근거 |
|---|---|
| 재개 시 샘플러는 `range(consumed_samples, total)` 로 **인덱스만** 이어 간다 (`dataloader_type=single`, 무작위 없음) | `legacy/data/data_samplers.py:119` |
| 블렌드 인덱스(어느 샘플이 어느 셋인가)는 **가중치·총량만**의 함수 | `helpers.build_blending_indices(…, weights, num_datasets, size)` |
| 각 셋의 셔플 순열은 (요청 샘플 수 = 가중치×총량, seed 1234)의 함수 | `gpt_dataset.py:420 RandomState(random_seed)` |

따라서 **가중치가 phase-1 과 비트 단위로 같으면** 24 미변경 셋의 순열이 그대로 재생성되고, 192,000번부터 읽는 것이 진짜 연속이다
(경계 중복·누락 0). 반대로 재정규화한 첫 swap yaml(517ca30)은 새 순열을 만들어 앞 1200 iters 와 무관한 샘플을 뽑는다 — 예컨대
SWE 1.00ep 앵커는 첫 0.49ep(무작위 49%)와 새 순열의 0.51ep 가 독립이라 기대상 **≈25% 문서가 2회, ≈25% 가 0회**로 깨진다.
교체 멤버 2개는 전임자의 가중치(자리)를 승계하고 내부 순열만 새로 만든다(내용이 새것이라 무해). 대가: opencode 부스트(0.20→0.16ep)와
identity 감량(0.3%→0.43%) 포기 — 순서 보존이 우선. 캐시는 `.cache/sft_128k_mixed_blend` 를 `_swap` 으로 복사해 미변경 셋 해시 재사용.

## 11. phase-2 재정의 — 능력 추가 스테이지, 절충안 M (2026-09-04 사용자 승인)

§1~§2 의 보정 목적(opencode 형식·identity 덮어쓰기)은 §10 의 iter 900 교체 재개가 phase-1 안에서 소화한다. 따라서 phase-2 는
**보정이 아니라 능력 추가**로 재정의한다. 발단은 Agentic-v2 검색 데이터였고(`SFT_RL_DATASETS.md` §2.3 번복 근거), 사용자가
"SFT 를 한 스테이지로 설계했으니 phase-2 의 비율이 곧 최종 분포"라고 못박아 미사용 16종 전체를 검토했다(같은 문서 §2.8).

### 11.1 원칙 세 줄

| 원칙 | 내용 | 근거 |
|---|---|---|
| 형제 대체 | 미사용 셋 중 phase-1 카테고리의 형제(Chat-v2, SWE-v2/v1, Science-v1)는 **리플레이를 대체**한다. 카테고리 비중은 유지하되 같은 문서 반복을 피한다 | Chat v2→v3 프롬프트 겹침 5.1%, Science/Agentic 0% — 형제는 독립 표본 |
| 카테고리별 리플레이 | verbatim 리플레이는 카테고리별로 정한다: 형제 있는 카테고리 = phase-1 비중의 **0.15**, 대체재 없는 대형(cp·math) **0.5**, 대체재 없는 소형(한국어·IF+effort·ml ko/ja/pt·identity) **1.0**, safety 는 E_max 상한 0.2ep | 평평한 15% 는 한국어 5.0%→0.7%, IF 4.7%→0.7% 로 붕괴 — 망각은 gradient 분포 이동에서 온다. cp·math 는 phase-1 소비 0.45/0.86ep 라 "리플레이"의 절반 이상이 새 문서 |
| 예산은 결과 | iters 는 손으로 정하지 않고 위 규칙에서 푼다(`gen_phase2_blend.py --solve-iters`) | 비율이 우선, 일수는 그 결과 |

구세대 대형 셋(Math-v2 71B·CP-v1 55B·Math-v3 51B)과 Chat-v1·Safety-v1·Agentic-v1 은 제외 — 같은 도메인의 최신 셋이 아직 sub-1 epoch 이라
epoch 를 올리는 편이 싸고 품질이 높다. LR 1e-5(phase-1 peak 의 0.4×)가 망각 방지 1차 장치라는 §2 원칙은 그대로.

### 11.2 신규 멤버와 epoch

| 멤버 | 원천 | epoch | 역할 |
|---|---|---|---|
| agentic_v2_search | Agentic-v2 search (held-out 300 제외 5,668행, real 135M) | **2.0** | 웹 검색 에이전트 — Ultra 가 retain 한 유일 공개 검색 셋 |
| agentic_v2_ia | Agentic-v2 interactive_agent (278,880행, real 1.56B, reasoning 93%) | 0.25 | 다중턴 고객응대 tool-use |
| agentic_v2_tc | Agentic-v2 tool_calling (707,052행, real 3.92B) | 0.10 | 단일/다중 스텝 함수 호출 |
| chat_v2_on / chat_v2_off | Chat-v2 reasoning_on 929k행 / reasoning_off 1.07M행 (off = 빈 `<think></think>` no-think 규약) | 0.3 / 0.3 | chat_v3 리플레이 대체 |
| swe_v2_openhands / swe_v2_agentless | SWE-v2 swe.jsonl(reasoning 0%) / agentless.jsonl | 0.3 / 0.3 | swe_v3 리플레이 대체 |
| swe_v1_r2e | SWE-v1 r2e_gym (reasoning 0%) | 0.3 | 〃 |
| science_v1 | Science-v1 MCQ+RQA (226k행, real 0.75B) | 1.0 | science_v2 리플레이 대체 |
| finance_v1 | Finance-v1 (trainable 4.9%) | 0.1 | 도메인 확장 |
| math_proofs_v1_lean | Math-Proofs-v1 lean (`messages=="[]"` ≈33% bad_row 드롭) | 0.05 | 형식 수학 — RL `math_formal_lean` 대비 |
| ml_super-v3_{code,math}_{de,es,fr,it,ja,zh} | Multilingual-v1 12 파일 (전부 `ALPHA_LANGS`) | 0.05 | 다국어 확장 |

리플레이 scale 은 §11.1 표대로 멤버별로 준다(정확한 인자는 `toolkits/sft_data_preprocessing/gen_sft_128k_mixed_blend_p2.sh`).

### 11.3 산출 (2026-09-04, `gen_sft_128k_mixed_blend_p2.sh` 실행)

**602 iters × GBS 160 = 12.62B bin-tok = 96,320 samples ≈ 2.3일(329.7 s/iter)**. 49 멤버, 가중치 합 0.999999, 전 경로 idx 확인.
phase-1 verbatim 리플레이 41.4%(헤더 기준, safety 0.2% 는 ep 고정으로 분류) / 신규(형제 대체·능력 추가) 37.4% / 확장 도메인 21.0%.

| 카테고리 | phase-1 | phase-2 | 신규 / verbatim | 토큰 |
|---|---|---|---|---|
| 에이전틱 (swe_v3·opencode·arc + SWE-v2/v1·Agentic-v2) | 26.4% | **27.2%** | 23.2 / 4.0 | 3.44B |
| CP | 23.7% | 11.9% | 0 / 11.9 | 1.50B |
| Chat (chat_v3 + Chat-v2 on/off) | 8.3% | 9.5% | 8.3 / 1.3 | 1.20B |
| Science (science_v2 + Science-v1) | 11.6% | 7.7% | 5.9 / 1.7 | 0.97B |
| Math (math_v4·proofs_v2) | 13.3% | 6.7% | 0 / 6.7 | 0.84B |
| ml ko/ja/pt | 5.7% | **5.8%** | 0 / 5.8 | 0.73B |
| 한국어 (kochat×3) | 5.0% | **5.0%** | 0 / 5.0 | 0.63B |
| IF+effort (if_me·budget×2) | 4.7% | **4.7%** | 0 / 4.7 | 0.60B |
| Identity | 0.4% | 0.4% | 0 / 0.4 | 0.05B |
| Safety (E_max 상한 0.2ep) | 0.9% | 0.2% | 0 / 0.2 | 0.02B |
| 확장: Multilingual-v1 / Finance-v1 / Lean | — | 9.3 / 7.5 / 4.2% | 전부 신규 | 1.18 / 0.95 / 0.53B |

누적 epoch(헤더 `cum`): swe_v3 1.04 · chat_v3 1.96 · cp 0.51 · math_v4 0.97 · science_v2 0.42 · kochat 2.36 · if_me 2.36 · ml ko/ja/pt 1.25 ·
safety 4.51 · 신규 멤버는 §11.2 의 epoch 그대로. 데이터 게이트: `verify_sft_bins --tree p2 --seq-length 131072` **49/49 OK PASS**,
`render_check.py` 23 신규 멤버 전부 봉투 흔적 0·`<think>` 수 = assistant 턴 수(각 멤버 `RENDER_CHECK.md`). 변환 드롭은 전부 설계 사유
(lean bad_row 455,782 = `messages=="[]"`, tool_calling render_error 21·bad_row 5, injection 7~81, too_long ≤149).

### 11.4 데이터 산출물·게이트

| 항목 | 내용 | 게이트 |
|---|---|---|
| 재다운로드 | `Agentic-v2/data/tool_calling.jsonl` 절단본(8,444행) → HF 정본 707,052행(14,941,561,688 B 일치), 절단본은 `.truncated_8444rows_20260904` | 크기·행수 일치 |
| held-out | `Agentic-v2/splits/search_heldout300.jsonl`(seed 20260904) — 학습 금지, 검색 게이트 전용. 학습 입력은 `splits/search_train.jsonl` 5,668행 | — |
| 변환 | `convert_sft_128k_mixed_p2b.sh`(sub1, NCORES 180) → p2 트리에 23 멤버 추가 | `verify_sft_bins --tree p2 --seq-length 131072` 전 PASS |
| 렌더 육안(규칙 9) | `render_check.py --tree p2 --write` → 각 멤버 `RENDER_CHECK.md` | 봉투 흔적 0, `<think>` 수 = assistant 턴 수(interleaved 보존) |
| 블렌드 | `gen_sft_128k_mixed_blend_p2.sh` → `sft_128k_mixed_blend_p2.yaml`(§1 의 보정 설계판을 대체) | 합 1.0·전 경로 idx·헤더 epoch 표 |
| 프리셋 | `sft_128k_full_p2.yaml`: load = 교체 재개 런 ckpt, train-samples 96,320(602 iters), 나머지 §2 | G-P5 = **main1 개시 시 첫 iteration 게이트**(아래) |
| **G-P5 PASS (2026-09-05 17:46~18:01 sub1, compat 570)** | `sub1_compat_smoke.sh --now`: iter1 lm loss **0.7114** → iter2 **0.7335**(phase-1 iter 1936 시점 ≈0.70~0.72, 기준 +0.1 이내), grad norm 0.56, nan 0, max allocated **55.9GB**(128K·CP8 포락선 55~59GB), iter2 322 s/iter·269 TFLOP/s(phase-1 323~333 s 동일), traceback 0. 종료 후 compat 595 자동 복원 확인. 로그 `outputs/smoke_p2_sub1_compat570_20260905_174643.{log,summary.txt}` |
| (경위) | 2026-09-04 sub1 스모크 3회 전부 첫 스텝 SIGABRT — **sub1 compat libcuda 595 스왑 때문(`KNOWN_ISSUES` 09-04), 데이터·프리셋 무관**(phase-1 데이터도 동일 실패). 로드·49 멤버 인덱스 빌드·블렌드 인덱스는 정상 완료 → 캐시 398 파일 프리빌드. 스모크는 `scripts/launch_p2_after_phase1.sh` 가 본 런 첫 iteration 에서 대신 판정(loss 유한·Traceback 0, 실패 시 `P2_CHAIN_ALERT.txt` + HOLD) | **사용자 결정(09-04): sub1 fleet 은 09-05 종료 → 그 뒤 정식 스모크.** `scripts/sub1_compat_smoke.sh --wait-after 1788566400`(= 09-05 09:00 KST) 를 **sub1 에서 사용자가 arm**(sudo 로 시스템 symlink 를 바꾸는 스크립트라 Claude 세션 권한 정책이 원격 기동을 차단): 20분 유휴 확인 → symlink 570(sudo) → 2-iter 스모크 → 판정 summary → 595 복원(trap). 로그 `outputs/sub1_compat_smoke_watcher.log`, 결과 `outputs/smoke_p2_sub1_compat570_<ts>.summary.txt` |

### 11.5 평가 게이트 (G-P6 기준선 → G-P7 100 iters 마다 → 종료)

| 게이트 | 내용 | 기준 |
|---|---|---|
| 검색 format | `eval_sft/search_agent_eval.py --gate format` — held-out 300 의 teacher-forced 접두부에 대한 tool-call/answer 파싱률 | ≥ 99% |
| 검색 live | `--gate live --backend tavily` — held-out 문항을 실제 검색으로 풀어 ground_truth 대조. phase-1 최종 ckpt 가 기준선 | phase-2 > phase-1. Tavily 는 문항당 평균 10.6회(예산 ≈13회). **키는 `examples/alpha/.env` 의 `TAVILY_API_KEY`(사용자 제공 09-04, 1회 호출로 유효 확인)** |
| 유지 | T1(MMLU-Pro·GPQA-D·IFEval) ±1pp, **LogicKor(한국어)·IFEval 무회귀**, 에이전틱 SWE·Terminal ≥ phase-1, 제작자 프로브 ≥95%·누출 0 | 기존 G-P7 + 한국어·IF 명시 |

### 11.6 일정

phase-1(교체 재개) 2448 종료 ≈ 09-07 17:30 KST(iter 1692 @ 09-04 10:28 UTC, 333 s/iter) → **자동 연계**(`scripts/launch_p2_after_phase1.sh`,
2026-09-04 main1 에 arm: 완주 대기 → GPU 유휴 → sanity → 본 런 → 첫 iteration 게이트) → 602 iters ≈ 2.3일 → ~09-10 판정.
G-P6 기준선(HF 변환·T1·에이전틱·프로브·검색 게이트)은 sub1 fleet 에서 병행. 해제: `pkill -f "[l]aunch_p2_after_phase1.sh"`.
데이터·블렌드·하니스·캐시 프리빌드는 09-04 완료.

### 11.7 주의·알려진 특성

- chat_v2_off·swe_v2_openhands·swe_v1_r2e 는 reasoning 이 없어 `<think></think>` no-think 타깃으로 렌더된다(opencode 선례와 동일 판단).
  agentic_v2_ia/tc 의 빈 think 턴 6.9%도 같다.
- finance_v1 은 trainable 4.9% — 예산 대비 신호가 작다. 0.1ep 는 도메인 노출 목적.
- `identity_v2`·`opencode_fixed` 의 헤더 ep_p1 은 전 구간 환산 명목값이다(실제 편입은 iter 900 부터).
- Agentic-v1 tool_calling 은 변환기가 `bool` 필드에서 크래시 — 제외했고 수정하지 않았다(편입 가치 낮음).
- 데이터 캐시: 셋별 인덱스 캐시의 키에 요청 샘플 수(= 가중치 × 총량)가 들어가므로 가중치가 바뀐 phase-2 에서는 `_swap` 캐시를 재사용할 수
  없다. 49 멤버 콜드 빌드(26 멤버 25분 실측 기준 ≈45분)를 첫 기동 시간에 넣는다. `distributed-timeout-minutes: 180` 유지.
