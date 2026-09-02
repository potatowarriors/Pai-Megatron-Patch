# 512K 추론 확장 — 판정 그리드 (2026-09-01 착수)

목표: **512K(524,288) 추론 창**. 학습이 아니라 추론 측 RoPE 스케일링(YaRN)으로 먼저 푼다.
결정 근거·경로 판정은 이 문서 §1~§2, 실행은 §3, 판정 규칙은 §4, 결과는 §5.

## 1. 출발점 (실측)

| 항목 | 값 | 출처 |
|---|---|---|
| 학습 최장 길이 | 131,072 (LC-B 128K@CP8 → SFT 128K 진행 중) | STATUS.md |
| NIAH, LC-B base, HF 경로 | 196K 100% · **262K 95%** · **393K 0%(전 깊이)** · 524K 측정 불가(fla 작업공간 73GB) | `lc_b_final_eval.md` §2·§5 |
| RULER, SFT iter600, vLLM (64K/128K/258K) | single_1 100/100/100 · single_2 100/90/**25** · multikey 50/45/25 · multivalue 26/29/24 | `eval_sft/results/…iter0000600` |
| vLLM 상한 | `max_model_len ≤ original_max_position_embeddings × factor` (yarn) 또는 `max_position_embeddings` | `vllm/config/model.py` |
| KV | 12KB/token → 512K 시퀀스 6.0GiB. H100 1장 KV 3.2M tok(실측) → 동시 6 | `SFT_BENCHMARKS.md` |
| 프리필 (추정) | attention 6층 O(n²) 13.5 PFLOP + linear 1.9 → H100 ~30s, A100 ~80s | 계산 |

393K 붕괴가 **깊이 무관 0%** 인 것은 attention 의 RoPE 외삽 실패 패턴이다(저주파 dim 의 OOD
로짓이 attention mass 를 가져간다). GDN state 붕괴라면 깊이 의존적이어야 한다. θ=10M·rotary 64dim
의 32 주파수 쌍 중 128K 에서 한 주기를 완주한 쌍은 20개 — 나머지 12쌍이 256K 너머에서 미학습
상대위치를 만난다. YaRN 은 정확히 그 저주파 쌍만 보간한다(orig 262144·β 32/1 → 램프 dim 14~21).

## 2. 경로 판정

| 경로 | 판정 | 근거 |
|---|---|---|
| A. 추론-only YaRN (config.json 프로파일) | **1순위 — 이 그리드** | 256K 실측 창을 2배 보간. 코드 변경 0 (HF·vLLM 모두 config 에서 읽음) |
| B. YaRN 적응 CPT (LC-C, 128K seq + s=4) | A 미달 시 | 128K@CP8 기존 인프라, ~320 iters ≈ 1.5일 + Megatron `mamba_model.py` yarn 배선 패치 |
| C. 네이티브 256K+ 학습 | 불가 | 128K@CP8 54.9~58.8GB(offload) + 실측 기울기 +20GB/8K tok·rank → 256K ≈ 95GB. CP>8 은 IB 없는 2노드 |

## 3. 인프라 (2026-09-01 구축)

| 구성요소 | 위치 | 검증 |
|---|---|---|
| 프로파일 도구 | `tools/set_long_context_config.py` — 가중치 symlink + config.json 교체. `ext`/`yarn`. `factor×original==max_pos` 강제, `--check --hf-probe` | CPU 프로브: transformers 5.16.1 `rope_parameters` 표준화 ✓ · vLLM 0.25.1 `ModelConfig` yarn2 @524288 수용, 원본 거절 ✓ · HF rotary 램프 dim 14~21 보간·저주파 ×1/factor ✓ |
| RULER 512K 태스크 | `eval_sft/tasks/ruler_niah_*_512k.yaml` (4종), 구간 `[131072, 258048, 393216, 520192]` — `_aa` 와 별개 태스크 | `ruler_utils.SEQ_SETS` 등록·센티넬 유닛 ✓, lm_eval 태스크 인식 ✓. `_aa` 경로 불변(진행 중 suite 무영향) |
| 레지스트리 | `eval_sft/bench_registry.py` — 대표지표 = 520192 단일 구간 | headline 유닛 ✓ |
| 그리드 런처 | `scripts/lc512k_grid.sh` — sub1 suite/fleet 종료·GPU 유휴 대기 → 셀별 fleet 524288 → 게이트 → RULER → (SFT) 단문 표본 → 요약 | 셀 순서: sft:yarn2 → sft:ext → sft:yarn4 → base:yarn2 → base:ext → base:yarn4 |
| 요약기 | `scripts/lc512k_summarize.py` → `outputs/lc512k_eval/SUMMARY.md` | — |

프로파일 3종: **yarn2** = s=2 / original 262144 (실측 창 기준, mscale 1.069) · **ext** = 순수 외삽
(393K 절벽 대조군) · **yarn4** = s=4 / original 131072 (학습 길이 기준, YaRN 정석, mscale 1.139).
모델 2종: SFT 최신 변환본(iter900) + LC-B base iter320 (SFT 가 롱컨텍스트를 깎는지 분리).
결과는 `eval_sft/results/` 밖(`outputs/lc512k_eval/`)에 둔다 — TRACKING 오염 방지.

```bash
# sub1
cd examples/alpha && mkdir -p outputs/lc512k_eval
setsid nohup bash scripts/lc512k_grid.sh > outputs/lc512k_eval/grid.log 2>&1 < /dev/null &
tail -f outputs/lc512k_eval/grid.log          # 진행
touch outputs/lc512k_eval/STOP                # 중단 (현재 셀 후)
python3 scripts/lc512k_summarize.py outputs/lc512k_eval
```

비용: 셀당 fleet 기동 ~5분 + RULER 320문항(512K prefill ~30s, 8레플리카) ~10분 + (SFT) 단문
표본 ~15분 → 6셀 ≈ 2시간.

## 4. 판정 규칙 (2026-09-02 재정의 — 사용자 승인)

첫 셀 결과(single_1 이 520K 에서 100)가 보여줬듯 **single_1 은 검색 난도 0 인 메커니즘 탐침**
(반복 haystack 속 유일 특이 문장)이라 능력 지표가 못 된다. 그래서 판정을 두 층으로 나눈다.

**A. 메커니즘 게이트 (그리드, `_512k` 4태스크·n=20)** — "YaRN 채택" 여부만 판정:
1. ext 가 393216 에서 붕괴 재현 ∧ yarn 이 393216/520192 single_1 ≥ 90%
2. yarn 의 131K/258K 가 ext 와 동급 (보간이 기존 창을 해치지 않음)
3. yarn 의 단문 표본(ifeval·gpqa 100문항)이 ext 대비 ±3pp (static YaRN mscale 부작용 체크)
→ 통과 시: YaRN 프로파일 채택, **≤256K 원본 · >256K yarn 분리 서빙** (Qwen3 관례).

**B. 능력 판정 (§6 스위트, 11태스크·n=50)** — "L 지원" 선언은 프론티어 규약으로만:
**11태스크 구간 평균 ≥ 85 인 최대 길이 = 유효 컨텍스트** (RULER 논문 규약; Nemotron 3 Ultra
카드 게재 방식 — 참고: Ultra 는 512K 84.5 / 1M 76.8). 여기 미달이면 "512K 지원" 을 선언하지
않고, LC-C(YaRN 적응 CPT) 또는 SFT 완주 후 재측정을 사용자가 결정한다.

## 5. 결과

(그리드 완료 후 `outputs/lc512k_eval/SUMMARY.md`, 능력 스위트 완료 후
`outputs/ruler_cap_eval/SUMMARY.md` 를 옮겨 적는다.)

중간 (2026-09-02, 셀 1/6 `sft_yarn2` — iter900 + yarn s=2): single_1 100/100/**95/100**
(무보정 393K 는 0% 였던 지점) · single_2 90/45/10/0 · multikey 65/10/20/10 · multivalue
21/29/16/14 (131K/258K/393K/520K, no_answer 0%) · 단문 표본 ifeval 57.1 / gpqa 34.3.
판독: 메커니즘 게이트 A-1 충족 방향. 어려운 태스크의 완만한 길이 하락은 절벽형이 아니라
검색·state 쪽 — A/B 분리는 ext·base 셀에서.

## 6. 능력 스위트 (RULER 11태스크, 2026-09-02 구축)

`_aa`/`_512k` 규약(Reasoning-Off·seq_set 센티넬)을 RULER-13 으로 확장한 **별도 태스크**
`ruler_cap_*` (n=50/구간, tag `ruler_cap_512k`): niah single 1/2/3 · multikey 1/2/3 ·
multiquery · multivalue · vt(멀티홉) · fwe(빈도) · qa_hotpot(다문서 QA).

- **"라벨만 긴 측정" 2종 구조적 제외** (사전 빌드 게이트 `scripts/ruler_cap_validate.py`
  fill≥0.85 기준): ① qa_squad — SQuAD dev 문서 풀 ~0.27M 토큰 실측, 393K+ 미충족.
  ② cwe — wonderwords 어휘 풀 8,166개 고갈로 입력이 ~130K 포화 (fill 실측
  1.00/0.51/0.33/0.25). 나머지 5개 생성 경로는 fill 0.89~1.00 PASS (validate.log 09-02).
  qa_hotpot 은 HF `hotpot_qa/distractor` validation(풀 ~9M tok)로 소싱
  (stock 의 curtis.ml.cmu.edu 직다운로드는 이 클러스터에서 타임아웃).
- **길이 탐색 incremental 을 길이 비례로 상향** (vt/cwe L//2048, qa L//8192): stock 기본
  10 은 520K 에서 수천 회 전체 재토크나이즈(O(N²), 시간 단위). 정밀도 손실 ~0.05%.
- 실행: `ruler_cap_validate.py && ruler_cap_run.sh [cell]` 체인 (기본 sft:yarn2; lc512k 그리드
  DONE 대기 후 자동 개시, 소요 ~2.5h) → `scripts/ruler_cap_summarize.py` 가 §4-B 판정(평균 ≥85)까지 계산.
- 후속(미착수): MRCR — `SFT_BENCHMARKS.md` T2 계획에 있던 것이 yarn 프로파일로 512K 까지
  열림. 능력 스위트 결과 확인 후 착수 여부 결정.
