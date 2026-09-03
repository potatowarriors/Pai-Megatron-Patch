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

## 5. 결과 — 그리드 완주 (2026-09-03 02:41, 6셀 전체)

정본 수치: `outputs/lc512k_eval/SUMMARY.md` (n=20/셀, no_answer 전 셀 0%). 아래는 판독용 발췌.
구간 순서 131K/258K/393K/520K.

| 태스크 | sft_ext (무보정) | sft_yarn2 | **sft_yarn4** |
|---|---|---|---|
| single_1 | 100/100/95/100 | 100/100/95/100 | 100/100/100/100 |
| single_2 | 75/35/5/0 | 90/45/10/0 | **90/70/30/25** |
| multikey_1 | 60/20/15/0 | 65/10/20/10 | 35/30/35/**40** |
| multivalue | 29/25/16/4 | 21/29/16/14 | 32/29/20/15 |
| 단문 ifeval/gpqa | 56/39 | 57/34 | 54/38 |

**판독** (SFT iter900 3셀 = 동일 하니스 내적 타당 A/B):

1. **single_1 은 무판별** — ext 도 520K 까지 95~100. 셀 1 시점의 "YaRN 이 절벽을 옮겼다"
   해석은 대조군이 기각했다. SFT iter900 은 무보정으로도 single_1 절벽이 없다.
2. **YaRN 효과는 어려운 검색에서, 보간 강도 순으로 단조**: single_2 520K 0→0→**25**,
   multikey 520K 0→10→**40** (ext→yarn2→yarn4). 위치 보간이 실제로 작동하는 증거는 이쪽.
3. **yarn4 단문 무회귀** (ifeval −2 / gpqa −1 vs ext, §4-A-3 통과). 유일한 관찰 대가는
   multikey@131K 35 (ext 60) — n=20 노이즈(±11pp)를 넘는 폭, 능력 스위트(n=50)에서 재확인.
4. **base 3셀은 판독 제한** — instruct 미학습 모델에 chat 템플릿 하니스라 포맷 교란이 지배
   (base single_1@131K 25~65% vs 같은 모델 HF raw-completion NIAH 200/200; base multivalue@131K
   51~65% 로 SFT 보다 높게 나오는 역전 = 장황 출력의 부분점수 인플레 의심). 기존 "393K 전 깊이
   0%"(base+HF 하니스)가 모델 차이였는지 하니스 차이였는지는 **이 데이터로 분리 불가** —
   필요 시 base 를 raw-completion 하니스로 재측정.
5. §4-A 게이트 판정: A-1 은 전제(ext 절벽 재현)가 single_1 기준으로 불성립 — 무판별 태스크로
   내려가고, **"YaRN 채택" 근거는 §4-B 능력 평균의 ext 대비 우세로 대체**한다(그리드 2번 항목이
   방향을 이미 보여줌). A-2: yarn2 유지 OK, yarn4 는 multikey@131K 관찰 건. A-3: 둘 다 통과.

절대 수준 주의: yarn4 로도 520K 는 single_2 25·multikey 40·multivalue 15 — "512K 지원" 선언
가능성은 낮고, 능력 스위트(§6)가 공식 수치를 낸다. 능력 런 ① sft:yarn2 (02:43 개시) →
② sft:yarn4 자동 체인.

### 5.1 능력 스위트 결과 + 종합 판정 (2026-09-03 11:07 완료 — 트랙 1차 종결)

11태스크 구간 평균 (n=50, no_answer 전 셀 0%; 정본 `outputs/ruler_cap_eval/SUMMARY.md`):

| cell | 131K | 258K | 393K | 520K |
|---|---:|---:|---:|---:|
| sft_yarn2 | **48.1** | 34.2 | 26.1 | 18.1 |
| sft_yarn4 | 45.7 | **37.1** | **31.8** | **24.8** |
| (판정 ≥85) | ❌ | ❌ | ❌ | ❌ |

**판정 1 — "L 지원" 선언은 전 구간 불가, 병목은 위치가 아니라 능력.** 학습 길이 131K 에서
이미 45~48 (기준 85; Nemotron 3 Ultra 128K 92.5). multikey_2/3 은 전 프로파일·전 구간 0~4
(교란-needle 속 정확한 key 결합 — 위치 무관 능력 항목, SFT 37% 지점의 미성숙). RoPE 처방으로
도달 불가한 영역이므로 **선언은 SFT 완주 ckpt 재측정(~2.5h/셀)으로 연기**.

**판정 2 — YaRN 은 채택, 프로파일은 yarn4, 단 >256K 전용 분리 서빙.**
- 용량-반응이 n=50 평균에서도 유지: 학습 길이 너머 전 구간 yarn4 > yarn2 (+2.9/+5.7/+6.7pp),
  그리드에서 yarn ≫ ext. 태스크 단위 극명한 예: single_3@520K 34 vs 4, single_2@258K 78 vs 56.
- **비용은 131K 집계형 태스크에 집중**: yarn4 의 vt 28(yarn2 44)·fwe 42.7(60.7) — 중대역
  주파수 압축이 창 내부의 미세 위치 분해능(추적·빈도 집계)을 깎는 것으로 해석. 그래서
  "항상 yarn4" 는 기각, **≤256K 원본 / >256K yarn4** 2-프로파일 확정 (Qwen3 관례와 동일).
- yarn2 는 중간값으로 실익 없음 — 채택 안 함.
- 그리드의 yarn4 multikey@131K 35 관찰(-25pp)은 n=50 에서 42 vs 46 으로 **재현 안 됨** — 노이즈 판정.

**미결 (SFT 완주 후 재평가)**: ① multikey_2/3 전멸이 능력인지 포맷 마찰인지 ② 기존 "393K
0%"(base+HF)의 모델/하니스 분리 ③ MRCR 착수 ④ LC-C(YaRN 적응 CPT) — 능력 병목이 해소된
뒤에야 위치 항의 잔여 크기를 잴 수 있으므로 보류.

**재측정 절차 (SFT 최종 ckpt)**: 변환 후
`bash scripts/ruler_cap_run.sh sft:yarn4` + 원본 대조는 `sft:ext` — 두 평균의 차가 위치 항,
131K 평균의 상승이 능력 항. 판정 사다리: 원본 131K ≥85 → "128K 지원", yarn4 258K/520K ≥85 →
해당 길이 지원 선언.

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
