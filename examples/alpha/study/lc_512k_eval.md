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

## 4. 판정 규칙

1. **520192 single_1 ≥ 90%** 그리고 single_2·multikey 가 같은 모델의 128K 수치 −10pp 이내
2. 393216 이 ext 프로파일의 0% 에서 유의미하게 회복 (절벽 이동 확인)
3. yarn 프로파일의 단문 표본(ifeval·gpqa 100문항)이 ext 프로파일 대비 노이즈 대역 (±3pp)

→ 통과: 추론-only 채택, **≤256K 는 원본 config·>256K 는 yarn 프로파일**로 분리 서빙 (Qwen3 관례 —
static YaRN 의 mscale 이 단문 attention 온도도 바꾼다). 미달: 경로 B(LC-C) 착수, 시점은 SFT 종료
(~09-06) 전후로 사용자 결정.

## 5. 결과

(그리드 완료 후 `outputs/lc512k_eval/SUMMARY.md` 를 옮겨 적는다.)
