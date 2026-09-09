# 벤치 추이 (eval_ckpt 집계)

각 체크포인트별 대표 점수(100분율). 매핑 정본은 `bench_registry.py`.

`무효` = 추출 실패율/사고 마감률이 임계를 벗어나 측정이 성립하지 않은 셀 (판정: `summarize.py`).

| run | iter | mmlu_pro | gpqa_diamond | aime25 | hmmt_feb_2025 | ifeval_prompt_strict | ruler_single_1_avg | ruler_single_2_avg | ruler_multikey_avg | ruler_multivalue_avg | simpleqa_verified | logickor | swe_verified | terminal_bench |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha_baseline_48L_sft_128k_full_20260828_081911 | 300 | 47.0 | 32.0 | 무효 | 무효 | 49.3 | 100.0 | 70.0 | 35.0 | 28.8 | 1.2 | 36.8 | 1.6 | 무효 |
| alpha_baseline_48L_sft_128k_full_20260828_081911 | 600 | 48.6 | 32.2 | 무효 | 무효 | 55.6 | 100.0 | 71.7 | 40.0 | 26.2 | 3.6 | 40.2 | 3.2 | 1.2 |
| alpha_baseline_48L_sft_128k_full_20260828_081911 | 900 | 49.1 | 33.6 | 무효 | 무효 | 59.4 | 100.0 | 76.7 | 43.3 | 24.2 | 3.5 | 43.1 | 4.4 | 1.7 |
| alpha_baseline_48L_sft_128k_full_swap_20260901_101523 | 1200 | 49.5 | 34.1 | 무효 | 무효 | 62.2 | 98.3 | 80.0 | 40.0 | 22.9 | 4.0 | 42.2 | 4.8 | 2.2 |
| alpha_baseline_48L_sft_128k_full_swap_20260901_101523 | 1500 | 50.3 | 33.9 | 무효 | 무효 | 60.1 | 98.3 | 78.3 | 43.3 | 22.9 | 3.9 | 43.0 | 6.2 | 1.1 |

## 중단 기록 — phase-2 iter500 (2026-09-09 05:43 UTC)

`alpha_baseline_48L_sft_128k_full_p2_20260907_073414_iter0000500` 스위트를 T1 도중 중단했다.

| | |
|---|---|
| 중단 시점 | T1 **9,590/19,864 (48%)**, 1시간 23분 경과 |
| 완료된 단계 | 변환(SIGSEGV 없음) · G1 · G2 · G3 |
| 산출물 | **없음** — lm_eval 은 종료 시점에만 결과를 쓴다 |
| 사유 | SFT 데이터 결함 2건(SWE-v3 도구 선언 부재, opencode tools 스키마 렌더 파손) 재변환·스모크가 sub1 GPU 를 필요로 함. phase-3 기동(≈21:00 KST) 전 처리. 사용자 결정, `pai-megatron-patch-c2` 세션 경유 |
| 재개 | 처음부터 (T1 부분 재개 불가). 변환본 `hfmodel_0000500` 은 남으므로 변환은 건너뛴다 |

재개 명령:
```bash
ssh sub1 "cd .../examples/alpha && nohup bash eval_sft/eval_new_ckpt.sh \
  .../outputs/alpha_baseline_48L_sft_128k_full_p2_20260907_073414 500 t1,t3,agentic,t2 \
  > /home/work/vidsearch/tools/bench_logs/p2_iter500_suite.log 2>&1 &"
```
