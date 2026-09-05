# 벤치 추이 (eval_ckpt 집계)

각 체크포인트별 대표 점수(100분율). 매핑 정본은 `bench_registry.py`.

`무효` = 추출 실패율/사고 마감률이 임계를 벗어나 측정이 성립하지 않은 셀 (판정: `summarize.py`).

| run | iter | mmlu_pro | gpqa_diamond | aime25 | hmmt_feb_2025 | ifeval_prompt_strict | ruler_single_1_avg | ruler_single_2_avg | ruler_multikey_avg | ruler_multivalue_avg | simpleqa_verified | logickor | swe_verified | terminal_bench |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha_baseline_48L_sft_128k_full_20260828_081911 | 300 | 47.0 | 32.0 | 무효 | 무효 | 49.3 | 100.0 | 70.0 | 35.0 | 28.8 | 1.2 | 36.8 | 1.6 | 무효 |
| alpha_baseline_48L_sft_128k_full_20260828_081911 | 600 | 48.6 | 32.2 | 무효 | 무효 | 55.6 | 100.0 | 71.7 | 40.0 | 26.2 | 3.6 | 40.2 | 3.2 | 1.2 |
| alpha_baseline_48L_sft_128k_full_20260828_081911 | 900 | 49.1 | 33.6 | 무효 | 무효 | 59.4 | 100.0 | 76.7 | 43.3 | 24.2 | 3.5 | 43.1 | 4.4 | 1.7 |
| alpha_baseline_48L_sft_128k_full_swap_20260901_101523 | 1200 | 49.5 | 34.1 | 무효 | 무효 | 62.2 | 98.3 | 80.0 | 40.0 | 22.9 | 4.0 | 42.2 | 4.8 | 2.2 |
