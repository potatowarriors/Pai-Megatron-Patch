# 벤치 추이 (eval_ckpt 집계)

각 체크포인트별 대표 점수(100분율). 매핑 정본은 `bench_registry.py`.

`무효` = 추출 실패율/사고 마감률이 임계를 벗어나 측정이 성립하지 않은 셀 (판정: `summarize.py`).

| run | iter | mmlu_pro | gpqa_diamond | aime25 | hmmt_feb_2025 | ifeval_inst_loose |
|---|---|---|---|---|---|---|
| alpha_baseline_48L_sft_128k_full_20260828_081911 | 300 | 47.0 | 32.0 | 무효 | 무효 | 65.6 |
