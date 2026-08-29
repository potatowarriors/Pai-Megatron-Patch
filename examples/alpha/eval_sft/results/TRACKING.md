# 벤치 추이 (eval_ckpt 집계)

각 SFT 체크포인트별 점수. `eval_ckpt.sh`/`eval_watch.sh` 가 갱신. 100분율.

| run | iter | mmlu_pro | gpqa_diamond_generative_n_shot | aime25 | hmmt_feb_2025 | ifeval | simpleqa_verified | logickor |
|---|---|---|---|---|---|---|---|---|
| alpha_baseline_48L_sft_128k_full_20260828_081911 | 300 | 36.9 | 23.2 | 0.0 | 0.0 | 43.2 | 1.4 | 13.2 |
| baseline_lcb | 320 | 24.1 | 19.2 | 0.0 | 0.0 | 39.0 | 0.8 | 14.2 |
| smoke_lcb | 320 | 23.8 | — | 0.0 | — | 20.0 | — | — |
| smoke_logickor | -1 | — | — | — | — | — | — | 12.5 |
| smoke_simpleqa | -1 | — | — | — | — | — | 0.0 | — |
