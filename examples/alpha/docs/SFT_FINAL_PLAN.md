# SFT 최종 단일 런 계획 (2026-09-13~)

**결정(사용자, 2026-09-13)**: SFT phase-1(2,448 iters 완주)·phase-2(iter 500)·phase-3(iter 30) 계보를 **폐기**하고, 그동안 만든 모든 SFT 데이터셋을
합쳐 LC-B 최종 ckpt(iter 320)에서 **한 번의 SFT** 로 끝낸다. 블렌드 비율은 **Nemotron 3 Ultra 공개 SFT 블렌드의 카테고리 비중을 그대로**,
총량은 Ultra 1단계와 같은 **60B**, 속도를 위해 **DiLoCo 2노드**(main1+sub1)를 적용한다. 폐기 ckpt 파일은 지시 전까지 삭제하지 않는다.
상태는 `STATUS.md`, 사고는 `KNOWN_ISSUES.md`, 이 문서는 설계·근거·프로토콜의 정본이다.

## 1. 블렌드 — Ultra 비율 토큰 기준 적용

근거 파일: NVIDIA-NeMo/Nemotron `recipes/ultra3/stage1_sft/config/data_prep/data_blend_raw.json` — 공개 12 항목(가중치 합 58.3) + 내부 13 범주(47.5) = **105.8**
("Weights unnormalized; normalized at runtime"). 앞선 보고의 "합 113.8" 은 요약 도구의 오류였고 원본을 직접 받아 정정했다(2026-09-13 02:00).

| 카테고리 | Ultra 원가중치 | Ultra 정규화 % | 우리 목표 %(장문맥 제외·identity 0.5 추가 후 재정규화) | 우리 멤버 (base_ep → 최종 ep) |
|---|---|---|---|---|
| chat / IF / 구조화 | chat_if 14.3 + structured_outputs 14.3 | 27.03 | **27.41** | 구조화 고정: `ifchat_v1_structured` 8ep · `usab_v1_think/nothink` 4ep. 나머지 공통 **1.70ep**: `ifchat_v1_chat_if`·`chat_v3_chat`·`chat_v3_if_fanout_me`·`chat_v2_on_scrub`·`chat_v2_off_scrub`·`kochat_v3_think/nothink` |
| 코드 | 경쟁 20.7 + infinibyte 2.0 + SWE 3.0 + agentic-programming 2.0 + CUDA 0.5 + SQL 0.5 | 27.13 | **27.51** | `cp_v2` 0.45→0.49 · `swe_v3_tools_keepthink` 0.20→0.22 · `opencode_tools` 0.155→0.17 · `cuda_v1` 4ep 고정 |
| 수학 | math 9.9 + math-w-tools 4.9 + lean 2.0 | 15.88 | **16.10** | `math_v4` 0.86→0.97 · `math_proofs_v2` 0.45→0.51 · `math_proofs_v1_lean` 0.10→0.11 |
| 과학 | 12.8 | 12.10 | **12.27** | `science_v2` 0.40→0.44 · `science_v1` 1.0→1.11 |
| 다국어 | 7.4 | 6.99 | **7.09** | `ml_ultra-v3_*` 9종 1.0→1.10 · `ml_super-v3_*` 12종 0.04→0.044 |
| 에이전틱 | terminal 1.5 + tool-calling 1.0 + interactive 1.0 + search 0.5 | 3.78 | **3.83** | `ntc_v1_*` 6종 0.14→0.149 · `agentic_v2_tc` 0.14→0.149 + `kotool_v1` 2→2.1 + `when2call_v1` 6→6.4 · `agentic_v2_ia` 0.35→0.37 · `agentic_v2_search` 1.0→1.06 + `search_ko_v1`·`research_ko_v1` 0.5→0.53 |
| 금융 | 3.0 | 2.84 | **2.88** | `finance_v1` 0.165→0.18 |
| 저노력 reasoning | 2.0 | 1.89 | **1.92** | `budget_trunc_v1_if/math` 2.5→3.06 |
| safety | 0.5 | 0.47 | **0.48** | `safety_v2` 2.4→2.81 |
| 장문맥 | 2.0 | 1.89 | 제외 | LC-A/LC-B 에서 기학습 |
| identity (alpha 추가) | — | — | **0.50** | `identity_v2_fanout` 18→18.1 |

- 생성기 `toolkits/sft_data_preprocessing/gen_sft_128k_final_blend.py` + 스펙 `sft_128k_final_spec.tsv` → `configs/data/sft_128k_final_blend.yaml`.
  카테고리 목표 토큰에서 고정(fixed) 멤버 소비를 빼고 잔여를 base_ep 상대 비율대로 공통 배율 k 로 채운다(표의 →). 집계는 bin 1표라 **토큰 비중 = gradient 비중**.
- 예산 **2,862 iters × GBS 160 × 131,072 = 60.02B bin-tok = 457,920 samples**. 단일 노드 324 s/iter 기준 ≈10.7일, DiLoCo 2노드 기준 ≈5.4일(+sync 오버헤드).
- Ultra 가중치 단위는 샘플(NeMo 관례)이라 긴 궤적(SWE·터미널)은 토큰 기준으로 Ultra 보다 크게 잡힌다. 사용자 결정은 토큰 기준 적용.
- 제외: kochat v1/v2(가짜 reasoning, 09-09 폐기), ARC-AGI(Ultra 대응 범주 없음), 구 변형(`chat_v2_on`·`chat_v2_on_fanout`·`chat_v2_off`·`opencode_fixed`·`swe_v3_keepthink`·`identity_v2`·`swe_v3_terminus_keephist`·`swe_v1_r2e`·`swe_v2_*`·`terminal_terminus2_synth`).
- 트리 `/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_final_pad16` (58 멤버, 기존 멤버는 실경로 symlink).

## 2. 신규 멤버 (2026-09-13 변환, 전부 verify PASS·render 봉투 0)

| 멤버 | 원천 | 행/샘플 | bins | 실/학습 tok | 비고 |
|---|---|---|---|---|---|
| `chat_v2_on_scrub` | Chat-v2 `reasoning_on.p2scrub.jsonl`(자기귀속 1,823행 드롭) + `--fanout-implicit-turns` | 927,414 → 1,197,270 | 21,200 | 2.77B / 2.11B | 기존 `chat_v2_on_fanout` 은 스크럽 전 원본이었음 |
| `chat_v2_off_scrub` | Chat-v2 `reasoning_off.p2scrub.jsonl`(829행 드롭) | 1,067,364 | 10,520 | 1.37B / 1.06B | no-think |
| `ifchat_v1_chat_if` | IF-Chat-v1 `chat_if.jsonl`(Ultra 최상위 가중치 셋, odc-by-1.0) + fan-out | 426,009 → 633,210 | 13,966 | 1.83B / 1.52B | 멀티턴 83%, reasoning 36% |
| `ifchat_v1_structured` | IF-Chat-v1 `structured_outputs.jsonl`(cc-by-4.0) | 4,969 | 157 | 20.4M / 11.3M | XML/JSON 스키마 준수, 8ep 고정 |
| `when2call_v1` | `nvidia/When2Call` train_sft(cc-by-4.0), `sdg/kotool/convert_when2call.py` | 14,876 | 66 | 8.4M / 0.41M | 전부 미호출(되묻기 7,007·불가 7,518·기타 351), no-think, BFCL 스키마→JSON Schema 정규화 |

변환 스크립트 `toolkits/sft_data_preprocessing/convert_sft_128k_final_new.sh`. 미호출:호출 비율과 When2Call epoch 논의는 `sdg/kotool/README.md` 마지막 절.

## 3. 프리셋

| 프리셋 | 용도 | 차이 |
|---|---|---|
| `configs/training/sft_128k_final.yaml` | 단일 노드 본 런 / A/B 기준선 | `sft_128k_full.yaml` 전체 복제 + train-samples 457,920 · warmup 21,440(134 iters, 4.7%) · save/eval 100 · `mid-level-dataset-surplus 0.05` · load LC-B + finetune + no-load-optim |
| `configs/training/sft_128k_final_diloco.yaml` | DiLoCo 2노드 | 위와 동일하되 train-samples **228,960(노드당 1,431 iters)** · warmup 10,720 · save/eval 50 · `pretrained-checkpoint:` 방식(런처가 `--load <노드 dir>` 를 덧붙이고, 비어 있으면 Megatron 이 pretrained 로 finetune·있으면 재개) — `finetune:`/`no-load-optim:` 없음(재개 리셋 방지) |

LR 은 phase-1 과 동일(2.5e-5 cosine → 1.5e-6, Ultra 비율 이식). DiLoCo 에서 노드당 samples 로 스케줄이 돌므로 warmup/decay 는 노드당 값으로 적는다.
`DILOCO_DATA_SHARD=1` 은 데이터셋을 world× 로 짓고 노드가 블록 순환으로 서로소 절반을 먹는다 → 두 노드 합 = 457,920 = 블렌드 예산.

## 4. 검증·게이트 체인

1. **2-iter 스모크**(필수, 09-09 교훈): main1 `sft_128k_final` → sub1 `scripts/sub1_jit595_smoke.sh sft_128k_final_diloco sft_128k_final_blend`(jit595 우회, NFS 인덱스 캐시 공유).
2. **DiLoCo A/B (150 iters)**: A = main1 단일 `sft_128k_final` 150 iters(ckpt 저장, 실패 시 본 런으로 그대로 이어감) → B = `launch_diloco.sh sft_final … sft_128k_final_diloco`(H=30, τ=2, SHARD_BLOCK=160, `NODE1_ENV=LD_LIBRARY_PATH=…/jit595`) 노드당 150 iters.
   판정: (i) B 의 노드 loss 곡선이 iter ≤ 60 에서 A 와 동일 궤적(±0.01), (ii) iter 150 에서 B ≤ A(2× 데이터 효과), (iii) outer sync 체크섬 일치·NaN 0·저장 성공, (iv) 벽시계 노드당 iter 시간이 A 의 1.1× 이내. 통과 시 B 를 그대로 본 런으로 재개(같은 명령 재실행 = 재개). 실패 시 A 의 iter-150 ckpt 에서 단일 노드로 계속.
3. **ckpt 마다(100 iters 상당)**: `eval_sft/probe_ckpt.sh` — MG→HF 변환(evaluate.sh 게이트: forward_sanity·eos 정합) → vLLM 단일 서버 → `identity_probe.py`(제작자 ≥95%·누출 0) + 유령 호출 재생(`results/reasoning_probe/bestcase_replay.py`, tools25 조건 유령률 ≤1/33).
4. **300 iters 마다** T1(`eval_ckpt.sh … t1`: MMLU-Pro·GPQA-D·IFEval·AIME·HMMT). 두 노드가 모두 학습 중이면 GPU 가 없으므로 **사후 일괄**(런 종료 후 ckpt 순회) — 단일 노드 런이면 sub1 에서 병행.
5. 조기중단 가드: valid loss 상승 전환 시 직전 ckpt 채택(phase-1 규칙 유지).

## 5. 열린 사용자 결정
- When2Call epoch: 기본 6.4ep(미호출:호출 ≈1:1 행 기준). 엄격 2:1 은 ≈13ep(과잉 거절 편향 위험) — 게이트 결과로 판정.
- DiLoCo A/B 실패 시 단일 노드 10.7일 진행 여부.
