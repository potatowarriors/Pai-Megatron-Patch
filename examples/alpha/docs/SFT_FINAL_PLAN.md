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
| chat / IF / 구조화 | chat_if 14.3 + structured_outputs 14.3 | 27.03 | **27.41** | 구조화 고정: `ifchat_v1_structured` 8ep · `usab_v1`(think+nothink 병합) 4ep. 나머지 공통 **1.70ep**: `ifchat_v1_chat_if`·`chat_v3_chat`·`chat_v3_if_fanout_me`·`chat_v2_on_scrub`·`chat_v2_off_scrub`·`kochat_v3_think/nothink` |
| 코드 | 경쟁 20.7 + infinibyte 2.0 + SWE 3.0 + agentic-programming 2.0 + CUDA 0.5 + SQL 0.5 | 27.13 | **27.51** | `cp_v2` 0.45→0.49 · `swe_v3_tools_keepthink` 0.20→0.22 · `opencode_tools` 0.155→0.17 · `cuda_v1` 4ep 고정 |
| 수학 | math 9.9 + math-w-tools 4.9 + lean 2.0 | 15.88 | **16.10** | `math_v4` 0.86→0.97 · `math_proofs_v2` 0.45→0.51 · `math_proofs_v1_lean` 0.10→0.11 |
| 과학 | 12.8 | 12.10 | **12.27** | `science_v2` 0.40→0.44 · `science_v1` 1.0→1.11 |
| 다국어 | 7.4 | 6.99 | **7.09** | `ml_ultra-v3_*` 9종 1.0→1.10 · `ml_super-v3_*` 12종 0.04→0.044 |
| 에이전틱 | terminal 1.5 + tool-calling 1.0 + interactive 1.0 + search 0.5 | 3.78 | **3.83** | `ntc_v1_*` 6종 0.14→0.149 · `agentic_v2_tc` 0.14→0.149 + `kotool_v1_x2` 1→1.06(고유 2.1ep) + `when2call_v1_x2` 3→3.19(고유 6.4ep) · `agentic_v2_ia` 0.35→0.37 · `agentic_v2_search` 1.0→1.06 + `search_ko_v1`·`research_ko_v1` 0.5→0.53 |
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
- 트리 `/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_final_pad16` (57 멤버, 기존 멤버는 실경로 symlink).
- **bins<100 멤버 금지**(split 99/1 에서 valid 문서 0 → GPTDataset 빌드 무한대기, `rules/sft-data.md`; 09-13 02:03 첫 스모크가 `usab_v1_nothink` 35 bins 에서 정지 — `outputs/smoke_failed_sft_128k_final_main1_20260913_020200_valid0doc_hang.log`). 해소: `usab_v1` = think+nothink 병합(307 bins), `kotool_v1_x2`·`when2call_v1_x2` = 행 2벌(161·131 bins, base_ep 는 고유 에폭의 1/2) — `toolkits/sft_data_preprocessing/convert_sft_128k_final_small.sh`.

## 2. 신규 멤버 (2026-09-13 변환, 전부 verify PASS·render 봉투 0)

| 멤버 | 원천 | 행/샘플 | bins | 실/학습 tok | 비고 |
|---|---|---|---|---|---|
| `chat_v2_on_scrub` | Chat-v2 `reasoning_on.p2scrub.jsonl`(자기귀속 1,823행 드롭) + `--fanout-implicit-turns` | 927,414 → 1,197,270 | 21,200 | 2.77B / 2.11B | 기존 `chat_v2_on_fanout` 은 스크럽 전 원본이었음 |
| `chat_v2_off_scrub` | Chat-v2 `reasoning_off.p2scrub.jsonl`(829행 드롭) | 1,067,364 | 10,520 | 1.37B / 1.06B | no-think |
| `ifchat_v1_chat_if` | IF-Chat-v1 `chat_if.jsonl`(Ultra 최상위 가중치 셋, odc-by-1.0) + fan-out | 426,009 → 633,210 | 13,966 | 1.83B / 1.52B | 멀티턴 83%, reasoning 36% |
| `ifchat_v1_structured` | IF-Chat-v1 `structured_outputs.jsonl`(cc-by-4.0) | 4,969 | 157 | 20.4M / 11.3M | XML/JSON 스키마 준수, 8ep 고정 |
| `when2call_v1_x2` | `nvidia/When2Call` train_sft(cc-by-4.0), `sdg/kotool/convert_when2call.py` → 행 2벌 | 14,876 ×2 | 131 | 16.9M / 0.83M | 전부 미호출(되묻기 7,007·불가 7,518·기타 351), no-think, BFCL 스키마→JSON Schema 정규화 |
| `usab_v1` | 트랙 U think 20,272 + nothink 8,539 병합 | 28,811 | 307 | 40.0M / 35.6M | 구 `usab_v1_think/nothink`(273+35 bins) 대체 |
| `kotool_v1_x2` | 트랙 T 6,134행 ×2 | 12,268 | 161 | 20.9M / 8.2M | 구 `kotool_v1`(81 bins) 대체 |

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
2. **DiLoCo A/B (150 iters)**: A = main1 단일 `sft_128k_final` 150 iters(`--exit-interval 150`, save 100·150)(ckpt 저장, 실패 시 본 런으로 그대로 이어감) → B = `launch_diloco.sh sft_final … sft_128k_final_diloco`(H=30, τ=2, SHARD_BLOCK=160, `NODE1_ENV=LD_LIBRARY_PATH=…/jit595`) 노드당 150 iters.
   **A′(복제 런, sub1, 100 iters, 사용자 승인 09-13 04:00)**: A 와 같은 프리셋·블렌드·seed 를 sub1 에서 재실행 → (a) 595 스왑 이후 sub1 장시간 학습 안정성 증거(B 는 sub1 이 5일+ 버텨야 함) (b) 실행 간 비결정 산포 포락선(CLAUDE.md "A/B 는 같은 구성 재실행의 산포로 판정") (c) A′ iter 100 ckpt(예비).
   판정(정정 09-13, 실측 확인 16:19): `DILOCO_DATA_SHARD=1`·SHARD_BLOCK=GBS 에서 **node0 의 iter k 배치 = A 의 iter 2k−1 배치, node1 의 iter k = A 의 iter 2k**(node0 iter1 loss 1.069033 = A iter1, node1 iter1 1.038442 ≈ A iter2 1.038569)이므로 곡선을 iter 번호로 겹쳐 보지 않는다.
   (i) 같은 배치 비교(B-node0 iter k ↔ A iter 2k−1, node1 iter k ↔ A iter 2k): 첫 outer sync(iter 30) 전에는 B 노드가 A 의 절반 업데이트만 했으므로 B − A ≈ +0.02 가 **정상**(실측 iter ≤30: node0 +0.005→+0.023, node1 +0.006→+0.025). sync 가 누적될수록 이 격차가 줄어 0 이하로 가야 2× 데이터 효과가 있는 것.
   (ii) 같은 iteration 비교(10-iter 평균, 배치는 다름): k > 30 에서 B 노드 평균 ≤ A(파일럿 "동일 iteration 기준 우위"), 격차가 k 에 따라 벌어져야 한다. 실측 iter ≤30(sync 전): B_avg − A = −0.004/−0.008/−0.025 (배치 구성 차이 수준).
   (iii) iter 141~150 평균: B 두 노드 평균 < A(0.882 부근) (iv) outer sync 체크섬 일치·NaN 0·저장 성공 (v) 벽시계 노드당 iter 시간 ≤ 1.1×A(313 s) — 실측 318 s(1.016×), iter 30 sync 포함 324 s. 통과 시 B 를 그대로 본 런으로 재개(같은 명령 재실행 = 재개). 실패 시 A 의 iter-150 ckpt 에서 단일 노드로 계속.
3. **ckpt 마다(100 iters 상당)**: `eval_sft/probe_ckpt.sh` — MG→HF 변환(evaluate.sh 게이트: forward_sanity·eos 정합) → vLLM 단일 서버 → `identity_probe.py`(제작자 ≥95%·누출 0) + 유령 호출 재생(`results/reasoning_probe/bestcase_replay.py`, tools25 조건 유령률 ≤1/33).
4. **300 iters 마다** T1(`eval_ckpt.sh … t1`: MMLU-Pro·GPQA-D·IFEval·AIME·HMMT). 두 노드가 모두 학습 중이면 GPU 가 없으므로 **사후 일괄**(런 종료 후 ckpt 순회) — 단일 노드 런이면 sub1 에서 병행.
5. 조기중단 가드: valid loss 상승 전환 시 직전 ckpt 채택(phase-1 규칙 유지).

## 5. 사용자 결정 (2026-09-13 02:50 확정)
- When2Call **6.4ep 기본안**(미호출:호출 ≈1:1 행 기준) — 엄격 2:1(≈13ep) 대신 유령 호출 게이트(≤1/33)로 판정.
- Ultra 비중 **27/27 적용 확인**(파일 원본 합 105.8 기준; 앞선 25/25 는 113.8 오류에서 나온 수치).
- DiLoCo A/B 결과 **단일 노드 확정**(09-14 01:00), phase-1 이어가기 대신 LC-B 새 런 유지(01:15).

## 6. 실행 기록
- 09-13 02:27 main1 2-iter 스모크 PASS: iter1 loss 1.069033 (515 s) → iter2 1.038569 (324.7 s, 267 TFLOP/s/GPU). 첫 시도(02:03)는 bins<100 valid 0-doc 정지(§1).
- 09-13 11:44(sub1 시계) sub1 jit595 2-iter 스모크 PASS: loss **비트 동일**(1.069033 → 1.038569), 321.7 s/iter, max alloc 55.3 GB, munmap 0 — 캐시 462 파일 공유.
- 09-13 02:29 A 기동 `outputs/alpha_baseline_48L_sft_128k_final_20260913_022854`(wandb alpha-posttraining 88yjmimx), iter1 loss 1.069033(스모크와 동일). 체인 `scripts/sft_final_chain.sh`(outputs/sft_final_chain.log): iter 100 + sub1 유휴 → sub1 probe, A 종료 → B 자동 기동.
- 09-13 04:00 **A′** sub1 기동(`outputs/run_sft_final_Aprime_sub1.log`, `--exit-interval 100`, ≈9h). B 기동 시각(≈16:00) 불변.
- 09-13 13:26(sub1 시계 기준 22:26) **A′ 100 iters 완주**(sub1 9.5 h 연속, munmap 0, ckpt iter 100). A 와 겹치는 100 iters loss 차이 평균 7e-5·최대 2e-4 → 비결정 포락선 ≈ 2e-4.
- 09-13 15:47 **A 150 iters 완주**(loss 1.069 → 0.882, valid iter100 0.824/PPL 2.28, ckpt iter 100·150).
- 09-13 15:47 **B 1차 기동 실패**: 런처가 `CUDA_DEVICE_MAX_CONNECTIONS=32` 를 강제(프리트레인 CP=1 전제) → CP8 프리셋에서 Megatron assert 로 양 노드 즉사. 런처가 프리셋의 CP/TP>1 이면 1 로 두도록 수정(1b7ffe5). **B 재기동 16:12** `outputs/diloco_sft_final/node{0,1}`, 로그 ~/run_diloco_sft_final_node{0,1}.log.
- probe 체인 실측: MG→HF 변환은 **main1 8 GPU** 에서 PASS(forward_sanity), sub1 2 GPU 는 저장 단계 SIGSEGV(원인 미상, 09-13 13:29). vLLM(alpha_serve_venv, CUDA 13 torch)은 **sub1 에서만** 뜬다(main1 드라이버 12.8 "too old"). → 사후 평가는 변환 main1 · 서빙 sub1 로 분담.
- A iter 100 probe(09-14 01:09 sub1 시계): 제작자 0/30(OpenAI/챗GPT 자칭), 유령 호출 4/33 — LC-B 시작점(FAIL·4/33)과 동일. iter 100 = 예산의 3.5%·워밍업 중·identity 0.5% 라 아직 판정 의미 없음, 이후 ckpt 추이로 본다.
- 09-14 00:55 **B 중단(iter 94, 사용자 결정)** — A/B 결과 `study/diloco_sft_ab.md`: 같은 데이터 효율 0.60~0.63, 벽시계 1.2×/2×GPU. 사용자 판단: outer momentum(0.6) 이 제대로 동작하는지 보려면 210 iters 이상 지켜봐야 하고, 그 실험 시간까지 더하면 단일 노드가 낫다.
- 09-14 01:15 **phase-1 이어가기 vs LC-B 새로 시작** 재검토 후 **A(LC-B 새로, 현재 런) 유지** 결정 — 근거: phase-1 은 폐기 데이터(kochat v1/v2 2.6B·identity_v1 반복·ARC)가 학습 스팬에 있고 코드 50% 혼합이라 Ultra 비율·계보·인과 추적이 깨짐; 시간 차 4~5일. iter 300 T1 이 phase-1(47.0/32.0/65.6) 대비 열세면 그때 phase-1 이어가기로 전환(손실 1.3일).
- 09-14 01:12 본 런 재개(iter 151 consumed 24,160·LR 2.4998e-5·loss 0.855 로 승계 확인) → 사용자 지적(100 iters 저장·평가 과함: 저장+valid 11분/회·4.3 TB) → **01:32 save 300 / valid 100 으로 재시작**(013219 런은 디렉터리가 실수로 삭제돼 tensorboard FileNotFoundError 로 종료, 사용자 확인) → **01:39 재기동** `outputs/alpha_baseline_48L_sft_128k_final_resume_20260914_013856` (프리셋 `sft_128k_final_resume.yaml`: load = A iter 150, finetune/no-load-optim 없음), 잔여 2,712 iters ≈ 9.8일 → 종료 ≈ 09-24 00:00. sub1 `scripts/sft_final_eval_watch.sh`: 300 iters 마다 probe + T1 (변환 sub1 8 GPU).
