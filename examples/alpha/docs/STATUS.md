# Alpha 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push**한다. 상태는 여기에만 쓴다 — Claude auto-memory에 쓰지 않는다
(메모리는 컨테이너·노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.

_마지막 갱신: 2026-09-01 07:00 (SFT phase-2 계획 착수 — 블렌드 실측 검증에서 결함 3건, 이전: 04:40 512K 추론 확장 트랙 착수)_

## 진행 중

| 트랙 | 상태 (2026-08-25) | 다음 할 일 | 정본 |
|---|---|---|---|
| **SFT 준비** | 64k 21종 + 128k 6종 변환·`verify_sft_bins` 전 PASS. `sft_40b_blend.yaml`(SWE 1-pass 앵커 40B) + `sft_128k_blend.yaml`(5.54B). preset `sft_64k.yaml`(CP4+offload, DP2) / `sft_128k.yaml`(CP8). interleaved-thinking 규약 정비 완결(a2f8894), agentic_v2 미편입 확정. **effort/budget NVIDIA 레시피 재현 확정(사용자, 08-25)**: 변환기 `--medium-effort`·`--truncate-reasoning-budget` 구현, 테스트 34/34·34/34, 스모크 bins 게이트 PASS (`SFT_RL_DATASETS.md` §2.6) | ① LongBlocks 변환기(1.5% 슬롯) ② RULER ③ 본 런 — LC 완료 후 ckpt 경로만 채움. ④ ~~effort 재변환~~ 완료(08-25): 3종 변환·게이트 PASS·블렌드 반영(chat 21→20+budget 1, ep 2.67/1.19). **GBS 192/96 확정(사용자, 2026-08-28)** — 12.58M tok/iter 전 스테이지 상수 관례 복원(초기 GBS 64는 Ultra 샘플 수만 미러한 오류; Ultra SFT 실제 18.9M). ETA 불변: **64k 3,179 iters ≈ 9.7일 + 128k 441 iters ≈ 1.7일**. `sft_64k.yaml` load = LC-B iter320 정본 기입 완료. **단일 128k 혼합 버킷으로 확정(사용자, 2026-08-28)** — 투버킷(64k 40B + 128k 꼬리) 폐기. 실측: 128k·CP8+offload·GBS 96 실블렌드 200.1s/iter = 62.9k tok/s·260 TFLOP/s(64k·CP4 GBS64 실측 47.5k 대비 1.32× — 12.58M 배치에서 iter 고정비 희석), 혼합 패킹 fill 98.7~100%, max-alloc 55~59GB. 본 런 = `sft_128k_full.yaml` × `sft_128k_mixed_blend.yaml`(26멤버, 51.34B = **2,448 iters @ GBS 160 = 20.97M tok/iter**(사용자, Ultra 18.9M 스케일), SWE 9.61B 정확 1ep, 전 꼬리 epoch 비례 편입) ≈ **9.3일**(실측 329.7s/iter·63.6k tok/s; GBS 96 대비 +1% — 처리량 동일, 근거는 최적화 설계). 64k·CP4·GBS192 비교 실측 65.4k tok/s = CP8 과 동률(1.53× 는 GBS64 착시). LR = Ultra 비율 이식 2.5e-5→1.5e-6 cosine(warmup 115). load = LC-B iter320. **본 런 개시 2026-08-28 08:19 KST** (`outputs/alpha_baseline_48L_sft_128k_full_20260828_081911`, wandb online, save/eval 300 iters = 8회+종료). KoChat 2차 트랜치 반영 검증 완료(v2 MANIFEST cut=tranche2 누적, `_t2` 3종만 편입). ETA 2,448 iters × 329.7s ≈ 9.3일 → **~09-06**. 조기중단 가드: valid loss 300 iters 마다. 구 sft_64k/sft_128k 프리셋은 superseded 표기 보존. **1노드 학습 확정(사용자, 2026-08-25)** — 2-node 단축안 기각. 유휴 노드는 SFT 후 RL 환경 검증용(별도 워크스페이스, 아래 보류 표) | `INTERLEAVED_THINKING.md`, `SFT_RL_DATASETS.md`, `DATA_PREP_LOG.md` |
| **SFT phase-2 (계획)** | **착수 2026-09-01.** phase-1 블렌드 실측 검증에서 데이터 결함 3건 발견(`KNOWN_ISSUES.md` 09-01): ① opencode_v1 tool 결과가 Python repr 로 렌더 + reasoning 0% ② identity_v1 원본 기준 ≈180회 반복 ③ chat_v3_chat 프롬프트 복원 잔여 11.8%. 비율·epoch·렌더 플래그·게이트는 설계대로(`SFT_RL_DATASETS.md` §2.7). **권고** = 3건 수정 후 LC-B iter320 부터 **재실행**(phase-1 = 대조군), phase-1 종료(~09-06) 직후 개시 ≈9.2일(323s/iter 실측) | ① 사용자 결정 4건(재실행/즉시전환/연속·identity 상한·opencode no-think 유지·WildChat-Full 접근) ② G-P0~P4 데이터 준비(phase-1 중 CPU, sub1) ③ G-P6 phase-1 최종 평가 = 기준선 | `SFT_PHASE2_PLAN.md` |

| **SFT 벤치마크 (sub1)** | **첫 유효 측정 완료 (2026-08-30).** iter300 T1: MMLU-Pro **47.0**, GPQA-D **32.0**, IFEval **65.6** (전부 유효 — 추출실패 1.4~1.7%, 사고마감 57~91%). AIME 0.6 / HMMT 0.2 는 **무효** — 사고마감 16%/11% 로 32K 예산 안에 답에 도달 못함(모델 미성숙, 12% 지점). 구 하니스 대비 MMLU-Pro +10.1 / GPQA +8.8 — 구 수치는 하니스 결함을 재고 있었다(추출실패 34%→1.7%). 인프라: T1·T2 재작성 + 게이트 G1~G3·A1~A3 + `run_suite.sh` 오케스트레이터 + `bench_registry.py` 매핑 정본. GPU0 복귀(레플리카 8). **진행 중**: T3 SimpleQA·LogicKor. **대기**: 에이전틱(TOOLS=1 fleet 필요), T2 RULER(롱 fleet) | ① T3 완료 ② 에이전틱 ③ T2 RULER ④ iter600 재측정 | `SFT_BENCHMARKS.md` §3.6·§3.8·§3.9·§7, `results/TRACKING.md` |

| **채팅 서빙 (main1 GPU3)** | **가동 중 (2026-08-31).** iter600 을 vLLM :8001(128K 창, KV 1.29M 토큰) + OpenWebUI :8080 으로 서빙. 검증: G1 PASS, `chat/smoke_chat.sh` 7/7 PASS, UI 경유 대화 확인. G2 는 reasoning 파서 때문에 설계상 FAIL(사유는 `chat/README.md` §3). 함정 6건 기록 (CUDA13 compat, open-webui 순환 import 로 마이그레이션 침묵 실패 등) | 새 ckpt 나오면 `chat/serve_chat.sh <ckpt>` 로 교체 | `examples/alpha/chat/README.md` |
| **512K 추론 확장 (sub1)** | **착수 2026-09-01.** 추론-only YaRN 프로파일로 512K 창 판정 (학습 없음; 네이티브 256K+ 학습은 메모리 불가). 인프라 완료: `tools/set_long_context_config.py`(config 프로파일, CPU 프로브 PASS) · RULER `_512k` 태스크 4종(구간 131K/258K/393K/520K, `_aa` 불변) · `scripts/lc512k_grid.sh`(sub1 suite 종료 대기 후 자동 실행, 6셀 ≈ 2h). **iter900 suite 와 GPU 경합 → suite 종료 후 자동 개시** | ① 그리드 결과 → `study/lc_512k_eval.md` §5 ② 판정(§4) ③ 통과 시 서빙 프로파일 분리, 미달 시 LC-C(YaRN CPT) 시점 결정(사용자) | `study/lc_512k_eval.md` |

## 보류·재평가 대기

| 항목 | 판정 | 재평가 시점 | 정본 |
|---|---|---|---|
| FlashQLA (TileLang GDN 커널) | 정확도 ✅, 성능 **채택 보류** — LC-A 형상(CP4 head-split)에서 fla 1.6~1.9× 우세, 128K 형상만 qla 2.2× 우세 | LC-B(128K) 국면 | `study/flashqla_poc.md` |
| Muon offload chunk/fraction 튜닝 | 256MB 기본으로 GO. 미튜닝 | LC-B | `MUON_OFFLOAD_BACKPORT.md` |
| **NeMo-RL post-training (RLVR→teacher RL→MOPD)** | **이 리포 범위 밖 — 보류** (사용자, 2026-08-25). 별도 워크스페이스 `project_s/NeMo-RL/`(브랜치 `alpha/post-train`, 최신 83976fe 08-24)에서 게이트 8종 통과: mcore+vLLM GRPO 퀵스타트, AlphaBridge 라운드트립 14,181/14,181, forward 패리티 cos≥0.99988, alpha vLLM 플러그인(stock 0.25.1), refit_verifier(mult_prob_err≈1.02), RL 블렌드, FlashQLA 커널. 남은 것 = GRPO 레시피 yaml(+ `env.nemo_gym.effort_levels` 마커 길이보상 설정, `SFT_RL_DATASETS.md` §2.6) + 8-GPU KL 게이트 | SFT 완료 후, 유휴 노드에서 재개 | `project_s/NEMO_RL_SETUP.md`(운영), `project_s/ALPHA_POSTTRAIN_PROGRESS.md`(경과), `NeMo-RL/examples/configs/alpha/README.md` |
| moe-recompute 해제 레버 | LC-A 본 런에서 OOM(68.9GB)으로 원복(9b1e7b1). 스모크 A/B(+15.5%)가 프로덕션 블렌드 꼬리를 못 봄 | resume 시 MoE 토큰 통계와 함께 | KNOWN_ISSUES 없음 — `gdn_cp_port.md` |

## 완료 (2026-08)

| 트랙 | 결과 | 정본 |
|---|---|---|
| **ko_chat 한국어 chat 합성** (sub1, 2026-08-23~28) | **종료 2026-08-28 (사용자 결정 — sub1 은 SFT 이후 RL 검증 자원으로 전환)**. 교사 gemma-4-31B(vLLM TP2×DP4, ~3.4k tok/s) + DataDesigner. 최종 = 2차 트랜치 `alpha-SFT-KoChat-v2/` → `sft_packed_64k_pad16/{kochat_if_fanout_me_t2, kochat_chat_t2, kochat_b_fanout_t2}`: **A 544,771행(IF 번역 164,560 fan-out·chat 재생성 380,211) + B 네이티브 33,831행(r1+r2 strict-facts) = 한국어 real 1.35B**, verify PASS, `sft_40b_blend.yaml` 반영(chat 슬롯 8.77B 불변, 한국어 chat = 영어 chat_v3 **동일 epoch 1.89**, 한국어 학습 2.55B = 슬롯 29.1%). **최종 yaml 스모크 PASS(08-28 13:22, sub1, LC-B iter320·GBS 192)**: 2-iter loss 1.167→1.147, max-alloc 55.2GB, 186 TFLOP/s/GPU, traceback 0 — 24종 캐시 프리빌드 완료, **sub1 GPU 8장 비움(vLLM 미재기동)**. 사고·교훈: reasoning 소실(폐기 후 재생성), OxAlpha 1,000/일 상한 → 심판 캘리브레이션 전용 → 08-27 퇴출(B 미검증), B r1 DD 정체(타임스탬프 dir 직접 resume), 특수토큰 4중 가드. 미소비: A 시드 ~200k·B r2 85k·리젝 16.6k(재개 절차는 README). 20개 언어 chat 보강 미착수 | `examples/alpha/sdg/ko_chat/README.md` |
| Pre-training stage2 (DiLoCo 2노드) | **완주 2026-08-22** iter 26,832, train 1.145/1.141, valid(P3) 1.1658. 커리큘럼 자연→P2→P2b→P3 + bias-sync + 블록-순환 샤딩 적용. **stage3 없음(사용자 확정)**. 1-노드 재저장본 `outputs/alpha_baseline_48L_stage2_20260822_123916/checkpoints`. 완주 벤치마크 평가 결과는 미기록 — 실행했다면 여기 갱신 | `STAGE2_CURRICULUM_LOG.md` |
| **LC-B 128K@CP8 CPT** | **완주 2026-08-27** 320 iters (4.03B tokens, GBS 96·LR 7.5e-6 constant), final valid 1.6744/PPL 5.34. **최종 ckpt = `outputs/alpha_baseline_48L_lc_b_resume_20260826_223836/checkpoints` iter 320** (optim 포함 153GB) — **SFT load는 반드시 이 경로** (run1 033011은 iter150에서 검증 중단, resume1 172926은 iter222 크래시 — 둘 다 최종본 아님). 중간 검증 5종 PASS: NIAH 200/200·위치별 NLL 위치-단조(+0.089@120k)·표준벤치 Δ−0.10pp(망각 無)·CP8 backward 등가·GBS 노이즈 분석 (`outputs/lc_b_midrun_eval/`). 사고 2건: 자동연계 NaN 게이트 오탐(수정 bb279c9)·재개1 rank6 cuDNN driver 실행실패(일과성 판정, 재발 시 GPU6 하드웨어 의심). **최종 평가 완료(08-27)**: 벤치 11종 Δ−0.26pp(망각 無)·NIAH 4k~131k 200/200·**256K 95%(LC-A 42.5% 대비 — LC-B가 만든 외삽력)**·384K 0%(실사용 창 ~256K)·512K는 fla 커널 층당 작업공간 한계로 측정 불가 — 정본 `study/lc_b_final_eval.md`. RULER 풀 하니스(13태스크)는 미구축 | `configs/training/lc_b.yaml`·`lc_b_resume.yaml` 헤더, `LC_DATASETS.md` |
| **LC-A 32K@CP4 THD CPT** | **완주 2026-08-26** iter 1113 (14B tokens), final valid **1.4420 / PPL 4.23** (iter100 1.5357 대비 −0.094). iter100 조기 검증 GO(NLL 위치-단조, NIAH 160/160), 사고 1건(iter170 리터럴 EOD) 수리 포함. 최종 ckpt `outputs/alpha_baseline_48L_lc_a_resume_20260823_070651/checkpoints`(weights-only). 롤링 검증(`outputs/lc_a_early_eval/run_evals.sh`)은 미실행 — 4-GPU 창구 필요, 사용자 판단 | `configs/training/lc_a.yaml`, `study/lc_a_early_eval.md`, KNOWN_ISSUES 08-23 |
| LC 진입 게이트 | 판정 1~6 통과, LC-A GO. qk-clip은 LC preset에서 제거(TE 2.9 thd 비호환, max logit 19.4 = 임계 1/5) | `LC_ENTRY_GATE.md` |
| GDN CP + varlen/THD 스티치 | main 흡수(c9f65ad). CP{1,2,4} 32K 실데이터 등가 1.2e-4. 잠복버그 4건 수정 | `gdn_cp_port.md` |
| Muon chunked offload 백포트 | S0~S5 완료, main 흡수. 128K@CP8 72.5GB(OOM) → 54.9~58.8GB GO | `MUON_OFFLOAD_BACKPORT.md` |
| Nemotron specialized 데이터 | P2/P2b/P3 커리큘럼으로 투입 완료 (stage2 완주에 포함). 라이선스 주의: Multiple-Choice/Generative subset은 DeepSeek-v3 산출물(의무 조항), Wiki-Rewrite/Scientific-Coding은 CC BY-SA/GFDL; CC-Code-v1은 NVIDIA Open Data | `STAGE2_CURRICULUM_LOG.md` |
| Claude Code 설정 영속화 | `~/.claude`가 휘발 경로였음 → `CLAUDE_CONFIG_DIR=/home/work/vidsearch/.claude-config/<노드>` (2026-08-25). 재시작 후 `setup_pai_megatron_env*.sh`가 자동 source | `/home/work/vidsearch/setup-claude.sh` |

## 열린 사용자 결정

(없음 — 2026-08-25 SFT 1노드 확정으로 종결)
