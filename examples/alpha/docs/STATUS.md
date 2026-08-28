# Alpha 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push**한다. 상태는 여기에만 쓴다 — Claude auto-memory에 쓰지 않는다
(메모리는 컨테이너·노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.

_마지막 갱신: 2026-08-27 12:30 (LC-B 최종 평가 완료 — **main1 GPU 유휴, SFT 개시 가능**. 결과 정본: `study/lc_b_final_eval.md`)_

## 진행 중

| 트랙 | 상태 (2026-08-25) | 다음 할 일 | 정본 |
|---|---|---|---|
| **SFT 준비** | 64k 21종 + 128k 6종 변환·`verify_sft_bins` 전 PASS. `sft_40b_blend.yaml`(SWE 1-pass 앵커 40B) + `sft_128k_blend.yaml`(5.54B). preset `sft_64k.yaml`(CP4+offload, DP2) / `sft_128k.yaml`(CP8). interleaved-thinking 규약 정비 완결(a2f8894), agentic_v2 미편입 확정. **effort/budget NVIDIA 레시피 재현 확정(사용자, 08-25)**: 변환기 `--medium-effort`·`--truncate-reasoning-budget` 구현, 테스트 34/34·34/34, 스모크 bins 게이트 PASS (`SFT_RL_DATASETS.md` §2.6) | ① LongBlocks 변환기(1.5% 슬롯) ② RULER ③ 본 런 — LC 완료 후 ckpt 경로만 채움. ④ ~~effort 재변환~~ 완료(08-25): 3종 변환·게이트 PASS·블렌드 반영(chat 21→20+budget 1, ep 2.67/1.19). **GBS 192/96 확정(사용자, 2026-08-28)** — 12.58M tok/iter 전 스테이지 상수 관례 복원(초기 GBS 64는 Ultra 샘플 수만 미러한 오류; Ultra SFT 실제 18.9M). ETA 불변: **64k 3,179 iters ≈ 9.7일 + 128k 441 iters ≈ 1.7일**. `sft_64k.yaml` load = LC-B iter320 정본 기입 완료. **개시는 합성 SFT 데이터 세션의 추가분 합류 후(사용자, 08-28) — 수동 기동**. **1노드 학습 확정(사용자, 2026-08-25)** — 2-node 단축안 기각. 유휴 노드는 SFT 후 RL 환경 검증용(별도 워크스페이스, 아래 보류 표) | `INTERLEAVED_THINKING.md`, `SFT_RL_DATASETS.md`, `DATA_PREP_LOG.md` |

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
