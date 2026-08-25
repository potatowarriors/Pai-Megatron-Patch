# Alpha 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push**한다. 상태는 여기에만 쓴다 — Claude auto-memory에 쓰지 않는다
(메모리는 컨테이너·노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.

_마지막 갱신: 2026-08-25 (SFT 1노드 확정, RL 트랙은 범위 밖으로 보류)_

## 진행 중

| 트랙 | 상태 (2026-08-25) | 다음 할 일 | 정본 |
|---|---|---|---|
| **LC-A** 32K@CP4 THD | 학습 중. run `outputs/alpha_baseline_48L_lc_a_resume_20260823_070651`(iter 100 재개본), 예산 1,113 iters·GBS 384·LR 7.5e-6 constant, ~4분/iter, **완주 08-26 새벽 예상**. iter100 조기 검증 GO (NLL 격차 위치-단조, NIAH 160/160). 사고 1건(iter170 리터럴 EOD) 수리 후 재개 | 완주 → LC-B 자동 연계(아래 행). 최종 ckpt 롤링 검증(`outputs/lc_a_early_eval/run_evals.sh`)은 LC-B와 GPU 겹침 — iter1113 ckpt가 보존되므로 사후 실행 (4-GPU 창구 필요, 사용자 판단) | `configs/training/lc_a.yaml`, `study/lc_a_early_eval.md`, KNOWN_ISSUES 08-23 |
| **LC-B** 128K@CP8+offload | 준비 완료 (1dfc01b): 128k pad16 4종 8.65B, `lc_b_128k_blend.yaml`(LC 46% + 32k filler 재사용 54%), `lc_b.yaml`(GBS 96·320 iters·load=LC-A ckpt) | **자동 연계 armed (08-25)**: `scripts/launch_lc_b_after_lc_a.sh`(가동 중) — LC-A 완주 감지(사망 시 연계 중단) → `lc_b_preflight.py --deep`(08-25 전 항목 PASS 사전 실증) → 10-iter 스모크 게이트(loss 유한·첫 loss<3·max-alloc ≤65GB·오프로더 배너) → 본 런. 실패는 `outputs/LC_B_CHAIN_ALERT.txt` 후 정지. 병행: RULER@128K 하니스 구축. 미착수(의도): EN 다문서 합성 | `LC_DATASETS.md`, `MUON_OFFLOAD_BACKPORT.md`, `configs/training/lc_b.yaml` 헤더 |
| **SFT 준비** | 64k 21종 + 128k 6종 변환·`verify_sft_bins` 전 PASS. `sft_40b_blend.yaml`(SWE 1-pass 앵커 40B) + `sft_128k_blend.yaml`(5.54B). preset `sft_64k.yaml`(CP4+offload, DP2) / `sft_128k.yaml`(CP8). interleaved-thinking 규약 정비 완결(a2f8894), agentic_v2 미편입 확정. **effort/budget NVIDIA 레시피 재현 확정(사용자, 08-25)**: 변환기 `--medium-effort`·`--truncate-reasoning-budget` 구현, 테스트 34/34·34/34, 스모크 bins 게이트 PASS (`SFT_RL_DATASETS.md` §2.6) | ① LongBlocks 변환기(1.5% 슬롯) ② RULER ③ 본 런 — LC 완료 후 ckpt 경로만 채움. ④ ~~effort 재변환~~ 완료(08-25): 3종 변환·게이트 PASS·블렌드 반영(chat 21→20+budget 1, ep 2.67/1.19). 실측 ETA **64k 9,537 iters ≈ 9.7일 + 128k 1,322 iters ≈ 1.7일**. **1노드 학습 확정(사용자, 2026-08-25)** — 2-node 단축안 기각. 유휴 노드는 SFT 후 RL 환경 검증용(별도 워크스페이스, 아래 보류 표) | `INTERLEAVED_THINKING.md`, `SFT_RL_DATASETS.md`, `DATA_PREP_LOG.md` |
| **ko_chat 합성** (sub1) | r2 무인 체인 가동 중 — 트랙A 번역+재생성 잔여 풀 745k(08-25 19:00 기준 122k 완료, 1.55 rec/s) + 트랙B r1 20k 진행(r2 100k 대기). reasoning 소실 결함 수정(5fd67cb) 후 재가동(08-24). **OxAlpha(OpenRouter 무료)는 1,000 요청/일 상한 실측(08-25)** → 생성 백엔드 불가, **심판 캘리브레이션 전용**으로 확정(사용자): `calib_daily.sh` 매일 09:05 KST 900건 재심판(effort high), 결과로 B export 임계 보정. 생성 경로는 구현·검증 후 스위치 off. 리터럴 special-token 4중 가드. r1_a 29,122행 think 보충 완료 | r2 완료(~08-29) → 캘리브레이션 리포트(`calibrate_judge.py --report`)로 export 임계 확정 → KoChat-v1 이관(export 게이트 필수) → idxmap 64k 변환 → 블렌드 한국어 비중(chat 21% 내). LC-B 2노드화 시 sub1 GPU 창구 종료 | `examples/alpha/sdg/ko_chat/README.md` |

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
| Pre-training stage2 (DiLoCo 2노드) | **완주 2026-08-22** iter 26,832, train 1.145/1.141, valid(P3) 1.1658. 커리큘럼 자연→P2→P2b→P3 + bias-sync + 블록-순환 샤딩 적용. **stage3 없음(사용자 확정)**. 1-노드 재저장본 `outputs/alpha_baseline_48L_stage2_20260822_123916/checkpoints`. 완주 벤치마크 평가 결과는 미기록 — 실행했다면 여기 갱신 | `STAGE2_CURRICULUM_LOG.md` |
| LC 진입 게이트 | 판정 1~6 통과, LC-A GO. qk-clip은 LC preset에서 제거(TE 2.9 thd 비호환, max logit 19.4 = 임계 1/5) | `LC_ENTRY_GATE.md` |
| GDN CP + varlen/THD 스티치 | main 흡수(c9f65ad). CP{1,2,4} 32K 실데이터 등가 1.2e-4. 잠복버그 4건 수정 | `gdn_cp_port.md` |
| Muon chunked offload 백포트 | S0~S5 완료, main 흡수. 128K@CP8 72.5GB(OOM) → 54.9~58.8GB GO | `MUON_OFFLOAD_BACKPORT.md` |
| Nemotron specialized 데이터 | P2/P2b/P3 커리큘럼으로 투입 완료 (stage2 완주에 포함). 라이선스 주의: Multiple-Choice/Generative subset은 DeepSeek-v3 산출물(의무 조항), Wiki-Rewrite/Scientific-Coding은 CC BY-SA/GFDL; CC-Code-v1은 NVIDIA Open Data | `STAGE2_CURRICULUM_LOG.md` |
| Claude Code 설정 영속화 | `~/.claude`가 휘발 경로였음 → `CLAUDE_CONFIG_DIR=/home/work/vidsearch/.claude-config/<노드>` (2026-08-25). 재시작 후 `setup_pai_megatron_env*.sh`가 자동 source | `/home/work/vidsearch/setup-claude.sh` |

## 열린 사용자 결정

(없음 — 2026-08-25 SFT 1노드 확정으로 종결)
