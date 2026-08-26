# Alpha 현재 상태판

**규칙**: 세션 종료 시 자기 트랙의 행을 갱신하고 **커밋·push**한다. 상태는 여기에만 쓴다 — Claude auto-memory에 쓰지 않는다
(메모리는 컨테이너·노드별이라 다른 세션이 못 본다). 날짜는 절대 표기. 끝난 트랙은 "완료" 절로 내리고 정본 링크만 남긴다.

_마지막 갱신: 2026-08-26 (LC-A 완주 → LC-B 본 런 가동)_

## 진행 중

| 트랙 | 상태 (2026-08-25) | 다음 할 일 | 정본 |
|---|---|---|---|
| **LC-B** 128K@CP8+offload | **본 런 가동 (08-26 03:30)**: run `outputs/alpha_baseline_48L_lc_b_20260826_033011`, 320 iters·GBS 96·~233s/iter → **완주 ~08-27 새벽**. 10-iter 스모크(THD×128K×CP8×offload 첫 실데이터) 전 게이트 실질 PASS: loss 1.43~1.59(LC-A 연속)·max-alloc 59.1GB·오프로더 8/8·222~231 TFLOP/s/GPU. 자동 연계는 판정 오탐 1건(정상 필드 "number of nan iterations: 0"에 nan 매치)으로 HOLD → 게이트 수정 후 수동 launch | 완주 → 평가. 병행: RULER@128K 하니스 구축. 미착수(의도): EN 다문서 합성 | `LC_DATASETS.md`, `MUON_OFFLOAD_BACKPORT.md`, `configs/training/lc_b.yaml` 헤더 |
| **SFT 준비** | 64k 21종 + 128k 6종 변환·`verify_sft_bins` 전 PASS. `sft_40b_blend.yaml`(SWE 1-pass 앵커 40B) + `sft_128k_blend.yaml`(5.54B). preset `sft_64k.yaml`(CP4+offload, DP2) / `sft_128k.yaml`(CP8). interleaved-thinking 규약 정비 완결(a2f8894), agentic_v2 미편입 확정. **effort/budget NVIDIA 레시피 재현 확정(사용자, 08-25)**: 변환기 `--medium-effort`·`--truncate-reasoning-budget` 구현, 테스트 34/34·34/34, 스모크 bins 게이트 PASS (`SFT_RL_DATASETS.md` §2.6) | ① LongBlocks 변환기(1.5% 슬롯) ② RULER ③ 본 런 — LC 완료 후 ckpt 경로만 채움. ④ ~~effort 재변환~~ 완료(08-25): 3종 변환·게이트 PASS·블렌드 반영(chat 21→20+budget 1, ep 2.67/1.19). 실측 ETA **64k 9,537 iters ≈ 9.7일 + 128k 1,322 iters ≈ 1.7일**. **1노드 학습 확정(사용자, 2026-08-25)** — 2-node 단축안 기각. 유휴 노드는 SFT 후 RL 환경 검증용(별도 워크스페이스, 아래 보류 표) | `INTERLEAVED_THINKING.md`, `SFT_RL_DATASETS.md`, `DATA_PREP_LOG.md` |
| **ko_chat 합성** (sub1) | r2 무인 체인 가동 중 — 트랙A 번역+재생성 잔여 풀 745k(08-25 19:00 기준 122k 완료, 1.55 rec/s) + 트랙B r1 20k 진행(r2 100k 대기). reasoning 소실 결함 수정(5fd67cb) 후 재가동(08-24). **OxAlpha(OpenRouter 무료)는 1,000 요청/일 상한 실측(08-25)** → 생성 백엔드 불가, **심판 캘리브레이션 전용**(사용자): `calib_daily.sh` 매일 09:05 KST 900건. **1일차 결과(08-26, 887행)**: Gemma 심판 무정보(전부 5) vs OxAlpha factuality 4.49·<4 가 10%, 원인은 지어낸 상호·가격·시설 정보(금융·부동산·여행 취약). Gemma-strict 프록시 가설 기각(재현율 0.23) → **원천 차단: B r2 `--strict-facts` 프롬프트 규칙** 적용(체인 B 재기동, r1 지문 보존). **08-26 13:45**: A r2 297k/745k **일시정지**(B 에 서버 양보, 컷 후 재개). **B r1 사고·복구**: 08-24 재시작 런이 새 타임스탬프 디렉토리(`ko_chat_b_r1_08-24-2026_200831`)에 쓰는 것을 계측이 놓쳐 이틀간 미감시 → DD 스케줄러 내부 정체(vLLM 요청 0)로 0.02 rec/s → 정체 드라이버 kill 후 **타임스탬프 디렉토리를 `--dataset-name` 으로 직접 지정해 `--resume`**(26/40 row group·12,765행 보존, 새 디렉토리 없음). 잔여 7,235행, 96 병렬 서버 독점 중. 08-26 캘리브레이션 887건은 폐기된 구 런 표본이었음(glob 누락) — 새 런 미측정. **사용자 결정(08-26)**: ① SFT 는 LC-B 완주(~00:10) 직후 시작, 1차 트랜치 = **A think 행 전량(~265k ≈ 0.58B) + B r1 완주분 전량(20k, 미검증)** — r1 완주 시 자동 컷(데드라인 23:30) ② **한국어 chat = 영어 chat_v3 와 동일 epoch**(chat 슬롯 8B 를 bin 토큰 수 비례, ≈2.2 ep). f1 세션 협조 요청은 상대 승인 대기(미전달). IF 61k reasoning 소급은 0.9% — 2차용 | r1 완주(오늘 저녁 예상) → `cut_tranche.py --b-mode all` → `alpha-SFT-KoChat-v1/{trackA,trackB}.jsonl` → 64k 변환(IF `--fanout-train-turns --medium-effort`)·verify·블렌드(f1 또는 ko_chat 세션) → 스모크 → SFT(~1.5h 지연 허용). 체인 B → r2 100k(strict-facts) 자동. 08-27 09:05 캘리브레이션 = B 새 런 첫 측정(r1/r2 출처 태깅). **함정**: 체인 B 의 자동 `--resume` 은 구 디렉토리 지문과 비교해 새 디렉토리로 새로 시작함 — 정체 시 반드시 타임스탬프 디렉토리 직접 지정 | `examples/alpha/sdg/ko_chat/README.md` |

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
| **LC-A 32K@CP4 THD CPT** | **완주 2026-08-26** iter 1113 (14B tokens), final valid **1.4420 / PPL 4.23** (iter100 1.5357 대비 −0.094). iter100 조기 검증 GO(NLL 위치-단조, NIAH 160/160), 사고 1건(iter170 리터럴 EOD) 수리 포함. 최종 ckpt `outputs/alpha_baseline_48L_lc_a_resume_20260823_070651/checkpoints`(weights-only). 롤링 검증(`outputs/lc_a_early_eval/run_evals.sh`)은 미실행 — 4-GPU 창구 필요, 사용자 판단 | `configs/training/lc_a.yaml`, `study/lc_a_early_eval.md`, KNOWN_ISSUES 08-23 |
| LC 진입 게이트 | 판정 1~6 통과, LC-A GO. qk-clip은 LC preset에서 제거(TE 2.9 thd 비호환, max logit 19.4 = 임계 1/5) | `LC_ENTRY_GATE.md` |
| GDN CP + varlen/THD 스티치 | main 흡수(c9f65ad). CP{1,2,4} 32K 실데이터 등가 1.2e-4. 잠복버그 4건 수정 | `gdn_cp_port.md` |
| Muon chunked offload 백포트 | S0~S5 완료, main 흡수. 128K@CP8 72.5GB(OOM) → 54.9~58.8GB GO | `MUON_OFFLOAD_BACKPORT.md` |
| Nemotron specialized 데이터 | P2/P2b/P3 커리큘럼으로 투입 완료 (stage2 완주에 포함). 라이선스 주의: Multiple-Choice/Generative subset은 DeepSeek-v3 산출물(의무 조항), Wiki-Rewrite/Scientific-Coding은 CC BY-SA/GFDL; CC-Code-v1은 NVIDIA Open Data | `STAGE2_CURRICULUM_LOG.md` |
| Claude Code 설정 영속화 | `~/.claude`가 휘발 경로였음 → `CLAUDE_CONFIG_DIR=/home/work/vidsearch/.claude-config/<노드>` (2026-08-25). 재시작 후 `setup_pai_megatron_env*.sh`가 자동 source | `/home/work/vidsearch/setup-claude.sh` |

## 열린 사용자 결정

(없음 — 2026-08-25 SFT 1노드 확정으로 종결)
