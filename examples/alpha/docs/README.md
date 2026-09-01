# Alpha 문서 색인 (docs/ + study/)

**규칙**: 새 문서는 여기 **한 줄**로 등록한다. `CLAUDE.md`에는 문서 요약을 쓰지 않는다.
현재 진행 상태는 [`STATUS.md`](STATUS.md) 한 곳에만 쓴다. 사고·수정 서사는 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)에 쓰고 CLAUDE.md 함정 표에는 한 줄만.

## 지금 볼 것

| 문서 | 한 줄 |
|---|---|
| [STATUS.md](STATUS.md) | 트랙별 현재 상태 · 다음 할 일 · 정본 링크. 세션 종료 시 갱신·커밋 |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 사고·수정 기록 전문 28건 (2026-05~08). CLAUDE.md 함정 표의 원문 |

## 가이드 (안정, 참조용)

| 문서 | 한 줄 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 하이브리드 GDN+Attention+MoE 아키텍처 상세 |
| [SETUP.md](SETUP.md) | 학습 환경 구축 가이드 |
| [CONVERSION.md](CONVERSION.md) | Megatron→HF 변환 + lm-eval 벤치 가이드 |
| [EVALUATION.md](EVALUATION.md) | LM-Evaluation-Harness 벤치마크 가이드 |
| [MUON.md](MUON.md) | Muon optimizer 사용 가이드 |
| [PARAMETERS.md](PARAMETERS.md) | 파라미터 구성·계산법 (`calculate_parameters.py`) |
| [DEBUGGING.md](DEBUGGING.md) | VSCode 디버거 시나리오 4종 + breakpoint 위치 |
| [WANDB.md](WANDB.md) | wandb 통합 가이드 |
| [MIGRATION_251125.md](MIGRATION_251125.md) | Megatron-LM 250908→251125 마이그레이션 |
| [REFACTORING.md](REFACTORING.md) | 2025-11-29 코드 리팩토링 변경 정리 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 실험 로그 템플릿 (실험 #1 baseline 24L만 기록) |

## 결정·측정 기록 (시간순)

| 문서 | 기간 | 한 줄 |
|---|---|---|
| [TRAINING_HISTORY.md](TRAINING_HISTORY.md) | 2026-02~05 | Stage 0 마이그레이션(05-20) + 레거시 Stage 1/2-1/2-2/2-3 이력 (현재 상태 아님) |
| [PRETRAIN_DATA_PIPELINE.md](PRETRAIN_DATA_PIPELINE.md) | 2026-05~07 | v5 토크나이즈 교훈(75h→4h), preflight 6-phase, stage2 재토크나이즈 1.686T, best-fit packing |
| [V2_PIPELINE_VERIFICATION.md](V2_PIPELINE_VERIFICATION.md) | 2026-05-26 | 평가 파이프라인 통합 + v1→v2 검증 감사 |
| [THROUGHPUT_INVESTIGATION.md](THROUGHPUT_INVESTIGATION.md) | 2026-06~07 | 처리량 전수조사 + 2노드 스케일링 실측 (§5: IB 없음, bf16-reduce 1.15×) |
| [throughput_optimization.md](throughput_optimization.md) | 2026-06 | nsys 프로파일·레버 A/B·mid-training 적용 프로토콜 (그림·생성 스크립트 동봉) |
| [A100_SINGLE_GPU_EVAL.md](A100_SINGLE_GPU_EVAL.md) | 2026-07-22 | A100 단일-GPU 평가 환경 + DiLoCo stage2 첫 벤치 |
| [STAGE2_CURRICULUM_LOG.md](STAGE2_CURRICULUM_LOG.md) | 2026-07-29~08-22 | stage2 P2/P2b/P3 커리큘럼 + DiLoCo bias-sync 수정 + 완주 기록 |
| [DATA_PREP_LOG.md](DATA_PREP_LOG.md) | 2026-07-31~ | LC·SFT 데이터 준비 상태 스냅샷·타임라인·결정·작업 큐 |
| [LC_DATASETS.md](LC_DATASETS.md) | 2026-08 | LC 데이터셋 전수 분석, 스테이지 계획, filler 표 |
| [LC_ENTRY_GATE.md](LC_ENTRY_GATE.md) | 2026-08-18~22 | LC 진입 게이트 판정 1~6 (GDN CP·THD·FlashQLA), 최종 LC-A GO |
| [LC_REPACK_RUNBOOK.md](LC_REPACK_RUNBOOK.md) | 2026-08-21 | THD+CP용 `--pad-doc-multiple 16` 재패킹 절차 + 검증 스크립트 |
| [LC_FILLER_HANDOFF.md](LC_FILLER_HANDOFF.md) | 2026-08-22~23 | filler specialized 인계·완료 기록 |
| [gdn_cp_port.md](gdn_cp_port.md) | 2026-08 | GDN Context Parallel 포팅 + THD 잠복버그 규명 분석노트 3 |
| [MUON_OFFLOAD_BACKPORT.md](MUON_OFFLOAD_BACKPORT.md) | 2026-08-22 | PR #6244 chunked offload 백포트 S0~S5, 128K@CP8 GO |
| [SFT_RL_DATASETS.md](SFT_RL_DATASETS.md) | 2026-08 | SFT·RL 데이터 자산 49종, Ultra 파이프라인 설계, 예산·epoch 근거 |
| [INTERLEAVED_THINKING.md](INTERLEAVED_THINKING.md) | 2026-08-24 | think-히스토리 규약 정비 (DSV4 분기·IF fan-out·keepthink) + 새 SFT 셋 규칙 8건 (8 = effort 렌더 플래그) |
| [SFT_BENCHMARKS.md](SFT_BENCHMARKS.md) | 2026-08-29~ | **SFT 벤치 스위트 정본** — 운영 절차(§2.5)·구성요소(§2.6)·프론티어 규약(§3.4)·태스크 정의(§3.6·§3.9)·측정 결과(§3.8)·게이트 G1~G3·A1~A4(§7) |
| [EVAL_DOCKER_NODE.md](EVAL_DOCKER_NODE.md) | 2026-08-29 | 에이전틱 벤치(SWE·Terminal) 실행용 외부 docker 호스트 gpu06 DinD — 접속·복구·재구축 runbook |

## study/ (실측·규명·스터디)

| 문서 | 한 줄 |
|---|---|
| [../study/gradient_reduce.md](../study/gradient_reduce.md) | gradient reduction 스터디 노트 |
| [../study/2node_project_report.md](../study/2node_project_report.md) | 2노드 프로젝트 종합 보고 (2026-07-12~15) |
| [../study/diloco_pilot.md](../study/diloco_pilot.md) | DiLoCo 파일럿 전체 실측·검증 (2026-07-14) |
| [../study/mirror_loss_aliasing.md](../study/mirror_loss_aliasing.md) | 샤드×blend 거울상 loss 시소 규명 + 재현 `mirror_loss_repro.py` (2026-08-17) |
| [../study/nondeterminism_probe.md](../study/nondeterminism_probe.md) | 실행 간 비결정 원천 = TE fused attn bwd + 재현 `.py` (2026-08-22) |
| [../study/flashqla_poc.md](../study/flashqla_poc.md) | FlashQLA GDN 커널 벤치 — 채택 보류, 128K에서 재평가 (2026-08-22) |
| [../study/lc_a_early_eval.md](../study/lc_a_early_eval.md) | LC-A iter100 위치별 NLL + NIAH 조기 검증 GO (2026-08-23) |
| [../study/lc_b_final_eval.md](../study/lc_b_final_eval.md) | LC-B 최종 평가 — 벤치 3자(망각 無)·NIAH 4K→384K 스펙트럼(실사용 창 ~256K)·NLL 곡선·512K 불가 원인 (2026-08-27) |
| [../study/lc_512k_eval.md](../study/lc_512k_eval.md) | 512K 추론 확장 — YaRN 프로파일 × RULER 512K 판정 그리드 (도구·태스크·런처·판정 규칙·결과) (2026-09-01~) |
| [../study/netbench/](../study/netbench/) | 노드 간 TCP/NCCL 실측 스크립트 (IB 부재 확인) |

## sdg/ (합성 데이터 파이프라인 — docs 밖, 코드와 동거)

| 문서 | 한 줄 |
|---|---|
| [../sdg/ko_chat/README.md](../sdg/ko_chat/README.md) | 한국어 SFT chat 합성 — 트랙A 번역+재생성 / 트랙B 네이티브, 교사 Gemma-4-31B + OxAlpha(무료), 무인 체인·게이트·GPU 창구·함정 (2026-08-23~) |
| [../sdg/identity/README.md](../sdg/identity/README.md) | 정체성 SFT/RL 데이터 생성 (DataDesigner, identity_card 단일 진실 원천) (2026-08-07~10) |

## chat/ (사람이 직접 대화 — docs 밖, 코드와 동거)

| 문서 | 한 줄 |
|---|---|
| [../chat/README.md](../chat/README.md) | SFT ckpt 채팅 서빙 (vLLM :8001 + OpenWebUI :8080, main1 GPU3) — 벤치 fleet 과의 설정 차이·G2 의도적 FAIL 사유·접속 경로 (2026-08-31~) |

## 리포 루트 docs/

| 문서 | 한 줄 |
|---|---|
| [../../../docs/CUSTOM_TRAINING_FEATURES.md](../../../docs/CUSTOM_TRAINING_FEATURES.md) | Megatron-LM-251125 비-upstream 기능 5건 전문 + 테스트 명령 |
| [../../../docs/MUON_CLIP_ANALYSIS.md](../../../docs/MUON_CLIP_ANALYSIS.md) | MuonClip 분석 |
| [../../../docs/NGC_ENV_REBUILD.md](../../../docs/NGC_ENV_REBUILD.md) | NGC 환경 재구축 |

## 이 리포 밖 (project_s 워크스페이스)

| 문서 | 한 줄 |
|---|---|
| `../../../../NEMO_RL_SETUP.md` | NeMo-RL post-training 환경 운영 가이드 (alpha 브리지·vLLM 플러그인·refit 검증) — RL은 이 리포 범위 밖, 보류 |
| `../../../../ALPHA_POSTTRAIN_PROGRESS.md` | post-training 인프라 준비 경과·게이트 8종 (2026-08-12~21) |
| `../../../../RESTORE_AFTER_REBOOT.md` | 컨테이너 재시작 후 전체 복원 runbook |
