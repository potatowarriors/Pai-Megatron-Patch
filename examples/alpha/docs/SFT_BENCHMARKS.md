# SFT 벤치마크 스위트 — 설계·판정·인프라 (2026-08-29)

SFT 본 런(`sft_128k_full`, 2,448 iters, save 300)의 체크포인트별 평가 체계.
기존 base-스타일 표준 11종([EVALUATION.md](EVALUATION.md))과 별개로, **chat/thinking 모드**의
instruct 능력을 측정한다. 참조 좌표는 DSV4 post-training 표 + Nemotron 3 Ultra Table 10
(SFT 데이터가 Ultra 레시피이므로 후자가 1차 준거). 진행 상태는 [STATUS.md](STATUS.md).

## 0. 결정 로그 (사용자, 2026-08-29)

| 결정 | 내용 |
|---|---|
| SWE-bench·Terminal-Bench | **필수 요구**. Backend.AI 노드는 docker 불가(§4) → **외부 docker 호스트 gpu06 DinD 컨테이너로 해결·검증 완료**(2026-08-29, [EVAL_DOCKER_NODE.md](EVAL_DOCKER_NODE.md)). 하니스 구축 대기 |
| judge | **gemini-3.7-flash** 확정(사용자, 2026-08-29). 키 검증 완료(`examples/alpha/.env`). 러너는 provider-agnostic(Gemini/OpenAI 호환) |
| 실행 노드 | sub1 (유휴 8×H100). main1은 SFT 학습 전용 |

## 1. DSV4 테이블 26종 판정

채택 8 / 대체·보강 5 / 차단·불가 13.

| 벤치 | 판정 | 근거 |
|---|---|---|
| MMLU-Pro | ✅ T1 | lm_eval 0.4.12 내장 (CoT EM) |
| GPQA Diamond | ✅ T1 | gated 승인·캐시 완료 |
| LiveCodeBench | ✅ T1 | 공식 하니스 로컬 subprocess 실행 — docker 불필요 |
| HMMT Feb / (AIME) | ✅ T1 | MathArena 공개분(2025 Feb) + lm_eval aime25 |
| SimpleQA-Verified | ✅ T3 | 1,000문항, mini judge 채점 |
| MRCR | ✅ T2 (≤256K) | openai/mrcr. 1M은 모델 창(262K, 실사용 ~256K) 초과 |
| SWE Verified | ⚠️ 프록시→풀 | 풀 에이전틱은 docker 경로 확정 후. 그 전엔 oracle-patch + sb-cli 채점 |
| Terminal Bench 2.0 | ⚠️ 대기 | docker 경로 확정 후 (프록시 없음) |
| IMOAnswerBench / HLE | 후순위 | 15B-A1.8B에 변별력 낮음. HLE는 gated(HF 토큰 필요) |
| Chinese-SimpleQA | 🔁 대체 | 중국어는 학습 언어 아님 → KMMLU 유지 + KoChat 판정(T3) |
| Codeforces / Apex / CorpusQA-1M / BrowseComp / HLE-tools / MCPAtlas / Toolathlon / GDPval-AA / SWE Pro·Multilingual | ❌ | 실시간 저지·비공개 scaffolding·1M 창·웹서치 스택·MCP 팜·유료 서비스·docker |

**보강 (SFT 데이터 정합)**: IFEval(IF-Chat fan-out), BFCL v4(tool-call 템플릿 분기, 로컬 AST 채점),
RULER 64/128K(STATUS의 "RULER 풀 하니스 미구축" 해소), 표준 11종(망각 게이트, T4).

## 2. 티어 구성·주기

ckpt 주기 = 300 iters ≈ 27h. 스위트는 그 안에 완주해야 추이가 그려진다.

| 티어 | 벤치 | 주기 | 예상 소요(sub1) |
|---|---|---|---|
| T1 코어 | MMLU-Pro, GPQA-D, AIME25+HMMT25Feb, LiveCodeBench, IFEval | 매 ckpt | ~4–6h |
| T2 롱 | RULER 64/128K(태스크당 100샘플), MRCR 128/256K, NIAH 연속성 | 격 ckpt | ~2h |
| T3 판정 | SimpleQA-Verified, KoChat 품질, (최종만) Arena-Hard류 | 격 ckpt | ~2h |
| T4 회귀 | 기존 표준 11종 (LC-B 63.72 대비 망각 감시) | ckpt 2–3개마다 | ~3h |
| SWE 프록시 | oracle-patch 생성 → sb-cli 클라우드 채점 | 격 ckpt | 생성 ~1h |

생성 규약: thinking 기본(reasoning parser로 `<think>` 분리), 최종 평가에서
medium-effort 렌더(`{reasoning effort: efficient}`) 2차 패스로 토큰 효율 곡선 측정.
**서빙·평가는 tokenizer_v5 apply_chat_template만 사용** (INTERLEAVED_THINKING.md §7-4).

## 3. 인프라 (sub1)

```
main1 SFT save → sub1 감시 → ① run_convert.sh(EP=8, ~30–40분)
  → ② vLLM 서빙: alpha 1-GPU 레플리카 DP4~6 (15B bf16 30GB, attn 6층이라 128K KV ~1.6GB)
  → ③ lm_eval(local-chat-completions) + 커스텀 러너(LCB·MRCR·SimpleQA·HMMT·SWE프록시)
  → ④ wandb alpha-evals 업로드 → 서버 종료, 다음 ckpt 대기
```

- 서빙 venv: sub1 `/tmp/alpha-eval-venv` — **vllm==0.25.1 핀** (08-24 서빙 패리티 게이트가
  이 버전 기준: argmax·top5 일치, KL ≤ 0.0013). 플러그인은
  `project_s/NeMo-RL/examples/configs/alpha/vllm_alpha_plugin/` editable 설치.
  /tmp은 재부팅 시 소멸 → requirements 스냅샷을 `vidsearch/tools/alpha_eval/`에 유지.
- 하니스: lm_eval 0.4.12 (`vidsearch/tools/lmeval0412` 격리 설치와 동일 버전).
- 러너·오케스트레이터 코드: `examples/alpha/eval_sft/` (이 리포).
- 베이스라인: 전 스위트를 **LC-B iter320 hfmodel**로 먼저 완주(하니스 검증 겸 기준점).

## 3.5 sub1 서빙 환경 — vllm 0.25.1 + CUDA 13 compat (2026-08-29)

서빙 venv: `/home/work/vidsearch/tools/alpha_serve_venv` (NFS, 재부팅 유지). vllm==0.25.1
(패리티 검증 버전) + `vllm_alpha_plugin` editable. 설치 시 함정 3건:

1. **PIP_CONSTRAINT 해제 필수.** NGC 이미지 전역 `/etc/pip/constraint.txt`가 torch를
   시스템 NGC 버전으로 고정해 vllm 의존성 해결이 깨진다. 모든 pip·serve에 `PIP_CONSTRAINT=`.
2. **transformers 5.16.1로 딸려 옴** → `regex`·`safetensors`·`typing_extensions` 핀 연쇄
   충돌. `pip install -U "regex>=2025.10.22" "safetensors>=0.8.0" "typing_extensions>=4.15"`로 해소.
   (HF remote-code modeling_alpha.py의 5.x 비호환은 **플러그인 경로와 무관** — 서빙은 플러그인 사용.)
3. **vllm 0.25.1 = CUDA 13 빌드** (`_C.so`가 libcudart.so.13). sub1 드라이버 535(CUDA 12.8)로는
   기동 불가(`driver too old 12080`). **해법: CUDA 13 forward-compat** — `setup_nemo_rl_env.sh`와
   동일하게 `cuda-compat-13-2_595.91.07` 설치, sub1 `/usr/local/cuda/compat/lib.real`의
   libcuda 570→595 교체(사용자 승인 2026-08-29). 595는 CUDA 12·13 모두 지원해 기존 Pai
   시스템 torch(cu12.8)도 회귀 없음(검증됨). **백업**: `/home/work/vidsearch/tools/cuda_compat13/backup_570_*`.
   **재부팅 시 컨테이너 초기화되면 이 교체를 재수행해야 함** (deb·백업은 NFS에 보존:
   `/home/work/vidsearch/tools/cuda_compat13/`).

서빙 기동: `bash eval_sft/serve_alpha.sh <hfmodel> [max_len] [DP] [port]` (§3).

## 4. docker 부재 — 실측과 경로

main1·sub1 공통 실측(2026-08-29): sudo는 passwordless로 존재하나 CapBnd에
`cap_sys_admin` 없음 + Seccomp filter 활성(`unshare` EPERM) + `/dev/fuse` 없음.
→ 컨테이너 안에서는 root여도 dockerd/rootless podman/apptainer 전부 기동 불가.
Backend.AI 세션 생성 시점 설정이라 내부에서 해결 불가.

경로 3택 (SWE·T-Bench 공통, 미정):

| 경로 | 비용 | 비고 |
|---|---|---|
| 관리자 privileged 세션(또는 docker socket 마운트) | 0 | 정석. 유휴 노드 1대면 충분 |
| Modal/Daytona 클라우드 샌드박스 | SWE 회당 ~$50–150, TB 회당 ~$10–40 | 하니스 공식 지원. 추론은 sub1 vLLM이라 무과금 |
| 별도 docker 호스트 (**CPU 전용으로 충분**) | 호스트 확보 | 16–32코어/64GB/300GB↑ 권장. sub1→호스트 ssh 역터널로 엔드포인트 연결 |

## 5. judge — gemini-3.7-flash (확정 2026-08-29)

키: `examples/alpha/.env` GEMINI_API_KEY (gitignore됨). generateContent 라이브 호출 검증 완료.
Google Generative Language API v1beta 엔드포인트
(`models/gemini-3.7-flash:generateContent`). 러너는 provider-agnostic으로 작성해
나중에 OpenAI-호환 judge로도 교체 가능. 비용은 flash급이라 SFT 런 전체 수 달러 내외.
공식 관례 참고: SimpleQA-Verified=Gemini 2.5 Pro, Arena-Hard-v2=프런티어 judge.

## 6. 작업 큐

- [x] sub1 유휴·환경 실사, docker 가능성 판정 (2026-08-29)
- [ ] Phase 0: vLLM 0.25.1+플러그인 설치 → LC-B iter320 서빙 스모크 + 패리티 + 템플릿/파서 검증
- [ ] Phase 1: lm_eval API 하니스 + T1 태스크 구성 (HMMT 커스텀 yaml 포함) → LC-B 베이스라인
- [ ] Phase 2: LCB·MRCR·RULER 러너 + T2 베이스라인
- [ ] Phase 3: judge 러너(T3, 키 수령 후) + SWE 프록시 레인
- [ ] Phase 4: 오케스트레이터(ckpt 감시→변환→스위트→wandb) 상시화
- [ ] docker 경로 확정 시: SWE-bench Verified·Terminal-Bench 2.0 온보딩
