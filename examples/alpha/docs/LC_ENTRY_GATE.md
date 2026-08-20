# LC Phase 진입 게이트 — P3 종료 직후 실행 체크리스트 (2026-08-18 작성)

**트리거**: P3(stage2) 학습 종료 → 8×H100 유휴 → **LC-A(32K) 학습 시작 전에 이 게이트를 통과해야 한다.**
GPU가 P3에 점유된 동안 준비만 완료해 둔 두 검증 트랙(GDN CP 클러스터 검증 + FlashQLA
커널 벤치)을 한 창구에서 순서대로 소화한다. 예상 소요: 반나절 내외 (§2가 대부분).

**왜 이 순서인가**: GDN CP는 LC의 전제 조건(32K 초과는 CP 필수 — logits+활성값 80GB 초과)이고,
FlashQLA는 그 위의 커널 최적화다. CP가 실패하면 FlashQLA 채택 여부와 무관하게 LC-B가
막히므로 CP 검증이 먼저다. FlashQLA 벤치의 CP 시나리오(lc_a_cp{4,8}, lc_b_cp8)는
gdn-cp head-split의 rank-local 형상(전체 seq × heads/cp)을 재현하므로 CP 검증과 독립적으로
돌릴 수 있다 — 단, 채택 판단은 두 결과를 함께 본다.

---

## §0. 사전 준비 (GPU 불필요 — P3 종료 전에 미리 확인 가능)

- [ ] `git fetch origin && git checkout feature/gdn-context-parallel` 가능한지 확인
      (브랜치는 origin에 push되어 있음; 로컬 worktree는 `.claude/worktrees/gdn-cp`)
- [ ] FlashQLA 격리 설치 확인: `ls /home/work/vidsearch/envs/flashqla_poc/pylibs/flash_qla`
      (없으면 `examples/alpha/study/flashqla_poc.md` §설치 재현 — GPU 불필요, ~5분)
- [ ] `CUDA_DEVICE_MAX_CONNECTIONS` override 미사용 확인 — **conn=8 throughput lever(+2.7%)는
      CP>1과 양립 불가** (Megatron assert). train.sh 기본값 1이면 OK.
- [ ] P3 마지막 체크포인트 경로 확보 (§1.3 로드 스모크에 사용)
- [ ] wandb 오염 방지 확인: 게이트의 모든 학습형 검증은 `mock` data preset → train.sh가
      자동으로 wandb 차단. `--save`도 타임스탬프 디렉토리로 자동 유도 — 라이브 체크포인트
      디렉토리 직접 지정 금지.

## §1. GDN CP 클러스터 검증 (전제 조건 트랙)

**전체 절차·명령·판정 기준은 [`gdn_cp_port.md`](gdn_cp_port.md) "H100 클러스터 검증 러너북"이
단일 정본이다** (feature 브랜치에 있음; 머지 후엔 이 디렉토리에 존재). 요약:

| 단계 | 내용 | 판정 | 예상 |
|---|---|---|---|
| §1 | 포팅 유닛테스트 (sm90 + 실 NCCL, torchrun 2/4-rank) | ALL PASSED, fwd max diff 0.0 | ~5분 |
| §2 | analysis_24L profile×mock, CP{1,2,4,8} × 16 iter + QK-Clip 강제 발동 | init 성공, loss 궤적 CP=1 근접, Muon/QK-Clip 무크래시 | ~1.5h |
| §3 | P3(stage2) ckpt를 CP=2로 로드 스모크 (read-only) | torch_dist 복제 로드 + loss 연속 | ~30분 |
| §4 | LC 메모리 프로파일: 32K@CP4, 32K@CP8, 128K@CP8 (MBS=1) | max-alloc < ~76GB, 실측치를 gdn_cp_port.md에 기입 | ~1h |

§4 결과가 **LC-A의 CP 값 선정 근거**가 된다 (CP=4 권장 기준선; 메모리 여유와 §2 처리량을
함께 보고 결정). 128K가 selective recompute로 OOM이면 full recompute 재시도(러너북 §4 참조).

**통과 시**: `feature/gdn-context-parallel`를 main에 머지하고 브랜치 삭제 (커밋 규칙 §2),
실측치가 기입된 gdn_cp_port.md 갱신 커밋 포함. **실패 시**: LC-A를 32K CP=1로 겨우 시작할 수
있는지(§4의 CP=1 32K 메모리 실측 추가 필요) 검토하되, LC-B(64K+)는 CP 수정 전까지 보류.

## §1.5. THD 문서 격리 검증 (LC-A 데이터 경로 전제 조건)

**배경 (2026-08-20 발견)**: 현행 문서 격리(`--reset-attention-mask` dense mask)는 비용이
O(seq²)라 32K에서 샘플당 **1 GiB**, 128K에서 17 GiB — LC에서 유지 불가. `LC_DATASETS.md`
§5의 "패킹 filler가 LC 신호를 오염시키지 않는다" 논거가 이 격리에 기대므로, LC-A는
**THD/cu_seqlens 방식 문서 격리**(varlen attention + GDN 커널 state 리셋)로 전환해야 한다.
격리 자체를 끄는 대안은 Llama 3 보고(LC CPT에서 cross-doc 차단 중요)와 상충 → 비권장.

**준비 완료 (GPU 불필요분, `feature/gdn-varlen-thd` 브랜치)**: MambaStack/MambaLayer/
GDN mixer에 PackedSeqParams 관통 배선(fla `cu_seqlens` + causal_conv1d `seq_idx`),
`AlphaMambaModel` 서브클래스(서브모듈 무수정), helper.py cu_seqlens 로직 공용화
(+단일 세그먼트 윈도우에서 `seqlens.max()` 빈 텐서 크래시 잠재버그 수정),
CPU 테스트 7종 통과. GPU 검증 항목은 `tests/test_gdn_varlen_thd.py`의 skipif 3종 +
아래 스모크.

| 단계 | 내용 | 판정 | 예상 |
|---|---|---|---|
| a | `python -m pytest tests/test_gdn_varlen_thd.py -v` (GPU 3종: fla varlen 등가성 / conv seq_idx 등가성 / 격리 네거티브 컨트롤) | ~~ALL PASSED~~ **✅ 2026-08-20 분석 노드에서 10/10 통과** (커널 테스트는 소메모리라 P3 대기 불필요했음). P3 후 재실행은 회귀 확인용 | ~2분 |
| b | 1-GPU 스모크: `bash train.sh baseline_48L smoke mock --reset-position-ids --no-create-attention-mask-in-dataloader --micro-batch-size 1` | iter 진행 + NaN 없음 (stage1.yaml 주석의 "iter 1 AssertionError"가 사라졌는지) | ~10분 |
| c | 4096 등가성 A/B: mock 30 iter, ①현행(reset-attention-mask dense) vs ②THD(위 플래그) — loss 궤적 비교 | 근접(비트일치 아님 — ②는 GDN 문서 간 state까지 격리하므로 미세 개선 방향의 차이만 허용) | ~30분 |
| d | **QK-Clip×THD**: b/c 실행 중 `max_attention_logit` 로그 정상 확인 (TE fused-attn return_max_logit이 thd 포맷에서 동작하는지 — cuDNN 엔진 부재 크래시 이력 있는 조합) | 무크래시 + 로그값 유효 | b/c에 포함 |

**⚠️ 스코프 한계 — THD+CP 미해결**: 이 브랜치는 **CP=1에서의 THD**만 검증한다.
§1.4에서 LC-A가 CP≥2로 결정되면 추가 작업 필요: megatron_patch helper의 `CP>1 패킹 금지`
가드 해제 + 업스트림 `get_thd_batch_on_this_cp_rank` 방식 배치 슬라이싱 이식.
(GDN CP는 head-split이라 rank가 전체 seq를 보므로 cu_seqlens 의미론은 그대로 —
attention 쪽 TE thd+CP 경로만 통합하면 됨.) §1.4 메모리 실측에서 **32K@CP1(MBS1,
recompute)** 항목을 추가 측정할 것 — 통과하면 LC-A를 CP=1+THD로 시작하고 THD+CP는
LC-B 전 과제로 미룰 수 있다.

**충돌 주의**: `feature/gdn-context-parallel`과 이 브랜치는 둘 다 `gated_deltanet.py`를
수정한다. 머지 순서: gdn-cp 먼저(§1 통과 시) → varlen-thd를 그 위로 rebase.

## §2. FlashQLA 커널 벤치 (최적화 트랙)

배경·설치·판정 기준의 정본: [`../study/flashqla_poc.md`](../study/flashqla_poc.md).

```bash
cd <repo-root>
PYTHONPATH=/home/work/vidsearch/envs/flashqla_poc/pylibs \
  python examples/alpha/study/flashqla_bench.py --device cuda:0
```

- 스크립트가 GPU 점유(>2GiB) 시 스스로 중단 — P3가 정말 끝났는지의 이중 안전장치.
- 첫 호출에 TileLang JIT 컴파일 포함(`first_call_s`로 분리 기록). 전체 ~30–60분.
- 결과 JSON은 `examples/alpha/study/`에 타임스탬프로 저장 — 커밋할 것.

**판정** (flashqla_poc.md §판정 기준의 요약):
1. 정확도: qla_vs_oracle 오차가 fla_vs_oracle과 동일 자릿수 (fwd + grads 전부)
2. `g≈0`(무감쇠, AutoCP 최악 조건)에서 qla_vs_fla 오차 급증 없음
3. 성능: `lc_a_cp8`/`lc_b_cp8`에서 fwd+bwd 유의미 개선 (참고 주장: fwd 2–3×, bwd 2×)

## §3. 게이트 판정표

| # | 항목 | 트랙 | 실패 시 |
|---|---|---|---|
| 1 | GDN CP 수치 정합 (H100) | §1.1 | LC 전면 보류, 포팅 디버그 |
| 2 | Muon+QK-Clip+EP8×CP full-stack | §1.2 | 〃 |
| 3 | P3 ckpt CP>1 로드 | §1.3 | 〃 (LC는 이 ckpt에서 시작하므로 하드 게이트) |
| 4 | 32K 메모리 (CP4 또는 CP8) | §1.4 | LC-A 보류 |
| 5 | 128K@CP8 메모리 | §1.4 | LC-A는 GO 가능, LC-B만 보류 (recompute 재시도 후) |
| 6 | THD 문서 격리 (커널 등가성 + 스모크 + QK-Clip 상호작용) | §1.5 | LC-A 보류 — dense mask는 32K에서 불가하므로 대체 경로 없음 (격리 포기 결정은 별도 레시피 승인 필요) |
| 7 | FlashQLA 정확도 | §2 | fla로 LC 진행 (성능 손해만, 기능 무손실) |
| 8 | FlashQLA 성능 | §2 | 〃 — 채택 포기 판단도 유효한 결론, poc.md에 기록 |

**1–4 + 6 통과 = LC-A GO.** 5는 LC-B 전용, 7–8은 채택 여부만 좌우한다.
단 LC-A의 CP가 ≥2로 결정되면 §1.5의 THD+CP 통합(스코프 한계 참조)까지 완료해야 GO.

## §4. 통과 후 작업 (각각 별도 커밋 단위)

1. **gdn-cp 머지**: 실측치 기입 → main 머지 → 브랜치 삭제 → push.
2. **FlashQLA 채택 시** (판정 6·7 통과):
   - `gated_deltanet.py:355` 호출부에 가드 스왑 구현 — `ALPHA_GDN_BACKEND=flashqla` env로
     선택, 기본은 fla 폴백 (`HAVE_FLA` 가드와 같은 패턴). eval용 A100 박스는 SM90 미지원이라
     폴백이 자동 적용됨.
   - 짧은 mock 학습 A/B(fla vs flashqla, 동일 seed 수십 iter)로 loss 궤적 근접 확인 후 커밋.
   - GVA-native 호출(q/k `repeat_interleave` 제거)은 **별도 후속 커밋** — fla 폴백 경로와
     형상이 달라지므로 스왑 커밋에 섞지 말 것.
   - 설치를 정식화: flashqla_poc.md의 절차를 클러스터 셋업 스크립트
     (`setup_pai_megatron_env_multinode.sh`)에 반영.
3. **LC-A 시작**: `docs/LC_DATASETS.md`의 32K 데이터(`cpt_lc_packed_32k`, 15.56B tok) +
   §1.4에서 선정한 CP로 LC training preset 작성. **커널 백엔드 전환은 stage 경계인 지금이
   적기** — LC 도중 교체 금지 (Muon QGKV 수정과 같은 원칙).

## 미결 사항 (게이트 범위 밖, LC 레시피 설계 시 결정)

- 2노드 사용 여부: CP는 노드 내로 한정(무IB), 노드 간은 DiLoCo DP. LC-A 예산(12–16B tok)이
  1노드로 충분한지 먼저 계산.
- THD/packed + CP (문서 격리): gdn_cp_port.md "Out of scope" 참조. LC 레시피에서 문서 격리
  채택 여부 결정 후 별도 안건.
- 16-rank 동시 TileLang JIT 첫 호출 컴파일 폭주 여부 (§2에서 단일 GPU만 확인됨) —
  FlashQLA 채택 시 mock A/B에서 함께 관찰.
