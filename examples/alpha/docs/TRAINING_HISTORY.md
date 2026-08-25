# Alpha 학습 이력 (Stage 0 마이그레이션 ~ 레거시 Stage 2-3)

`examples/alpha/CLAUDE.md` "Training Plan"에서 2026-08-25 이관. **역사 기록이며 현재 상태가 아니다** —
현재 진행 중인 트랙과 다음 할 일은 [`STATUS.md`](STATUS.md). 이후 이력:
stage2 커리큘럼(P2/P2b/P3, 2026-08-22 완주)은 `STAGE2_CURRICULUM_LOG.md`, LC 게이트는 `LC_ENTRY_GATE.md`.

## Stage 0 — Qwen3.5 dimensioning + DSV3 MoE migration (2026-05-20 완료 ✅)

**목표**: Qwen3-Next reference에서 Qwen3.5 dimensioning + DeepSeek-V3 MoE routing으로 baseline 정렬, alpha 전용 v5 tokenizer 채택, RMSNorm 표준화.

**5번의 smoke test** (각 단계, 2-iter, 8 H100, mock data, exit 0 / NaN 0 검증):

| # | 변경 | 검증 결과 |
|---|------|-----------|
| 1 | Qwen3.5 dim (head_dim 256, 184 experts × 512) + DSV3 routing | per-rank 3.21B, total 15.03B, loss 11.99→11.44 |
| 2 | `apply-layernorm-1p` 제거 (Qwen3.5 정렬) | iter 1 forward 동등 (loss 11.99280 동일), iter 2 분기 |
| 3 | WD policy `apply_wd_to_qk_layernorm` (Stage 1+2 통일, Qwen3-Next NVIDIA 레시피) | grad norm 무영향, loss curve 안정 |
| 4 | tokenizer v5 (beta path 참조) | per-rank 3.26B, total 15.08B, loss 12.07→11.51 |
| 5 | tokenizer v5 (in-repo `tokenizer_v5/`) | iter 1 lm_loss byte-perfect 일치 → migration 완전 동등 입증 |

**옛 checkpoint 호환성**: Stage 2-3까지의 모든 checkpoint(`outputs/alpha_baseline_48L_*`)는 (1) head_dim, (2) expert 수, (3) MoE FFN dim, (4) vocab size가 모두 변경되어 새 baseline과 **structurally incompatible**합니다. 새 stage 1을 from-scratch로 시작해야 함.

**Stage 1 (post-migration) 시작 명령**:
```bash
bash train.sh baseline_48L pretrain_auxfree stage1_v5_korean_web
# (또는 다른 v5-tokenized data preset)
```

---

## Pre-migration history (legacy, 호환 X)

> 아래 Stage 1/2-1/2-2/2-3은 Qwen3-Next-호환 baseline (128 experts × 768 FFN, vocab 151,936, 1p RMSNorm) 시절의 학습 기록입니다. 새 baseline과 구조적으로 호환되지 않으므로 **재현 또는 continual learning 불가**. Historical reference (학습 곡선, 발견된 bug fix 등)로만 보존.

## Stage 1: Initial Pre-training (완료)
- **Dataset**: kormo_50pct (~1.13T tokens)
- **Iterations**: 400k + 40k cooldown = 440k
- **Run**: `bash train.sh baseline_48L pretrain_auxfree kormo_50pct` (current preset, with auxfree routing)
- **Checkpoint**: `outputs/alpha_baseline_48L_cooldown_20260209_200711/checkpoints`

## Stage 2-1: Continual Pre-training (200k/400k에서 중단)
- **Dataset**: stage2 blend (~3.1T tokens) — Korean Web + Math + Nemotron CC-HQ + Nemotron Code v2
- **전략**: WSD continual learning (12k warmup + 348k stable + 40k decay)
- **All-to-All dispatcher** — DeepEP 대비 ~7% 빠름 (단일 노드 벤치마크)
- **QK LayerNorm WD**: `no-weight-decay-cond-type: apply_wd_to_qk_layernorm` — Stage 1에서 발견된 gamma 폭발 방지
- **Checkpoint**: `outputs/alpha_baseline_48L_stage2_20260301_015403/checkpoints` (200k iter)
- **중단 사유**: swap memory 포화로 throughput 불안정

## Stage 2-2: Continual Pre-training (200k→400k, cosine) (진행 예정)
- **Dataset**: stage2 blend 이어서 (consumed_samples 유지)
- **전략**: cosine decay (500 warmup + 199.5k cosine decay), WSD에서 변경
- **Config**: `configs/training/stage2_2.yaml` + `configs/data/stage2_2.yaml`
- **변경사항 (vs Stage 2-1)**:
  - LR scheduler: WSD → cosine warmup+decay
  - num-workers: 32 → 8 (swap memory 문제 해결)
  - Nesterov 버그 자동 수정 (yaml_to_flags가 `muon-use-nesterov: true` → `--muon-use-nesterov` 정확히 emit)
  - `no-load-optim: true`로 optimizer/scheduler 리셋, data position 유지
- **실행**: `bash train.sh baseline_48L stage2_2 stage2_2`

## Stage 2-3: Continual Pre-training (375k→800k, 4× LR boost)
- **Dataset**: stage2 blend (consumed_samples 유지, `data/stage2_2.yaml` 그대로)
- **전략**: cosine decay (2k warmup + 423k cosine), 4× LR boost
- **Config**: `configs/training/stage2_3.yaml`
- **변경사항 (vs Stage 2-2)**:
  - LR: 1e-4 → 4e-4 (4× boost), min-lr: 1e-5 → 4e-5
  - Warmup: 500 → 2000 (LR jump 안정화)
  - LayerNorm WD: `apply_wd_to_qk_layernorm` → `apply_wd_to_all_layernorm`
- **실행**: `bash train.sh baseline_48L stage2_3 stage2_2`

