# Custom Training Features (Alpha, Megatron-LM-251125 비-upstream 기능)

루트 `CLAUDE.md`에서 2026-08-25 이관. 기능 5건의 CLI·구현 위치·테스트 명령 전문.
새 비-upstream 기능을 추가하면 여기에 절을 추가하고 루트 CLAUDE.md 표에는 한 줄만.

Non-upstream features added on Megatron-LM-251125 (Stage1 3건 + 2026-08-22 LC 준비 2건).
(4번째 비-upstream 기능인 **DiLoCo 2노드 학습** — IB 없는 클러스터용 저통신 분산 — 은
`megatron_patch`/submodule이 아니라 `examples/alpha/diloco_patch.py`에 살며,
[`examples/alpha/CLAUDE.md`](examples/alpha/CLAUDE.md) § Multi-Node Training 참조.)

## 1. Step-wise Global Batch Size Schedule

Backport of upstream main's `--step-batch-size-schedule`. Lets GBS step up at token thresholds during training (more flexible than the linear `--rampup-batch-size`).

**CLI**:
```bash
--step-batch-size-schedule "0:768 250B:1536 500B:3072 750B:6144"
```

- Format: `T0:BS0 T1:BS1 ...`. Suffixes K/M/B/T (1e3/1e6/1e9/1e12) supported.
- Thresholds are **tokens** (converted to samples via `--seq-length`). First threshold must be `0`.
- `--global-batch-size` is auto-derived as the max BS in the schedule. Setting it explicitly to anything other than that max is a hard error.
- **Mutually exclusive** with `--rampup-batch-size`.
- Sample-based training only (`--train-samples`, not `--train-iters`).

**Implementation**:
- `backends/megatron/Megatron-LM-251125/megatron/core/num_microbatches_calculator.py` — `parse_step_batch_size_schedule()`, `StepBatchSizeScheduleCalculator`
- `backends/megatron/Megatron-LM-251125/megatron/training/arguments.py` — argparse + validation
- `backends/megatron/Megatron-LM-251125/megatron/training/training.py::update_train_iters` — segment-wise iteration count
- Note: this is one of the rare cases where `backends/megatron/Megatron-LM-*/` is patched directly (the num-microbatches calculator is core infrastructure not exposed for patching from `megatron_patch/`).

## 2. Progressive Auxiliary Dataset Blending

Smoothly ramp up an arbitrary number of auxiliary datasets (e.g. code, math, multi-lingual) on top of a static base blend. Each aux has its own start/end thresholds and start/final ratios. Lives entirely in `megatron_patch/data/` — Megatron core is untouched for the dataset side.

**CLI**:
```bash
--progressive-blend-config /path/to/blend.yaml
# Optional: tag every sample with its source dataset for diagnostic logging
--progressive-blend-emit-source
```

- **Mutually exclusive** with `--data-path`, `--data-args-path`, `--train-data-path`, etc.
- The base blend is used for validation/test splits. Progressive ramping applies to train only.

**YAML config format**:
```yaml
base:
  data_path: ["0.7", "/path/dclm", "0.3", "/path/korean_web"]

# Default unit for all aux schedules (overridable per-aux). One of: tokens|samples|iterations.
schedule_unit: tokens   # default if omitted

auxiliary:
  - name: code
    data_path: ["1.0", "/path/code_corpus"]
    schedule:
      start_at: "100B"        # 100B tokens
      reach_full_at: "500B"
      start_ratio: 0.0
      final_ratio: 0.10

  - name: math
    data_path: ["1.0", "/path/math_corpus"]
    schedule:
      unit: iterations        # override default for this aux
      start_at: 50000
      reach_full_at: 150000
      start_ratio: 0.01
      final_ratio: 0.08

  - name: multilingual
    data_path: ["0.5", "/path/lang_a", "0.5", "/path/lang_b"]
    schedule:
      unit: samples
      start_at: 0
      reach_full_at: "100M"
      start_ratio: 0.02
      final_ratio: 0.20
```

**Schedule unit semantics**:
| `unit` | Meaning | Robust against |
|---|---|---|
| `tokens` (recommended) | `sample_idx * seq_length` | seq_length and GBS changes |
| `samples` | global sample index | GBS changes (not seq_length) |
| `iterations` | training iteration | (precomputed from `--step-batch-size-schedule` or constant GBS) |

`iterations` is intuitive (matches wandb x-axis) but each iter represents different work amounts under step-wise GBS.

**Per-aux schedule semantics** (linear ramp):
- `sample_idx <= start_at` → `start_ratio`
- `start_at < sample_idx < reach_full_at` → linear interpolation
- `sample_idx >= reach_full_at` → `final_ratio` (constant thereafter)
- Constraint: `Σ aux_ratios <= 1.0` at every idx (validated at build time on a 200-point grid).

**Reproducibility**:
- Each `__getitem__(idx)` uses `np.random.default_rng(seed=[args.seed, idx])` for source selection — bit-reproducible across runs.
- Checkpoint resume: Megatron's `MegatronPretrainingSampler` advances from `consumed_train_samples`, so the same `idx` is replayed and the schedule continues seamlessly.

**Implementation**:
- `megatron_patch/data/progressive_mix_dataset.py` — `LinearRampSchedule`, `ProgressiveMixDataset`, `parse_progressive_blend_config`, `build_iter_to_samples_table`
- `megatron_patch/data/__init__.py::_build_progressive_blend_dataset` — builds base + N aux `BlendedMegatronDataset` and wraps them
- `backends/megatron/Megatron-LM-251125/megatron/training/arguments.py` — argparse + mutual-exclusion checks

## Combining Both Features

Typical Alpha Stage1 recipe:
```bash
--seq-length 4096 \
--train-samples <total> \
--step-batch-size-schedule "0:768 250B:1536 500B:3072 750B:6144" \
--progressive-blend-config configs/stage1_blend.yaml
```
GBS scales with budget; one or more auxiliary datasets ramp in at chosen thresholds independently.

## 3. Muon QGKV Split for Gated Attention

Fix to a silent bug in the upstream Muon optimizer's QKV-split path. The upstream
detector matched only `linear_qkv.weight` (standard 3-way fused projection),
which **silently failed** for Alpha's Gated Attention parameter `linear_qgkv.weight`
(4-way fused: Q, **Gate**, K, V). As a result every Alpha checkpoint trained
before this fix ran Newton-Schulz on the entire `[Q|Gate|K|V]` block as one
matrix instead of orthogonalizing each sub-projection independently.

**No CLI changes required** — the fix is automatic when `optimizer: dist_muon`
is selected and `--muon-no-split-qkv` is not passed (default).

**What the fix does**:
- Recognizes `linear_qgkv.weight` and assigns 4-way split shapes
  `[q_per_group, q_per_group, kv_channels, kv_channels]`.
- Per-parameter `qkv_split_shapes` attribute lets one optimizer instance
  serve both standard 3-way and gated 4-way attention in the same model.
- Adds an INFO log of matched/unmatched counts at startup, plus a WARNING when
  `split_qkv=True` is on but no fused QKV name pattern matches (catches future
  attention layouts like MLA before they go unnoticed).

**Implementation**:
- `backends/megatron/Megatron-LM-251125/megatron/core/optimizer/muon.py`
  — `get_megatron_muon_optimizer` (param marking + warning), `TensorParallelMuon.orthogonalize`
  (per-param shapes via `getattr`).
- Same precedent as features #1 and #2: optimizer is core infra not exposed for
  patching from `megatron_patch/`, so the submodule is edited directly.

**Behavioral note** for in-flight Alpha runs: applying this fix changes optimizer
dynamics mid-training (K/V updates were previously underweighted relative to
Q/Gate). Prefer a stage boundary for adoption.

**Out of scope**: MLA (DeepSeek-V3-style latent attention) is still untouched —
the WARNING log is what alerts you if a future MLA model is trained with Muon.

## 4. Muon Chunked Optimizer-State Offload (PR #6244 백포트, 2026-08-22)

Muon의 momentum과 fp32 master weight를 **CPU 상주(pinned) 캐노니컬**로 두고,
optimizer step에서만 청크 단위로 GPU에 스트리밍(H2D→갱신→D2H)한다.
128K@CP8의 Muon 바닥짐(~45GB/rank, dp=1이라 샤딩 불가) 해소가 목적.

**CLI**:
```bash
--chunked-optimizer-state-offload --optimizer-state-offload-chunk-size-mb 256
# 선택: --optimizer-state-offload-fraction 0.5   (0이면 비활성)
```

- 지원 경로: `--optimizer dist_muon`(LayerWise) 또는 `--use-distributed-optimizer`
  (Adam). plain `muon`은 validate_args가 거부(무음 no-op 방지).
- 제약: optimizer state를 저장/로드하면 `--ckpt-format torch_dist` 필수,
  `--async-save` 불가, full-iteration CUDA graph 불가.
- dist_muon 체인 [Muon, Adam*]에서 **Muon 자식만 오프로드**, Adam 자식(임베딩·
  norm·router)은 GPU 상주 — upstream(형제 DistOpt 라우팅)과의 의도적 구조 분기.
- 실측: 128K@CP8 max-alloc 72.5GB(OOM) → **54.9~58.8GB GO**; analysis_24L에서
  −31% 메모리 / +5~7% per-iter; 체크포인트는 ON저장↔OFF/ON재개 상호 호환
  (스위치 3방향 save→load→save 비트 동일).
- **A/B 판정 주의**: alpha 풀모델은 실행 간 비결정(원천 = TE fused attention bwd,
  `examples/alpha/study/nondeterminism_probe.md`) — 등가 검증은 동일 구성 재실행의
  자기 산포 포락선으로 판정하고, 비트 검증은 결정론적 유닛 골든이 담당한다.

**Implementation** (단계별 검증 로그: `examples/alpha/docs/MUON_OFFLOAD_BACKPORT.md`):
- `megatron/core/optimizer/cpu_offloading/chunked_optimizer_state_offload.py` — 코어(신규, dev 원본)
- `megatron/core/optimizer/{optimizer,optimizer_config,distrib_optimizer,layer_wise_optimizer}.py` — 통합 훅
- `megatron/training/{arguments,training,checkpointing}.py` — 플래그·train_step 수명주기·저장 가드
- QK-Clip이 `main_param`(fp32 master)에 쓰는 계약은 train_step의
  `ensure_master_weights_for_param_sync()`로 고정 (LC preset은 qk-clip 제거라 실전 무관)

## 5. THD+CP 코어 잠복버그 수정 3건 (2026-08-22)

THD(packed)+CP 경로가 이 스냅샷에서 한 번도 실행된 적이 없어 잠복해 있던
upstream 버그들 — GDN THD+CP 스티치(megatron_patch 쪽)가 처음 경로를 밟으며 검출.
규명 전 과정: `examples/alpha/docs/gdn_cp_port.md` 분석노트 3.

- `megatron/core/utils.py::get_thd_batch_on_this_cp_rank` — `PackedSeqParams`
  import 누락(NameError) + 슬라이싱 루프 None 가드 부재(dense 버전에는 있음).
- `megatron/core/models/mamba/mamba_model.py` — **THD rope 미배선(진범)**:
  gpt_model과 달리 `get_rotary_seq_len`/`rotary_pos_emb()`에 packed_seq_params를
  안 넘겨 rope 테이블이 dense CP 관례로 rank별 zigzag 사전 슬라이스됨 → THD rope
  함수(자체 CP 슬라이싱)가 테이블 밖을 읽어 q/k NaN → 하류 MoE 라우터의 CUDA
  topk가 NaN에서 중복 인덱스 반환 → dispatcher a2a split 불일치 크래시.
  수정 = upstream gpt_model 미러(forward에 `packed_seq_params` 파라미터 추가 포함).

## Tests

```bash
# Step-wise GBS calculator (8 tests)
cd backends/megatron/Megatron-LM-251125 && \
  WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 \
  python -m pytest tests/unit_tests/test_step_batch_size_schedule.py -v --noconftest

# Progressive mix dataset (13 tests)
cd <repo-root> && python -m pytest tests/test_progressive_mix_dataset.py -v

# DiLoCo data-shard mappings — parity/block-cyclic + switch exactness (9 tests)
cd <repo-root> && python -m pytest tests/test_diloco_shard_view.py -v

# Muon QGKV split — full suite incl. oracle per-block independence (20 tests)
cd backends/megatron/Megatron-LM-251125 && \
  NVIDIA_PYTORCH_VERSION=25.06 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  MASTER_ADDR=localhost MASTER_PORT=29500 \
  python -m pytest tests/unit_tests/test_muon_optimizer.py -v --noconftest

# Muon chunked offload — 코어+플러밍(38) / layer_wise 통합·골든(8) (1 GPU)
cd backends/megatron/Megatron-LM-251125 && \
  CUDA_VISIBLE_DEVICES=0 NVIDIA_PYTORCH_VERSION=25.06 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  MASTER_ADDR=localhost MASTER_PORT=29500 \
  python -m pytest tests/unit_tests/test_chunked_offload_s1.py \
    tests/unit_tests/test_chunked_offload_s3_layerwise.py -v --noconftest

# Muon chunked offload — 체크포인트 스위치 3방향 비트동일 (3, conftest 필요)
cd backends/megatron/Megatron-LM-251125 && \
  CUDA_VISIBLE_DEVICES=0 NVIDIA_PYTORCH_VERSION=25.06 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  MASTER_ADDR=localhost MASTER_PORT=29500 \
  python -m pytest tests/unit_tests/dist_checkpointing/test_chunked_offload_s4.py -v

# GDN varlen/THD + CP — 유닛 33종 + 분산 등가 (CP{2,4}는 torchrun, THD는 비트동일 기대)
cd <repo-root> && python -m pytest tests/test_gdn_varlen_thd.py \
  tests/test_gdn_context_parallel.py tests/test_gdn_thd_cp_perm.py -v
torchrun --nproc_per_node=2 tests/test_gdn_context_parallel.py   # 또는 4
```

