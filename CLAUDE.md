# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pai-Megatron-Patch is a production-grade deep learning training toolkit for Large Language Models (LLMs) and Vision Language Models (VLMs) using NVIDIA's Megatron framework. It bridges high-level model definitions (HuggingFace) with high-performance distributed training (Megatron-LM/Megatron-Core).

**Core Design Philosophy**: Non-invasive patching. Functions are provided as patches rather than modifying Megatron-LM source code, allowing users to stay current with Megatron-LM updates.

## Module-Specific Guides

| Module | CLAUDE.md Location | Description |
|--------|-------------------|-------------|
| **Alpha Model** | [`examples/alpha/CLAUDE.md`](examples/alpha/CLAUDE.md) | GatedDeltaNet hybrid architecture, training, validation |
| **Checkpoint Converter** | [`toolkits/distributed_checkpoints_convertor/CLAUDE.md`](toolkits/distributed_checkpoints_convertor/CLAUDE.md) | HF↔Megatron conversion |
| **Megatron Patch** | [`megatron_patch/CLAUDE.md`](megatron_patch/CLAUDE.md) | Core library, Muon optimizer |
| **경로 스코프 규칙** | `.claude/rules/{megatron-submodule,sft-data,pretrain-data}.md` | 해당 경로를 건드릴 때만 자동 로드 |
| **Alpha 문서 색인 / 상태판** | [`examples/alpha/docs/README.md`](examples/alpha/docs/README.md) / [`STATUS.md`](examples/alpha/docs/STATUS.md) | 무엇이 어디에 · 지금 어디까지 |

## Architecture

### Directory Structure

- **`megatron_patch/`**: Core library (model/, data/, ssm/, training.py, arguments.py)
- **`examples/`**: Model-specific training scripts (run_mcore_*.sh)
- **`toolkits/`**: Utilities (checkpoint converters, data preprocessing)
- **`backends/`**: Git submodules (Megatron-LM versions, ChatLearn, verl)

### Training Pipeline Flow

```
examples/{model}/run_mcore_*.sh
  ↓
megatron_patch/arguments.py → initialize.py → training.py::pretrain()
  ↓
backends/megatron/Megatron-LM-*/megatron/core/ (Distributed execution)
```

## Common Commands

### Environment Setup
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
export PYTHONPATH=/path/to/Pai-Megatron-Patch:/path/to/backends/megatron/Megatron-LM-250908:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
```

### Data Preprocessing
```bash
# Pre-training
cd toolkits/pretrain_data_preprocessing/
bash run_make_pretraining_dataset.sh <vocab_file> <input_jsonl> <output_prefix> <workers>

# SFT
cd toolkits/sft_data_preprocessing/
bash convert_sft_dataset.sh <input_jsonl> <output_dir> <tokenizer_path>
```

### Checkpoint Conversion
```bash
# HuggingFace → Megatron
cd toolkits/model_checkpoints_convertor/{model}/
bash hf2mcore_*.sh <model_size> <hf_dir> <output_dir> <tp> <pp>

# Megatron → HuggingFace
bash mcore2hf_*.sh <model_size> <megatron_dir> <hf_output_dir> <tp> <pp>
```

### Training
```bash
cd examples/{model}/
bash run_mcore_{model}.sh <ENV> <MODEL_SIZE> <BATCH_SIZE> <GLOBAL_BATCH_SIZE> <LR> ...
```

## Key Technical Details

### Parallelism Strategy
| Type | Description | When to Use |
|------|-------------|-------------|
| TP (Tensor) | Splits weights across GPUs | Large models (TP=4 or 8) |
| PP (Pipeline) | Splits layers across GPUs | Memory constraints |
| EP (Expert) | Splits MoE experts | MoE models |
| CP (Context) | Splits sequences | Ultra-long contexts (>32K) |

**Rule of thumb**: 8×GPU with 70B model → TP=8, PP=1

### Checkpoint Formats
- **Legacy**: `model_optim_rng.pt` files
- **torch_dist**: Distributed sharded (recommended for 100B+)
- **HuggingFace**: `.safetensors` or `.bin`

### Optimizers
- **Adam/AdamW**: Standard, supports all parallelism
- **Muon** (`dist_muon`): ~2x faster convergence, requires TP support, NO CPU offloading
  - See [`megatron_patch/CLAUDE.md`](megatron_patch/CLAUDE.md) for details

## Model-Specific Notes

### Qwen Models
- Use `NullTokenizer`, GQA support, RoPe theta: 1000000

### MoE Models (DeepSeek-V3, Mixtral)
- Require `--moe-grouped-gemm`, set EP and ETP

### Hybrid SSM (Qwen3-Next, Alpha)
- **TP=1 required** (Mamba layers don't support TP > 1)
- See [`examples/alpha/CLAUDE.md`](examples/alpha/CLAUDE.md) for Alpha-specific guide

### LLaMA Models
- Use `SentencePieceTokenizer`, RoPe theta: 500000

## Custom Training Features (Alpha, Megatron-LM-251125 비-upstream)

전문·CLI·구현 위치·테스트 명령: [`docs/CUSTOM_TRAINING_FEATURES.md`](docs/CUSTOM_TRAINING_FEATURES.md). 여기에는 한 줄씩만.

| # | 기능 | CLI / 트리거 | 위치 |
|---|---|---|---|
| 1 | Step-wise GBS 스케줄 (토큰 임계마다 GBS 계단) | `--step-batch-size-schedule "0:768 250B:1536 …"` (`--rampup-batch-size`와 배타, `--train-samples` 전용) | submodule `num_microbatches_calculator.py`, `arguments.py`, `training.py` |
| 2 | Progressive auxiliary blend (aux 데이터셋 선형 램프) | `--progressive-blend-config blend.yaml` (`--data-path`류와 배타) | `megatron_patch/data/progressive_mix_dataset.py` |
| 3 | Muon QGKV 4-way split (Gated Attention `linear_qgkv`) | 자동 (`dist_muon`, `--muon-no-split-qkv` 아님). 시작 로그 `Muon QKV matcher` 확인 | submodule `optimizer/muon.py` |
| 4 | Muon chunked optimizer-state CPU offload (PR #6244 백포트) | `--chunked-optimizer-state-offload --optimizer-state-offload-chunk-size-mb 256` (torch_dist·동기 save 필수) | submodule `optimizer/cpu_offloading/`, `layer_wise_optimizer.py` |
| 5 | THD+CP 잠복버그 3건 (utils NameError·None 가드, mamba_model THD rope 미배선) | 자동 | submodule `core/utils.py`, `models/mamba/mamba_model.py` |
| + | DiLoCo 2노드 (IB 없는 클러스터) | `examples/alpha/launch_diloco.sh` | `examples/alpha/diloco_patch.py` (submodule 아님) |

테스트 전체 목록은 위 문서 § Tests. 대표: `tests/test_progressive_mix_dataset.py`, `tests/test_diloco_shard_view.py`,
submodule `tests/unit_tests/test_step_batch_size_schedule.py`, `test_muon_optimizer.py`, `test_chunked_offload_s*.py`, `tests/test_gdn_*.py`.

## Debugging Tips

### Common Issues
| Issue | Solution |
|-------|----------|
| OOM | Reduce batch size, enable AC=1 or AC=2, increase PP |
| Slow training | Enable Flash Attention (FL=true), check CUDA_DEVICE_MAX_CONNECTIONS=1 |
| Checkpoint load error | Verify TP/PP match, check Megatron version compatibility |
| Data loading error | Ensure .bin/.idx exist together, check tokenizer vocab |

### Logging
- TensorBoard: `OUTPUT_BASEPATH/tensorboard/`
- Checkpoints: `OUTPUT_BASEPATH/checkpoints/`

## Version Compatibility

### Megatron-LM Versions
| Version | Status | Use Case |
|---------|--------|----------|
| **251125** | Dev | Alpha, Muon optimizer |
| **250908** | Stable | Qwen3, DeepSeek-V3 |
| **250624** | Stable | Qwen3, Moonlight |

**Version Selection**: Set PYTHONPATH in training scripts (line 6 of `run_mcore_*.sh`)

### Framework Requirements
- PyTorch: ≥2.0 (2.3+ recommended)
- Transformer Engine: ≥2.9.0 (for Muon QK-Clip)
- CUDA: 11.8+ (12.1+ for FA3)

## Development Workflow

### Adding a New Model
1. Create `megatron_patch/model/{model_name}/`
2. Add tokenizer in `megatron_patch/tokenizer/`
3. Create conversion scripts in `toolkits/model_checkpoints_convertor/{model_name}/`
4. Add training script in `examples/{model_name}/`

### Modifying Training Logic
- **DON'T** modify `backends/megatron/Megatron-LM-*/` (submodule) by default.
- **DO** modify `megatron_patch/training.py` or add patches.
- **Exception**: core infrastructure not exposed for patching (e.g. the num-microbatches calculator) may be patched directly — see [Custom Training Features](#custom-training-features-alpha-stage1) for the precedent.

### 검증 규칙 (adopted 2026-08-29)

검증을 거치지 않은 결과는 믿을 수 없다. 검증을 건너뛰어 얻은 "성공"은 잠재 문제를 안은 채 다음 단계로 넘기는 것이다.

1. **검증이 깨지면 검증을 고친다 — 절대 건너뛰지 않는다.** 검증 스크립트·게이트가 환경 문제(라이브러리 몽키패치,
   버전 충돌 등)로 실패하면 그 원인을 제거해 검증을 복원한다. `--skip-validate`류 우회 플래그, 검증 단계 주석 처리,
   게이트 임계 완화로 "해결"하지 않는다. 우회 플래그는 **이미 통과한 산출물을 재사용**할 때만 쓴다.
   선례: 2026-08-29 NGC modelopt가 HF 로드를 깨뜨리자 `--skip-validate`로 우회하던 것을 되돌리고 modelopt import
   차단으로 검증을 복원 (`757fe93`).
2. **모든 작업은 검증을 완료한 뒤 진행한다.** 코드·설정·데이터·체크포인트 어느 단계든 "검증 없이 그냥 진행"은 없다.
   검증 수단이 없으면 먼저 **검증 방법을 제시**하고(단위 테스트, ON/OFF differential, smoke run, 수치 게이트 등)
   완료한 뒤 다음 단계로 간다. 보고에는 무엇을 어떻게 검증했고 어떤 수치가 나왔는지 그대로 적는다
   (예: `14181/14181 matched, ppl 5.91 PASS`). 실패·부분 통과·미실행을 통과처럼 쓰지 않는다.

### Commit & History Rules (adopted 2026-08-18)

이 저장소의 커밋 이력 관리 규칙. Claude Code 세션에서도 동일하게 적용한다.

1. **작업 단위 즉시 커밋.** 기능 하나가 완료·검증되면 그 자리에서 커밋한다.
   여러 주 작업을 한 번에 몰아 커밋하지 않는다. 커밋 메시지는
   `type(scope): summary` (feat/fix/docs/chore/perf/experiment/test) 관례를 따른다.
2. **브랜치 규율.** 실험은 `experiment/*` 브랜치에서. 결론이 나면 — 기각이어도 —
   기록 커밋으로 main에 흡수하고 브랜치는 삭제한다 (`TESTED & REJECTED` 커밋 선례).
   검증 대기 중인 기능 브랜치(`feature/*`)만 장기 유지하되 main 위로 주기적 rebase.
3. **서브모듈 수정 시 vendored patch 동시 갱신.** `backends/megatron/Megatron-LM-251125`
   를 건드린 커밋에는 `backends/submodule_patches/` 재생성이 반드시 포함되어야 한다
   (재생성법은 그 디렉토리 README). superproject의 서브모듈 포인터는 항상 **upstream
   커밋**을 가리켜야 하며, 로컬-전용 커밋 SHA를 기록하면 클론이 깨진다 —
   `git status`의 251125 `M` 표시는 설계상 정상이므로 절대 커밋하지 않는다.
4. **main 커밋 후 즉시 push.** 원격(origin)이 유일한 백업이다. 로컬에만 수십 커밋을
   쌓아두지 않는다.
5. **공유 워킹트리.** 여러 Claude 세션이 같은 워킹트리에서 동시에 커밋한다. 커밋은 항상
   **명시 pathspec**(`git add <파일>`), `git add .`/`-A`/`commit -a` 금지. 커밋 전 `git status`로
   staged에 내 파일만 있는지 확인. `git reset`류 HEAD 이동 금지(잘못 커밋했으면 revert/수정 커밋).
   서브모듈 포인터가 staged면 반드시 unstage. 2026-08-23 사고(타 세션 커밋 낙마·포인터 오커밋)의 교훈.

### 메모리·문서 규칙 (adopted 2026-08-25)

기록은 많을수록 좋지만 **매 세션 로드되는 것**은 작아야 한다. 기록의 양이 아니라 로드 시점이 비용이다.

1. **한 사실은 한 곳에.** 우선순위 docs > CLAUDE.md/rules > auto-memory. 다른 곳에는 링크만.
2. **CLAUDE.md는 지침만.** 루트 ≤150줄, `examples/alpha/CLAUDE.md` ≤300줄. 사고 서사는 `examples/alpha/docs/KNOWN_ISSUES.md`에
   쓰고 CLAUDE.md 함정 표에는 **한 줄**. 상대 시점("이번 세션") 금지, 날짜는 절대 표기.
3. **상태는 `examples/alpha/docs/STATUS.md`에 커밋한다.** auto-memory에 진행 상태를 쓰지 않는다 — 메모리는 노드·컨테이너별이라
   다른 세션이 못 보고 갱신 규율이 없어 모순이 쌓인다. 메모리는 사용자 피드백·선호 같은 진짜 개인화 정보에만.
4. **새 문서는 `examples/alpha/docs/README.md`에 한 줄 등록.** CLAUDE.md에 문서 요약을 쓰지 않는다. "단일 진입점" 문서를
   새로 만들기 전에 기존 문서에 절을 추가할 수 있는지 먼저 본다.
5. **스테이지 경계마다 정리.** 두 스테이지 이상 지난 Known Issue는 `docs/archive/`로 옮기고 함정 표에서 제거. STATUS.md의
   끝난 트랙은 "완료" 절로 내린다.

### 보고 문체

- 결론 먼저, 근거는 뒤에. 한 문장에 한 주장. 괄호 속 부연 최소화, 수치 나열은 표로.
- 긴 합성 명사구("A + B + C 정렬 스위치")를 피하고 풀어 쓴다. 보고는 의사결정 자료다 — 다시 읽게 만들면 요약의 의미가 없다.

## References

- Main README: [README.md](README.md)
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Model guides: `examples/{model}/README.md`
