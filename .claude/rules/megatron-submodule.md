---
paths:
  - "backends/**"
---

# Megatron-LM 서브모듈 수정 규칙

- `backends/megatron/Megatron-LM-*/`는 **기본적으로 수정하지 않는다.** 학습 로직은 `megatron_patch/`에 패치로 얹는다.
- 예외는 패치 지점이 노출되지 않는 코어 인프라뿐. 현재 5건: step-wise GBS 스케줄, progressive blend argparse,
  Muon QGKV split, Muon chunked offload, THD+CP 잠복버그 3건 — 전문은 `docs/CUSTOM_TRAINING_FEATURES.md`.
- 서브모듈을 건드린 커밋은 **`backends/submodule_patches/` 재생성을 반드시 포함**한다 (재생성법은 그 디렉토리 README).
- superproject의 서브모듈 포인터는 항상 **upstream 커밋**을 가리킨다. `git status`의 251125 `M` 표시는 설계상 정상이며
  **절대 커밋하지 않는다** — staged로 보이면 unstage. 로컬 전용 SHA를 기록하면 클론이 깨진다.
- 251125 = Alpha/Muon 개발 스냅샷, 250908/250624 = Qwen3·DeepSeek-V3 안정판. 버전 선택은 `run_mcore_*.sh`의 PYTHONPATH.
