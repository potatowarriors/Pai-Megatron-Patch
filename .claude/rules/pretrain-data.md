---
paths:
  - "toolkits/pretrain_data_preprocessing/**"
  - "examples/alpha/configs/data/**"
---

# Pre-training 데이터 작업 규칙 (Alpha v5 tokenizer)

- **100GB 이상 토크나이즈는 `preprocess_data_megatron.py`를 쓰지 않는다.** `fast_tokenize_v5.py`(encode_batch, 프로세스≫스레드,
  `RAYON_NUM_THREADS=8`)를 쓴다. 예산 25M tok/s, 실측이 2× 느리면 구조가 틀린 것. 근거·표:
  `examples/alpha/docs/PRETRAIN_DATA_PIPELINE.md` §Pre-tokenization.
- 새 스크립트는 100-doc 샘플로 legacy와 `.bin`/`.idx` **byte-perfect 비교** 후에만 장시간 실행.
- v5 `.bin`을 블렌드에 넣기 전 **EOD가 id 0인지 확인**(Phase B). 2026-05-12 이전 토크나이즈본은 id 3 — `remap_eod.py`.
- packed 산출물은 학습 투입 전 `scan_internal_eod.py`로 원문 내 리터럴 EOD 오염 스캔 (LC-A iter170 사고).
- Best-fit packing 불변량: bin capacity = seq_length(L+1 아님), `add_document(arr, [L])` 단일 길이, pad = EOD(id 0), 데이터셋별 실행.
  THD+CP용은 `--pad-doc-multiple 16` 필수 (`LC_REPACK_RUNBOOK.md`).
- 초단문 셋의 emit은 NFS가 아니라 per-doc Python 천장(~35k docs/s)에 걸린다 — 시간 예산은 문서 길이 분포부터.
- 블렌드 가중치는 `compute_blend_weights.py`의 epoch 표로 over-epoch을 확인한다. 가중치 합 ±1e-6 잔차가 DiLoCo 샤딩과
  간섭한 전례(`study/mirror_loss_aliasing.md`).
- `/home/work/Datasets/LL_preprocessed/mmap/`(옛 Qwen3 토큰화, 8.4T)는 **다른 사용자 소유 — 삭제 금지.**
