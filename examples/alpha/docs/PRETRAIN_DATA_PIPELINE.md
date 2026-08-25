# Pre-training 데이터 파이프라인 — v5 토크나이즈·Preflight·Stage2 재토크나이즈·Best-fit Packing

`examples/alpha/CLAUDE.md`에서 2026-08-25 이관 (원문 그대로, 2026-05~07 작업 기록).
LC 단계 재패킹(pad16)은 `LC_REPACK_RUNBOOK.md`, LC/SFT 데이터 상태는 `DATA_PREP_LOG.md` 참조.

## Pre-tokenization Performance (v5 tokenizer) — Critical Lessons (2026-05)

**75시간 DCLM 토큰화가 4시간으로 끝났어야 했음.** 이 섹션은 v5 (HF `PreTrainedTokenizerFast`) 토크나이저로 대규모 데이터 토큰화 시 같은 실수 반복 방지용. **옛 Qwen tokenizer (slow path, `.encoder` dict 있음) 사용 시 해당 없음** — 그 때는 `preprocess_data_megatron.py` 그대로 OK.

### What went wrong
`toolkits/pretrain_data_preprocessing/preprocess_data_megatron.py` 를 default `--workers 64 --partitions 8` (`preprocess_koreanweb_half.sh` 템플릿 default — 74 GB 작업용으로 튜닝됨)로 1.9 TB DCLM에 적용:
- 75시간 wall clock
- 8개 partition 중 data_2가 라인 분포 편차로 5-7h 단독 tail
- Per-worker effective throughput **~13K tok/s** (Rust BPE 잠재력 500K tok/s의 ~3%)

### Root cause
1. **`pool.imap(encoder.encode, fin, chunksize=32)` per-doc IPC overhead** — 각 doc마다 pickle/unpickle, Pool master 단일 스레드 dispatch. 워커가 99% CPU여도 실제 토큰화는 그 시간의 일부.
2. **HF 75× speedup은 fast vs slow 비교** — 우리는 이미 fast 사용 중. 진짜 lever는 **`encode_batch([texts])` batched encoding**인데 코드에서 안 씀.
3. **너무 적은 partition 수 (8)** — 큰 데이터셋에서 한 partition slow가 전체 tail-effect 만듦.
4. **Default RAYON_NUM_THREADS** (= num_cpus = 224) — thread contention으로 single-process throughput 50% 감소.

### Architectural rules (>100 GB 입력 토큰화 시 필수 적용)
1. **`preprocess_data_megatron.py` 사용하지 말 것** for v5 tokenizer + 큰 입력. 직접 `tokenizers.Tokenizer.from_file(tokenizer.json)` + `encode_batch` 사용. (둘 다 byte-perfect 동일 토큰 ID 검증됨 — English/Korean/Mixed/Code 4종 샘플)
2. **Optimal config**: 16-28 Python processes × **8 Rayon threads** each, batch_size **5000 docs**.
3. **`RAYON_NUM_THREADS=8` 명시** before `from tokenizers import Tokenizer`. Default는 thread contention 유발.
4. **Process count >> thread count** for aggregate throughput. 64 cores 사용 시 8p×8t = 18M tok/s, 4p×16t = 10.8M tok/s, 2p×32t = 5.5M tok/s — 같은 cores지만 3× 차이.
5. **Output**: `IndexedDatasetBuilder.add_document(arr, [len(arr)]) + finalize(idx_path)` (in `backends/megatron/Megatron-LM-251125/megatron/core/datasets/indexed_dataset.py`). Per-process parts merge via `IndexedDatasetBuilder.add_index(part_prefix)` — deterministic 순서.
6. **Append EOD (im_end, id=3)** at doc end if `len(ids) > 0` — matches `preprocess_data_megatron.py` semantics for `--append-eod`.

### Verification protocol (mandatory before any multi-hour run)
새 토큰화 스크립트 작성 시 100-doc 샘플로 byte-perfect compare to `preprocess_data_megatron.py`:

```bash
head -n 100 <input>.jsonl > /tmp/sample.jsonl

# Legacy (correctness baseline)
python preprocess_data_megatron.py --input /tmp/sample.jsonl \
  --output-prefix /tmp/legacy --patch-tokenizer-type AlphaTokenizer \
  --load examples/alpha/tokenizer_v5 --workers 1 --partitions 1 --append-eod

# New (under test)
python <new_script.py> --input /tmp/sample.jsonl --output-prefix /tmp/new ...

cmp /tmp/legacy_text_document.bin /tmp/new_text_document.bin && echo BIN_OK
cmp /tmp/legacy_text_document.idx /tmp/new_text_document.idx && echo IDX_OK
```

BIN_OK + IDX_OK 둘 다 통과해야 production run. 영어/한국어/Arabic/CJK 다양한 스크립트 cover.

### Throughput rule of thumb (Intel Xeon 8480+, 224 logical cores, 2 TB RAM, v5 tokenizer)

| Config | Cores busy | Aggregate throughput |
|---|---|---|
| `preprocess_data_megatron.py` (per-doc imap) 128w × 8p | 128 | **1.7M tok/s** ← 옛 방식 |
| 1 proc × 1 Rayon thread | 1 | 0.94M tok/s |
| 1 proc × 8 Rayon threads | 8 | **4.7M tok/s** ← single-proc sweet spot |
| 1 proc × 192 Rayon threads | 192 | 2.2M tok/s ← contention |
| 4 procs × 8 threads | 32 | 10.6M tok/s |
| 8 procs × 8 threads | 64 | 18.0M tok/s |
| 16 procs × 8 threads | 128 | 23.8M tok/s |
| **28 procs × 8 threads** | 224 | **32.4M tok/s** ← best measured |
| 4 procs × 56 threads | 224 | 8.3M tok/s ← thread-heavy 나쁜 예 |

**예산 (보수적)**: **25M tok/s aggregate**. 계획 단계에서 `T` total tokens 예상 wallclock = `T / 25M / 3600` hours.

**STOP rule**: 예산 대비 실측이 >2× 느리면 architecture 잘못된 것. 1+ hour 작업 commit 전에 1 GB sample throughput 실측 후 재검토.

### Why this matters specifically for alpha v5
- v5 tokenizer는 `tokenizer.json` 만 ship — `vocab.json/merges.txt` 없음. HF가 `use_fast=False` 무시하고 **항상 `PreTrainedTokenizerFast` 반환**.
- 옛 `_Qwen3Tokenizer` wrapper는 `.encoder` dict 접근 가정 (slow path) → fast tokenizer로는 `AttributeError`. 2026-05에 `_AlphaTokenizer` wrapper를 신규 추가한 이유.
- 즉 v5 → fast tokenizer 강제 → batched API (`encode_batch`) 가 진짜 lever. Per-doc API는 fast tokenizer 잠재력의 ~3%만 활용.

### Reference: optimized pipeline location
`toolkits/pretrain_data_preprocessing/fast_tokenize_v5.py` (2026-05 작성, korean_web + FineWeb workloads — production-ready Rust encode_batch pipeline with multi-process scaling).

## Stage 1 Pre-flight Verification — Methodology (2026-05-12)

이 섹션은 multi-month run 전 데이터 + 토크나이저 + config 정합성을 끝까지 점검하는 **재사용 가능한 protocol**입니다. Stage 2/3에서 데이터 패치하거나 새 baseline 만들 때 같은 phase 구조를 그대로 활용 가능. 실제 실행 산출물은 `tests/preflight_stage1/`에 보존되어 있음.

### 왜 이런 protocol이 필요한가

Multi-month 학습은 silent failure가 가장 위험. 학습 중반에 발견된 토크나이저/데이터 bug는 전체 run 폐기로 이어짐. 79시간 DCLM tokenization이 끝나고도 **EOD가 stream에 없는 것을 미리 잡지 못했다면 학습 시작 후 cross-doc 잡음을 학습하다 한참 후 깨달았을 것**. 자동화된 verification net을 한 번 짜두면 같은 사고 재발 시 즉시 차단.

### 6-Phase 구조

각 phase가 standalone runnable script + markdown artifact + (선택) JSON report 페어로 구성. 모두 `tests/preflight_stage1/` 아래.

| Phase | 목적 | 산출물 |
|---|---|---|
| **0** | Bug fix + config 정합 (tokenizer 파일 3종 + alpha_config.py + 데이터 remap) | `00_eod_bug_diagnosis.md`, `01_eod_repro.md`, `02_audit_grep.md`, `03_eod_regression_tests.md` |
| **A** | Tokenizer round-trip + frontier deviation matrix | `A_tokenizer.md`, `A_roundtrip_report.json`, `run_phase_a.py` |
| **B** | `.idx`/`.bin` 구조적 audit (header, dtype, size consistency, token ID range, doc-boundary EOD, empty docs) | `B_dataset_integrity.md`, `B_<src>_report.json`, `run_phase_b.py` |
| **C** | Decoded sample 사람-눈 sanity check | `C_decoded_samples.md`, `C_decoded_samples.txt`, `run_phase_c.py` |
| **C-loader** | 실제 `GPTDataset.__getitem__` 흐름을 ON/OFF differential로 검증 (reset flags가 정말로 작동하는지) | `C_loader_audit.md`, `C_loader_report.json`, `run_phase_c_loader.py` |
| **D** | Training-time data flow (단일 packed sample의 tokens/position_ids/loss_mask/attention_mask 4중 snapshot) | `D_dataflow.md`, `D_sample_snapshot.txt`, `run_phase_d.py` |
| **E** | 100-iter smoke (model + optimizer + multi-GPU 통합 동작 확인) | `run_phase_e_smoke.sh` |
| **F** | Decisions log — 모든 deviation을 intentional/will-fix/accepted로 분류 | `F_decisions.md` |

### Phase C-loader가 가장 가치 있는 도구 — Differential ON/OFF 검증

이 한 가지 패턴이 verification net의 핵심 invention:

```python
# 같은 데이터를 두 번 로드하되 reset flags만 토글
ds_on  = GPTDataset(..., config=GPTDatasetConfig(reset_position_ids=True, ...))
ds_off = GPTDataset(..., config=GPTDatasetConfig(reset_position_ids=False, ...))
# 100 samples × 4 metrics × 2 configs 비교
```

만약 flag가 silent하게 무시되고 있다면 (예: 코드 회귀, 데이터 EOD 누락 등) ON/OFF가 같은 결과를 냄. 만약 진짜 작동한다면 다음 4가지 invariant이 ON/OFF 간에 명확히 다름:

| Metric | ON 기대값 | OFF 기대값 | 의미 |
|---|---|---|---|
| Cross-doc attention block rate | 1.0 | 0.0 | `reset-attention-mask` 작동 |
| Max position_id 평균 | < seq_len (~doc 길이) | seq_len - 1 (4095) | `reset-position-ids` 작동 |
| Loss mask coverage | < 1.0 (1 - eod_density) | 1.0 | `eod-mask-loss` 작동 |
| EOD count per sample | 둘 다 동일 | (data identity 확인) |

수치적 일치까지 검증 가능: loss mask coverage drop = EOD count / seq_len이 수학적으로 일치해야 함.

### Verification가 잡은 silent failure 종류 (2026-05-12 실측)

- **Type 1 — Designation drift**: `tokenizer_config.json:eos_token`이 chat-only 토큰으로 잘못 지정. HF는 lucky하게 우선 처리하지만 `special_tokens_map.json`이 stale → 일부 도구만 silent하게 wrong 동작.
- **Type 2 — Stale hardcoded defaults**: HF config 생성기에 다른 모델 family의 default token IDs 박혀있음 (alpha_config.py Qwen3 IDs). 학습 영향 0이지만 inference deployment 시점에 silent breakage.
- **Type 3 — Empirical exploration mistake**: 사람 (또는 LLM agent) 의 sampling-based empirical check이 wrong 결론에 도달 ("0/100 docs end in EOD" 라고 잘못 보고). Differential / 다중 seed 재검증이 정정.
- **Type 4 — Data ↔ runtime mismatch**: 데이터엔 EOD 있고 runtime flag도 ON인데 `tokenizer.eod`가 다른 ID를 반환 → flag는 silent하게 no-op. Phase C-loader의 ON/OFF differential이 즉시 잡음.

### 새 데이터셋에 protocol 적용하는 법 (Stage 2/3 사용 예상)

```bash
# 1. 데이터 패치 (필요 시)
python toolkits/pretrain_data_preprocessing/remap_eod.py \
  --prefix /path/to/stage2_dataset/data_text_document --dry-run    # 먼저 dry
python toolkits/pretrain_data_preprocessing/remap_eod.py \
  --prefix /path/to/stage2_dataset/data_text_document              # 실제 적용

# 2. tests/preflight_stage1/run_phase_b.py 의 DATA dict 갱신 + 실행
# 3. tests/preflight_stage1/run_phase_c_loader.py 갱신 + 실행
# 4. 새 stage용 training preset yaml 생성 (stage1.yaml 기반)
# 5. Phase E smoke launch
```

회귀 테스트 `tests/test_alpha_tokenizer_eod.py` (9개) 는 CI에 통합하면 EOD designation drift / Qwen3 default 회귀 / preprocess 패스 회귀를 영구적으로 차단.

## Stage 2 v5 Re-tokenization (2026-06-26, 완료 ✅)

레거시 Stage-2 blend(`configs/data/arxive/stage2.yaml`의 9-dataset, 옛 Qwen3 vocab
151,936)를 alpha v5 tokenizer(163,968)로 **전부 재토크나이징**. Stage 1과 동일한
fast-path 파이프라인을 재사용했고, 결과는 **1.686T tokens / 10 datasets**
(레거시 9개 + FineWeb2-HQ 2nd half — stage1/stage2 complementary-halves 설계상 추가).

### 결과 & blend config

산출물: `/home/work/Datasets/LL_preprocessed/v5/stage2/<name>/data_text_document.{bin,idx}`.
Blend: `configs/data/stage2_v5_blend.yaml` (weight-less → Megatron이 `.idx` 크기 비례
auto-mix; 실측 토큰 비율은 YAML 주석에 기록).

| 데이터셋 | tokens | 비중 | 출처 |
|---|---|---|---|
| nemotron_cc_hq/actual | 530.2B | 31.6% | MinIO 복원 |
| nemotron_cc_hq/qa_pairs | 475.7B | 28.3% | MinIO 복원 → decompress |
| code/question_answering | 241.1B | 14.4% | 로컬 jsonl |
| math | 205.0B | 12.2% | 로컬 (Nemotron-CC-Math-v1) |
| code/code_review | 77.1B | 4.6% | 로컬 |
| code/rewriting | 77.2B | 4.6% | 로컬 |
| code/transpilation | 29.3B | 1.7% | 로컬 |
| code/student_teacher | 25.8B | 1.5% | 로컬 |
| korean_web | 19.0B | 1.1% | 재사용(remap 3→0) |
| fineweb2hq | 5.7B | 0.3% | 재사용 2nd half(remap 3→0) |
| **합계** | **1.686T** | CC-HQ 59.7% / Code 26.7% / Math 12.2% / Korean 1.1% / FineWeb 0.3% | |

학습 시작 (**채택된 기본 방식** — 자연 blend + budget 제한):
```bash
bash train.sh baseline_48L <stage2_training_preset> stage2_v5_blend --train-samples 170898438
```
(training preset은 `stage2_2`/`stage2_3` 등 별도 선택 — 이 데이터 작업 범위 밖.)

**학습량(token budget) 제어**: 코퍼스 1.680T 전부 돌릴 필요 없이 `--train-samples`로
원하는 토큰만 학습. seq_length 4096 → `train_samples = tokens / 4096` (0.5T→122.07M,
0.6T→146.48M, **0.7T→170,898,438**). **iter 직접 지정보다 `--train-samples` 권장** —
step-wise GBS 스케줄에서 iter당 작업량이 달라져 "iter=토큰"이 깨지므로(GBS-invariant).
weight-less 자연 blend @ 0.7T는 **모든 소스 0.42 epoch**(반복 0, 코퍼스가 budget의
2.4배) → CC-HQ 60%/Code 27%/Math 12%/Korean 1% 비율 그대로 (CC-HQ 419B / Code 188B
/ Math 85B / Korean 7.9B). 이 비율이 적절하면 reweighting 불필요.

### (옵션) Blend 비율 조정 — CC-HQ down-weight

작은 budget에서 curated(math/code/korean) 비중을 자연비율 이상으로 키우고 싶을 때만.
data-path 각 경로 앞에 명시적 weight를 붙이면 크기와 무관하게 샘플링 비율 지정 가능
(Megatron normalize; **런타임 검증으로 CC-HQ 40% 반영 확인됨** — weights honored).

**도구**: `tools/compute_blend_weights.py` — CC-HQ를 목표치로 cap하고 남는 몫을
math/code/korean의 **자연 상대비율대로 재분배**(세 카테고리 내부 믹스 유지 + 동일
epoch). 토큰 수는 `bin_bytes/4`로 즉시 산출(GPU 불필요).
```bash
python tools/compute_blend_weights.py --cap-cchq 40 --budget-t 0.6 \
  --write configs/data/stage2_v5_blend_cc40.yaml
```
산출물 예 `configs/data/stage2_v5_blend_cc40.yaml` (옵션, CC-HQ 40% cap, 0.6T 기준):

| 카테고리 | share | tokens@0.6T | epoch |
|---|---|---|---|
| CC-HQ | 40.0% | 240.0B | 0.24 |
| Code | 40.1% | 240.4B | 0.53 |
| Math | 18.2% | 109.4B | 0.53 |
| Korean | 1.7% | 10.2B | 0.53 |

`★ 핵심 트레이드오프 — over-epoch`: 명시적 weight는 크기와 sampling을 분리하므로,
**작은 데이터셋을 자연 share 이상으로 키우면 epoch>1(반복)** 됨. "CC-HQ cap +
비례 재분배" 방식은 어느 카테고리도 자연 share를 *초과*하지 않아 0.6T에서 셋 다
0.53 epoch(<1)로 안전. 반대로 flat 비율(예: Korean 5% 고정)은 Korean(19B뿐)을
1.5~2.2 epoch로 반복시킴 — `compute_blend_weights.py`의 epoch 표로 항상 확인할 것.
학습: `bash train.sh baseline_48L <preset> stage2_v5_blend_cc40`.

### Tooling (전부 `toolkits/pretrain_data_preprocessing/`)

레거시 `preprocess_kormo_*.sh`/`preprocess_nemotron_*.sh`는 **재사용 불가**(stale:
`--patch-tokenizer-type Qwen2Tokenizer`가 fast-only v5에서 크래시, 느린 per-doc 경로,
옛 250624 backend). 대신 다음을 신규 작성/확장:

| 파일 | 역할 |
|---|---|
| `run_stage2_v5.sh` | 정규 레시피. sub-targets: `restore` / `local` / `cchq` / `all` / `<dataset>` |
| `preprocess_stage2_v5.sh` | 범용 multi-shard FAST 드라이버. 파일 round-robin → P개 `fast_tokenize_v5.py` 병렬 → `merge_indices.py --dtype int32`. 기본 `PROCS=12` ×8 rayon ≈ 110-core cgroup의 96. `AUTO_CLEAN_PARTS=1`로 merge 검증 후 parts 자동 삭제 |
| `fast_tokenize_v5.py` | `--input-list-file` 추가(멀티 jsonl 파일 → 1 part). parity-preserving |
| `minio_restore.py` | boto3 다운로더(`preprocess_data_megatron.py`엔 없던 restore). `/home/work/Datasets/LL_datasets/minio_backup.py`를 import해 creds + `_patch_botocore_time` 재사용. **MinIO 서버 clock ~9h skew → time-patch 필수**(없으면 SignatureDoesNotMatch). resumable(size-match skip + `.part` atomic rename) |
| `tests/preflight_stage2/run_phase_b.py` | Stage-1 Phase B 적응판. **B4가 ~100% doc-end=id 0을 assert**(Stage 1은 pre-injection이라 0을 기대 — 정반대). per-dataset try/except로 non-halting |

### 핵심 발견 (재현 주의)

1. **재사용 v5 데이터는 EOD가 id 3.** `korean_web`·`fineweb2hq`는 2026-05-12 designation
   수정 *전*(eos=`<|im_end|>` id 3)에 토크나이즈돼 doc 끝이 id 3. Stage 1 데이터는
   remap됐지만 이 둘은 안 됨. Phase B가 즉시 검출 → **둘 다 `remap_eod.py --old-eod 3
   --new-eod 0`로 수정 완료**(dry-run이 200k doc-end 전부 id 3 보유를 pre-verify; post-verify
   200k 전부 id 0 + first/last 100 확인). 신규 토크나이즈 데이터(`--append-eod`)는 id 0
   직접 부여 → remap 불필요(Stage 1과 달라진 점). **교훈**: 다른 stage/session에서
   토크나이즈된 v5 `.bin`을 blend에 넣기 전 반드시 EOD(id 0 vs 3)를 Phase B로 확인할 것.

2. **smoke test로는 이 blend를 검증할 수 없음.** `configs/{model,training}/smoke.yaml`은
   toy preset이고 251125 backend와 **기존 config drift**가 있음: ① GBS=1이 multi-GPU에서
   `÷ data_parallel_size` 안 나눠짐, ② `moe-router-topk:1`이 `--moe-router-pre-softmax`
   요구. 둘 다 **dataset 로드 *전* model-build에서 크래시**라 blend와 무관. 데이터 정합성은
   `BlendedMegatronDataset`를 직접 빌드해 검증함(CPU-only):
   - `torch.distributed.init_process_group("gloo", rank=0, world_size=1)` 필요
   - tokenizer stub에 `.eod`(=0) + `.unique_identifiers` property 필요(후자는 cache-key
     JSON 직렬화용)
   - 검출 항목: blend auto-weight(= 실측 비율 일치), 샘플 token range <163968,
     EOD 기반 `reset-position-ids`/`eod-mask-loss` 작동(maxpos<4095, loss_mask cov<1.0)
   - 결과: 9-source 빌드 + 샘플 조립 OK. 실제 학습은 `baseline_48L`이라 smoke drift 무영향.

### 디스크

검증 완료 후 회수: CC-HQ raw `.jsonl`(actual 2.5T + qa_pairs 2.3T), 모든 `_parts`(~2.4T),
qa_pairs `.jsonl.zstd`(814G) — **~8TB**. CC-HQ는 MinIO에서 `run_stage2_v5.sh restore`로
재복원 가능(~3h). **옛 Qwen3 `/home/work/Datasets/LL_preprocessed/mmap/`(8.4T)는
다른 사용자 소유 — 삭제 금지.** 로컬 raw math/code는 MinIO 백업이 없어 보존.

## Best-fit Packing — 문서 truncation 최소화 (2026-06-30, 구현 ✅)

논문 **"Fewer Truncations Improve Language Modeling"** (arXiv 2404.10830, ICML 2024)의
**Best-fit Packing (BFP)** 를 offline 전처리로 구현. Megatron-LM은 이 기능을 제공하지 않음
(기본 `GPTDataset`은 concat-and-chunk로 문서를 seq_length 경계에서 무차별 절단).

### 문제 & 해법
- **문제**: 학습 시 `GPTDataset`이 모든 문서를 이어붙여 4096씩 자름 → 경계에 걸친 문서가
  매 sample마다 잘림. (alpha의 `--reset-attention-mask`는 *한 sequence 안*의 cross-doc
  attention만 막을 뿐 절단 자체는 못 막음 — BFP와 **직교/상보**.)
- **해법**: 문서 길이 배열만 읽어 **Best-Fit-Decreasing**(segment tree, O(N log L))로 문서를
  4096 bin에 통째로 packing → 각 bin을 EOD(id 0)로 4096에 padding → **bin 1개 = .idx 문서
  1개**로 재출력. 학습의 concat-and-chunk가 정확히 bin 경계에서 잘려 문서가 안 잘림
  (4096 초과 문서만 불가피하게 분할 — 논문도 동일).

### 사용
```bash
# stats만 (truncation 감소/fill ratio 확인, 쓰기 없음)
bash toolkits/pretrain_data_preprocessing/run_stage2_v5.sh pack-dry
# 전체 10개 데이터셋 packing → /home/work/Datasets/LL_preprocessed/v5/stage2_packed/
bash toolkits/pretrain_data_preprocessing/run_stage2_v5.sh pack
# 1개만: bash ... pack code/transpilation   (또는 직접 bestfit_pack.py --input … --output … --dry-run)
# 학습: data preset을 packed blend으로
bash train.sh baseline_48L <preset> stage2_v5_blend_packed --train-samples <N>
```

### 실측 결과 — 전체 10개 완료 (2026-07-02 ✅, `stage2_packed/`)
| 데이터셋 | fill | 감소 | 크기 |
|---|---|---|---|
| fineweb2hq | 99.95% | −72.8% | 22 G |
| korean_web | 99.99% | −78.5% | 72 G |
| code/transpilation | 99.01% | −100% | 111 G |
| code/student_teacher | 99.05% | −100% | 98 G |
| code/code_review | 99.95% | −99.5% | 289 G |
| code/rewriting | 99.78% | −100% | 290 G |
| math | 99.69% | −82.5% | 770 G |
| code/question_answering | 99.08% | −100% | 911 G |
| nemotron_cc_hq/actual | 100.00% | −81.0% | 2.0 T |
| nemotron_cc_hq/qa_pairs | 99.81% | −100% | 1.8 T |
| **합계** | **전부 ≥99%** | code −100% / web·math·CC-HQ-actual −73~82% (긴 문서 잔존) | **6.2 TiB** |

각 셋 자체 post-verify(전부 4096 / 토큰 보존 / round-trip 20/20) 통과 + 두 CC-HQ는 독립
재로드로 bins·토큰 재확인(actual 129,442,937 / qa 116,351,363 bins). 학습: `train.sh
baseline_48L <preset> stage2_v5_blend_packed --train-samples N`.

**Loader differential** (실제 `GPTDataset` 경로, fineweb2hq 3000 samples,
`tests/preflight_stage2/run_pack_loader_check.py`): **bad-truncation
(whole docs + 잘린 small doc) 82.0% → 0.0%**. ends-on-doc-boundary 0.03% → 72.6%
(나머지 27.4%는 4096 초과 문서의 *순수 단일-문서 chunk* — fragmentation 아님).

### emit 성능 — threaded prefetch (`--emit-threads`, 기본 48)
emit 병목은 소스 `.bin`에서 bin 멤버 문서를 **무작위로 읽는 NFS latency**(대역폭 아님).
`os.pread`(syscall이 GIL 해제)로 read 48개 동시 발행해 latency를 숨기고, 쓰기는 메인
스레드에서 순서대로 → **출력 byte-identical**(회귀 테스트로 잠금). 실측: cchq_actual
**8,183 bins/s = serial(~1,000) 대비 ~8.2×**(단독). 병렬 2개 동시엔 합계 쓰기 ~132 MB/s로
수렴(그 시점 병목 = NFS write 대역폭). **대형 셋 필수** — 없으면 CC-HQ가 각 ~30h+, 있으면
두 개 병렬로 ~반나절.

### 핵심 설계 (정합성, helpers.cpp/gpt_dataset.py 소스 대조 검증)
- **bin capacity = seq_length = 4096 (L+1 아님).** Megatron이 sample당 4097 토큰을 읽어
  "+1"을 스스로 공급(공유 경계). 4097로 packing하면 정렬이 깨짐.
- **`add_document(bin_arr, [L])` — 길이 리스트는 단일 원소 `[L]`.** GPTDataset은 *sequence*
  단위로 자르고 `document_indices`를 무시 → 문서별 길이를 넣으면 packing이 조용히 무효화.
  (tool에 hard assert.)
- **pad = EOD(id 0).** `--eod-mask-loss`가 pad 구간 loss를 masking, `--reset-attention-mask`가
  pad를 격리. 별도 pad id는 masking 안 됨(모델이 pad 예측 학습) → EOD-pad가 유일 정답.
- **데이터셋별로** 실행. Megatron의 `BlendedDataset`은 한 constituent에서 *통째* sequence를
  샘플 → per-dataset packing이 blend로 보존. (cross-dataset packing은 blend 비율 오염.)
- **놓치기 쉬운 점**: 4096 초과 문서의 full-L head chunk는 content로 끝나 1토큰 eod-unmasked
  leak(fineweb2hq ~0.0066% positions, 논문 수용 범위). `--strict-eod`로 제거 가능(대신 mid-doc
  false EOD 도입 — 기본은 accept+report). 또 weight-less blend는 dataset별 padding(<1%) 차이로
  실토큰 비율이 sub-% drift — 정확히 맞추려면 `compute_blend_weights.py`로 명시 weight.

### 도구
| 파일 | 역할 |
|---|---|
| `toolkits/pretrain_data_preprocessing/bestfit_pack.py` | BFP packer (segment-tree BFD + pad-to-L emit + pre/post-verify + round-trip + `--dry-run`/`--strict-eod`). BFD ~0.25M docs/s(pure Python); 대형 셋은 emit가 I/O bound. **`--pad-doc-multiple 16`** (2026-08-21): 문서별 EOD 패딩으로 THD+CP 세그먼트 %2cp 정렬 — LC 32k/128k 패킹 필수, 재패킹 절차는 `docs/LC_REPACK_RUNBOOK.md` |
| `toolkits/pretrain_data_preprocessing/split_by_doclen.py` | unpacked 문서를 길이 기준 lt/ge 분리 (기본 T=65536). LC 스테이지 배분 정책용 — 32k는 lt만 패킹, ≥64k는 128k 보존. `docs/LC_DATASETS.md` §5.1 |
| `run_stage2_v5.sh` `pack`/`pack-dry` sub-target | 10개 blend member 일괄 packing (env `SEQLEN`/`EOD`/`OUT_PACKED`) |
| `examples/alpha/configs/data/stage2_v5_blend_packed.yaml` | packed blend (stage2_v5_blend의 packed 트리 미러) |
| `tests/test_bestfit_pack.py` | 16 unit tests (segtree↔brute, BFD↔naive parity, piece coverage, baseline 추정, 실 IndexedDataset round-trip) |
| `tests/preflight_stage2/run_pack_loader_check.py` | packed vs unpacked GPTDataset loader differential |

