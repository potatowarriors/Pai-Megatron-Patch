# LC 재패킹 러너북 — THD+CP용 `--pad-doc-multiple 16` (2026-08-21 작성)

**목적**: LC 학습 데이터를 THD+CP 문서 격리 요건에 맞게 재패킹한다.
**GPU 불필요** — CPU/NFS 작업만이므로 P3 학습과 병행 가능한 별도 컴퓨팅 세션용.
이 문서 하나로 사전조건 확인 → 실행 → 검증 → 후처리까지 완결되도록 작성했다.

## 0. 왜 필요한가 (배경 요약)

LC는 CP≥2로 학습한다. 문서 격리는 dense `--reset-attention-mask`가 O(seq²)라
32K에서 불가(샘플당 1 GiB)하여 **THD/cu_seqlens 방식**으로 전환했는데
(`LC_ENTRY_GATE.md` §1.5 — **머지 전까지는 `feature/gdn-varlen-thd` 브랜치의 버전**에
있음; 이 러너북은 그 참조 없이도 자기완결), TE의 `thd_get_partitioned_indices`와
GDN THD a2a 순열은
**모든 packed 세그먼트 길이가 `2 × cp_size`로 나눠떨어질 것**을 요구한다.
`bestfit_pack.py --pad-doc-multiple 16`(main 커밋 70220d7)이 각 문서를 EOD로
16의 배수까지 패딩해 이를 보장한다 — 16 = 2×CP8, CP≤8 전부 커버.

- 기존 `cpt_lc_packed_32k/`(pad 없음)는 **THD+CP 학습에 사용 불가** — 재패킹 대상.
- per-doc pad 오버헤드는 문서당 평균 8토큰 — LC 문서는 수천~수만 토큰이라 <0.1% 수준
  (dry-run이 정확한 수치를 출력한다).
- pad는 EOD(id 0): `--eod-mask-loss`가 loss에서 제외하고, causal 아래에서 내용
  토큰이 pad를 볼 수 없으므로 학습 의미론 무변.

## 1. 사전조건 (5분)

```bash
cd /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch
git checkout main && git pull
# --pad-doc-multiple이 있는지 (70220d7 이후):
grep -q "pad-doc-multiple" toolkits/pretrain_data_preprocessing/bestfit_pack.py && echo TOOL_OK
# 도구 회귀 테스트 (19개, ~1분):
python -m pytest tests/test_bestfit_pack.py -q
# 소스(lt64k unpacked) 존재 확인 — 4종 전부 있어야 함:
for ds in longblocks pg19 edgar pes2o; do
  ls -lh /home/work/Datasets/LL_preprocessed/v5/cpt_lc/$ds/lt64k_text_document.bin
done
# 디스크: 신규 산출 ~60G 필요 (2026-08-21 실측: lt64k 합계 ~59G, NFS 여유 19T)
df -h /home/work/Datasets | tail -1
```

2026-08-21 실측 소스 크기: longblocks 16G / pg19 1.5G / edgar 21G / pes2o 21G.

## 2. 작업 1 — LC 4종 32k 재패킹 (확정, 즉시 실행 가능)

### 2.1 dry-run (쓰기 없음, 데이터셋당 수 분)

```bash
cd toolkits/pretrain_data_preprocessing
for ds in longblocks pg19 edgar pes2o; do
  python3 bestfit_pack.py \
    --input  /home/work/Datasets/LL_preprocessed/v5/cpt_lc/$ds/lt64k \
    --output /dev/null --dry-run \
    --seq-length 32768 --eod 0 --pad-doc-multiple 16
done
```

확인할 것: `per-doc pad to %16: +N tokens (~0.0x% of real)` 라인과 fill ratio가
기존 §6.5 실측(98.8~99.99%)에서 크게 떨어지지 않는지 (pad 오버헤드만큼 소폭 하락 정상).

### 2.2 실행

**출력은 반드시 신규 디렉토리** — 구 packed는 검증·블렌드 전환 완료까지 보존한다.
스크립트의 pack 스테이지는 exists-skip이라 신규 `PACKED_DIR` 지정이 곧 재실행 스위치다
(convert/tokenize/split 스테이지는 산출물이 이미 있으므로 자동 skip).

```bash
PACKED_DIR=/home/work/Datasets/LL_preprocessed/v5/cpt_lc_packed_32k_pad16 \
PAD_DOC_MULTIPLE=16 \
  bash run_cpt_lc_v5.sh all 2>&1 | tee /tmp/lc_repack_$(date +%m%d).log
```

- 예상 시간: 수 시간 내 (추정 — 원 패킹은 4종 일괄 처리에 포함되어 개별 실측 없음.
  emit는 NFS 랜덤 read가 병목, 총 쓰기 ~60G. `[STAGE-DONE] <ds>/pack` 마커로 진행 확인).
- 도중 실패 시: bestfit_pack은 출력 존재 시 거부(refusing to overwrite)하므로
  **실패한 데이터셋의 부분 산출물(`.bin/.idx`)을 삭제 후 재실행**.
- NFS 대역폭상 **동시 실행은 2개까지** (BFP 세션 실측: 2개 병렬 시 write ~132MB/s 포화).

### 2.3 검증

도구가 post-verify(전 bin=32768, 토큰 보존, round-trip 20 샘플)를 내장 실행한다.
추가로 THD 요건인 **세그먼트 경계 %16 정렬**을 직접 확인:

```bash
python3 - <<'EOF'
# 각 bin에서 EOD 런(연속 EOD 구간)의 끝 위치가 전부 %16인지 표본 검사.
# (tests/test_bestfit_pack.py::test_end_to_end_pad_doc_multiple과 동일 로직.
#  전체가 EOD 없이 끝나는 bin = 32k 초과 문서의 단일 청크 — 허용, bin 끝이 %16.)
import sys, numpy as np
sys.path.insert(0, "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/backends/megatron/Megatron-LM-251125")
from megatron.core.datasets import indexed_dataset
M, EOD = 16, 0
for ds in ["longblocks", "pg19", "edgar", "pes2o"]:
    p = f"/home/work/Datasets/LL_preprocessed/v5/cpt_lc_packed_32k_pad16/{ds}/data_text_document"
    out = indexed_dataset.IndexedDataset(p)
    rng = np.random.default_rng(7)
    bad = 0
    for b in rng.choice(len(out), size=min(500, len(out)), replace=False):
        seq = np.asarray(out[int(b)])
        is_eod = seq == EOD
        # EOD 런의 끝(exclusive) = EOD 다음이 non-EOD이거나 시퀀스 끝인 위치
        run_end = np.where(is_eod[:-1] & ~is_eod[1:])[0] + 1
        if is_eod[-1]:
            run_end = np.append(run_end, len(seq))
        bad += int((run_end % M != 0).sum())
    print(f"{ds}: sampled 500 bins, misaligned EOD-run boundaries = {bad}  "
          + ("OK" if bad == 0 else "**FAIL**"))
EOF
```

전 데이터셋 `= 0 OK`가 통과 기준. 추가 sanity: real token 수가 구 packed와 동일한지
(`[bestfit_pack] OK: ... (real N + pad M)` 로그의 real N을 구 실행 §6.5 표와 대조 —
lt64k 합계 15.56B).

### 2.4 후처리

1. `LC_DATASETS.md` §6.5 표 아래에 재패킹 실측(신규 bins 수·fill·pad 오버헤드) 추기.
2. `DATA_PREP_LOG.md` §1 스냅샷의 LC 행 갱신 (pad16 경로로 교체).
3. LC-A 블렌드 yaml(작성 시)은 **`cpt_lc_packed_32k_pad16/` 경로**를 사용.
4. 구 `cpt_lc_packed_32k/`(59G)는 블렌드 yaml이 신규 경로로 확정된 뒤 삭제 가능
   (unpacked canonical이 남아 있어 언제든 재생성 가능).

## 3. 작업 2 — LC-A filler(일반 replay ~54%) 32k 패킹 (블렌드 설계 결정 필요)

**⚠️ 전제 붕괴 발견 (2026-08-21)**: 원 계획(큐 B)은 "stage2 v5 블렌드 32k 재패킹"
이었으나, **stage2 unpacked 원본이 디스크에서 소실**됐다 — `v5/stage2/`에는
`specialized/`만 남았고 나머지 10종은 4k-packed(`stage2_packed/`)만 존재한다.
4k-packed는 재패킹 불가(장문이 이미 4k 조각 + pad/문서 EOD 구분 불가,
`LC_DATASETS.md` §6 체크리스트 2). filler 확보는 아래 옵션 중 결정해야 한다:

**2026-08-21 추기 — raw 잔존 실사로 옵션 B가 크게 저렴해짐** (같은 날 2차 정정 포함):
소실된 것은 v5 *unpacked mmap* 계층이고 **raw는 CC-HQ만 삭제**(MinIO 복원 가능)임을
확인했다. replay가 미러링해야 할 조성은 stage2 코퍼스 자연 비율(CC-HQ 60%...)이
아니라 **모델이 마지막으로 본 P3 decay 블렌드**
(`configs/data/stage2_v5_blend_packed_p3.yaml` 실측 집계):

| P3 구성 | 비중 | 32k filler 소스 상태 (08-21 실사) |
|---|---|---|
| **specialized v1/v1.1/v1.2** | **35% (최대)** | raw 잔존 `LL_datasets/pretraining/stage3/Nemotron-Pretraining-Specialized-*` (unpacked는 빈 스캐폴딩 — .bin 0개) |
| CC-HQ (actual 4 + qa 16) | 20% | raw 삭제 → **MinIO 복원 ~3h (유일한 복원 비용)** |
| math (+CC-Math) | 18% | raw 잔존 `pretraining/stage2/{math,Nemotron-CC-Math-v1}` (718G+244G) |
| cc_code | 10% | raw 잔존 `Nemotron-CC-Code-v1` (1.1T) |
| code 5종 | 8% | raw 잔존 `pretraining/stage2/code` (1.9T) |
| korean_web / fineweb2hq | 6% / 3% | stage1 unpacked 잔존 → 재패킹만 |

LC-A filler 필요량은 ~7-9B뿐이라 **전량이 아니라 샤드 서브셋만 재토크나이즈**하면
된다(처리량 ~25M tok/s 기준 토크나이즈는 분 단위; 지배 비용은 CC-HQ 복원 ~3h).
Nemotron 3 Ultra의 LC 54%가 "직전 phase 블렌드 재사용"이었으므로 P3 미러가 레시피
정합 기본값이고, 조성 조정 여부는 블렌드 yaml 설계에서 확정.

| 옵션 | 내용 | 비용 | 비고 |
|---|---|---|---|
| A. stage1 unpacked 사용 | `v5/stage1/{dclm,korean_web,fineweb2hq}/data`(보존 확인)를 32k+pad16 패킹 | korean_web+fw2hq ~23B, 수 시간 | filler가 web-heavy (math/code 부재) — P3 분포와 괴리 |
| **B. raw 서브셋 재토크나이즈 (실사 후 권고)** | code/math는 잔존 raw에서 필요분 샤드만 토크나이즈→32k+pad16 패킹; CC-HQ는 MinIO 복원 후 동일; korean은 stage1 unpacked 재사용 | 반나절 내외 (CC-HQ 복원 ~3h 지배) | **P3 조성 재현 = replay 목적에 최적.** 128k filler도 같은 raw에서 SEQ=131072로 재생산 가능 |
| C. 4k-packed에서 whole-doc 복원 도구 신규 | EOD split로 온전 문서만 추출(4k 초과 조각은 드롭/수용) | 도구 개발+검증 | raw 잔존 확인으로 존재 이유 소멸 — **탈락** |

**✅ 결정 (2026-08-21, 사용자 확정): P3 미러로 진행.** 근거: replay의 목적은
기존 학습 내용을 파괴적 망각으로부터 지키는 것이므로, 모델이 마지막으로 본 분포
(P3 decay 블렌드)를 **조성 변경 없이 그대로** 재현한다. filler 블렌드 가중치는
`configs/data/stage2_v5_blend_packed_p3.yaml`의 상대 가중치를 재사용(경로만
pad16 산출물 트리로 치환).

### 3.1 물량 산정

LC-A 예산 12~16B × filler 54% ≈ **6.5~8.6B**. 카테고리 목표 = P3 비중 × filler 총량.
재토크나이즈 소스는 **목표의 ≥2× 풀** 확보(epoch ≤0.5로 반복 회피). filler 8B 기준:

| 카테고리 | P3 비중 | 목표 | 확보 방식 |
|---|---|---|---|
| specialized | 35% | 2.8B | **전체 재생산** (§3.2 — subset 가중 보존·128k 재사용 겸) |
| CC-HQ | 20% | 1.6B | MinIO 복원 후 샤드 서브셋 ≥3.2B (actual:qa = 4:16 비율) |
| math(+CC-Math) | 18% | 1.44B | 샤드 서브셋 ≥2.9B |
| cc_code | 10% | 0.8B | 샤드 서브셋 ≥1.6B |
| code 5종 | 8% | 0.64B | 샤드 서브셋 ≥1.3B (subset별 P3 가중 비례 배분) |
| korean_web / fineweb2hq | 6% / 3% | 0.48B / 0.24B | stage1 unpacked **전체** 패킹 (아래 명령) |

### 3.2 카테고리별 절차

**① specialized — `run_stage3_v5.sh` 재사용 (전체 재생산 권장)**: 필요 풀은 ~5.6B지만
15개 subset의 P3 내부 가중치 보존 + parquet 샤드 순서 편향 회피를 위해 전체 처리가
안전하다 (멱등·무인, 산출 unpacked는 128k filler 재패킹에도 재사용):

```bash
OUT_DIR=/home/work/Datasets/LL_preprocessed/v5/lc_filler/specialized \
OUT_PACKED_DIR=/home/work/Datasets/LL_preprocessed/v5/lc_filler_packed_32k_pad16/specialized \
SEQLEN=32768 PAD_DOC_MULTIPLE=16 KEEP_UNPACKED=1 \
  bash run_stage3_v5.sh all
```

(**`KEEP_UNPACKED=1` 필수** — 128k 재패킹 대비. 예상 비용: 토크나이즈 324B ≈ 반나절
+ pack, 디스크 ~2.6TB — NFS 여유 19T 내. env 노브는 2026-08-21 추가분.)

**② math / code / cc_code — 샤드 파일 결정적 해시 샘플링**으로 목표 풀만 토크나이즈
(순서 편향 방지 — pes2o filler 티어의 해시 샘플링 선례). jsonl은
`preprocess_stage2_v5.sh` 직행, parquet(cc_code)은 변환 후. 이어서:

```bash
python3 bestfit_pack.py --input <unpacked_prefix> --output <...>/lc_filler_packed_32k_pad16/<ds>/data \
  --seq-length 32768 --eod 0 --pad-doc-multiple 16 --emit-threads 48
```

**③ CC-HQ** — `run_stage2_v5.sh restore`로 MinIO 복원(~3h, 시계 skew time-patch 내장)
후 ②와 동일.

**④ filler 블렌드 yaml** — P3 yaml 가중치 복사 + 경로만 pad16 트리 치환 →
LC 46% 쪽(cpt_lc_packed_32k_pad16 + KO 2종)과 합성해 LC-A 최종 yaml (작업 ③).

**korean_web + fineweb2hq 선행 패킹** (즉시 실행 가능):

```bash
cd toolkits/pretrain_data_preprocessing
for ds in korean_web fineweb2hq; do
  python3 bestfit_pack.py \
    --input  /home/work/Datasets/LL_preprocessed/v5/stage1/$ds/data \
    --output /home/work/Datasets/LL_preprocessed/v5/lc_filler_packed_32k_pad16/$ds/data \
    --seq-length 32768 --eod 0 --pad-doc-multiple 16 --emit-threads 48
done
```

(먼저 `--dry-run`으로 fill·pad 확인. stage1 데이터는 짧은 웹 문서 위주라 32k bin당
수십 문서 — per-doc pad 오버헤드가 LC 4종보다 큼: 문서당 평균 8토큰 × bin당 문서 수.
dry-run 수치로 수용 여부 판단, 대략 0.5~2% 예상.)

filler에는 길이 분할(lt/ge) 불필요 — ≥64k 보존 정책은 LC 희소 자원용이고 일반
웹 데이터의 ≥64k는 무시 가능량.

## 4. 작업 3 — 128k 재패킹 (미래, 큐 D와 연동)

128k 스테이지 재료(ge64k 보존분 8.65B + 다문서 합성 산출물)가 준비되면:

```bash
PACKED_DIR=.../cpt_lc_packed_128k_pad16 PAD_DOC_MULTIPLE=16 SEQ=131072 ...
# (ge64k 소스는 run_cpt_lc_v5.sh의 pack_ds가 lt64k를 하드코딩하므로
#  bestfit_pack.py 직접 호출 — 입력만 ge64k/합성 산출물로)
```

pad 16은 CP8까지 커버하므로 128k에서도 동일. 다문서 합성 샘플은 **내부 EOD 금지
불변량**(`LC_DATASETS.md` §5.1)을 그대로 지킬 것 — THD 세그먼트 경계가 곧 EOD이므로
내부 EOD가 있으면 합성 샘플이 attention/GDN 수준에서 조각난다.

## 5. 함정 요약

1. **기본 `PACKED_DIR`로 실행하면 아무 일도 안 일어남** — pack 스테이지가 exists-skip.
   재패킹 = 신규 경로 지정이 유일한 트리거.
2. `--strict-eod`와 병용하지 말 것 (LC 표준 레시피는 비-strict; strict는 mid-doc
   false EOD를 만들어 THD 세그먼트를 오염).
3. pad id는 EOD(0) 고정 — `--eod` 값을 바꾸지 말 것 (Phase B 검증으로 전 데이터
   doc-end=id 0 확인 완료 상태).
4. 학습 시 이 데이터는 **THD 경로 전용 플래그**와 함께 사용:
   `--reset-position-ids` + `--no-create-attention-mask-in-dataloader` + MBS=1
   (기존 `--reset-attention-mask` dense 방식과 혼용 금지 — 32k에서 마스크 1GiB/샘플).
   전체 플래그 셋은 LC preset 작성 시 `LC_ENTRY_GATE.md` §1.5 기준으로 확정.
5. 패킹된 pad 런은 학습 배치에서 `merge_eod_pad_segments`
   (`megatron_patch/data/utils.py`, varlen-thd 브랜치)가 문서 세그먼트로 흡수한다 —
   데이터 쪽에서 추가로 할 일 없음.

## 6. 완료 보고 체크리스트

- [ ] 4종 dry-run pad 오버헤드 기록 (각 %)
- [ ] 4종 pack `[STAGE-DONE]` + 내장 post-verify 통과
- [ ] §2.3 %16 정렬 스크립트 전 데이터셋 `bad=0`
- [ ] real token 수 == 15.56B (§6.5 대조)
- [ ] `LC_DATASETS.md` §6.5 / `DATA_PREP_LOG.md` §1 갱신 커밋 (+push)
- [ ] filler 옵션 결정 사항을 `DATA_PREP_LOG.md` 큐 B에 기록
