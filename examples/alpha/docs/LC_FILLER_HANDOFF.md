# LC-A filler 작업 인계 — 100코어 노드 이어서 실행 (2026-08-22 18:00 작성)

> **✅ 완료 (2026-08-23 새벽, 220코어 분석노드)** — §2 재개 실행으로 잔여 7종 전부
> 빌드·검증 통과. **specialized 15/15 PACKED**, yaml 미빌드 경로 0 (§3-(a) 통과).
> 상세는 아래 §6 완료 보고. §3-(b) %16 표본검사도 검사 세션이 통과시킴
> (15종 × 50 bins bad=0 + 32멤버 deep preflight; `scripts/lc_a_preflight.py`).
> **LC-A 학습 개시됨**: `outputs/alpha_baseline_48L_lc_a_20260822_191424`
> (P3 iter 26832에서 CP4·32K·THD·GBS384·LR 7.5e-6, ~3일 예상).
> 남은 것: (c)~(e) 문서 추기·커밋(사용자 승인 대기).

**목적**: 8코어 세션에서 진행하던 LC-A filler 구축(러너북 `LC_REPACK_RUNBOOK.md` §3,
P3 미러 결정)을 **100코어 노드에서 이어서 완료**하기 위한 상태 스냅샷 + 재개 절차.
남은 것은 specialized 재생산 후반부와 마무리 검증·문서화뿐이다.

## 0. 완료된 것 (재실행 금지 — 전부 검증 통과)

| 산출물 | 위치 (`LL_preprocessed/v5/`) | 실측 |
|---|---|---|
| 32k pad16 재패킹 6종 (EN4+KO2) | `cpt_lc_packed_32k_pad16/{longblocks,pg19,edgar,pes2o,ko_news,ko_grounded}` | EN 15.56B + KO 468M real, %16 검증 전 종 0 miss |
| filler 11종 | `lc_filler_packed_32k_pad16/{korean_web,fineweb2hq,math,code_review,question_answering,rewriting,student_teacher,transpilation,cc_code,cchq_actual,cchq_qa}` | 합 **42.9B real** (korean_web 16.96B · fw2hq 5.72B · math 4.53B · code5종 7.04B · cc_code 4.94B · cchq 3.71B) |
| filler 블렌드 yaml 초안 | `examples/alpha/configs/data/lc_filler_32k_pad16.yaml` | P3 가중 26멤버, specialized 8경로만 미빌드 |
| specialized 완료분 | `lc_filler_packed_32k_pad16/specialized/{v1.1 5종, v1/infinibyte_reasoning, v1/math_textbooks, v1/rqa}` | 8/15 (rqa 단독 15.2h — packed 516GB) |

- 서브셋 샤드 선택은 결정적 해시(`syn_data/scripts/lc_filler_subset.py`) — 재현 가능,
  스테이징은 `lc_filler/_subset_jsonl/`에 **하드링크**(심링크는 preprocess가 못 따라감).
- 로그: `syn_data/outputs/lc_filler_logs/` (specialized_0821.log, queue_0821.log 등).

## 1. 구 노드(8코어) 정지 — 새 노드 시작 전 필수

같은 NFS 출력에 **동시 쓰기 금지**. 구 세션(8코어)에서:

```bash
pkill -f "run_stage3_v5.sh"; sleep 2; pkill -f fast_tokenize_v5   # 잔여 워커까지
# 진행 중이던 stem_sft의 부분 산출물 정리 (tokenize 도중 킬 → _parts 불완전):
rm -rf /home/work/Datasets/LL_preprocessed/v5/lc_filler/specialized/v1/stem_sft
# (jsonl 변환본은 stage3/_jsonl/ 에 남아 있어 convert는 자동 skip됨)
```

## 2. 새 노드(100코어)에서 재개 — 명령 하나

파이프라인은 **서브셋 단위 멱등** (packed 존재→skip / unpacked만 존재→pack만 /
둘 다 없음→convert(기존물 skip)→tokenize→pack). 완료된 8종은 자동 skip된다.

```bash
cd /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing

# GPU 없는 노드면 필수 (TE→triton 크래시 방지; GPU 노드면 생략 가능하나 있어도 무해):
export PYTHONPATH=/home/work/vidsearch/repos/project_s/syn_data/src/syn_data/_cpu_shims

NCORES=96 \
OUT_DIR=/home/work/Datasets/LL_preprocessed/v5/lc_filler/specialized \
OUT_PACKED_DIR=/home/work/Datasets/LL_preprocessed/v5/lc_filler_packed_32k_pad16/specialized \
SEQLEN=32768 PAD_DOC_MULTIPLE=16 KEEP_UNPACKED=1 \
  nice -n 10 bash run_stage3_v5.sh all 2>&1 | tee -a /home/work/vidsearch/repos/project_s/syn_data/outputs/lc_filler_logs/specialized_resume_$(date +%m%d).log
```

- `NCORES=96`: 토크나이즈가 코어 수에 거의 선형 (8코어에서 13~17G raw/h였음 → ~10배 기대).
  PROCS/RAYON은 스크립트가 NCORES에서 자동 산출.
- **`KEEP_UNPACKED=1` 필수** — 128k filler 재패킹(러너북 작업 3)에 unpacked 재사용.
- 남은 작업량: stem_sft(91G, 처음부터) + scientific_coding(0.5G) + wiki_rewrite(9.3G)
  + v1.2 4종(51G) ≈ **152G raw** → 96코어면 토크나이즈는 짧고 pack emit(NFS ~132MB/s
  포화, 동시 2개 제한)이 지배 — 수 시간 예상.

## 3. 완료 후 체크리스트 (러너북 §6 + filler 확장)

```bash
# (a) yaml 미빌드 8경로가 전부 생겼는지:
grep -o '/home[^ "]*data_text_document' \
  /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/configs/data/lc_filler_32k_pad16.yaml \
  | while read p; do [ -f "$p.bin" ] || echo "MISSING: $p"; done   # 출력 없어야 통과

# (b) %16 정렬 표본검사 — 러너북 §2.3 스크립트의 데이터셋 목록을
#     lc_filler_packed_32k_pad16 의 specialized 서브셋 경로로 바꿔 실행 (bad=0 전부)

# (c) real 토큰 집계 → LC_DATASETS.md §6.5 재패킹 표 아래 filler 표 추기
#     + DATA_PREP_LOG.md 큐 B에 P3-미러 완료 기록
# (d) docs 커밋(+push) — ⚠️ 재패킹분 문서 수정도 아직 미커밋 (사용자 승인 대기 상태였음):
#     LC_DATASETS.md / DATA_PREP_LOG.md / LC_FILLER_HANDOFF.md(본 문서) / lc_filler_32k_pad16.yaml
# (e) 구 cpt_lc_packed_32k/ 삭제는 LC-A 최종 yaml이 pad16 경로로 확정된 뒤 (러너북 §2.4)
```

## 4. 이후 큐 (이번 인계 범위 밖, 참고)

- **LC-A 최종 yaml**: 본 filler yaml(54% 슬롯) + `cpt_lc_packed_32k_pad16` 6종(46% 슬롯) 합성
  — 블렌드 설계는 학습팀, THD 플래그 프리셋은 `LC_ENTRY_GATE.md` §1.5.
- **128k pad16 재패킹** (러너북 작업 3): ge64k 8.65B + `syn_data/outputs/nikl_news/ko_news_128k`
  (KO ≥64K 다문서 샘플, 내부 EOD 없음 준수) + specialized unpacked 재사용, SEQ=131072.
  100코어 노드에서 도는 게 적합.
- syn_data 쪽 후속: 그라운디드 트랜치 2 go/no-go (롱 이벌 A/B 후), 판례/DART 도메인 보강.

## 5. 함정 요약 (이번에 실제로 밟은 것)

1. **preprocess의 파일 탐색은 심링크 불추종** — 서브셋 스테이징은 하드링크
   (`lc_filler_subset.py --stage`가 처리).
0. *(2026-08-23 추가)* **emit 병목은 두 종류** — 러너북 함정 5번("emit은 NFS 상한")은
   긴 문서 셋에만 맞다. 초단문 셋(fact_seeking, 평균 ~57tok/doc)은 bin당 멤버 ~512개의
   per-doc Python 오버헤드(pread+frombuffer+pad concat, ~35k docs/s)가 천장 —
   **~9MB/s**로 NFS 상한의 1/20. 페이지 캐시 워밍 무효(이미 캐시 히트 상태였음).
   128k filler 재패킹 시간 예산 산정 시 문서 길이 분포를 먼저 볼 것.

## 6. 완료 보고 (2026-08-23, 이번 세션 실행분 7종)

`run_stage3_v5.sh all` (NCORES=192, 220코어 노드) 총 ~9.5h. 전부 내장 post-verify
(전 bin 32768 · 토큰 보존 · round-trip 20/20) 통과, exit 0.

| 서브셋 | real tokens | fill | packed 크기 |
|---|---|---|---|
| v1/stem_sft | 81.81B | 99.69% | 305.7 GB |
| v1/scientific_coding | 1.21B | 99.24% | 4.5 GB |
| v1/wiki_rewrite | 7.25B | 99.28% | 27.2 GB |
| v1.2/fact_seeking | 32.79B | **88.43%** | 138.1 GB |
| v1.2/generative | 0.67B | 98.34% | 2.5 GB |
| v1.2/moral_scenarios | 0.015B | 97.76% | 0.06 GB |
| v1.2/multiple_choice | 6.89B | 97.44% | 26.4 GB |
| **합계 (신규 7종)** | **130.64B** | | ~504 GB |

- fact_seeking fill 88.4% = pad16 구조 비용(문서 5.74억 개 × 평균 ~57 real tok →
  문서당 평균 ~7.5tok 패딩). 블렌드 가중/토큰 집계는 반드시 **real tokens** 기준.
- BFP truncation: fact_seeking 0, multiple_choice 10(>32k 장문 — eod-unmasked leak
  ~0.0000%, 수용). KEEP_UNPACKED=1로 unpacked 전부 보존(128k 재패킹 재료).
- ⚠️ 실행 중 노드 시계가 NFS 서버 시간대(~9h 뒤)로 스텝 보정됨 — 일부 로그의
  음수 소요시간(-31783s)과 mtime 혼동은 이 때문. 산출물 무결성 무관.
2. **tokenize 파티션도 megatron→TE→triton import** — GPU 없는 노드는 `_cpu_shims`를
   env 전체에 export (bestfit_pack 라인에만 걸면 tokenize가 죽는다).
3. 생성 커맨드 체인엔 fail-fast/exists-skip 가드 명시 (`A && B`만 쓰면 실패가 다음
   카테고리로 조용히 넘어간다).
4. NFS 쓰기 무거운 작업 동시 2개까지 (pack emit 2개에서 write 포화 실측).
5. rqa 같은 대형 서브셋은 pack emit이 지배 — NCORES를 올려도 emit은 NFS 상한.
