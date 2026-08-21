#!/bin/bash
#
# run_cpt_lc_v5.sh — LC(CPT) 데이터셋 4종 전처리 오케스트레이터.
#
# 데이터셋별 체인 (docs/LC_DATASETS.md §6 체크리스트 구현):
#   convert(jsonl) -> tokenize(unpacked mmap, KEEP) -> split_by_doclen(lt/ge 64k)
#   -> bestfit_pack(lt만, seq 32768)
#
#   longblocks : _jsonl 준비 완료(191,758 samples) — convert 생략
#   pg19       : train-*.parquet만 (val/test 150권 제외) -> convert_parquet_to_jsonl
#   edgar      : 섹션 결합 (convert_edgar_to_jsonl, 전 연도·전 스플릿)
#   pes2o      : s2orc & >=77k chars 필터 (convert_pes2o_lc_to_jsonl, train만)
#
# 산출 (unpacked는 canonical — 삭제 금지, §5.1 배분 정책):
#   /home/work/Datasets/LL_preprocessed/v5/cpt_lc/<ds>/data_text_document.{bin,idx}
#   /home/work/Datasets/LL_preprocessed/v5/cpt_lc/<ds>/{lt64k,ge64k}_text_document.*
#   /home/work/Datasets/LL_preprocessed/v5/cpt_lc_packed_32k/<ds>/data_text_document.*
#
# 멱등: 각 스테이지는 산출물 존재 시 스킵. 로그 마커: [STAGE-DONE] [STAGE-FAIL].
#
# Usage: bash run_cpt_lc_v5.sh [all|longblocks|pg19|edgar|pes2o]
#   PROCS(기본 4) RAYON_THREADS(기본 2) — 8코어 분석 노드 기준.
#   PACKED_DIR         packed 산출 디렉토리 override (재패킹 시 신규 경로 지정 —
#                      기본 경로는 exists-skip이라 덮어쓰지 않음)
#   PAD_DOC_MULTIPLE   bestfit_pack --pad-doc-multiple (기본 1 = 종전과 동일.
#                      THD+CP 문서 격리는 16 필요 — docs/LC_REPACK_RUNBOOK.md)
#   SEQ                packed seq-length (기본 32768; 128k 스테이지는 131072)

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
RAW="/home/work/Datasets/LL_datasets/longcontext/en"
OUT="/home/work/Datasets/LL_preprocessed/v5/cpt_lc"
PACKED="${PACKED_DIR:-/home/work/Datasets/LL_preprocessed/v5/cpt_lc_packed_32k}"
PROCS="${PROCS:-4}"
export RAYON_THREADS="${RAYON_THREADS:-2}"
SPLIT_T=65536
SEQ="${SEQ:-32768}"
PAD_DOC_MULTIPLE="${PAD_DOC_MULTIPLE:-1}"

fail() { echo "[STAGE-FAIL] $1"; exit 1; }

tokenize() {  # $1=ds $2=jsonl_dir $3=batch_size
    # batch_size는 문서 "개수" 단위 — 장문 데이터셋(책/공시)은 반드시 작게.
    # (BATCH_SIZE=5000 × PG19 평균 400KB 문서 → encode_batch 메모리 폭발로 OOM,
    #  2026-08-06 pg19 파티션 1·3 실측. 문서가 클수록 배치를 줄인다.)
    local ds="$1" jdir="$2" bs="${3:-2000}"
    if [ -f "${OUT}/${ds}/data_text_document.bin" ]; then
        echo "[skip] ${ds}/tokenize (exists)"; return 0
    fi
    mkdir -p "${OUT}/${ds}"
    BATCH_SIZE="${bs}" bash "${DIR}/preprocess_stage2_v5.sh" "${jdir}" "${OUT}/${ds}/data" "${PROCS}" text \
        || fail "${ds}/tokenize"
    echo "[STAGE-DONE] ${ds}/tokenize"
}

split_ds() {  # $1=ds
    local ds="$1"
    if [ -f "${OUT}/${ds}/lt64k_text_document.bin" ]; then
        echo "[skip] ${ds}/split (exists)"; return 0
    fi
    python3 "${DIR}/split_by_doclen.py" --input "${OUT}/${ds}/data" \
        --threshold "${SPLIT_T}" \
        --output-lt "${OUT}/${ds}/lt64k" --output-ge "${OUT}/${ds}/ge64k" \
        || fail "${ds}/split"
    echo "[STAGE-DONE] ${ds}/split"
}

pack_ds() {  # $1=ds
    local ds="$1"
    if [ -f "${PACKED}/${ds}/data_text_document.bin" ]; then
        echo "[skip] ${ds}/pack (exists)"; return 0
    fi
    mkdir -p "${PACKED}/${ds}"
    python3 "${DIR}/bestfit_pack.py" --input "${OUT}/${ds}/lt64k" \
        --output "${PACKED}/${ds}/data" --seq-length "${SEQ}" --eod 0 \
        --pad-doc-multiple "${PAD_DOC_MULTIPLE}" \
        || fail "${ds}/pack"
    echo "[STAGE-DONE] ${ds}/pack"
}

do_longblocks() {
    tokenize longblocks "${RAW}/LongBlocks/_jsonl" 1000
    split_ds longblocks; pack_ds longblocks
}

do_pg19() {
    local tdir="${RAW}/pg19/_train_parquet"
    if ! ls "${RAW}/pg19/_jsonl/"*.done >/dev/null 2>&1; then
        mkdir -p "${tdir}"
        for f in "${RAW}/pg19/data/"train-*.parquet; do
            ln -sf "$f" "${tdir}/$(basename "$f")"
        done
        python3 "${DIR}/convert_parquet_to_jsonl.py" --input-dir "${tdir}" \
            --output-dir "${RAW}/pg19/_jsonl" --workers "${PROCS}" \
            || fail "pg19/convert"
        echo "[STAGE-DONE] pg19/convert"
    else
        echo "[skip] pg19/convert (done markers exist)"
    fi
    tokenize pg19 "${RAW}/pg19/_jsonl" 64
    split_ds pg19; pack_ds pg19
}

do_edgar() {
    python3 "${DIR}/convert_edgar_to_jsonl.py" --input-root "${RAW}/edgar-corpus" \
        --output-dir "${RAW}/edgar-corpus/_jsonl" --workers 6 \
        || fail "edgar/convert"
    echo "[STAGE-DONE] edgar/convert"
    tokenize edgar "${RAW}/edgar-corpus/_jsonl" 256
    split_ds edgar; pack_ds edgar
}

do_pes2o() {
    # --filler-rate 0.03: 주 필터(>=16k tok)만으로는 fill 77%에 그침(bin당 1문서)
    # → 4k~14k tok 문서를 3% 해시 샘플링해 갭 충전 (converter docstring 참조)
    python3 "${DIR}/convert_pes2o_lc_to_jsonl.py" \
        --input-dir "${RAW}/peS2o/data/v3" \
        --output-dir "${RAW}/peS2o/_jsonl_lc" --workers 8 --filler-rate 0.03 \
        || fail "pes2o/convert"
    echo "[STAGE-DONE] pes2o/convert"
    tokenize pes2o "${RAW}/peS2o/_jsonl_lc" 256
    split_ds pes2o; pack_ds pes2o
}

TARGET="${1:-all}"
case "${TARGET}" in
    longblocks) do_longblocks ;;
    pg19)       do_pg19 ;;
    edgar)      do_edgar ;;
    pes2o)      do_pes2o ;;
    all)        do_longblocks; do_pg19; do_edgar; do_pes2o ;;
    *) echo "usage: $0 [all|longblocks|pg19|edgar|pes2o]"; exit 1 ;;
esac
echo "[PIPELINE-DONE] ${TARGET}"
