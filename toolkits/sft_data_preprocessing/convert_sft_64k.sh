#!/usr/bin/env bash
# alpha SFT 64k 버킷 일괄 변환 — build_alpha_sft_idxmap.py, 세트당 순차 실행.
#
# 블렌드 설계 (2026-08-23 확정, SWE 1-pass 앵커 / 총 40B bin-tokens):
#   swe_v3 24.5%(1.0ep) · chat 21.0%(if+chat[복원후]) · cp_v2 14.5%(0.22ep) ·
#   math_v4 11.0%(0.66ep) · science_v2 10.0%(0.27ep) · multilingual 6.5%(0.88ep) ·
#   agentic(opencode+cuda) 5.0%(0.28ep) · arc_agi 3.0%(0.15ep) ·
#   proofs≤64k 1.5% · longblocks_sft 1.5%(변환기 별도 — 후속) ·
#   safety 1.0% · identity 0.5%(결정 #9 상한 내)
#   근거: Ultra3 함의 epoch 역산(SWE-v1 1.1ep) + 카테고리 비율 + chat E_max 4~5.
#
# NFS 주의: emit(쓰기)이 지배 — 동시 1개(순차)로 고정. LC-A 학습(main1)이
# 같은 NFS를 읽는 중 (러너북 함정 4: 동시 쓰기 2개에서 포화).
# 멱등: <name>/data_text_document.idx 존재 시 스킵.
#
# 실행 (유휴 노드): NCORES=96 bash convert_sft_64k.sh
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SFT=/home/work/Datasets/LL_datasets/posttraining/SFT
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_64k_pad16}
NCORES=${NCORES:-96}
TOK=$REPO/examples/alpha/tokenizer_v5
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$REPO/backends/megatron/Megatron-LM-251125${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"

run() {
  local name=$1 input=$2
  shift 2   # 나머지 인자는 변환기 추가 플래그 (예: --fanout-train-turns)
  if [ -f "$OUT/$name/data_text_document.idx" ]; then
    echo "== $name: SKIP (idx 존재)"; return
  fi
  echo "== $name: $input  ($(date +%H:%M:%S))"
  mkdir -p "$OUT/$name"
  nice -n 10 python "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" \
    --input "$input" --tokenizer "$TOK" \
    --output-prefix "$OUT/$name/data" \
    --seq-length 65536 --pad-doc-multiple 16 --workers "$NCORES" "$@" \
    || echo "!! $name FAILED (exit $?)"
}

# 작은 셋 먼저 (이른 신호), 대형 셋 후반
run identity_v1      "$SFT/alpha-SFT-Identity-v1/data/train.jsonl"
run cuda_v1          "$SFT/Nemotron-SFT-CUDA-v1"
run safety_v2        "$SFT/Nemotron-SFT-Safety-v2"
run math_proofs_v2   "$SFT/Nemotron-Math-Proofs-v2"
# swe_v3: 2026-08-24 부터 keepthink 재변환본(swe_v3_keepthink)이 정본 — 템플릿
#   DSV4 tool-시나리오 분기로 멀티 user 턴 궤적(행 2.3%)의 히스토리 reasoning 이
#   보존됨 (verify_chat_template.py §6). 구 swe_v3 디렉토리는 대조·롤백용 보존.
run swe_v3_keepthink "$SFT/Nemotron-SFT-SWE-v3/data"
# arc_agi_v1: 2026-08-24 keepthink 재변환 (tool 멀티턴 22,957행 = 9.1% —
#   swe 주석 참조; 구 디렉토리 보존)
run arc_agi_v1_keepthink "$SFT/Nemotron-SFT-ARC-AGI-v1"
run math_v4          "$SFT/Nemotron-SFT-Math-v4"
run opencode_v1      "$SFT/Nemotron-SFT-OpenCode-v1"
run science_v2       "$SFT/Nemotron-SFT-Science-v2"
run cp_v2            "$SFT/Nemotron-SFT-Competitive-Programming-v2"

# Multilingual-v2: hi 제외, ko/ja/pt 파일별
for f in "$SFT"/Nemotron-SFT-Multilingual-v2/*_ko_* \
         "$SFT"/Nemotron-SFT-Multilingual-v2/*_ja_* \
         "$SFT"/Nemotron-SFT-Multilingual-v2/*_pt_*; do
  [ -f "$f" ] || continue
  run "ml_$(basename "$f" .jsonl)" "$f"
done

# chat_v3_if: 2026-08-24 부터 fan-out 재변환본(chat_v3_if_fanout)이 정본 —
#   IF split 은 60.9%가 multi-True 라 중간 학습 턴의 reasoning 이 단일 렌더에서
#   소실됨 (build_alpha_sft_idxmap.py 의도적 차이 #2). 구 chat_v3_if 디렉토리는
#   대조·롤백용 보존, 블렌드(sft_40b_blend.yaml)는 fanout 을 가리킨다.
# chat_v3_chat 은 fan-out 불요 (train_turns 전수 last-only, docs §2.5) —
#   복원 완료 후 별도 실행:
#   run chat_v3_chat "$SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data/chat.with_prompts.jsonl"
# 2026-08-25 부터 medium-effort 렌더본(chat_v3_if_fanout_me)이 정본 — IF split 은
#   GPT-OSS-120B medium-effort 생성분(컬렉션 README:50)이라 Ultra 레시피상
#   medium_effort=True 로 렌더해야 마커↔trace 조건부가 학습된다. 마커는 렌더
#   kwarg 라 데이터에 없다 (변환기 docstring §Effort/Budget, docs/SFT_RL_DATASETS.md
#   §2.6). 구 chat_v3_if_fanout 은 대조·롤백용 보존, 블렌드는 _me 를 가리킨다.
run chat_v3_if_fanout_me "$SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data/instruction_following.jsonl" \
  --fanout-train-turns --medium-effort

# budget_trunc_v1: Ultra §3.1.1 두 번째 컴포넌트 — 학습 턴 reasoning 을 무작위
#   예산(U(0.1,0.9)·L 토큰)으로 절단(응답 불변) + 잘린 자리 </think> 비학습.
#   IF(fan-out) 1/4 행 + math_v4 1/20 행 → 실측 369.8M real (블렌드 1% 슬롯,
#   2026-08-25 yaml 반영 — 헤더 규칙으로 재산출). 원본 셋과
#   행이 겹치지만 절단본은 다른 분포(강제 </think>)를 가르치므로 의도된 중복.
run budget_trunc_v1_if "$SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data/instruction_following.jsonl" \
  --fanout-train-turns --truncate-reasoning-budget --row-stride 4
run budget_trunc_v1_math "$SFT/Nemotron-SFT-Math-v4" \
  --truncate-reasoning-budget --row-stride 20

echo "== ALL DONE ($(date +%H:%M:%S)) -> $OUT"
