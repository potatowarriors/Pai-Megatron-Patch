#!/bin/bash
# restore_bench_env.sh — 컨테이너 세션 재생성 후 벤치 평가 환경 복원 (sub1).
#   NFS(영속): serve venv·lm_eval0412·compat deb·참조로짓·HF캐시 → 그대로 살아남음.
#   휘발(재적용): ① CUDA13 compat 시스템 스왑 ② ifeval leaf 의존성. 이 스크립트가 처리.
# 멱등: 이미 돼 있으면 skip. 사용: bash eval_sft/restore_bench_env.sh
set -uo pipefail
T=/home/work/vidsearch/tools
VENV=$T/alpha_serve_venv
COMPAT_SRC=$T/cuda_compat13/extracted/usr/local/cuda-13.2/compat
REAL=/usr/local/cuda/compat/lib.real
SUDO=$(command -v sudo >/dev/null && echo "sudo -n" || echo "")

echo "== 1) CUDA 13 compat (570→595) 재적용 =="
if [ "$(readlink $REAL/libcuda.so.1 2>/dev/null)" = "libcuda.so.595.91.07" ]; then
    echo "  이미 595 — skip"
else
    if [ ! -f "$COMPAT_SRC/libcuda.so.595.91.07" ]; then
        echo "  ❌ compat 소스 없음 ($COMPAT_SRC). deb 재추출 필요:"
        echo "     cd $T/cuda_compat13 && dpkg-deb -x cuda-compat-13-2_*.deb extracted"
        exit 1
    fi
    $SUDO cp -P "$COMPAT_SRC"/libcuda.so.595.91.07 "$COMPAT_SRC"/libnvidia-*.so.595.91.07 "$REAL/"
    $SUDO ln -sf libcuda.so.595.91.07 "$REAL/libcuda.so.1"
    $SUDO ln -sf libcuda.so.1 "$REAL/libcuda.so"
    echo "  → 595 적용: $(readlink $REAL/libcuda.so.1)"
fi

echo "== 2) ifeval leaf 의존성 (user-site, 휘발) =="
python3 -c "import langdetect, immutabledict; from importlib.metadata import version; assert version('nltk')>='3.9.1'" 2>/dev/null \
  && echo "  이미 설치됨 — skip" \
  || { PIP_CONSTRAINT= pip install -q --user "nltk>=3.9.1" langdetect immutabledict 2>&1 | tail -1; echo "  → 설치 완료"; }

echo "== 3) 검증 =="
PIP_CONSTRAINT= $VENV/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA 미인식(compat 실패?)'; import vllm; import vllm_alpha_plugin; vllm_alpha_plugin.register(); from vllm import ModelRegistry; assert 'AlphaForCausalLM' in ModelRegistry.get_supported_archs(); print('  serve venv OK: torch', torch.__version__, '| vllm', vllm.__version__, '| 플러그인 등록 True')" || { echo "  ❌ serve venv 검증 실패"; exit 1; }
PYTHONPATH=$T/lmeval0412 python3 -c "import lm_eval; print('  lm_eval', lm_eval.__version__, 'OK')" 2>/dev/null | tail -1
[ -f $T/alpha_ref_logits_iter320.pt ] && echo "  참조로짓 OK" || echo "  ⚠️ 참조로짓 없음(재생성: tools/gen_ref_logits_iter320.py)"
[ -d /home/work/Datasets/benchmarks ] && echo "  HF 벤치캐시 OK" || echo "  ⚠️ HF 캐시 없음"
echo "== 복원 완료. 서빙: bash eval_sft/serve_fleet.sh <hfmodel> 49152 8 =="
