#!/bin/bash
# install_tau2.sh — τ³-bench(tau2-bench) 하니스를 NFS 에 설치·검증한다. 멱등. sub1 에서 실행.
#
# 왜 sub1 직접인가: tau2-bench 는 docker 가 필요 없다(도구가 JSON DB 위의 순수 파이썬).
# gpu06 컨테이너·역터널 없이 fleet(:8100) 옆에서 돈다. 설치물은 NFS 라 세션 재구축 후에도 남는다.
#
# 사용: bash eval_sft/install_tau2.sh            # 기본 v1.0.1
#       TAU2_TAG=v1.0.1 bash eval_sft/install_tau2.sh
set -euo pipefail
TOOLS=/home/work/vidsearch/tools
TAU2_HOME="${TAU2_HOME:-$TOOLS/tau2-bench}"
TAG="${TAU2_TAG:-v1.0.1}"
# NGC 전역 PIP_CONSTRAINT 가 venv 를 오염시킨다 (serve_alpha.sh 와 같은 이유)
export PIP_CONSTRAINT=
export PATH="$TOOLS/bin:$PATH" UV_CACHE_DIR="$TOOLS/uv_cache" UV_INSTALL_DIR="$TOOLS/bin"
mkdir -p "$TOOLS/bin"

echo "== 1) uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh
fi
uv --version

echo "== 2) clone @ $TAG"
if [ ! -d "$TAU2_HOME/.git" ]; then
  git clone --branch "$TAG" --depth 1 https://github.com/sierra-research/tau2-bench "$TAU2_HOME"
fi
cd "$TAU2_HOME"
git fetch -q --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true
git checkout -q "$TAG"
git rev-parse HEAD > .pinned_commit
echo "   commit $(cat .pinned_commit)"

echo "== 3) uv sync (core extras: airline/retail/telecom/mock)"
uv sync --python /usr/bin/python3.12 --frozen 2>&1 | tail -3

echo "== 3b) litellm 모델 레지스트리 훅 (.pth)"
# litellm 은 미등록 모델의 비용 계산에서 예외를 낸다(tau2 는 잡아서 0 으로 두지만 호출마다 ERROR 로그).
# LITELLM_MODEL_REGISTRY_PATH 는 mini-swe-agent 의 기능이지 litellm 의 것이 아니다 → 이 venv 에서는 site 초기화 때
# .pth 의 import 줄이 그 파일을 읽어 litellm.register_model() 한다. (sitecustomize.py 는 데비안 시스템 파이썬의
# /usr/lib/python3.12/sitecustomize.py 가 먼저 잡혀 동작하지 않는다.) 파일은 eval_sft/configs/alpha_model_registry.json.
SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
rm -f "$SITE/sitecustomize.py"
cat > "$SITE/_tau2_litellm_registry.py" <<'PYS'
# tau2-bench venv 전용 (install_tau2.sh 가 생성, zz_tau2_litellm_registry.pth 가 import). LITELLM_MODEL_REGISTRY_PATH 의 JSON 을 litellm 에 등록한다.
import json, os
_p = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
if _p and os.path.exists(_p):
    try:
        import litellm
        _d = {k: v for k, v in json.load(open(_p)).items() if not k.startswith("_") and isinstance(v, dict)}
        litellm.register_model(_d)
    except Exception as _e:  # noqa: BLE001
        import sys
        print(f"[tau2 registry hook] litellm register_model 실패: {_e}", file=sys.stderr)
PYS
echo "import _tau2_litellm_registry" > "$SITE/zz_tau2_litellm_registry.pth"
echo "   $SITE/zz_tau2_litellm_registry.pth → _tau2_litellm_registry.py"

echo "== 4) 검증"
.venv/bin/tau2 --help >/dev/null && echo "   tau2 CLI OK"
.venv/bin/tau2 check-data 2>&1 | tail -5
echo "   -- run 플래그 실재 확인"
.venv/bin/tau2 run --help 2>&1 | grep -oE -- '--(task-split-name|auto-resume|timeout|max-concurrency|num-trials|num-tasks|task-ids|save-to|enforce-communication-protocol|max-steps|max-errors|seed|log-level)\b' | sort -u | tr '\n' ' '; echo
echo "   -- litellm / 레지스트리"
.venv/bin/python -c "import importlib.metadata as m, tau2; print('   litellm', m.version('litellm'))"
echo "   -- 과제 수 (base split)"
.venv/bin/python "$(dirname "$0")/tau_tasks_count.py" 2>&1 | sed 's/^/   /' || echo "   (과제 수 확인 스크립트 실패 — tau_tasks_count.py 를 소스에 맞춰 수정)"
echo "== 완료: $TAU2_HOME (.venv/bin/tau2)"
