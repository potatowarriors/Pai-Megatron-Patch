# NGC 환경 재빌드 가이드

## 배경

2026-04-07, SGLang 의존성 설치 과정에서 vanilla PyTorch(2.8.0)가 user-level(`~/.local`)에 설치되어
NGC 커스텀 torch(`2.8.0a0+...nv25.5`)를 가림(shadow). 이로 인해 **모든 CUDA 확장의 ABI가 깨짐**.

### 오염 경로

```
pip install sgl-kernel (또는 pip install sglang)
  → torch 2.8.0 (vanilla PyPI) 가 ~/.local에 설치
  → NGC torch 2.8.0a0+nv25.5 를 shadow
  → causal_conv1d, mamba-ssm, flash-attn, transformer-engine 전부 ABI 불일치
  → 추가로 transformers도 4.57.0.dev0 → 4.56.1로 다운그레이드
```

### 설치 스크립트 수정 사항 (이미 반영)

`setup_pai_megatron_env_A100.sh`에 적용된 3중 방어:

1. **스크립트 초반**: NGC torch 버전 검증 (`nv` 문자열 포함 확인)
2. **Step 14**: sgl-kernel/flashinfer에 `--no-deps` 추가 (torch 유입 차단)
3. **Step 14.5**: transformers를 마지막에 `--force-reinstall --no-deps`로 dev commit 핀닝
4. **설치 검증**: NGC 무결성 검사 (user-level shadow 검출)

추가로 Step 5(transformers 조기 설치)를 삭제하고, Step 11.5(떔질 re-pin)도 제거.

## 재빌드 절차

### 1. 새 NGC 컨테이너 시작

```bash
# NGC pytorch:25.06-py3 기반
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.06-py3
```

### 2. 설치 스크립트 실행

```bash
cd /path/to/workspace
git clone <repo> Pai-Megatron-Patch  # 또는 기존 repo 마운트
bash setup_pai_megatron_env_A100.sh --workspace /path/to/workspace
```

스크립트가 자동으로 NGC torch 검증 + SGLang 안전 설치 + transformers 핀닝 + 무결성 검사를 수행.

### 3. 재빌드 후 검증 (수동)

설치 스크립트의 자동 검증에 추가로, 아래를 직접 확인:

```bash
# A. NGC torch 확인
python -c "import torch; assert 'nv' in torch.__version__; print(f'NGC torch: {torch.__version__}')"

# B. transformers dev 확인 (is_flash_linear_attention_available 존재해야 함)
python -c "from transformers.utils.import_utils import is_flash_linear_attention_available; print('transformers dev: OK')"

# C. CUDA 확장 전체 import
python -c "
from causal_conv1d import causal_conv1d_fn; print('causal_conv1d: OK')
import mamba_ssm; print('mamba_ssm: OK')
from flash_attn import flash_attn_func; print('flash_attn: OK')
import transformer_engine; print('transformer_engine: OK')
from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print('fla: OK')
import sgl_kernel; print('sgl_kernel: OK')
"

# D. user-level shadow 없음 확인 (출력 없어야 정상)
pip list --user | grep -E "^(torch |triton |transformers )"

# E. check_attention_logits.py 동작 확인
cd examples/alpha/scripts
python check_attention_logits.py \
  --model-path ../outputs/alpha_baseline_48L_stage2_20260330_160918/hfmodel_0250000 \
  --seq-len 4096 --num-samples 4
```

### 4. 실패 시 대처

| 증상 | 원인 | 해결 |
|------|------|------|
| `assert 'nv' in torch.__version__` 실패 | user-level torch가 NGC를 가림 | `pip uninstall torch torchvision torchaudio` |
| `is_flash_linear_attention_available` import 에러 | transformers가 stable로 교체됨 | Step 14.5 재실행 또는 수동: `sudo pip install --force-reinstall --no-deps git+https://github.com/huggingface/transformers.git@5f6e278a5177d8b85945a2cdb6b776dacee34914` |
| `undefined symbol: _ZN3c10...` | CUDA 확장 ABI 불일치 | user-level torch 제거 후 해당 패키지 재빌드 |
| `pip list --user`에 torch 출력 | 외부에서 pip install 실행됨 | `pip uninstall torch` (user-level 제거) |

## 금지 사항

```bash
# 절대 하지 마세요:
pip install torch              # NGC torch가 vanilla로 교체됨
pip install sglang             # torch/transformers가 의존성으로 딸려옴
pip install sglang[runtime-common]  # transformers==4.56.1 핀닝
pip install --user <pkg>       # user-level이 system을 가림
```

SGLang은 **git submodule**로 관리됩니다 (`backends/sglang/sglang-v0.5.2/`).
CUDA 커널(sgl-kernel, flashinfer)만 `--no-deps`로 설치합니다.
