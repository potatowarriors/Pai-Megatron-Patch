#!/bin/bash
#
# Alpha 프로젝트 환경 검증 스크립트
# 학습 시작 전 모든 필수 요소를 확인합니다
#
# 사용법: bash scripts/validate_environment.sh

ALPHA_DIR="$( cd "$( dirname "$0" )/.." && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${ALPHA_DIR}))

cd ${MEGATRON_PATCH_PATH}

echo "=========================================="
echo "Alpha 프로젝트 환경 검증"
echo "=========================================="

#==============================================================================
# 1. 소프트웨어 환경
#==============================================================================

echo -e "\n[1] 소프트웨어 환경"
python3 << 'PYEOF'
import sys

# Flash Attention 3
try:
    import flash_attn_3
    from flash_attn_3.flash_attn_interface import flash_attn_func
    print("  ✓ Flash Attention 3: 사용 가능")
except ImportError:
    print("  ✗ Flash Attention 3: 설치 필요")
    print("    pip install flash-attn-3 --no-build-isolation")

# Transformer Engine
try:
    import transformer_engine.pytorch as te
    print("  ✓ Transformer Engine PyTorch: 사용 가능")
except ImportError:
    print("  ✗ Transformer Engine PyTorch: 사용 불가")

# Megatron-Core
sys.path.insert(0, '/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch')
sys.path.insert(0, '/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250908')

try:
    from megatron.core import parallel_state
    from megatron.core.transformer.moe import moe_layer
    print("  ✓ Megatron-Core (250908): 사용 가능")
except ImportError as e:
    print(f"  ✗ Megatron-Core: import 실패 ({e})")

# PyTorch & GPU
import torch
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"  ✓ GPU: {gpu_count}개 사용 가능")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"    - GPU {i}: {gpu_name}")
else:
    print("  ✗ GPU: 사용 불가")

# PyYAML (설정 파일용)
try:
    import yaml
    print("  ✓ PyYAML: 사용 가능")
except ImportError:
    print("  ✗ PyYAML: 설치 필요")
    print("    pip install pyyaml")
PYEOF

#==============================================================================
# 2. 설정 파일
#==============================================================================

echo -e "\n[2] 설정 파일"

CONFIG_DIR="${ALPHA_DIR}/configs"
CONFIG_FILES=(
    "model/baseline_48L.yaml"
    "training/pretrain.yaml"
    "training/h100x8.yaml"
    "data/kormo_1pct.yaml"
    "env.yaml"
)

for config in "${CONFIG_FILES[@]}"; do
    if [ -f "${CONFIG_DIR}/${config}" ]; then
        echo "  ✓ ${config}"
    else
        echo "  ✗ ${config}: 없음"
    fi
done

#==============================================================================
# 3. 데이터셋
#==============================================================================

echo -e "\n[3] 데이터셋"

# YAML에서 데이터 경로 읽기
DATA_PATH=$(python3 -c "
import yaml
with open('${CONFIG_DIR}/data/kormo_1pct.yaml', 'r') as f:
    data = yaml.safe_load(f)
print(data['data']['train_path'])
" 2>/dev/null)

if [ -z "$DATA_PATH" ]; then
    echo "  ✗ 데이터 경로를 읽을 수 없습니다"
else
    if [ -f "${DATA_PATH}.bin" ] && [ -f "${DATA_PATH}.idx" ]; then
        BIN_SIZE=$(du -sh "${DATA_PATH}.bin" 2>/dev/null | cut -f1)
        echo "  ✓ 데이터셋: ${DATA_PATH}"
        echo "    크기: ${BIN_SIZE}"
    else
        echo "  ✗ 데이터셋 파일 없음: ${DATA_PATH}.{bin,idx}"
        echo "    📥 전처리 명령:"
        echo "    cd toolkits/pretrain_data_preprocessing/"
        echo "    bash preprocess_kormo_subset.sh 1"
    fi
fi

#==============================================================================
# 4. 토크나이저
#==============================================================================

echo -e "\n[4] 토크나이저"

TOKENIZER_PATH=$(python3 -c "
import yaml
with open('${CONFIG_DIR}/model/baseline_48L.yaml', 'r') as f:
    data = yaml.safe_load(f)
print(data['model']['tokenizer_path'])
" 2>/dev/null)

if [ -z "$TOKENIZER_PATH" ]; then
    echo "  ✗ 토크나이저 경로를 읽을 수 없습니다"
else
    if [ -d "$TOKENIZER_PATH" ]; then
        echo "  ✓ 토크나이저: ${TOKENIZER_PATH}"
        # tokenizer.json이나 vocab.json 확인
        if [ -f "${TOKENIZER_PATH}/tokenizer.json" ] || [ -f "${TOKENIZER_PATH}/vocab.json" ]; then
            echo "    설정 파일 확인됨"
        fi
    else
        echo "  ✗ 토크나이저 없음: ${TOKENIZER_PATH}"
    fi
fi

#==============================================================================
# 5. 스크립트 실행 권한
#==============================================================================

echo -e "\n[5] 스크립트 실행 권한"

SCRIPTS=(
    "${ALPHA_DIR}/train.sh"
    "${ALPHA_DIR}/scripts/validate_environment.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "  ✓ $(basename $script): 실행 가능"
        else
            echo "  ⚠ $(basename $script): 실행 권한 없음"
            echo "    chmod +x $script"
        fi
    else
        echo "  ✗ $(basename $script): 파일 없음"
    fi
done

#==============================================================================
# 6. 디스크 공간
#==============================================================================

echo -e "\n[6] 디스크 공간"

OUTPUT_DIR="${ALPHA_DIR}/outputs"
FREE_SPACE=$(df -h ${ALPHA_DIR} | tail -1 | awk '{print $4}')
echo "  사용 가능 공간: ${FREE_SPACE}"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "  ✓ 출력 디렉토리 생성: ${OUTPUT_DIR}"
else
    echo "  ✓ 출력 디렉토리: ${OUTPUT_DIR}"
fi

#==============================================================================
# 요약
#==============================================================================

echo -e "\n=========================================="
echo "검증 완료"
echo "=========================================="
echo ""
echo "다음 명령으로 학습을 시작할 수 있습니다:"
echo "  cd ${ALPHA_DIR}"
echo "  bash train.sh"
echo ""
