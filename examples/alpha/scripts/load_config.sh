#!/bin/bash
#
# YAML 설정 파일 로드 헬퍼 함수
# Alpha 프로젝트의 YAML 설정을 Shell 변수로 변환
#
# 사용법:
#   source scripts/load_config.sh
#   value=$(yaml_get "configs/model/baseline_24L.yaml" "model.num_layers")
#

# yaml_get: YAML 파일에서 값 추출
# 인자:
#   $1: YAML 파일 경로
#   $2: Key path (점으로 구분, 예: "model.num_layers")
# 리턴:
#   추출된 값 (없으면 빈 문자열)
#
# 예시:
#   NUM_LAYERS=$(yaml_get "configs/model/baseline_24L.yaml" "model.num_layers")
#   echo "Layers: ${NUM_LAYERS}"
yaml_get() {
    local file=$1
    local key=$2

    if [ ! -f "$file" ]; then
        echo "" >&2
        return 1
    fi

    python3 -c "
import yaml
import sys

try:
    with open('$file', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    keys = '$key'.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, None)
        else:
            value = None
            break

    if value is not None:
        print(value)
    else:
        print('')
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
}

# yaml_get_array: YAML 배열을 공백으로 구분된 문자열로 변환
# 인자:
#   $1: YAML 파일 경로
#   $2: Key path
# 리턴:
#   배열 요소들을 공백으로 연결한 문자열
#
# 예시:
#   SPLIT=$(yaml_get_array "configs/data/kormo_1pct.yaml" "data.split")
#   # 결과: "99 1 0"
yaml_get_array() {
    local file=$1
    local key=$2

    if [ ! -f "$file" ]; then
        echo "" >&2
        return 1
    fi

    python3 -c "
import yaml
import sys

try:
    with open('$file', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    keys = '$key'.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, None)
        else:
            value = None
            break

    if isinstance(value, list):
        print(' '.join(map(str, value)))
    elif value is not None:
        print(value)
    else:
        print('')
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
}

# yaml_to_csv: YAML 배열을 CSV 형식으로 변환
# 인자:
#   $1: YAML 파일 경로
#   $2: Key path
# 리턴:
#   배열 요소들을 쉼표로 연결한 문자열
#
# 예시:
#   SPLIT=$(yaml_to_csv "configs/data/kormo_1pct.yaml" "data.split")
#   # 결과: "99,1,0"
yaml_to_csv() {
    local file=$1
    local key=$2
    local array=$(yaml_get_array "$file" "$key")
    echo "$array" | tr ' ' ','
}

# yaml_exists: YAML 파일에서 키 존재 여부 확인
# 인자:
#   $1: YAML 파일 경로
#   $2: Key path
# 리턴:
#   0 (존재) 또는 1 (없음)
yaml_exists() {
    local file=$1
    local key=$2
    local value=$(yaml_get "$file" "$key")
    [ ! -z "$value" ]
}

# 사용 예시 (이 스크립트를 직접 실행하면 테스트 실행)
if [ "${BASH_SOURCE[0]}" -eq "${0}" ]; then
    echo "=========================================="
    echo "YAML 설정 로드 헬퍼 테스트"
    echo "=========================================="

    SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
    ALPHA_DIR=$(dirname "$SCRIPT_DIR")
    CONFIG_DIR="${ALPHA_DIR}/configs"

    # 테스트 1: 단순 값 읽기
    echo -e "\n[테스트 1] 모델 레이어 수 읽기"
    NUM_LAYERS=$(yaml_get "${CONFIG_DIR}/model/baseline_24L.yaml" "model.num_layers")
    echo "  model.num_layers = ${NUM_LAYERS}"

    # 테스트 2: 배열 읽기
    echo -e "\n[테스트 2] 데이터 split 읽기"
    SPLIT=$(yaml_get_array "${CONFIG_DIR}/data/kormo_1pct.yaml" "data.split")
    echo "  data.split (배열) = ${SPLIT}"

    # 테스트 3: CSV 변환
    echo -e "\n[테스트 3] CSV 변환"
    SPLIT_CSV=$(yaml_to_csv "${CONFIG_DIR}/data/kormo_1pct.yaml" "data.split")
    echo "  data.split (CSV) = ${SPLIT_CSV}"

    # 테스트 4: 키 존재 확인
    echo -e "\n[테스트 4] 키 존재 여부"
    if yaml_exists "${CONFIG_DIR}/model/baseline_24L.yaml" "model.num_layers"; then
        echo "  ✓ model.num_layers 존재"
    else
        echo "  ✗ model.num_layers 없음"
    fi

    if yaml_exists "${CONFIG_DIR}/model/baseline_24L.yaml" "model.nonexistent_key"; then
        echo "  ✓ model.nonexistent_key 존재"
    else
        echo "  ✗ model.nonexistent_key 없음 (정상)"
    fi

    echo ""
fi
