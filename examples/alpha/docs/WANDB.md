# Weights & Biases (WANDB) 통합 가이드

Alpha 프로젝트에 WANDB를 통합하여 실험 추적 및 로그 관리를 개선합니다.

## 목차

1. [개요](#개요)
2. [설정 방법](#설정-방법)
3. [사용 방법](#사용-방법)
4. [테스트](#테스트)
5. [대시보드 활용](#대시보드-활용)
6. [문제 해결](#문제-해결)

---

## 개요

### WANDB란?

[Weights & Biases](https://wandb.ai)는 머신러닝 실험 추적 플랫폼으로, 다음 기능을 제공합니다:

- **실험 추적**: 학습 메트릭, 하이퍼파라미터, 시스템 리소스 자동 로깅
- **비교**: 여러 실험 간 성능 비교 및 시각화
- **협업**: 팀원과 실험 결과 공유
- **재현성**: 모든 실험 설정 및 결과 보존

### Megatron-LM 지원

- **Megatron-LM-250908** 버전에 WANDB 지원이 내장되어 있습니다
- 추가 코드 수정 없이 설정만으로 활성화 가능합니다
- TensorBoard와 병행 사용 가능합니다

### Alpha 프로젝트 통합 구현

Alpha 프로젝트에서는 YAML 설정 기반 WANDB 통합을 구현했습니다:

1. **YAML 설정 파일** (`configs/training/pretrain.yaml`)
2. **환경 설정 스크립트** (`scripts/setup_wandb.sh`)
3. **통합 테스트 스크립트** (`scripts/test_wandb.sh`)
4. **자동 인자 주입** (`train.sh`)

---

## 설정 방법

### 1. WANDB 계정 및 API 키

키는 **어떤 트래킹 파일에도 하드코딩하지 않는다** (2026-08-18 유출 정리 이후 규칙).
`scripts/setup_wandb.sh`가 아래 순서로 키를 해석한다:

1. 이미 export된 `WANDB_API_KEY` (그대로 존중)
2. `$WANDB_KEY_FILE`이 가리키는 파일
3. `scripts/.wandb_key` (gitignored, chmod 600 — **표준 위치**)
4. `~/.wandb_key`

키 설정/교체는 파일 한 줄 교체로 끝난다:
```bash
printf '%s' '<YOUR_WANDB_API_KEY>' > scripts/.wandb_key && chmod 600 scripts/.wandb_key
```

사용자: `kide004` (https://wandb.ai/kide004) — 키 발급/폐기: https://wandb.ai/settings → API keys

### 2. 환경 변수 설정

WANDB API 키를 환경 변수로 설정:

```bash
# 방법 1: 스크립트 사용 (권장 — scripts/.wandb_key 파일에서 키를 읽음)
source ./scripts/setup_wandb.sh

# 방법 2: 수동 설정 (env가 파일보다 우선)
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"   # 셸 히스토리 주의; 파일 방식 권장
export WANDB_MODE="online"
```

### 3. YAML 설정

`configs/training/pretrain.yaml`에서 WANDB 활성화:

```yaml
training:
  # ... 기타 설정 ...

  # Weights & Biases
  wandb:
    enabled: true                    # WANDB 활성화
    project: "alpha-pretraining"     # 프로젝트 이름
    entity: ""                       # 팀/조직 이름 (선택사항)
    save_dir: ""                     # 로그 저장 경로 (선택사항)
    notes: "Alpha project baseline experiments"
```

**주의**: 기본값은 `enabled: false`입니다. 학습 시 `true`로 변경하세요.

### 4. 설치 확인

WANDB 패키지가 설치되어 있는지 확인:

```bash
python -c "import wandb; print(wandb.__version__)"
# 출력: 0.23.0 (또는 최신 버전)
```

설치가 필요한 경우:
```bash
pip install wandb
```

---

## 사용 방법

### 기본 워크플로우

#### Step 1: 환경 설정

```bash
cd /path/to/Pai-Megatron-Patch/examples/alpha

# WANDB 환경 변수 설정
source ./scripts/setup_wandb.sh
```

#### Step 2: YAML 설정 활성화

`configs/training/pretrain.yaml` 수정:

```yaml
wandb:
  enabled: true  # false → true 로 변경
```

#### Step 3: 학습 실행

```bash
bash train.sh \
  baseline_48L \
  h100x8 \
  pretrain \
  kormo_1pct \
  env \
  /path/to/outputs
```

WANDB가 활성화되면 자동으로:
- 실험 이름 생성: `{model}_{infra}_{data}_{timestamp}`
- 프로젝트에 연결: `alpha-pretraining`
- 메트릭 실시간 로깅

#### Step 4: 대시보드 확인

학습 중 브라우저에서 실시간 모니터링:

```
https://wandb.ai/kide004/alpha-pretraining
```

### 고급 설정

#### 커스텀 프로젝트/실험 이름

`configs/training/pretrain.yaml`:

```yaml
wandb:
  enabled: true
  project: "alpha-ablation-study"     # 프로젝트 변경
  entity: "my-team"                   # 팀 계정 사용
  notes: "Experiment: MoE expert count ablation"
```

#### 오프라인 모드

네트워크 없이 로컬에 로그 저장:

```bash
export WANDB_MODE="offline"
```

나중에 동기화:
```bash
wandb sync /path/to/wandb/offline-run-*
```

#### 선택적 WANDB 비활성화

특정 실행에서만 WANDB 비활성화:

```bash
export WANDB_DISABLED=true
bash train.sh ...
```

---

## 테스트

### 통합 테스트 실행

WANDB 통합이 올바르게 작동하는지 확인:

```bash
bash ./scripts/test_wandb.sh
```

**테스트 항목**:
1. WANDB 환경 변수 설정 확인
2. YAML 설정 파싱 확인
3. WANDB Python API 로그인 테스트
4. WANDB 초기화 및 로깅 테스트

**예상 출력**:
```
==========================================
WANDB Integration Test
==========================================

1️⃣  Setting up WANDB environment...
✅ WANDB 환경 변수 설정 완료

2️⃣  Loading YAML parser...

3️⃣  Testing WANDB configuration parsing...
  - enabled: True
  - project: alpha-pretraining

4️⃣  Testing WANDB Python API...
  ✅ WANDB_API_KEY is set
  ✅ WANDB login successful
  ✅ WANDB init successful
  ✅ WANDB log successful
  ✅ WANDB finish successful

🎉 All WANDB tests passed!
```

### 실제 학습 테스트

짧은 학습으로 WANDB 로깅 확인:

```bash
# 1. WANDB 활성화
source ./scripts/setup_wandb.sh

# 2. pretrain.yaml에서 enabled: true 설정

# 3. 짧은 학습 실행 (train_tokens을 작게 설정)
# configs/training/pretrain.yaml:
#   train_tokens: 1000000  # ~1M tokens만 학습

bash train.sh baseline_48L h100x8 pretrain kormo_1pct env /tmp/test_wandb

# 4. WANDB 대시보드 확인
# https://wandb.ai/kide004/alpha-pretraining
```

---

## 대시보드 활용

### 로깅되는 메트릭

Megatron-LM이 자동으로 로깅하는 주요 메트릭:

#### 학습 메트릭
- `train/loss`: 학습 손실
- `train/learning_rate`: 현재 학습률
- `train/grad_norm`: 그래디언트 노름
- `train/loss-scale`: 손실 스케일 (mixed precision)

#### 성능 메트릭
- `train/throughput`: 처리량 (tokens/sec)
- `train/samples_per_sec`: 초당 샘플 수
- `train/elapsed_time`: 경과 시간

#### 검증 메트릭
- `validation/loss`: 검증 손실
- `validation/ppl`: Perplexity

#### 시스템 메트릭
- `system/gpu_memory_allocated`: GPU 메모리 사용량
- `system/gpu_utilization`: GPU 활용률

### 대시보드 기능

#### 1. Workspace 탐색

프로젝트 페이지: https://wandb.ai/kide004/alpha-pretraining

- **Runs**: 모든 실험 목록
- **Charts**: 커스텀 차트 생성
- **Reports**: 실험 보고서 작성
- **Sweeps**: 하이퍼파라미터 최적화

#### 2. 실험 비교

여러 실험을 선택하여 메트릭 비교:

1. Runs 탭에서 비교할 실험 체크
2. "Compare" 버튼 클릭
3. 메트릭 차트 자동 생성

#### 3. 커스텀 차트

특정 메트릭 시각화:

1. "Add chart" 클릭
2. 메트릭 선택 (예: `train/loss`, `validation/ppl`)
3. 차트 타입 선택 (Line, Scatter, Bar 등)

#### 4. 필터링 및 검색

실험 필터링:

```
# 특정 설정 검색
config.model_size = "24L"

# 성능 기준 필터
metrics.train/loss < 2.0

# 날짜 범위
created_at > "2025-01-17"
```

---

## 문제 해결

### 일반적인 문제

#### 1. WANDB_API_KEY not set

**증상**:
```
⚠️  경고: WANDB_API_KEY가 설정되지 않았습니다.
```

**해결**:
```bash
source ./scripts/setup_wandb.sh
```

또는 키 파일이 없는 경우 생성:
```bash
printf '%s' '<YOUR_WANDB_API_KEY>' > scripts/.wandb_key && chmod 600 scripts/.wandb_key
```

#### 2. WANDB not logging

**증상**: 학습 중 WANDB에 메트릭이 나타나지 않음

**확인 사항**:
1. `pretrain.yaml`에서 `enabled: true` 확인
2. 환경 변수 설정 확인: `echo $WANDB_API_KEY`
3. 학습 로그에서 WANDB 관련 메시지 확인

**해결**:
```bash
# 1. WANDB 모드 확인
echo $WANDB_MODE  # "online" 이어야 함

# 2. 재로그인
wandb login --relogin

# 3. 테스트 실행
bash ./scripts/test_wandb.sh
```

#### 3. Permission denied

**증상**:
```
wandb: ERROR Unable to create wandb directory
```

**해결**:
```bash
# WANDB 디렉토리 수동 생성
mkdir -p ./wandb
chmod 755 ./wandb

# 또는 커스텀 경로 지정
export WANDB_DIR="/tmp/wandb"
```

#### 4. Network errors

**증상**:
```
wandb: ERROR Network error
```

**해결**:
```bash
# 오프라인 모드로 전환
export WANDB_MODE="offline"

# 나중에 동기화
wandb sync ./wandb/offline-run-*
```

### 디버깅

#### WANDB 로그 레벨 증가

```bash
export WANDB_DEBUG=true
```

#### Megatron 로그 확인

학습 로그에서 WANDB 관련 메시지 확인:

```bash
tail -f /path/to/outputs/logs/train_*.log | grep -i wandb
```

#### 수동 WANDB 테스트

Python에서 직접 테스트:

```python
import wandb
import os

# 로그인
wandb.login(key=os.environ['WANDB_API_KEY'])

# 테스트 런
run = wandb.init(
    project="test-project",
    name="test-run",
    config={"test": True}
)

wandb.log({"metric": 42})
wandb.finish()

print("✅ WANDB test successful!")
```

---

## 참고 자료

### 공식 문서

- **WANDB 공식 문서**: https://docs.wandb.ai/
- **Megatron-LM WANDB 통합**: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py (WANDB 관련 인자 참조)
- **WANDB Python API**: https://docs.wandb.ai/ref/python

### Alpha 프로젝트 관련 파일

- **YAML 설정**: [configs/training/pretrain.yaml](../configs/training/pretrain.yaml)
- **환경 설정 스크립트**: [scripts/setup_wandb.sh](../scripts/setup_wandb.sh)
- **테스트 스크립트**: [scripts/test_wandb.sh](../scripts/test_wandb.sh)
- **학습 스크립트**: [train.sh](../train.sh)

### 유용한 명령어

```bash
# WANDB 상태 확인
wandb status

# 로그인
wandb login

# 오프라인 런 동기화
wandb sync ./wandb/offline-run-*

# 프로젝트 목록
wandb projects

# 런 목록
wandb runs alpha-pretraining
```

---

## 요약

Alpha 프로젝트의 WANDB 통합은 다음과 같이 구성됩니다:

1. **설정**: `configs/training/pretrain.yaml`에서 `wandb.enabled: true`
2. **환경**: `source ./scripts/setup_wandb.sh`로 API 키 설정
3. **실행**: `train.sh` 실행 시 자동으로 WANDB 활성화
4. **모니터링**: https://wandb.ai/kide004/alpha-pretraining

**핵심 장점**:
- ✅ 설정 파일 기반 (코드 수정 불필요)
- ✅ TensorBoard와 병행 사용 가능
- ✅ 실시간 메트릭 추적 및 비교
- ✅ 팀 협업 및 실험 재현성

**시작 방법**:
```bash
# 1. 환경 설정
source ./scripts/setup_wandb.sh

# 2. WANDB 활성화
vim configs/training/pretrain.yaml  # enabled: false → true

# 3. 학습 실행
bash train.sh baseline_48L h100x8 pretrain kormo_1pct env ./outputs

# 4. 대시보드 확인
# https://wandb.ai/kide004/alpha-pretraining
```

즐거운 실험 되세요! 🚀
