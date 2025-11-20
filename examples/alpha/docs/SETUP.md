# Alpha 프로젝트 환경 세팅 가이드

Alpha 프로젝트 학습 환경 구축을 위한 상세 가이드

---

## 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [소프트웨어 설치](#소프트웨어-설치)
3. [데이터 준비](#데이터-준비)
4. [환경 검증](#환경-검증)
5. [학습 실행](#학습-실행)
6. [트러블슈팅](#트러블슈팅)

---

## 시스템 요구사항

### 하드웨어

**최소 요구사항**:
- GPU: 8× NVIDIA H100 (80GB) 또는 동급
- CPU: 32+ cores
- RAM: 512GB+
- Storage: 1TB+ (NVMe SSD 권장)

**권장 사양**:
- GPU: 8× H100 SXM5 (NVLink)
- CPU: 220+ cores (AMD EPYC or Intel Xeon)
- RAM: 1TB+
- Storage: 2TB+ NVMe SSD (데이터셋 + 체크포인트)

### 소프트웨어

**필수**:
- OS: Ubuntu 22.04 LTS or later
- CUDA: 12.1+
- Python: 3.10+
- PyTorch: 2.3+
- Transformer Engine: 1.0+
- Flash Attention: 3.0+

---

## 소프트웨어 설치

### 1. CUDA Toolkit 설치

```bash
# CUDA 12.1 설치
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 설치 확인
nvcc --version
nvidia-smi
```

### 2. Python 환경 구성

```bash
# Conda 환경 생성
conda create -n alpha python=3.10 -y
conda activate alpha

# 또는 venv 사용
python3.10 -m venv ~/envs/alpha
source ~/envs/alpha/bin/activate
```

### 3. PyTorch 설치

```bash
# PyTorch 2.3 with CUDA 12.1
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 설치 확인
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

### 4. Transformer Engine 설치

```bash
# Transformer Engine (from source for latest)
git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git checkout v1.0
git submodule update --init --recursive

# Build and install
pip install .

# 설치 확인
python -c "import transformer_engine.pytorch as te; print('TE installed')"
```

### 5. Flash Attention 3 설치

```bash
# Flash Attention 3 (H100 optimized)
pip install flash-attn-3 --no-build-isolation

# 또는 source에서 빌드
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v3.0.0
python setup.py install

# 설치 확인
python -c "from flash_attn_3 import flash_attn_func; print('FA3 installed')"
```

### 6. 기타 의존성

```bash
# 필수 패키지
pip install \
    pyyaml \
    numpy \
    regex \
    einops \
    packaging \
    tensorboard \
    wandb \
    huggingface-hub \
    sentencepiece \
    tiktoken

# Megatron 의존성
pip install \
    apex \
    h5py \
    lm-dataformat
```

### 7. Pai-Megatron-Patch 설치

```bash
# 저장소 클론
cd ~/repos/
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch

# Submodule 업데이트
git submodule update --init --recursive

# PYTHONPATH 설정
echo "export PYTHONPATH=~/repos/Pai-Megatron-Patch:~/repos/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250908:\$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
```

---

## 데이터 준비

### 1. 토크나이저 다운로드

```bash
cd ~/repos/Pai-Megatron-Patch/models/

# Qwen3-Next 토크나이저 다운로드 (Hugging Face)
pip install modelscope
modelscope download \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --include "tokenizer*" "vocab*" \
    --local_dir Qwen3-Next-tokenizer
```

### 2. 원시 데이터 준비

```bash
# 예시: KORMo 데이터셋
# JSONL 형식:
# {"text": "문장 1"}
# {"text": "문장 2"}

# 데이터셋 위치
DATA_DIR=/home/work/Datasets/KORMo_raw/
mkdir -p $DATA_DIR

# 데이터 다운로드 또는 복사
# (데이터셋 소스에 따라 다름)
```

### 3. 데이터 전처리

#### 방법 1: 작은 데이터셋 (< 100GB)

```bash
cd ~/repos/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing/

python preprocess_data.py \
    --input /path/to/data.jsonl \
    --output-prefix /path/to/output/dataset \
    --dataset-impl mmap \
    --patch-tokenizer-type Qwen3Tokenizer \
    --load ~/repos/Pai-Megatron-Patch/models/Qwen3-Next-tokenizer \
    --workers 32 \
    --append-eod \
    --extra-vocab-size 0
```

#### 방법 2: 대용량 데이터셋 (> 1TB)

```bash
cd ~/repos/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing/

# Step 1: Arrow → JSONL (if needed)
python convert_arrow_to_jsonl_v2.py \
    --input-dir /data/arrow_dataset/ \
    --output-file /data/dataset.jsonl \
    --text-column text \
    --workers 224

# Step 2: Random sampling (optional, for testing)
python streaming_random_sample.py \
    --input /data/dataset.jsonl \
    --output /data/dataset_1pct.jsonl \
    --sample-rate 0.01 \
    --seed 42 \
    --method reservoir

# Step 3: Binary preprocessing
python preprocess_data.py \
    --input /data/dataset_1pct.jsonl \
    --output-prefix /data/mmap/dataset_1pct \
    --dataset-impl mmap \
    --patch-tokenizer-type Qwen3Tokenizer \
    --load ~/repos/Pai-Megatron-Patch/models/Qwen3-Next-tokenizer \
    --workers 64 \
    --append-eod
```

#### KORMo 1% subset (Alpha baseline)

```bash
cd ~/repos/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing/

# 간편 스크립트 사용
bash preprocess_kormo_subset.sh 1  # 1% subset

# 출력:
# /home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/kormo_content_document.bin
# /home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/kormo_content_document.idx
```

### 4. 데이터 검증

```bash
# 데이터셋 메타정보 확인
ls -lh /home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/

# 첫 몇 샘플 확인
cd ~/repos/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing/
python test_dataset.py \
    --data-path /home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/kormo_content_document \
    --num-samples 10
```

---

## 환경 검증

### 1. 자동 검증 스크립트

```bash
cd ~/repos/Pai-Megatron-Patch/examples/alpha/
bash scripts/validate_environment.sh
```

**출력 예시**:
```
==========================================
Alpha 프로젝트 환경 검증
==========================================

[1] 소프트웨어 환경
  ✓ Flash Attention 3: 사용 가능
  ✓ Transformer Engine PyTorch: 사용 가능
  ✓ Megatron-Core (250908): 사용 가능
  ✓ GPU: 8개 사용 가능
    - GPU 0: NVIDIA H100 80GB HBM3
    ...
  ✓ PyYAML: 사용 가능

[2] 설정 파일
  ✓ model/baseline_24L.yaml
  ✓ training/pretrain.yaml
  ✓ training/h100x8.yaml
  ✓ data/kormo_1pct.yaml
  ✓ env.yaml

[3] 데이터셋
  ✓ 데이터셋: /home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/kormo_content_document
    크기: 1.2G

[4] 토크나이저
  ✓ 토크나이저: /home/work/.../Qwen3-Next-tokenizer
    설정 파일 확인됨

[5] 스크립트 실행 권한
  ✓ train.sh: 실행 가능
  ✓ validate_environment.sh: 실행 가능

[6] 디스크 공간
  사용 가능 공간: 1.5T
  ✓ 출력 디렉토리: /home/work/.../examples/alpha/outputs

==========================================
검증 완료
==========================================
```

### 2. 수동 검증

#### GPU 확인
```bash
nvidia-smi

# 출력 확인:
# - GPU 수: 8
# - GPU 모델: H100
# - 메모리: 80GB each
# - CUDA Version: 12.1+
```

#### PyTorch & CUDA
```bash
python << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
```

#### Flash Attention
```bash
python << EOF
from flash_attn_3.flash_attn_interface import flash_attn_func
print("Flash Attention 3: OK")
EOF
```

#### Megatron-Core
```bash
python << EOF
import sys
sys.path.insert(0, '/home/work/repos/Pai-Megatron-Patch')
sys.path.insert(0, '/home/work/repos/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250908')

from megatron.core import parallel_state
from megatron.core.transformer.moe import moe_layer
print("Megatron-Core: OK")
EOF
```

---

## 학습 실행

### 1. 첫 학습 (Baseline)

```bash
cd ~/repos/Pai-Megatron-Patch/examples/alpha/

# 기본 설정으로 학습
bash train.sh

# 또는 명시적으로
bash train.sh baseline_24L pretrain h100x8 kormo_1pct
```

### 2. 학습 모니터링

#### TensorBoard 실행 (별도 터미널)
```bash
cd ~/repos/Pai-Megatron-Patch/examples/alpha/

# 최신 실험 찾기
LATEST_RUN=$(ls -td outputs/alpha_* | head -1)

# TensorBoard 실행
tensorboard --logdir ${LATEST_RUN}/tensorboard --port 6006 --bind_all
```

브라우저에서 `http://[서버IP]:6006` 접속

#### 로그 확인
```bash
# 실시간 로그
tail -f outputs/alpha_*/logs/train.log

# Loss 추출
grep "loss:" outputs/alpha_*/logs/train.log

# Throughput 확인
grep "throughput" outputs/alpha_*/logs/train.log
```

#### GPU 모니터링
```bash
# 실시간 GPU 상태
watch -n 1 nvidia-smi

# GPU utilization 로그
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > gpu_monitor.log
```

### 3. 학습 재개 (체크포인트에서)

```bash
# train.sh에 체크포인트 경로 추가 필요
# (현재는 YAML에서 설정 불가, 수동 수정 필요)

# 임시 방법: training.yaml 수정
# training:
#   load_checkpoint: "/path/to/checkpoint"

# 또는 run_mcore_qwen3.sh 직접 사용
```

### 4. 다중 노드 학습 (DLC)

```bash
# 환경 변수 설정
export WORLD_SIZE=4  # 노드 수
export RANK=0        # 현재 노드 순위 (0, 1, 2, 3)
export MASTER_ADDR=node0.hostname
export MASTER_PORT=6000

# 각 노드에서 실행
bash train.sh
```

---

## 설정 커스터마이징

### 1. 모델 아키텍처 변경

새 모델 설정 파일 생성:

```bash
cd ~/repos/Pai-Megatron-Patch/examples/alpha/configs/model/

# baseline_24L.yaml 복사
cp baseline_24L.yaml my_model.yaml

# 편집
vim my_model.yaml

# 변경 예시:
# - num_layers: 48 (더 깊은 모델)
# - num_experts: 128 (더 적은 experts)
```

사용:
```bash
bash train.sh my_model pretrain h100x8 kormo_1pct
```

### 2. 하이퍼파라미터 튜닝

```bash
cd configs/training/

# pretrain.yaml 복사
cp pretrain.yaml my_training.yaml

# 편집
vim my_training.yaml

# 변경 예시:
# - lr: 5.0e-4 (더 높은 learning rate)
# - weight_decay: 0.1 (더 강한 regularization)
```

### 3. 인프라 최적화

```bash
cd configs/training/

# h100x8.yaml 복사
cp h100x8.yaml my_infra.yaml

# 편집
vim my_infra.yaml

# 변경 예시:
# - pipeline_parallel: 2 (메모리 부족 시)
# - micro_batch_size: 4 (throughput 증가)
```

---

## 트러블슈팅

### OOM (Out of Memory)

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결책**:

1. **Micro Batch Size 감소**:
   ```yaml
   # configs/training/h100x8.yaml
   infrastructure:
     batch:
       micro_batch_size: 1  # 2 → 1
   ```

2. **Pipeline Parallel 증가**:
   ```yaml
   # configs/training/h100x8.yaml
   infrastructure:
     parallelism:
       pipeline_parallel: 2  # 1 → 2 (메모리 1/2)
   ```

3. **Activation Checkpointing 강화**:
   ```yaml
   # Full checkpointing (모든 레이어)
   infrastructure:
     activation_checkpointing:
       granularity: full  # selective → full
   ```

4. **Sequence Length 감소** (임시):
   ```yaml
   # configs/training/h100x8.yaml
   infrastructure:
     batch:
       seq_length: 2048  # 4096 → 2048
   ```

### NCCL Timeout

**증상**:
```
Watchdog caught collective operation timeout
```

**해결책**:

1. **Timeout 연장**:
   ```yaml
   # configs/training/pretrain.yaml
   training:
     distributed_timeout_minutes: 120  # 60 → 120
   ```

2. **NCCL 설정 조정**:
   ```yaml
   # configs/env.yaml
   environment:
     nccl:
       debug: "INFO"
       timeout_ms: 600000  # 10 minutes
   ```

3. **버그 수정 확인**:
   ```bash
   bash scripts/verify_fixes.sh
   ```

### Slow Training

**증상**: Tokens/sec이 예상보다 낮음

**진단**:

1. **GPU Utilization 확인**:
   ```bash
   nvidia-smi dmon -s u -d 1
   # GPU utilization이 90% 미만이면 병목 존재
   ```

2. **DataLoader 병목 확인**:
   ```yaml
   # configs/data/kormo_1pct.yaml
   data:
     num_workers: 64  # 32 → 64 (CPU 코어 수에 맞춤)
   ```

3. **Micro Batch Size 증가**:
   ```yaml
   # configs/training/h100x8.yaml (메모리 여유 있을 때)
   infrastructure:
     batch:
       micro_batch_size: 4  # 2 → 4
   ```

### Loss not Converging

**증상**: Loss가 감소하지 않음 또는 NaN

**해결책**:

1. **Learning Rate 감소**:
   ```yaml
   # configs/training/pretrain.yaml
   training:
     lr: 1.0e-4  # 3.0e-4 → 1.0e-4
   ```

2. **Gradient Clipping 강화**:
   ```yaml
   training:
     clip_grad: 0.5  # 1.0 → 0.5
   ```

3. **Weight Initialization 확인**:
   ```yaml
   training:
     init_method_std: 0.01  # 0.006 → 0.01
   ```

4. **Mixed Precision 안정성**:
   ```bash
   # Loss scaling 확인 (TensorBoard)
   # Loss spike 시 학습 재시작
   ```

### 데이터 로딩 오류

**증상**:
```
FileNotFoundError: [Errno 2] No such file or directory: '...bin'
```

**해결책**:

1. **파일 존재 확인**:
   ```bash
   ls -lh /home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/
   # .bin과 .idx 모두 있어야 함
   ```

2. **경로 확인**:
   ```yaml
   # configs/data/kormo_1pct.yaml
   data:
     train_path: "/correct/path/to/dataset"  # .bin/.idx 제외
   ```

3. **재전처리**:
   ```bash
   cd toolkits/pretrain_data_preprocessing/
   bash preprocess_kormo_subset.sh 1
   ```

---

## 성능 벤치마크

### 예상 성능 (H100 8-GPU)

| 항목 | Baseline 24L | 비고 |
|------|-------------|------|
| **Throughput** | ~XXX tokens/sec | MBS=2, GBS=256 |
| **GPU Memory** | ~XX GB/GPU | Peak usage |
| **Time/Iteration** | ~X.X sec | 평균 |
| **TFLOPs/GPU** | ~XXX | Model FLOPs utilization |

### 벤치마크 실행

```bash
# 100 iterations 실행 후 측정
cd examples/alpha/
bash train.sh

# 로그에서 throughput 추출
grep "throughput" outputs/alpha_*/logs/train.log | tail -20

# TensorBoard에서 확인
# - scalars/throughput_samples_per_sec
# - scalars/throughput_tokens_per_sec
```

---

## 참고 자료

- **Pai-Megatron-Patch 메인 문서**: [../../README.md](../../README.md)
- **CLAUDE.md**: [../../CLAUDE.md](../../CLAUDE.md)
- **Alpha Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Experiments Log**: [EXPERIMENTS.md](EXPERIMENTS.md)

---

## 체크리스트

학습 시작 전:

- [ ] GPU 8개 확인 (nvidia-smi)
- [ ] Flash Attention 3 설치 확인
- [ ] Megatron-Core 설치 확인
- [ ] 데이터셋 전처리 완료 (.bin, .idx)
- [ ] 토크나이저 다운로드
- [ ] YAML 설정 파일 5개 확인
- [ ] 디스크 공간 충분 (1TB+)
- [ ] 환경 검증 스크립트 실행
- [ ] TensorBoard 접근 가능 확인

---

**마지막 업데이트**: 2025-01-17
