# Alpha 모델 평가 가이드

LM-Evaluation-Harness를 사용한 벤치마크 평가 완전 가이드

---

## 목차

1. [사전 준비](#사전-준비)
2. [빠른 시작](#빠른-시작)
3. [벤치마크 태스크](#벤치마크-태스크)
4. [고급 사용법](#고급-사용법)
5. [트러블슈팅](#트러블슈팅)
6. [결과 분석](#결과-분석)

---

## 사전 준비

### 1. HuggingFace 토큰 설정

벤치마크 데이터셋을 다운로드하려면 HuggingFace 토큰이 필요합니다.

#### 토큰 생성
1. https://huggingface.co/settings/tokens 방문
2. "New token" 클릭
3. Token 타입: **Read** 권한 선택
4. 생성된 토큰 복사 (`hf_...`)

#### 토큰 설정 방법

**방법 1: 환경 변수 (권장)**
```bash
export HF_TOKEN="hf_your_token_here"
```

**방법 2: CLI 로그인 (영구 저장)**
```bash
huggingface-cli login
# 토큰 입력 후 엔터
```

**방법 3: 스크립트에 직접 설정 (비권장)**
```bash
# scripts/run_benchmarks.sh 편집
export HF_TOKEN="hf_your_token_here"
```

> ⚠️ **보안 주의**: 토큰을 Git에 커밋하지 마세요!

### 2. HuggingFace 모델 준비

벤치마크는 **HuggingFace 포맷 모델**이 필요합니다.

**Megatron 체크포인트를 변환한 경우**:
```bash
ls outputs/alpha_baseline_48L_*/hfmodel
# config.json, model-*.safetensors, tokenizer.json 등이 있어야 함
```

**변환이 필요한 경우**: [CONVERSION.md](CONVERSION.md) 참고

---

## 빠른 시작

### 기본 실행

```bash
cd /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha

# 기본 벤치마크 (hellaswag)
bash scripts/run_benchmarks.sh outputs/alpha_baseline_48L_*/hfmodel

# 표준 벤치마크 세트
bash scripts/run_benchmarks.sh \
  outputs/alpha_baseline_48L_*/hfmodel \
  "mmlu,hellaswag,arc_easy,arc_challenge,winogrande"
```

### 출력 예시

```
================================================================
Running benchmarks with LM-Evaluation-Harness
Model: outputs/alpha_baseline_48L_20251117_234326/hfmodel
Tasks: mmlu,hellaswag,arc_easy
Batch Size: auto
Device: cuda:0
================================================================

Loading model...
Evaluating...
|████████████████████| 100%

Results:
{
  "results": {
    "mmlu": {"acc": 0.45},
    "hellaswag": {"acc_norm": 0.62},
    "arc_easy": {"acc": 0.72}
  }
}
```

---

## 벤치마크 태스크

### 영어 벤치마크

#### 1. **MMLU** (Massive Multitask Language Understanding)
- **설명**: 57개 과목의 대학 수준 지식 평가
- **메트릭**: Accuracy
- **예상 시간**: ~30분 (8 GPU)
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "mmlu"
  ```

#### 2. **HellaSwag**
- **설명**: 상식 추론 (이야기 완성)
- **메트릭**: Normalized Accuracy
- **예상 시간**: ~10분
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "hellaswag"
  ```

#### 3. **ARC** (AI2 Reasoning Challenge)
- **설명**: 초등/중등 과학 질문
- **변형**:
  - `arc_easy`: 쉬운 질문
  - `arc_challenge`: 어려운 질문
- **메트릭**: Accuracy
- **예상 시간**: ~5분 (각각)
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "arc_easy,arc_challenge"
  ```

#### 4. **Winogrande**
- **설명**: 문장 이해 및 상식 추론 (대명사 해결)
- **메트릭**: Accuracy
- **예상 시간**: ~5분
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "winogrande"
  ```

#### 5. **BoolQ**
- **설명**: Yes/No 질문 답변
- **메트릭**: Accuracy
- **예상 시간**: ~5분
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "boolq"
  ```

#### 6. **PIQA** (Physical Interaction QA)
- **설명**: 물리적 상식 추론
- **메트릭**: Accuracy
- **예상 시간**: ~3분
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "piqa"
  ```

#### 7. **Social IQA**
- **설명**: 사회적 상황 이해
- **메트릭**: Accuracy
- **예상 시간**: ~5분
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "social_iqa"
  ```

#### 8. **OpenBookQA**
- **설명**: 과학 지식 추론 (교과서 기반)
- **메트릭**: Accuracy
- **예상 시간**: ~3분
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "openbookqa"
  ```

#### 9. **GSM8K**
- **설명**: 초등학교 수학 문제 (8000문항)
- **메트릭**: Exact Match
- **예상 시간**: ~15분
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "gsm8k"
  ```

### 한국어 벤치마크

#### 1. **KMMLU** (Korean MMLU)
- **설명**: 한국형 MMLU (한국사, 한국어 등 포함)
- **메트릭**: Accuracy
- **예상 시간**: ~30분
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "kmmlu"
  ```

### Gated Datasets (접근 권한 필요)

일부 데이터셋은 별도 접근 승인이 필요합니다:

#### **GPQA** (Graduate-Level Google-Proof Q&A)
- **설명**: 대학원 수준 과학 질문
- **접근**: https://huggingface.co/datasets/Idavidrein/gpqa
- **절차**:
  1. 위 링크 방문
  2. "Request Access" 클릭
  3. 승인 대기 (수 시간 ~ 수 일)
  4. 승인 후 사용 가능
- **사용법**:
  ```bash
  bash scripts/run_benchmarks.sh MODEL_PATH "gpqa_main_zeroshot"
  ```

---

## 고급 사용법

### 1. 배치 크기 조정

**자동 배치 크기 (기본)**:
```bash
bash scripts/run_benchmarks.sh MODEL_PATH TASKS auto
```

**수동 지정**:
```bash
# 메모리 부족 시 1로 설정
bash scripts/run_benchmarks.sh MODEL_PATH TASKS 1

# 고성능 GPU는 더 큰 값 사용
bash scripts/run_benchmarks.sh MODEL_PATH TASKS 8
```

### 2. GPU 개수 조정

`scripts/run_benchmarks.sh` 편집:

```bash
# 기본 (8 GPU)
accelerate launch --multi_gpu --num_processes=8 -m lm_eval ...

# 4 GPU로 변경
accelerate launch --multi_gpu --num_processes=4 -m lm_eval ...

# 단일 GPU
python -m lm_eval ...
```

### 3. 커스텀 태스크 조합

**표준 세트**:
```bash
TASKS="mmlu,hellaswag,arc_easy,arc_challenge,winogrande,boolq,piqa"
bash scripts/run_benchmarks.sh MODEL_PATH "$TASKS"
```

**수학 집중**:
```bash
TASKS="gsm8k,mathqa"
bash scripts/run_benchmarks.sh MODEL_PATH "$TASKS"
```

**한국어 + 영어 혼합**:
```bash
TASKS="kmmlu,mmlu,hellaswag"
bash scripts/run_benchmarks.sh MODEL_PATH "$TASKS"
```

### 4. lm_eval 직접 사용

더 세밀한 제어가 필요한 경우:

```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=MODEL_PATH,trust_remote_code=True,dtype=bfloat16 \
    --tasks mmlu \
    --batch_size auto \
    --output_path results/ \
    --log_samples
```

**추가 옵션**:
- `--num_fewshot 5`: Few-shot 예제 개수
- `--limit 100`: 샘플 개수 제한 (테스트용)
- `--output_path PATH`: 결과 저장 경로
- `--log_samples`: 개별 샘플 결과 저장
- `--device cuda:0`: 특정 GPU 지정

---

## 트러블슈팅

### 1. Rate Limit 에러

**에러**:
```
HfHubHTTPError: 429 Client Error: Too Many Requests
```

**해결**:
- HuggingFace 토큰 설정 확인
- 토큰이 유효한지 확인: https://huggingface.co/settings/tokens

### 2. Gated Dataset 에러

**에러**:
```
DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated dataset
```

**해결**:
- 해당 데이터셋 페이지에서 접근 권한 요청
- 또는 해당 태스크를 제외하고 실행

### 3. Out of Memory (OOM)

**에러**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**해결**:
```bash
# 배치 크기 감소
bash scripts/run_benchmarks.sh MODEL_PATH TASKS 1

# 또는 GPU 개수 증가 (run_benchmarks.sh 편집)
```

### 4. 모델 로딩 에러

**에러**:
```
ValueError: Unrecognized model in ...
```

**원인**: 잘못된 모델 경로 (Megatron 체크포인트 경로 사용)

**해결**:
```bash
# 올바른 경로: hfmodel 디렉토리
bash scripts/run_benchmarks.sh outputs/alpha_*/hfmodel TASKS

# 잘못된 경로: checkpoints 디렉토리 (X)
# bash scripts/run_benchmarks.sh outputs/alpha_*/checkpoints TASKS
```

### 5. 토큰 인식 안 됨

**에러**:
```
Invalid username or password
```

**해결**:
```bash
# 환경 변수 확인
echo $HF_TOKEN

# 토큰 재설정
export HF_TOKEN="hf_your_token_here"

# 또는 로그인
huggingface-cli login
```

### 6. 데이터셋 다운로드 실패

**해결**:
```bash
# 캐시 디렉토리 권한 확인
ls -ld ~/.cache/huggingface/datasets/
chmod -R u+w ~/.cache/huggingface/

# 또는 캐시 위치 변경
export HF_DATASETS_CACHE=/path/to/writable/directory
```

---

## 결과 분석

### 결과 형식

LM-Eval은 JSON 형식으로 결과를 출력합니다:

```json
{
  "results": {
    "mmlu": {
      "acc": 0.4523,
      "acc_stderr": 0.0089,
      "acc_norm": 0.4501,
      "acc_norm_stderr": 0.0087
    },
    "hellaswag": {
      "acc": 0.5234,
      "acc_norm": 0.6187,
      "acc_norm_stderr": 0.0048
    }
  },
  "config": {
    "model": "hf",
    "batch_size": "auto"
  }
}
```

### 주요 메트릭

- **acc** (Accuracy): 정확도
- **acc_norm** (Normalized Accuracy): 정규화된 정확도 (길이 보정)
- **acc_stderr**: 표준 오차
- **exact_match**: 정확히 일치 (생성 태스크)

### 참고 점수

**Qwen2.5 시리즈 (공식)**:

| 모델 | MMLU | HellaSwag | ARC-C | Winogrande |
|------|------|-----------|-------|------------|
| Qwen2.5-0.5B | 45.4 | 48.9 | 34.2 | 55.0 |
| Qwen2.5-1.5B | 56.5 | 58.3 | 44.4 | 60.3 |
| Qwen2.5-3B | 63.6 | 64.5 | 53.7 | 65.1 |
| Qwen2.5-7B | 70.3 | 78.5 | 84.8 | 75.4 |
| Qwen2.5-14B | 79.9 | 87.0 | 92.9 | 82.7 |

**Alpha 모델 기대 범위** (baseline_48L, 초기 학습):
- MMLU: 30-50% (학습 진행도에 따라)
- HellaSwag: 40-60%
- ARC-Easy: 50-70%
- Winogrande: 50-65%

> **Note**: Alpha는 실험적 모델로, 학습 토큰 수와 데이터 품질에 따라 성능이 크게 달라질 수 있습니다.

### 결과 저장

**출력 파일 저장**:
```bash
bash scripts/run_benchmarks.sh MODEL_PATH TASKS auto > results_$(date +%Y%m%d).txt
```

**JSON 형식 저장** (lm_eval 직접 사용):
```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=MODEL_PATH,trust_remote_code=True \
    --tasks mmlu,hellaswag \
    --output_path results/
```

결과는 `results/results.json`에 저장됩니다.

---

## 데이터셋 위치

LM-Eval은 HuggingFace Hub에서 자동으로 데이터셋을 다운로드합니다.

**기본 캐시 위치**:
```
~/.cache/huggingface/datasets/
```

**커스텀 캐시 위치**:
```bash
export HF_DATASETS_CACHE="/data/benchmark_datasets"
bash scripts/run_benchmarks.sh ...
```

**캐시 확인**:
```bash
ls -lh ~/.cache/huggingface/datasets/
```

**디스크 사용량**:
- MMLU: ~166 MB
- HellaSwag: ~44 MB
- 전체 (주요 10개): ~1-2 GB

---

## 참고 자료

- **LM-Evaluation-Harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **벤치마크 목록**: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
- **Qwen2.5 평가 결과**: https://qwenlm.github.io/blog/qwen2.5/
- **KMMLU 논문**: https://arxiv.org/abs/2402.11548

---

## FAQ

### Q: 모든 벤치마크를 한 번에 실행할 수 있나요?

A: 가능하지만 시간이 오래 걸립니다 (~2-3시간). 주요 벤치마크만 선택 권장:
```bash
bash scripts/run_benchmarks.sh MODEL_PATH \
  "mmlu,hellaswag,arc_easy,arc_challenge,winogrande"
```

### Q: 한국어 성능만 평가하고 싶어요.

A: KMMLU를 사용하세요:
```bash
bash scripts/run_benchmarks.sh MODEL_PATH "kmmlu"
```

### Q: 수학 능력을 평가하고 싶어요.

A: GSM8K를 사용하세요:
```bash
bash scripts/run_benchmarks.sh MODEL_PATH "gsm8k"
```

### Q: 결과가 너무 낮게 나와요.

A: 정상입니다. Alpha는 초기 학습 모델로:
1. 학습 토큰 수가 부족할 수 있음
2. 데이터 품질 영향
3. 모델 크기 영향 (24L은 작은 모델)

학습을 더 진행하거나 데이터를 개선하면 성능이 향상됩니다.

### Q: Megatron 체크포인트로 직접 평가할 수 있나요?

A: 아니요. 반드시 HuggingFace 포맷으로 변환 필요. [CONVERSION.md](CONVERSION.md) 참고.

---

**업데이트**: 2025-11-20
