# Training Sample Extraction Tool

이 도구는 특정 training iteration에서 사용된 샘플들을 추출하고 분석합니다. Grad norm spike 등의 이상 현상을 디버깅할 때 유용합니다.

## 사용법

### 기본 사용

```bash
cd toolkits/data_extraction

# 기본 추출 (iteration 324065)
python extract_training_samples.py

# 특정 iteration 추출
python extract_training_samples.py --iteration 324065

# 분석 포함
python extract_training_samples.py --iteration 324065 --analyze

# 읽기 쉬운 텍스트 파일로 출력
python extract_training_samples.py \
    --iteration 324065 \
    --output ./iter_324065_samples.json \
    --output-text ./iter_324065_readable.txt

# 개별 샘플 파일로 저장
python extract_training_samples.py \
    --iteration 324065 \
    --output-dir ./samples/
```

### 래퍼 스크립트 사용

```bash
./extract_iteration.sh 324065           # 기본 추출
./extract_iteration.sh 324065 --analyze  # 분석 포함
```

## 데이터 흐름

스크립트는 Megatron의 데이터 파이프라인을 역으로 추적합니다:

```
global_sample_idx (82960384~82960639 for iteration 324065)
    ↓ BlendedDataset: dataset_index[idx] → dataset_id (0=DCLM, 1=Korean Web)
    ↓ BlendedDataset: dataset_sample_index[idx] → within_dataset_idx
    ↓ GPTDataset: shuffle_index[within_dataset_idx] → shuffled_idx
    ↓ GPTDataset: sample_index[shuffled_idx] → (doc_idx, offset)
    ↓ GPTDataset: document_index[doc_idx] → actual_doc_id
    ↓ IndexedDataset: .bin 파일에서 토큰 읽기
    ↓ Tokenizer: 토큰을 텍스트로 디코딩
```

## 파일 구조

### 캐시 인덱스 파일

```
examples/alpha/configs/data/.cache/kormo_50pct/
├── 7a70c9f2e20b95da0f51ba7d24be187f-BlendedDataset-train-dataset_index.npy
├── 7a70c9f2e20b95da0f51ba7d24be187f-BlendedDataset-train-dataset_sample_index.npy
├── 8efc4b85c35229927727698046e57a21-GPTDataset-train-{document,sample,shuffle}_index.npy
└── 9a9c17d8c2970cae8e1e80c4b00d94d1-GPTDataset-train-{document,sample,shuffle}_index.npy
```

### 데이터셋 파일

```
datasets/processed/qwen3_50pct/
├── dclm/dclm_content_document.{bin,idx}
└── korean_web/korean_web_content_document.{bin,idx}
```

## 출력 형식

### JSON 출력 (`samples.json`)

```json
{
  "statistics": {
    "iteration": 324065,
    "sample_range": [82960384, 82960640],
    "total_samples": 256,
    "dataset_distribution": {
      "DCLM": 231,
      "Korean Web": 25
    },
    "total_tokens": 1048576,
    "avg_tokens_per_sample": 4096.0
  },
  "samples": [
    {
      "global_idx": 82960384,
      "dataset_id": 0,
      "dataset_name": "DCLM",
      "within_dataset_idx": 74665,
      "shuffled_idx": 12345,
      "doc_indices": [987654],
      "doc_offsets": [[0, 4097]],
      "token_count": 4097,
      "tokens": [123, 456, ...],
      "text_preview": "The quick brown fox..."
    }
  ]
}
```

### 텍스트 출력 (`samples_readable.txt`)

사람이 읽기 쉬운 형식으로 각 샘플의 전체 텍스트와 메타데이터를 포함합니다.

## 분석 기능

`--analyze` 옵션을 사용하면 다음을 검사합니다:

1. **토큰 길이 통계**: min, max, mean, 4096 미만/초과 샘플 수
2. **다중 문서 샘플**: 여러 문서에 걸친 샘플 목록
3. **잠재적 이상 패턴**:
   - 반복되는 토큰 패턴 (50 토큰 이상)
   - 높은 null 문자 비율
   - 비정상적으로 짧은 샘플

## 검증 방법

추출된 샘플이 올바른지 확인하려면:

```python
import json

# 결과 로드
with open('./iter_324065_samples.json') as f:
    data = json.load(f)

# 샘플 수 확인 (256개여야 함)
assert len(data['samples']) == 256

# 데이터셋 분포 확인 (DCLM ~90%, Korean Web ~10%)
dist = data['statistics']['dataset_distribution']
print(f"DCLM: {dist['DCLM']} ({dist['DCLM']/256*100:.1f}%)")
print(f"Korean Web: {dist['Korean Web']} ({dist['Korean Web']/256*100:.1f}%)")

# 토큰 수 확인 (각 샘플 ~4097 토큰)
token_counts = [s['token_count'] for s in data['samples']]
print(f"Token counts: min={min(token_counts)}, max={max(token_counts)}")
```

## Grad Norm Spike 분석

추출된 샘플에서 spike 원인을 찾으려면:

1. **비정상적인 토큰 패턴**: 반복, null, 특수문자
2. **극단적 문서 길이**: 매우 짧거나 긴 문서
3. **언어 혼합**: 한글과 영어의 갑작스러운 전환
4. **데이터 손상**: 깨진 인코딩, 잘못된 토큰

```bash
# 분석 실행
python extract_training_samples.py --iteration 324065 --analyze

# 의심스러운 샘플 수동 검토
python -c "
import json
d = json.load(open('./extracted_samples.json'))
for s in d['samples']:
    if s['token_count'] < 4000:  # 짧은 샘플
        print(f\"Short sample {s['global_idx']}: {s['token_count']} tokens\")
"
```

## 주의사항

1. **메모리 사용**: 대규모 캐시 파일을 memory-map으로 로드하므로 큰 메모리는 필요 없음
2. **토크나이저**: `transformers` 패키지가 설치되어 있어야 텍스트 디코딩 가능
3. **캐시 해시**: 다른 학습 실행은 다른 해시를 가질 수 있음 - 필요시 `--cache-dir` 사용
