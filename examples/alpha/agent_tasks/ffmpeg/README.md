# ffmpeg 편집 에이전트 task — 벤치·데이터·RL 환경

사내 업무 에이전트 트랙의 1번 도메인. 자연어 편집 요청을 받아 셸에서 ffmpeg로 처리하고, 출력 파일을 결정론적으로 채점한다.
같은 검증기를 벤치 채점, SFT 데이터 rejection sampling, RL 보상에 쓴다.

## 왜 이 task인가 (2026-09-18 사용자 결정)

목적은 alpha 프로젝트 지속의 근거를 만드는 것이다. 주장은 "학습 과정을 소유해야 회귀 없이 점진적으로 개선할 수 있다"이다.
따라서 실험은 task 점수 한 축이 아니라 아래 세 축으로 설계한다.

| 축 | 측정 | 주장이 예측하는 결과 |
|---|---|---|
| task 성능 | 이 벤치 | Alpha, Gemma4 fine-tune 모두 상승 |
| 일반 능력 유지 | 기존 T1·T3·프로브 (`docs/SFT_BENCHMARKS.md`) | Gemma4 fine-tune은 하락, Alpha는 블렌드 재실행으로 유지 |
| 개선 궤적 | 실패 진단 → 데이터 수정 → 재실행 2~3회 | Alpha만 원인을 데이터까지 추적 |

비교 대상은 Gemma4-12B부터 시작한다. baseline은 "단순 fine-tune"이 아니라 최선의 fine-tune이어야 한다.
비교표에는 frontier API 행을 반드시 넣는다. 회의실 예약류 관리 업무는 같은 틀의 2번 도메인으로 미룬다.

## 선행 조사 (2026-09-18)

| 대상 | 사실 | 판정 |
|---|---|---|
| ELLMPEG (arXiv 2602.00028, github `zoha-az/ELLMPEG`, MIT) | ffmpeg 질의 380 + VVenC 100, 단발 질문, LLM 판정(GPT-4o·Gemini 2.0 Flash), 입력 영상·출력 검사 없음. 기본 모델 Qwen2.5-7B 65%, 나머지 20~25% | 과제 분류 체계의 씨앗으로만 |
| remyxai/ffmperative-7b | Llama 2 7B + 도구 조합 500건 | 참고 |
| CutVerse · Aurora · LAVE | GUI 또는 멀티모달 편집 | 범위 다름 |
| Terminal-Bench | ffmpeg 과제 목록 미확인 | 미확인 |
| 로컬 `Nemotron-Terminal-Corpus` 366,154행 전수 스캔 | ffmpeg·ffprobe 등장 297행(0.08%). 과제 설명 237 · assistant 호출 170. SWE 어댑터 152행은 리포 버그 수정, 편집에 가까운 것은 file_operations 약 100행 | Alpha는 ffmpeg 편집을 거의 배우지 않았다 |

이 벤치의 차별점은 실제 입력 파일, 출력 파일 속성의 결정론적 검사, 다중 턴 에이전트 세 가지다.

## 구성

| 파일 | 역할 |
|---|---|
| `media.py` | 합성 입력 생성. lavfi `testsrc2` + `sine`이라 실제 영상이 없고 8코어에서도 가볍다. VFR · 회전 플래그 · 무음 구간 · 검은 구간 · 다중 오디오 · 내장 자막 변형 지원 |
| `checks.py` | 검사 15종과 `verify()`. LLM 판정 없음 |
| `env.py` | 작업 디렉터리 준비. 기준 출력은 `work/` 밖 `ref/`에 둬서 셸을 가진 에이전트가 읽지 못한다 |
| `tasks_pilot.py` → `tasks/pilot_v0.jsonl` | pilot 50과제 (L1 20 · L2 18 · L3 12). **난이도 보정용이며 동결 벤치가 아니다** |
| `selftest.py` | 검증기 관문 |

과제 스키마: `instruction`(한국어 자연어 요청) · `inputs` · `output` · `reference`(정답) · `variants`(프레임이 정당하게 다른 추가 기준 출력) ·
`alts`(다른 방식의 정답, PASS 해야 함) · `negatives`(그럴듯한 오답, FAIL 해야 함) · `checks`.

난이도: **L1** 단일 연산 · **L2** 복합 연산 · **L3** 규격 준수, 입력 진단, 오류 복구(입력 속성을 지시문에 알려주지 않는다).

검사 종류: `format` `vstream` `astream` `duration` `size_max` `bitrate_max` `frames_match` `audio_tones` `mean_volume` `loudness`
`faststart` `cfr` `file_count` `text_contains` `json_intervals`.

## 검증기 관문 — 모델 채점 전에 반드시 통과

```bash
cd examples/alpha/agent_tasks/ffmpeg
python3 tasks_pilot.py          # tasks/pilot_v0.jsonl 재생성
python3 selftest.py --workers 6 # 약 2분 (8코어)
```

과제마다 정답은 PASS, `alts`는 PASS, `negatives`와 빈 출력은 FAIL 이어야 한다. 하나라도 어긋나면 관문 실패다.
과제나 검사를 고치면 다시 돌린다. 관문을 통과하지 못한 과제 세트로 모델 점수를 기록하지 않는다.

**2026-09-18 결과: 정답 50/50 · alts 13/13 · negatives 107/107 → PASS.**

### 검증기 보정

`frames_match`는 후보와 기준 출력을 같은 해상도·fps로 맞춰 PSNR을 잰다. 앞뒤 1프레임 정렬 중 최댓값을 쓴다.

| 실측 (320x240 `testsrc2`, 2~5초 자르기) | PSNR |
|---|---|
| 정답, 화질만 낮춤(crf 30) | 36.9 dB |
| 시작점 2초 어긋남 | 18.0 dB |
| 시작점 0.2초 어긋남 | 17.5 dB |
| 좌우 반전 | 6.4 dB |

기본 임계는 25 dB다. 과제별 예외는 `l1_remux` 50(재인코딩 금지), `l1_gif` 22(팔레트 양자화), `l2_logo`·`l2_pip` 30,
`l2_burn_subs` 하단 영역 31이다.

### 관문이 잡아낸 것 (첫 실행 4건, 전부 과제 쪽 오류)

| 증상 | 원인 | 조치 |
|---|---|---|
| `l2_segments` 정답이 2조각(8초+4초) | segment muxer가 정확히 4.0초인 키프레임 경계를 놓친다 | 정답에 `-segment_time_delta 0.02`. 빠뜨린 명령은 negative로 등록 |
| `l1_fps`에서 `-r 15` 탈락(23.5 dB) | `fps` 필터와 `-r`이 서로 다른 원본 프레임을 고른다. 둘 다 정답 | `variants` 도입 — 기준 출력 여러 개 중 최댓값 |
| `l1_trim` stream copy가 통과 | MP4 edit list 덕분에 디코딩 결과가 실제로 정확 | negative → alt로 이동 |
| `l2_burn_subs` 여유 0.6 dB | 자막은 작은 영역이라 전체 프레임 PSNR로 안 드러남 | `region` 옵션. 하단 영역에서 오답 26.4 / 정답 36.5 dB |

### 알려진 얇은 여유

`l3_vfr_to_cfr` 정답 27.1 dB, `l2_vertical` 정답 27.4 dB (임계 25). baseline 측정 때 이 두 과제의 실패는 수동으로 열어 본다.

## 다음 단계

1. 에이전트 하니스 — Terminus-2 형식 셸 루프. Alpha가 터미널 코퍼스 3.35B 토큰을 이 형식으로 학습했다.
2. baseline 측정 — Gemma4-12B(`localhost:8000`) · Alpha 최신 ckpt · frontier API. **go/no-go 관문**: Gemma4가 40~60% 구간에 오도록 난이도 재조정.
3. 과제 생성기 확장 → 벤치 v0 규격과 held-out 테스트셋 동결(커밋으로 기록).
4. SFT 데이터 파이프라인 — 교사 궤적을 이 검증기로 rejection sampling. 납품 규격 문서를 구하면 L3에 반영.
5. NeMo-Gym 리소스 서버로 이식 → RL 보상 (`../../gym/README.md`).
