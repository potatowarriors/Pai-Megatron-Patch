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
| `checks.py` | 검사 19종과 `verify()`. LLM 판정 없음 |
| `env.py` | 입력 생성(`make_inputs`)과 작업 디렉터리 준비(`prepare`). 기준 출력은 `work/` 밖 `ref/`에 둬서 셸을 가진 에이전트가 읽지 못한다 |
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

## gpu06 실행 — Harbor + Terminus-2 (사용자 결정 2026-09-18)

이 노드에는 격리 수단이 없다(docker·bwrap 없음, `unshare` 권한 없음). NFS 를 공유하므로 모델이 만든 셸 명령을 여기서 실행하지 않는다.
TB-2 와 같은 경로를 쓴다: gpu06 `alpha-eval` 컨테이너의 Harbor 0.22 가 과제마다 docker 환경을 띄우고 Terminus-2 가 tmux 로 조작한다.

| 파일 | 역할 |
|---|---|
| `harbor/export_harbor.py` | pilot → Harbor 로컬 데이터셋. 기준 정답은 `tests/` 에만 넣는다. Harbor 는 `tests/` 를 채점 시점에만 올리므로 에이전트가 볼 수 없다. 채점 때 입력과 기준 출력을 `/ref_root` 에 새로 만든다(에이전트가 `in/` 을 건드렸을 수 있다) |
| `harbor/base.Dockerfile` | `alpha-ffmpeg-base:1`. ubuntu:24.04 = ffmpeg 6.1.1, 검증기를 보정한 빌드와 동일. debian bookworm 의 5.1 은 `-display_rotation` 이 없다 |
| `harbor/tunnel.sh` | 모델 엔드포인트를 컨테이너 `localhost:8299` 로 역터널. **사람이 직접 실행한다.** 8199 는 SFT 평가 fleet 용 |
| `harbor/run_harbor.sh` | Terminus-2 실행 + 과제별 보상·실패 검사 회수 → `results/<RUN>/trials.json`. `API_BASE` 로 컨테이너가 직접 닿는 엔드포인트를 주면 터널 불필요. 2026-09-18 스모크 2/2 뒤 전량 실행 |

```bash
S=/tmp/ffb && python3 harbor/export_harbor.py --out $S/pilot_v0
tar -C $S -cf - pilot_v0 | ssh -F /home/work/vidsearch/.ssh-keys/config alpha-eval 'mkdir -p /opt/ffbench && tar -C /opt/ffbench -xf -'
ssh -F … alpha-eval 'cd /opt/ffbench && docker build -f base.Dockerfile -t alpha-ffmpeg-base:1 .'
# 파이프라인 관문 (과제·검증기를 바꾸면 다시)
ssh -F … alpha-eval 'cd /opt/harbor && HOME=/opt/harbor ./venv/bin/harbor run -p /opt/ffbench/pilot_v0 -a oracle -n 12 -o /opt/ffbench/jobs --job-name ffb-oracle -y -q'
API_BASE=https://gemma4.withai.cj.net:10206/v1 bash harbor/run_harbor.sh gemma4_12b_pilot_v0 gemma-4-12B-it 4 12
# 컨테이너가 직접 못 닿는 로컬 서버일 때만: bash harbor/tunnel.sh 8000 8299  (별도 터미널, 사람이 실행)
```

**파이프라인 관문 2026-09-18: oracle 50/50 (4분 5초, W=12) · nop 0/50 (3분 30초) → PASS.** 종료 후 잔류 컨테이너 0.

Alpha 는 `run_harbor.sh` 를 그대로 쓰면 안 된다. 이전 턴 추론 복원에 `tau_proxy` + `interleaved_thinking` 이 필요하다
(`eval_sft/run_terminal_tb2.sh` 의 restore 절). Alpha 용 분기는 baseline 단계에서 추가한다.

## baseline 기록

### Gemma4-12B zero-shot, pilot_v0 (2026-09-18)

조건: Terminus-2(json) · 50과제 × 4회 = 200 트라이얼 · W=12 · temp 1.0 / top_p 0.95 / max_tokens 8192 · thinking 끔 ·
엔드포인트 `https://gemma4.withai.cj.net:10206/v1`(gpu06 컨테이너에서 직접 도달, 터널 불필요) · 예외 0.

| 난이도 | 통과 | 4/4 과제 | 0~1/4 과제 |
|---|---|---|---|
| L1 | 76/80 = 95.0% | 18/20 | `l1-bitrate` 1/4 |
| L2 | 63/72 = 87.5% | 14/18 | `l2-vertical` 0/4 · `l2-segments` 1/4 |
| L3 | 38/48 = 79.2% | 8/12 | `l3-bake-rotation` 0/4 · `l3-cut-black` 1/4 |
| 전체 | **177/200 = 88.5%** | 40/50 | |

**판정: pilot_v0 은 너무 쉽다 (목표 구간 40~60%).** 0~1/4 과제의 궤적을 열어 본 결과 검증기 오판은 없었다.

| 과제 | 모델이 한 것 | 실패 유형 |
|---|---|---|
| `l3-bake-rotation` | 4회 모두 `-vf transpose=1`. ffmpeg 는 회전 플래그를 자동 적용하므로 두 번 돈다 | 입력 진단 없이 관성으로 명령 |
| `l3-cut-black` | `-c copy` 로 4초·6초에서 잘라 concat → 키프레임에 끌려감 | 정확성 제약과 stream copy 의 충돌 |
| `l2-segments` | 4.0초 경계 누락(관문이 잡았던 그 함정) | 도구의 수치 경계 동작 |
| `l1-bitrate` | 영상만 400k 로 맞춰 오디오를 더하면 403~453 kbps | 제약을 전체가 아닌 일부에만 적용 |
| `l2-vertical` | 180x320 창을 그대로 crop (축소 없음) | **지시문이 모호함** — v0 에서 "세로 전체를 살린 채" 로 고친다 |

변별력은 단일·복합 연산이 아니라 **입력 진단 · 분석 후 편집 · 수치 제약의 정확한 충족**에서 나온다. v0 은 이 축으로 다시 짠다.

### Gemma4-12B zero-shot, v0 (2026-09-18)

조건은 pilot 과 동일(Terminus-2 · 50과제 × 4회 · temp 1.0 · thinking 끔). 환경 불일치 0 · AgentTimeoutError 6(실패로 집계).

| 묶음 | 통과 | 비고 |
|---|---|---|
| `d_` 입력 진단 | 32/48 = 66.7% | 회전 3종 3/12 · `d_mixed_concat` 1/4 · 트랙·채널 선택 5과제는 전부 4/4 |
| `a_` 분석 후 편집 | 23/44 = 52.3% | 검은 띠 제거 2과제 0/8 (cropdetect 기본 16배수 반올림 → 176) |
| `c_` 수치 제약 | 29/44 = 65.9% | `c_hls` 0/4 · `c_delivery_strict` 0/4 · 단일 제약 4과제는 4/4 |
| pilot 실패 과제 6 | 11/24 = 45.8% | `l3_bake_rotation` 0/4 · `l3_cut_black` 0/4 재현 |
| **앵커를 뺀 40과제** (d 12 · a 11 · c 11 · pilot 실패 6) | **95/160 = 59.4%** | 목표 구간 40~60% 안 |
| pilot 앵커 10 (l1 5 · l2 5) | 40/40 = 100% | 회귀 확인용 |
| 전체 | 135/200 = 67.5% | |

**판정: GO.** 앵커를 뺀 40과제가 59.4%로 목표 구간에 들어왔다. 과제별 분포는 0/4 7개 · 1~3/4 22개 · 4/4 11개로 학습 신호가 있는 중간대가 가장 넓다.
의심 과제 `c_delivery_strict`(키프레임 1.96초)의 궤적을 확인: 모델이 `-sc_threshold 0` 을 빠뜨려 노이즈 입력에서 장면 전환 키프레임이 끼어든 것 — 검증기 정상.

실패 유형 상위: ① 회전 플래그 자동 적용을 모르고 이중 회전 ② 도구 기본값 함정(cropdetect 반올림, segment·HLS 키프레임 정렬, scenecut)
③ stream copy 로 정확한 절단 시도 ④ 다단계 분석에서 중간 결과를 검증하지 않음(빈 JSON, 절반만 제거).

## v0 과제 세트 (2026-09-18, `tasks_v0.py` → `tasks/v0.jsonl`)

pilot baseline 의 실패 분석에 따라 세 축으로 다시 짰다. 50과제.

| 접두 | 축 | 수 | 예 |
|---|---|---|---|
| `d_` | 입력 진단 — 지시문이 숨긴 속성을 ffprobe 로 찾아야 한다 | 12 | 회전 플래그 90/180/270 · 10-bit 입력 · 비정사각 픽셀 · 영상보다 긴 오디오 · 다중 오디오/자막 트랙 · 좌우 채널 · 이질 클립 연결 |
| `a_` | 분석 후 편집 — 매체에서 무언가를 검출한 뒤 정확히 자른다/보고한다 | 11 | 검은 구간 여러 개 제거 · 무음 점프컷 · 레터박스 제거 · 키프레임/매체 보고서 · 규격 미달 파일만 수정 |
| `c_` | 수치 제약 — 규격을 정확히 충족한다 | 11 | 300,000바이트 · GOP 정확히 2초 · HLS 2초 세그먼트 · 프레임 60~149 · 10초 패딩 · 스프라이트 시트 · 다중 제약 납품 규격 |
| `l1_` `l2_` | pilot 회귀 앵커 | 10 | |
| `l1_` `l2_` `l3_` | pilot 에서 Gemma4 가 실패한 과제 | 6 | |

입력 생성기 추가: `content`(구워진 검은 띠) · `sar` · 오디오 `dur`·`hz_right` · `sub_tracks` · 검은 구간 여러 개 · libx265 10-bit.
검사 추가: `gop` · `silence_intervals` · `json_records` · `json_numbers` · `vstream` 의 `width_any`/`level`/`sar`/`rotation`/`frames` · `audio_tones` 의 `channel` · `frames_match` 의 `exact`.

**검증기 관문: 정답 50/50 · alts 11/11 · negatives 119/119 → PASS** (`python3 selftest.py --set v0`). pilot 재실행도 PASS(회귀 없음).
**파이프라인 관문: oracle 50/50 · nop 0/50 · 환경 불일치 0 → PASS.**

관문이 잡은 검증기 구멍 2건:

| 구멍 | 조치 |
|---|---|
| `gop` 검사가 키프레임이 0초 하나뿐인 파일을 "2초 간격"으로 통과시킴 | 길이에서 기대 키프레임 목록을 만들어 개수까지 대조 |
| `frames_match` 의 ±1 프레임 허용이 "프레임 60~149" 과제에서 61~150 을 통과시킴 | 프레임 정확도가 과제인 경우 `exact=True` 로 허용을 끈다 |

### 환경 불일치 가드 (2026-09-18)

v0 첫 oracle 실행에서 `d-rot180` 1건이 실패했다(49/50). 출력이 240x320 이었으므로 그 트라이얼의 입력은 180도가 아니라 90/270도 회전 파일이었다.
같은 과제군을 4회씩 16 트라이얼 재실행해도 재현되지 않았고 베이스 이미지 안 수동 재현도 정상이라 **원인은 특정하지 못했다.**
대응: `run_verify.py` 가 에이전트가 쓴 입력과 채점 시 새로 만든 입력의 지문(스트림 구성·해상도·회전·길이)을 대조해 `env_mismatch` 를 기록한다.
`run_harbor.sh` 는 해당 트라이얼을 모델 실패로 세지 않고 제외하며 건수를 표시한다. 에이전트가 `in/` 을 고친 경우도 같은 표시가 붙는다.

## 다음 단계

1. ~~에이전트 하니스~~ — Harbor + Terminus-2 재사용, 파이프라인 관문 PASS(위).
2. baseline 측정 — Gemma4-12B(`localhost:8000`) · Alpha 최신 ckpt · frontier API. **go/no-go 관문**: Gemma4가 40~60% 구간에 오도록 난이도 재조정.
3. 과제 생성기 확장 → 벤치 v0 규격과 held-out 테스트셋 동결(커밋으로 기록).
4. SFT 데이터 파이프라인 — 교사 궤적을 이 검증기로 rejection sampling. 납품 규격 문서를 구하면 L3에 반영.
5. NeMo-Gym 리소스 서버로 이식 → RL 보상 (`../../gym/README.md`).
