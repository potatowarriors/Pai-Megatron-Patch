# NeMo-Gym — alpha 환경·설정·게이트 (우리 리포 쪽)

**결정 (사용자, 2026-09-15)**: NeMo-Gym 을 SFT 단계부터 채택한다. 범위는 "RL 에서 재사용될 층"으로 자른다 — 에이전틱 평가(SWE·TB-2·τ)
이관, 학습 데이터 게이트(유령 호출·BFCL·검색·kotool·identity)를 Gym 리소스 서버로, SDG 교사 궤적 생성은 다음 트랜치부터.
T1/T2/T3(lm-eval `_aa`)와 SFT 데이터 변환기는 유지. 버전은 **v0.6.0 하나로 고정**, 우리 환경은 Gym 을 포크하지 않고 **이 디렉토리**에 둔다.
근거·대안 비교: [`../study/nemo_gym_assessment.md`](../study/nemo_gym_assessment.md).

## 배치

| 무엇 | 어디 | 비고 |
|---|---|---|
| Gym 클론 (v0.6.0, 3045a79) | `/home/work/vidsearch/tools/Gym` (NFS) | `uv venv --python 3.13.14 && uv sync --extra dev`. NeMo-RL 벤더 사본(0.5.0-dev)과 별개 — RL 재개 때 같은 태그로 맞춘다 |
| uv · Python · 캐시 | `/home/work/vidsearch/tools/{bin/uv, uv_python, uv_cache}` | HOME(`/home/work`, 49 GB 루프 볼륨) 회피. 항상 `UV_CACHE_DIR`·`UV_PYTHON_INSTALL_DIR` 지정 |
| 우리 설정·환경·게이트 | 이 디렉토리 | Gym CLI 는 `--config <절대경로>` 로 읽고, 이름으로 찾는 컴포넌트(리소스 서버·에이전트)는 `--search-dir <이 디렉토리>` (= `NEMO_GYM_EXTRA_ROOTS`) |
| `env.yaml` | Gym 루트 (gitignored) | 템플릿 `env.yaml.example`. CLI `++policy_base_url=…` 로 대체 가능 |
| 리소스 서버별 venv | `<Gym>/resources_servers/<name>/.venv` 등 (첫 기동 시 uv 가 생성) | `skip_venv_if_present` 로 재사용 |

노드 제약: Gym 은 CPU 전용이지만 **Ray 를 띄운다**. Backend.AI 훅(`/etc/ld.so.preload` 의 libcudahook) 때문에 GPU 드라이버가
깨진 노드에서는 `ray.init()` 이 C++ 예외로 즉사한다 (2026-09-15 main1 GPU 장애 중 실측: `Failed to fill in device global IDs`).
정상 노드(sub1)에서는 문제없다. main1·sub1 은 docker 를 못 돌리므로 SWE·TB-2 의 Agent/Resources 서버는 gpu06 `alpha-eval` 컨테이너 안에 둔다.

```bash
export PATH=/home/work/vidsearch/tools/bin:$PATH \
       UV_CACHE_DIR=/home/work/vidsearch/tools/uv_cache UV_PYTHON_INSTALL_DIR=/home/work/vidsearch/tools/uv_python RAY_TMPDIR=/tmp
cd /home/work/vidsearch/tools/Gym && source .venv/bin/activate
```

## 파일

| 파일 | 용도 |
|---|---|
| `configs/alpha_vllm_model.yaml` | alpha fleet 용 모델 서버 (평가). `uses_reasoning_parser: true` — fleet 는 `TOOLS=1 REASONING_PARSER=nemotron_v3` 로 떠야 한다 |
| `configs/alpha_vllm_model_tb2.yaml` | TB-2(terminus-2, 도구 미선언) 전용: `truncate_history_thinking: false` 명시 |
| `configs/alpha_vllm_model_gate.yaml` | 게이트 전용: 롤아웃에 `prompt_token_ids` 포함 (모델이 본 프롬프트 복호) |
| `configs/tau2_alpha.yaml` | τ² 벤치 — 정책 alpha + 사용자 시뮬레이터 Gemini(`inference_provider`) |
| `smoke/run_tau2_roundtrip.sh` | 2단계 게이트 러너: preflight(T1b 상당) → 서버 기동 → τ² N문항 → `roundtrip_gate.py` |
| `smoke/roundtrip_gate.py` | 판정기: vLLM 전송 로그(`NEMO_GYM_VLLM_TRANSPORT_LOG`)의 필드 증거 + 프롬프트 복호 증거 → PASS/FAIL |
| `smoke/data/strip_2turn.jsonl` | 무도구 2턴 mcqa 2행 — strip 판정용 (이력 assistant 턴에 reasoning 항목 포함) |

## 게이트 (§6 2단계) — fleet 여유가 생기면

restore (도구 시나리오, τ²):
```bash
FLEET_URL=http://sub1:8100/v1 bash examples/alpha/gym/smoke/run_tau2_roundtrip.sh 5 2
```
strip (무도구 멀티턴, mcqa):
```bash
cd /home/work/vidsearch/tools/Gym; A=<repo>/examples/alpha/gym; export NEMO_GYM_VLLM_TRANSPORT_LOG=$A/results/strip/vllm_transport.jsonl
gym env start --resources-server mcqa --config $A/configs/alpha_vllm_model_gate.yaml ++policy_base_url=$FLEET_URL ++policy_api_key=dummy ++policy_model_name=alpha &
gym eval run --no-serve --agent mcqa_simple_agent --input $A/smoke/data/strip_2turn.jsonl --output $A/results/strip/rollouts.jsonl
/home/work/vidsearch/tools/alpha_serve_venv/bin/python $A/smoke/roundtrip_gate.py --transport $NEMO_GYM_VLLM_TRANSPORT_LOG \
  --rollouts $A/results/strip/rollouts.jsonl --tokenizer <repo>/examples/alpha/tokenizer_v5 --expect strip
```
판정 기준은 09-14 `tau_proxy` 스모크와 같다: 이력 assistant 턴에 `reasoning_content` 가 실려 갔는가(필드) + 실제 프롬프트가
`<think>\n…</think>`(restore) / `<think></think>`(strip) 인가(바이트). `/tokenize` 는 증거가 아니다.

## 기록

| 날짜 | 내용 |
|---|---|
| 2026-09-15 | v0.6.0 설치(uv sync 34 s). 단위 테스트 main1: 3,934 passed / 3 failed / 119 skipped — 실패 3건은 프로세스 스폰 테스트로 sub1 재실행 3/3 PASS(main1 GPU 장애 영향), E2B 1건은 선택 의존성(`httpx_aiohttp`) 부재로 deselect. **CPU 스모크 sub1 PASS**: `mcqa` × `simple_agent` × Gemini(`inference_provider`) 3/3 rollouts, `mean/reward 1.0`, 산출물 4종 생성. main1 은 Ray 기동 즉사(GPU 장애). τ² 데이터 278행 준비. 게이트 판정기 합성 자기검증 restore/strip PASS·교차 FAIL |
