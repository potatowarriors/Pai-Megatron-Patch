# alpha 채팅 서빙 (vLLM + OpenWebUI)

SFT 체크포인트와 **사람이 직접 대화**하기 위한 최소 구성. 벤치 스위트(`../eval_sft/`)와는
목적이 달라 설정이 다르다 — 아래 §3 이 그 차이의 근거다.

구성: 브라우저 → OpenWebUI(:8080) → vLLM OpenAI 호환 API(:8001) → alpha (main1 GPU 3, 40GB A100 슬라이스).
창 128K, KV 캐시 1,311,257 토큰, 가중치 29.9GiB.

## 1. 기동

```bash
cd examples/alpha
bash chat/serve_chat.sh                       # vLLM  (기본: hfmodel_0000600, 128K, :8001, GPU3)
bash chat/run_openwebui.sh                    # UI    (기본: :8080)
```

체크포인트를 바꾸려면 첫 인자로 준다: `bash chat/serve_chat.sh outputs/<run>/hfmodel_<iter>`.

## 2. 접속

포트 8080 은 이 컨테이너의 `BACKENDAI_SERVICE_PORTS` 중 `nniboard` preopen 슬롯과 겹친다.
Backend.AI 앱 프록시로 열리지 않으면 SSH 터널을 쓴다:

```bash
ssh -N -L 8080:localhost:8080 main1     # ~/.ssh/config 의 Host main* (포트 2200)
```

## 3. 벤치 fleet 과 무엇이 다른가

| 항목 | 벤치 (`eval_sft/serve_alpha.sh`) | 채팅 (`chat/serve_chat.sh`) | 이유 |
|---|---|---|---|
| reasoning 파서 | off | **`nemotron_v3`** | UI 가 사고 과정을 접어서 보여주려면 `reasoning_content` 로 분리돼야 한다 |
| 레플리카 | DP 8 (H100 fleet) | 단일 GPU | 1인 사용 |
| GPU | sub1 H100 | main1 GPU3 (40GB A100 슬라이스) | 벤치와 자원 분리 |
| `--max-num-seqs` | 기본 | 8 | 1인 사용. KV 여유는 충분하다(창의 10배) |
| 모델명 | `alpha` | `alpha-v2-sft` + 별칭 `alpha` | UI 표시는 구체적으로, 게이트 호환은 별칭으로 |

**게이트 G2 는 채팅 fleet 에서 의도적으로 FAIL 한다.** 이유가 둘이다:
① reasoning 파서가 `</think>` 를 본문에서 떼어간다, ② `check_gates.py` 는 `reasoning_content`
필드를 보는데 vLLM 0.25.1 이 실제로 내보내는 이름은 **`reasoning`** 이다. 벤치는 파서를 끄고
돌리므로 그쪽에서는 문제가 되지 않는다. 채팅 fleet 의 검증은 G1 + G3 + `smoke_chat.sh` 다 (§5).

## 4. chat template

`hfmodel_*/tokenizer_config.json` 의 `chat_template` 필드에 내장되어 있다 — vLLM 이 자동으로
집어가므로 `--chat-template` 플래그가 필요 없고, 학습·평가·서빙이 같은 렌더러를 쓴다.
템플릿 자체의 규약(think 히스토리, tool 분기)은 `../docs/INTERLEAVED_THINKING.md`,
검증은 `../tools/verify_chat_template.py`.

## 5. 검증

```bash
python3 tools/emit_generation_config.py <CKPT> --check                  # G1
bash chat/smoke_chat.sh                                                 # 7항목 (엔드포인트·종료·reasoning 분리·멀티턴)
python3 eval_sft/check_gates.py --base-url http://localhost:8001/v1     # G3 (G2 는 위 사유로 FAIL 정상)
```

2026-08-31 iter600 실측: G1 PASS, smoke **7/7 PASS**.

## 6. 종료

```bash
pkill -TERM -f "openwebui_venv/bin/open-webui"
pkill -TERM -f "alpha_serve_venv/bin/vllm"      # GPU 메모리 회수 확인은 eval_sft/stop_fleet.sh 3,7 절 참조
```

## 7. 구축하며 밟은 함정 (2026-08-31)

| 증상 | 원인 | 대응 |
|---|---|---|
| `driver too old (found version 12080)` | vllm 0.25.1 은 CUDA 13 빌드, main1 compat lib 은 570(=12.8) | `eval_sft/restore_bench_env.sh` 로 570→595 교체. 실행 중이던 Gemma 서빙은 무영향 |
| open-webui `no such table: config` | `config.py` 80행 `run_migrations()` 가 순환 import 로 실패하는데 예외를 **삼킨다**. 되돌아온 import 가 같은 파일 1103행 `ENABLE_LOCAL_WEB_FETCH` 를 요구 | `init_openwebui_db.py` 로 분리 — config 를 끝까지 로드한 뒤 alembic 실행 |
| `.webui_secret_key` 가 리포에 생성 | 키를 안 주면 **현재 작업 디렉토리**에 떨군다 | 런처가 `DATA_DIR/secret_key` 를 만들어 `WEBUI_SECRET_KEY` 로 주입 |
| `/api/models` 가 빈 배열 | 인증 세션 없이 호출 | 브라우저는 자동 로그인. CLI 확인은 `/api/v1/auths/signin` 토큰으로 |
| 게이트가 404 | `check_gates.py` 가 모델명 `alpha` 를 하드코딩 | `--served-model-name` 에 별칭 `alpha` 추가 |
| `Using default MoE config ... E=192,N=512` | 192-expert 튜닝 설정 부재 + Backend.AI 가 GPU 이름을 `CUDA_GPU` 로 마스킹 | 정확성 무관, MoE 처리량만 손해. 1인 채팅에서는 무시 |
