# Interleaved Thinking 규약 정비 — 템플릿 분기 + SFT 데이터 재정렬 (2026-08-24)

**단일 진입점**: DeepSeek-V4 tech report의 Interleaved Thinking 절 검토에서 출발해,
alpha의 think-히스토리 규약을 3계층(템플릿 / SFT 데이터 / 서빙)에서 정비한 작업의
전체 기록. 커밋 2건: **e3c484d** (IF fan-out) + **aada536** (DSV4 tool-시나리오
분기 + swe/arc 재변환). 관련: [`SFT_RL_DATASETS.md`](SFT_RL_DATASETS.md) §2.5,
`toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py` docstring 의도적 차이 #2.

## 0. TL;DR — 다른 세션이 알아야 할 것

1. **템플릿은 더 이상 Nemotron 3 Ultra 바이트 동일 사본이 아니다** (의도적 이탈 1건).
   `tokenizer_v5/chat_template.jinja`: tool 흔적(tools 선언 또는 tool_calls/tool 턴)이
   있는 대화는 `truncate_history_thinking` 기본값이 **False** — reasoning이 user 턴
   경계 너머로 보존된다 (DSV4 Fig.7(a)). 일반 대화는 기존 제거 유지 (7(b)).
   명시 kwarg가 항상 우선. `tokenizer_config.json` 동기화됨.
2. **변환기에 `--fanout-train-turns` 추가** — multi-True `train_turns` 행을 True 턴별
   `messages[:k+1]` 서브샘플로 전개. IF 계열 전용 (chat split은 전수 last-only라 불요).
3. **정본 bins 교체 3건** (구 디렉토리는 대조·롤백용 보존, 본 런 검증 전 삭제 금지):
   `chat_v3_if` → `chat_v3_if_fanout` / `swe_v3` → `swe_v3_keepthink` (64k+128k) /
   `arc_agi_v1` → `arc_agi_v1_keepthink` (64k+128k). 블렌드 yaml 2종 재산출 반영.
4. **agentic_v2는 편입하지 않는다** (2026-08-24 사용자 확정, §5).
5. 검증 정본: `tools/verify_chat_template.py` **31 tests** (§6 = 시나리오 분기),
   `tests/test_alpha_sft_idxmap.py` **25 tests** (§3 = fan-out), `verify_sft_bins.py`
   게이트 keepthink 4트리 PASS.

## 1. 배경 — 규약 분류와 판정 원칙

업계 수렴 표준(Kimi-K3/Qwen3.5/GLM-5.2/Nemotron)은 히스토리 think를 렌더 시 제거한다.
DSV3.2는 tool 라운드 사이만 보존·새 user 턴에서 flush, DSV4는 tool-calling
시나리오에 한해 **user 턴 경계 너머까지 보존**으로 확장했다 (1M ctx 전제).

판정 원칙은 하나다: **학습시킨 조건부 = 배포에서 질의되는 조건부.**
- 일반 chat의 히스토리 think는 클라이언트 경계에서 구조적으로 소실된다
  (OpenAI 호환 wire format이 reasoning_content를 재전송하지 않음) → 보존 학습은
  존재하지 않을 정보에 의존하는 모델을 만든다 → **제거 유지 + fan-out**이 정답.
- tool 루프/에이전트 세션의 think는 하네스가 세션 안에 쥐고 있어 재현 가능
  → **보존**이 정답. DSV4의 이분법은 이 재현 가능성의 경계선과 일치한다.
- Terminus류(툴 결과를 role=user 텍스트로 주입, 구조 없음)는 tool 시나리오로
  판정되지 않아 보존 혜택이 없다 — DSV4 report의 경고와 동일 (verify §6이 잠금).

## 2. 실측 (2026-08-24, 원본 전수/샘플 스캔)

렌더 변경 = (tool 시나리오) ∧ (멀티 user 턴) ∧ (마지막 user 이전 reasoning 존재).
세 조건의 교집합만 재변환하면 된다는 것이 스캔의 결론.

| 셋 | tool 행 | 렌더 영향 행 | 판정 |
|---|---|---|---|
| chat_v3_if | 0% | 60.9% (152k행, **fan-out 트랙**) | fan-out 재변환 |
| swe_v3 | 97.3% | 2.3% | keepthink 재변환 |
| arc_agi_v1 | 30.3% | **9.1% (22,957행)** | keepthink 재변환 |
| math_v4 / opencode / science / cuda | 4.7~100% | **0** (전부 단일 user 턴) | 불변 |
| cp / ml / proofs / safety / identity / chat_v3_chat | 0% | 0 | 불변 |

핵심 함정: **"tool 행 존재" ≠ "렌더 변경"**. 단일 user 턴 + tool 루프는 기존
템플릿도 이미 think를 전부 보존한다 (`last_user_idx` 이후는 안 지움).

렌더 토큰 증분 (V3.2식 → 보존, 영향 행 한정): swe +12.9% / agentic류 +13.8% /
IF +37.9%. IF의 지워지던 reasoning은 전수 26.7% chars ≈ 1.3억 토큰.

## 3. 결정 1 — IF 멀티턴 fan-out (e3c484d)

**문제**: IF split은 60.9%가 multi-True(중간 assistant 턴도 loss 학습)인데, 단일
시퀀스 렌더는 그 턴의 think를 지운 채 학습시켰다 → reasoning 소실 + 빈
`<think></think>`를 정답으로 학습(no-think 오신호). 기존 "등가" 논거(마지막 턴만
학습이라 fan-out 불요)는 chat split에만 성립 — §2.5 실측과 모순이었다.

**해법**: `--fanout-train-turns` — True 턴 k마다 `messages[:k+1]` 서브샘플.
잘린 시점의 렌더에서 턴 k는 마지막 user 이후가 되어 think가 보존되고, 앞 턴들은
제거된다 = **추론에서 그 턴이 라이브였던 순간의 컨텍스트와 토큰열 일치**
(마지막 True 턴 서브샘플 == 전체 렌더 — 인수분해 성질, 유닛으로 잠금).
retention(전체 보존 학습)이 아니라 fan-out인 이유는 §1의 클라이언트 경계 논거.

**결과** (`chat_v3_if_fanout`): 249,748행 → 529,472샘플. fanout_rows **151,988 ==
진단 스캔의 multi-True 행수와 정확 일치** (독립 구현 교차검증). trainable
+109.2M(+18.9%) = 회수된 reasoning. 블렌드는 chat 카테고리 합 불변, 내부
real-token 재비례 (if 0.072972 / chat 0.157242, 동일 epoch 2.81).

## 4. 결정 2 — DSV4 tool-시나리오 템플릿 분기 (aada536)

**분기 시맨틱** (jinja 상단, `tsns.tool_scenario`):
```
tool_scenario := (tools 선언 비어있지 않음) ∨ (어느 메시지든 tool_calls) ∨ (role=tool 턴)
truncate_history_thinking 기본값 := not tool_scenario     # 명시 kwarg가 우선
```
- "tools 선언"을 판정에 포함한 이유: 에이전트 세션은 1턴부터 tools를 선언하므로
  시나리오가 세션 수명 내내 **안정** — 첫 tool 호출 시점에 렌더가 소급 변경되는
  (= prefix cache 무효화) 플립을 방지.
- **템플릿 내부 분기인 이유**: vLLM/NeMo RL/lm-eval은 kwargs 없이
  `apply_chat_template`을 부른다. 하네스별 kwargs 분기였다면 배포 분포가 하네스
  수만큼 갈라짐. 템플릿이 소유하면 학습 데이터도 정확히 같은 규칙으로 구워진다.
- 부수 이득: tool 시나리오 렌더가 **append-only**가 되어 user 턴 경계에서도
  서빙 prefix KV 캐시가 유지된다 (기존엔 매 user 턴 전면 무효화 — verify §4
  prefix-diff 함정의 서빙측 함의).

**데이터 재변환** (§2에서 확정된 2셋 × 2버킷, 게이트 4트리 PASS):

| | 64k | 128k | 여집합 정합 |
|---|---|---|---|
| swe_v3_keepthink | kept −83, trainable +8.5M | kept +80, trainable +2.7M | 82 이월 = 80 도착 + 2 드롭(>128k) ✓ |
| arc_agi_v1_keepthink | kept −9,387, trainable +88.4M | kept +7,832, trainable **+591.5M** | 9,387 = 7,832 + 1,555 드롭(0.6%) ✓ |

- **64k/128k는 같은 템플릿 세대로 동시 재변환해야 여집합이 정확하다** (렌더 증가로
  64k 경계를 넘는 행이 128k 버킷으로 이월되므로).
- arc 1,555행(0.6%)은 보존 렌더가 128k도 초과해 드롭 — truncation 유해 원칙대로
  드롭 수용. swe 재변환은 구 bins에 잠입해 있던 `<|endoftext|>` 리터럴 1행도 정화
  (구 변환이 endoftext 스캔 커밋 이전 실행분이었음).
- 블렌드: swe 1.0-epoch 앵커 재산출 w=0.188891, 잔여 20엔트리 전역 리스케일
  f=0.99985198; 128k는 bins 비 (swe 0.576167 / proofs 0.423833).

## 5. 결정 3 — agentic_v2 미편입 (2026-08-24 사용자 확정)

`used_in` 실측상 Agentic-v2는 **super** post-training 셋이고 Ultra 목록에 없다
(Agentic-v3은 부재 — NVIDIA는 Ultra의 agentic을 SWE-v3 + RL 단계로 커버).
**ultra_v3 재현 원칙을 우선해 편입하지 않는다.** 참고로 남기는 사실: user-경계
tool reasoning의 최대 질량(interactive_agent 278,880행 중 74.9%, reasoning 63%)이
이 셋에 있어, 향후 에이전트 능력 보강이 필요해지면 재검토 1순위다 (그 경우
keepthink 렌더 + agentic 카테고리 5% 내부 재비례가 경로).

## 6. 산출물·정본 경로

| 계층 | 정본 | 비고 |
|---|---|---|
| 템플릿 | `examples/alpha/tokenizer_v5/chat_template.jinja` (+config json 동기화) | 분기 주석 상단 |
| 변환기 | `build_alpha_sft_idxmap.py` (`--fanout-train-turns`) | docstring 의도적 차이 #2 |
| bins 64k | `sft_packed_64k_pad16/{chat_v3_if_fanout, swe_v3_keepthink, arc_agi_v1_keepthink}` | 구 디렉토리 보존 |
| bins 128k | `sft_packed_128k_pad16/{swe_v3_keepthink, arc_agi_v1_keepthink}` | 〃 |
| 블렌드 | `configs/data/sft_40b_blend.yaml`, `sft_128k_blend.yaml` | 헤더에 재산출 근거 |
| 드라이버 | `convert_sft_64k.sh`, `convert_sft_128k.sh` | keepthink/fanout 엔트리 |
| 테스트 | `verify_chat_template.py` 31 / `test_alpha_sft_idxmap.py` 25 | |

## 7. 다른 세션을 위한 규칙·함정

1. **새 SFT 셋 추가 시**: tool 흔적이 있으면 새 템플릿이 자동으로 보존 렌더 —
   별도 조치 불요. `train_turns` multi-True가 있으면 `--fanout-train-turns` 필수
   (전수 스캔으로 확인 — 파일 앞 30k행은 last-only라 샘플 검사가 오판함, §2.5).
2. **템플릿 수정 시**: `tokenizer_config.json` 동기화 + verify 31종 통과 필수.
   비-tool 렌더를 바꾸는 수정은 기존 bins 전체를 무효화하므로 각별 주의.
3. **64k/128k 버킷은 항상 같은 템플릿 세대로 쌍 변환** (§4 여집합 규칙).
4. **서빙/RL/평가는 tokenizer_v5 디렉토리의 apply_chat_template만 사용** —
   커스텀 프롬프트 조립은 시나리오 분기를 우회해 분포를 깨뜨린다. Terminus식
   (tool 결과를 role=user로 주입) 하네스는 보존 혜택이 없다.
5. **에이전트 하네스 요건**: 보존 혜택을 받으려면 하네스가 assistant 턴의
   reasoning_content를 messages에 유지·재전달해야 한다 (없으면 남길 게 없음).
6. **구 bins 디렉토리 삭제 금지** — 본 런 개시·검증 후 공간 회수 시점에 정리.
7. 블렌드 yaml 갱신 시 헤더의 재산출 규칙(카테고리 설계단위 고정 + 내부
   real-token 비례 + swe 1.0ep 앵커)을 따를 것 — 수동 가중치 수정 금지.

## 8. 검증 명령

```bash
# 템플릿 (31 tests — §6 = 시나리오 분기)
python3 examples/alpha/tools/verify_chat_template.py

# 변환기 (25 tests — §3 = fan-out)
python -m pytest tests/test_alpha_sft_idxmap.py -v

# bins 게이트 (셋 단위는 심링크 트리로)
python toolkits/sft_data_preprocessing/verify_sft_bins.py \
  --tree /home/work/Datasets/LL_preprocessed/v5/sft_packed_64k_pad16 --seq-length 65536
```
