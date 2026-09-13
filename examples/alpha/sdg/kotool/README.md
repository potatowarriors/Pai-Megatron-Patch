# 트랙 T — 한국어 도구 호출/미호출 SFT 셋 (kotool_v1, When2Call 형)

**목적**: phase-2 회귀 "도구 선언 시 무관한 질문에도 호출"(유령 호출 8/33, `KNOWN_ISSUES` 09-09) 교정 항목 중 **한국어 도구 행**. 한국어 chat 행에는 도구 선언이 전혀 없어 템플릿 tool 분기가 영어로만 학습됐다.
**비율**: 호출 : 미호출 = 1 : 2 (사용자 결정). 미호출 = 직접 답변(무관·도구 없이 가능) / 되묻기(필수 정보 부재) / 불가 안내(선언 도구로 불가능). **규모** 5k 행 목표.

## 레시피 (`t_generate.py`)
- 도구 세트: Nemotron-SFT-Agentic-v2 `tool_calling` 행의 실제 API 스키마(영어, 행당 1~8개)를 그대로 선언.
- 요청 작성: GLM(저효율 사고)이 사례 규칙(call/direct/clarify/infeasible, 가중치 2:2:2:1)에 맞춰 한국어 요청을 씀. 되묻기 사례는 "기본값으로 대신할 수 없는 필수 인자를 비운 짧고 막연한 요청".
- 정책 응답: GLM(v2 지시)·DSV4 1:1, tools 선언 + `tool_choice auto`, n=2 → **검증기**: call = tool_calls 있음·선언된 이름·필수 인자 충족 / 미호출 = tool_calls 없음 + (clarify: 질문형·≤300자, infeasible: 한계 인정 어구) → 상대 교사 심판.
  call 사례는 도구 결과를 DSV4 무사고로 모의 생성 → 최종 답변 턴(Agentic-v2 형, 전 assistant 턴 학습).
- **생성 시점 시스템 규칙**(학습 행 미포함): "꼭 필요할 때만 호출·핵심 정보 없으면 되묻기·불가하면 한계 명시". 규칙 없이는 되묻기 수율 0, 유령 호출 리젝 30%였다(P0/P1 실측 2026-09-12).

## 실측
- P0 60잡: 채택 43(call 13·direct 17·infeasible 13·clarify 0), 유령 호출 샘플 19를 검증기가 리젝.
- 본생성 1차(규칙 없음) 750잡: 채택 451(clarify 0) → 중단·보존(`out/p1/t_main.jsonl`). 2차(규칙 적용, conv_id 오프셋 100000) 750잡: 채택 533(**clarify 65**), 유령 호출 리젝 235→128.

## 실행
```
GEN_THOROUGH=2 GEN_TEACHERS=glm53-flash,dsv4-flash JUDGE=other python3 t_generate.py --n 8000 --out out/p1/t_main2.jsonl --workers 96 --seed 8 --id-offset 100000
```
출력 행: `{messages(system 없음), tools, conv_id, source: kotool_v1, teacher, case, metadata}` → 변환기 tool 시나리오 분기, `render_check` 로 `<tools>` 선언·`<tool_response>` 확인. bins `kotool_v1`.

## 완료 (2026-09-12 08:45 KST)
1차 480 + 2차 5,655 = 6,135행(자기귀속 오탐 1 제외 → 6,134): call 1,973 · direct 2,403 · clarify 642 · infeasible 1,117 = 호출:미호출 1:2.11. 교사 DSV4 3,041 / GLM 3,094, 사고 중앙값 681자.
**bins** `sft_packed_128k_terminal_pad16/kotool_v1` 81 bins(실 10.45M·학습 4.10M tok; 도구 선언·결과는 비학습). `verify_sft_bins` PASS 81/81, `render_check` clean(`<tools>` 선언 1~8개·`<tool_response>` JSON).

## When2Call 영어 미호출 셋 편입 (`when2call_v1`, 2026-09-13)
`nvidia/When2Call` `train/when2call_train_sft.jsonl`(15,000행, cc-by-4.0)은 **전부 미호출** 예시다(실측: 되묻기 7,110 · 불가/한계 안내 7,540 · 기타 350, tool_calls 0건, 도구 0개 행 2,223). 단일턴·reasoning 없음 → no-think 규약(빈 `<think></think>`).
`convert_when2call.py` 가 BFCL 형 스키마(`"type": "dict"`, `"str, optional"`, `List[int]`)를 JSON Schema 형(object/string/integer/number/boolean/array)으로 정규화해 Agentic-v2·kotool_v1 과 같은 `{"type":"function","function":{…}}` 선언으로 맞춘다. 중복(user+assistant+tools 동일) 124행 제외 → **14,876행** → `when2call_v1` 66 bins(실 8.43M·학습 0.41M tok — 학습 스팬은 짧은 답 1턴뿐, 선언부는 비학습). bins<100 은 valid 0-doc 무한대기라 최종 트리에는 행 2벌 `sft_packed_128k_final_pad16/when2call_v1_x2`(131 bins, 스펙 base_ep 3 = 고유 6ep)로 넣었고 `kotool_v1` 도 같은 이유로 `kotool_v1_x2`(161 bins, base_ep 1 = 고유 2ep). `verify_sft_bins` PASS, `render_check` 봉투 0·tools 결함 0(`<tools>` 선언 1~8개 확인).
미호출:호출 비율(행 기준, 최종 블렌드 소비량): 호출 ≈ agentic_v2_tc 0.149ep × 707k ≈ 105k + kotool call 4k / 미호출 = when2call 6.4ep × 14.9k ≈ 95k + kotool 8.7k → **≈1:1**. 2:1 로 올리려면 when2call ≈13ep 가 필요한데 정형 거절문 반복이라 과잉 거절 편향 위험이 있어 6ep 를 기본으로 두고 유령 호출 게이트(≤1/33)로 판정한다(`docs/SFT_FINAL_PLAN.md`).
