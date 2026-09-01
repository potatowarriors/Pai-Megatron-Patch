# SFT phase-2 계획 — phase-1 데이터 결함 3건 수정 재학습

**작성 2026-09-01.** phase-1(`alpha_baseline_48L_sft_128k_full_20260828_081911`, 2,448 iters, ~09-06 종료 예정)의 블렌드를
실측 검증하다 나온 결함 3건(`KNOWN_ISSUES.md` 2026-09-01)을 고친 SFT 를 phase-2 로 진행한다. 상태는 `STATUS.md`,
검증 수치는 `SFT_RL_DATASETS.md` §2.7, 여기는 **수정 스펙·시작점 판단·게이트·일정**만 쓴다.

## 0. 결론

- **권고: 3건을 고친 블렌드로 LC-B iter320 부터 재실행한다(옵션 A).** 레시피(preset·LR·예산·load)는 phase-1 과 동일, 데이터만 바꾼다.
  phase-1 최종 체크포인트와 벤치 결과가 대조군이 되어 "수정이 효과가 있었나"를 A/B 로 판정할 수 있다.
- phase-1 종료(~09-06) 직후 개시하면 ≈9.2일(323 s/iter 실측) → **~09-16 종료**. 데이터 준비(G-P0~P4)는 phase-1 중 CPU 로 끝낸다.
- 연속 학습(옵션 B)은 2.3일로 싸지만 ①·② 는 이미 학습된 뒤라 부분 완화에 그친다. 9일이 불가하면 B 를 택하되 그 한계를 기록한다.

## 1. 수정 3건 스펙

### F1. opencode_v1 tool 결과 평문화

| 항목 | 내용 |
|---|---|
| 위치 | `toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py::normalize_row`, `role == "tool"` 분기 |
| 입력 형태 | `content: [{"type":"tool-result","toolCallId":…,"toolName":…,"output":{"type":"text","value":str}}]` — 표본 12,359건 100% list, 1-item 12,358·3-item 1 |
| 변환 | `content = "\n".join(item["output"]["value"] for item in content)`. `type != "tool-result"` 또는 `output.type != "text"` 또는 value 가 str 아님 → `bad_row` 드롭 + 카운트(**조용한 str() 금지**) |
| 불변 | str content 는 그대로(swe_v3·arc_agi 경로 무변경). assistant/user/system 은 무관 |
| 테스트 | `tests/test_alpha_sft_idxmap.py` +4: 1-item → 평문 렌더 `<tool_response>\n<value>\n</tool_response>`, 3-item join, str 통과, 미지 형식 bad_row |
| 육안 | `agent_skills_question_tool/data.jsonl` 1행 렌더 → tool_response 가 평문·실제 줄바꿈인지 확인, 결과를 stats 옆 `RENDER_CHECK.md` 에 붙임 (규칙 9) |
| 부수 | reasoning 0% 는 데이터 사실 — no-think 에이전틱으로 **유지**가 기본안(swe_v3 66% reasoning 과 혼합 학습). 제외는 사용자 결정 |

### F2. identity_v1 반복 상한

| 항목 | 내용 |
|---|---|
| 현행 | 0.4271% 비중 → 219M tok 소비 = 원본 7,315행(1.18M tok) 기준 ≈180회 |
| 제안 | **원본 기준 20회 상한** → 소비 ≈23.6M tok, 비중 ≈0.046%. ×12 파일·114 bins 는 그대로(bins≥100 요건) |
| 구현 | `gen_mixed_blend.py` 에 `--consume-override identity_v1=2.36e7` 추가(현재 `--fixed-consume` 은 64k 소비량 고정만 가능) |
| 확정 근거 | phase-1 최종 ckpt 정체성 프로브(G-P6): 정체성 무관 20문항 중 자기소개 혼입 수. 혼입 ≥2 면 20회 유지, 0 이고 정체성 질문 정답률이 떨어지면 40회로 상향 |
| 문서 | `DATA_PREP_LOG.md` 결정 #9 에 "비중 상한은 반복 상한과 함께 본다" 보강 (phase-2 착수 커밋에서) |

### F3. chat_v3_chat 프롬프트 복원 재시도

| 항목 | 내용 |
|---|---|
| 현행 | 637,663행 중 75,287(11.8%) 첫 user null → 드롭. WildChat 26.6%·lmsys 12.2% 미매칭 |
| 절차 | ① `prepare_chat_prompts.py` 를 `allenai/WildChat-1M-Full`(gated) 로 재실행 시도 ② lmsys 미매칭 16,310건 해시 표본 10개를 원천에서 직접 검색해 리댁션/필터 여부 판정 ③ 복원율 재측정 |
| 성공 시 | `chat_v3_chat` 재변환(플래그 없음) → real_tokens 증가 → `gen_mixed_blend` 가 1.89ep 를 보존하며 소비량 재산출 |
| 실패 시 | 88.2% 를 정본으로 `SFT_RL_DATASETS.md` §2.7 에 기록하고 종결. 블렌드 비율에는 영향 없음 |

## 2. 시작점·예산 옵션

실측: 323 s/iter @ GBS 160(20.97M tok/iter). phase-1 잔여 1,401 iters ≈ 5.2일 → 종료 ~09-06 12:00.

| 옵션 | 시작점 | 예산 | 소요 | ①② 효과 | ③ 효과 | 대조군 | 비고 |
|---|---|---|---|---|---|---|---|
| **A 재실행 (권고)** | LC-B iter320 | 2,448 iters(±, 블렌드 재산출) | 9.2일, 09-06 개시 → ~09-16 | **완전** — 처음부터 올바른 형식·반복 | 완전 | phase-1 최종 = 동일 레시피 A/B | RL 개시 ~10일 지연. 데이터 준비는 phase-1 중 병행 |
| A' 즉시 전환 | LC-B iter320 | 동일 | 준비 1일 + 9.2일 → ~09-11 | 완전 | 완전 | **iter2448 대조군 상실**(iter300/600/900 만 남음) | 5일 절약 vs 판정 기준선 상실. 사용자 결정 |
| B 연속 보정 | phase-1 최종 | 25% = 612 iters, LR 1.0e-5→1.5e-6 cosine, warmup 30 | 2.3일 | **부분** — opencode 정상 형식 0.08ep 만 추가 노출(오형식 0.31ep 는 이미 학습), identity 기억은 희석만 | 완전 | 없음(phase-1 위에 쌓임) | 누적 SWE 1.25·chat 2.36·cp 0.57·math 1.08·science 0.51 — 앵커 원칙 이탈 |
| C 패치 스테이지 | phase-1 최종 | opencode 정상 1pass 7.15B(341 iters) + chat 복원분 ~0.3B + 리플레이 3B ≈ 500 iters | 1.9일 | ① 완전·② 희석 | 완전 | 없음 | phase-2 의 68% 가 에이전틱 → chat/IF 분포 이동 위험, Ultra 레시피 선례 없음 |

권고가 A 인 이유 세 가지. 첫째, 세 결함은 "형식·반복" 결함이라 처음부터 고쳐야 효과가 있다. 둘째, 이 리포의 원칙은 Ultra 레시피
재현과 A/B 규율이고, A 만 동일 레시피 대조가 성립한다. 셋째, RL 은 SFT 완료 후 개시라 어차피 대기 중이며, GRPO yaml·effort_levels
설정(`SFT_RL_DATASETS.md` §2.6)은 유휴 노드에서 병행할 수 있다.

## 3. 실행 절차와 게이트

검증 규칙(`CLAUDE.md`)대로 각 단계는 게이트를 통과한 뒤 다음으로 간다. G-P0~P4 는 CPU 작업이라 phase-1 중 sub1 에서 진행한다.

| 게이트 | 내용 | 통과 기준 |
|---|---|---|
| G-P0 | F1 코드 + 유닛 4종 + `gen_mixed_blend --consume-override` | `pytest tests/test_alpha_sft_idxmap.py` 38/38, 렌더 육안 1건 기록 |
| G-P1 | opencode_v1 재변환 → 새 트리 `sft_packed_128k_mixed_p2_pad16/`(미변경 셋은 symlink) | real_tokens 감소·trainable ≈1.21B 불변·drops 불변, `verify_sft_bins --tree … --seq-length 131072` PASS |
| G-P2 | F3 복원 재시도 → 변경 시 chat_v3_chat 재변환 | null 비율 재측정 기록, verify PASS |
| G-P3 | identity 소비량 override | 재산출 표에서 identity ep(원본) ≤ 20, bins 114 |
| G-P4 | `gen_mixed_blend.py --base-yaml sft_40b_blend.yaml --tree-128k <p2 트리> --gbs 160 --consume-override identity_v1=… --out sft_128k_mixed_blend_p2.yaml` | diff: SWE 1.00·chat 1.89·budget 1.19·opencode 0.31 불변, identity 만 변경. 헤더의 샘플 수로 `train-samples` 확정 |
| G-P5 | preset `sft_128k_full_p2.yaml`(phase-1 복사, train/decay-samples·wandb 이름만 변경, load = LC-B iter320) → 2-iter 스모크 | loss 1.0~1.2 범위, traceback 0, max-alloc ≤ 60GB |
| G-P6 | **phase-1 최종 평가 = 기준선**: T1(MMLU-Pro·GPQA-D·IFEval), T2 RULER, 에이전틱(SWE·Terminal, `TOOL_PARSER=qwen3_xml`), 정체성 프로브 20문항 | `results/TRACKING.md` 기록. 프로브 결과로 F2 상한 확정 |
| G-P7 | phase-2 300 iters 마다 동일 스위트 + 프로브; 종료 시 phase-1 대비 판정 | 에이전틱 ≥ phase-1, 자기소개 혼입 < phase-1, T1 ±1pp 이내(퇴행 없음) |

phase-2 에 **넣지 않는 것**: LongBlocks 변환기(1.5% 슬롯), agentic_v2 재검토, identity_v2 재생성, 하이퍼파라미터 변경. 셋 수정 3건만
바꿔야 A/B 해석이 성립한다. 각각은 별도 결정으로 다룬다.

## 4. 일정

| 시점 | 일 |
|---|---|
| 09-01~09-03 | G-P0~P4 (sub1 CPU). F3 gated 접근은 사용자 토큰 필요 |
| ~09-06 12:00 | phase-1 종료 → HF 변환 → G-P6 기준선 평가(~1일, sub1 fleet) |
| 09-06~07 | G-P5 스모크 → phase-2 개시 (main1) |
| ~09-16 | phase-2 종료 → G-P7 최종 판정 → RL 입력 ckpt 확정 |

## 5. 열린 사용자 결정

1. 시작점: **A 재실행(권고)** / A' 즉시 전환(5일 절약, 대조군 상실) / B 연속(2.3일, 부분 효과).
2. identity 반복 상한: **20회(제안)** — G-P6 프로브 후 확정.
3. opencode reasoning 0%: **no-think 에이전틱으로 유지(기본)** / 제외(에이전틱 5% 슬롯을 swe_v3 로 — 단 SWE 1-pass 앵커 이탈).
4. F3: `WildChat-1M-Full` gated 접근 시도 여부(HF 약관 동의·토큰 필요).

## 6. 변경 파일(예정)

`build_alpha_sft_idxmap.py`(normalize_row) · `tests/test_alpha_sft_idxmap.py`(+4) · `gen_mixed_blend.py`(+`--consume-override`) ·
`convert_sft_128k_mixed.sh`(p2 트리 경로) · `configs/data/sft_128k_mixed_blend_p2.yaml` · `configs/training/sft_128k_full_p2.yaml` ·
`DATA_PREP_LOG.md` 결정 #9 보강 · `STATUS.md`.
