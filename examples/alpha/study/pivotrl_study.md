# PivotRL 스터디 — 방법·공개 자산 실사·alpha agentic SFT 이후 적용 설계

_작성 2026-09-22. 원전: PivotRL 논문(arXiv 2603.21383 v1), Nemotron 3 Ultra 기술 보고서(arXiv 2606.15007 v1), NeMo-RL main·NeMo Gym main 파일 트리(2026-09-22 조회), HF `nvidia/Nemotron-RL-Agentic-*-Pivot-v1` 4종 카드. 로컬 대조본: Gym v0.6.0 `/home/work/vidsearch/tools/Gym`(3045a79), NeMo-RL `project_s/NeMo-RL`(브랜치 `alpha/post-train`)._

## 0. 결론

1. **PivotRL 은 SFT 궤적의 "한 assistant turn" 을 학습 샘플로 쓰는 단일 스텝 GRPO 다.** 피벗은 task 가 아니라 결정 지점(state)이고, 보상은 최종 과제 성공이 아니라 "전문가가 그 지점에서 낸 action 과 기능적으로 교환 가능한가"를 보는 로컬 검증기다. 환경을 끝까지 굴리지 않으므로 롤아웃 비용이 E2E RL 의 1/4(논문 SWE-Bench 실측)이다.
2. **방법은 완전히 공개되어 있다.** 알고리즘·선별 기준·도메인별 검증기 설계는 논문에, 검증기 구현·피벗 행 포맷·변환 도구는 NeMo Gym 에 있다. 우리 SFT 데이터로 돌리는 데 Ultra 의 산출물은 필요 없다.
3. **Ultra 의 PivotRL "단계 설정"은 NeMo-RL 에 없다.** 공개 Ultra 레시피의 `swe_teacher.yaml` 은 3단계 중 마지막 E2E SWE-RL 만이고, 터미널 티처 설정은 없다. 피벗 환경들은 MOPD 단계의 라우팅 표에만 등장한다(§3).
4. **HF 피벗 데이터셋 4종은 이름만 보고 쓰면 안 된다.** `SWE-Pivot-v1` 은 단일 스텝 피벗이 아니라 E2E SWE-RL 과제 세트(6,436 이슈)다. `Terminal-Pivot-v1` 은 진짜 피벗 행 31,111건이지만 **프로파일링 없이 성공 궤적의 전 turn** 을 담았다(§4.3).
5. **alpha 적용 루프**: agentic SFT ckpt 를 참조 정책으로 고정 → SFT 원본 궤적에서 turn 추출 → 후보당 8회 로컬 롤아웃으로 프로파일링 → 분산>0·평균<τ 만 피벗 → 단일 turn GRPO → 검증 정확도 포화 시 현재 정책으로 재프로파일링(§5). GRPO 학습기·환경은 NeMo-RL·Gym 이 준비되어 있다(사용자 확인 2026-09-22).

## 1. 방법 (논문 §2·§3·부록 A 정리)

### 1.1 용어

| 용어 | 정의 | 비고 |
|---|---|---|
| turn / action | model-call 경계에서 assistant 가 낸 **완성 전체**. 토큰이 아니다 | 자연어만 있는 turn 도 action (τ²) |
| 피벗 후보 | 궤적을 assistant 결정 지점마다 잘라 만든 (직전까지의 전체 기록 h, 전문가 action a*) 쌍 | 궤적 1개 → turn 수만큼 후보 |
| 피벗 | 후보 중 오프라인 프로파일링을 통과한 부분집합 | 학습은 이 집합에서만 샘플 |
| 로컬 롤아웃 | h 를 프롬프트로 주고 **다음 한 turn 만** 생성 | 환경 상호작용 없음(채점에 필요한 짧은 실행만) |

### 1.2 세 단계 (Algorithm 1)

1. **오프라인 프로파일링.** 참조 정책 π_ref(보통 PivotRL 시작 ckpt)를 고정. 후보 h 마다 G 개 로컬 롤아웃을 뽑아 검증기로 0/1 채점 → 평균 μ(h)·분산 σ²(h). 논문 기본 G = 8(Gym 스킬 권고 "최소 8").
2. **선별(식 5).** σ² > 0 **그리고** μ < τ 인 후보만 남긴다. 분산 0 은 GRPO 정규화 advantage 가 0 이라 그래디언트가 없다(Proposition 3.1). 평균이 낮은 것만 남기는 것은 "아직 못 하는 지점"에 예산을 몰기 위함. 논문은 무작위 turn 의 상당수가 분산 0 이라고 보고한다(수치는 HTML 수식이 이미지라 미추출, PDF 확인 필요).
3. **단일 turn GRPO(식 7).** 피벗에서 G 개 샘플 → 검증기 보상 → 그룹 정규화 advantage → 정책 업데이트. E2E RL 과의 차이는 롤아웃이 한 turn 이라는 것뿐.

Ultra 보고서가 추가한 것: **재프로파일링**. 학습이 진행되면 피벗의 보상 분산이 무너져(논문 Figure 3) 신호가 사라지므로, 검증 정확도가 포화하면 현재 정책으로 1~2 단계를 다시 돌린다. 논문 기본은 1회 프로파일링이다.

### 1.3 검증기(식 6) — 도메인별

보상 = 샘플 action 이 "그 지점에서 로컬하게 받아들일 만한 집합" 에 속하면 1. 정확 일치(식 2)는 miss rate 가 높아 기각됐다(같은 데이터 SFT 대비 이득 미미).

| 도메인 | 후보 | 검증기 (논문 A.2) | Gym 구현 |
|---|---|---|---|
| τ²-Bench 대화형 도구 | 모든 assistant turn | 툴콜 인자 비교 + "툴 부를 차례 vs 말할 차례" 판정 | `single_step_tool_use_with_argument_comparison` |
| Terminal-Bench | 모든 bash action | 출력 스키마 검사 + 정규화 문자열 유사도 + 동등성 LLM-judge(명령과 즉각 효과) | `terminus_judge` (JSON 파싱 → 스키마 → task_complete → keystrokes 유사도·judge) |
| SWE-Bench | 오류 아닌 모든 tool-call | **tool-call 이름만** 비교("deliberately coarse"). 인자·패치 품질 미채점 | `swe_pivot` (이름 → 대상 파일 → 인자 유사도 → diff 크기 shaping 으로 확장) |
| BrowseComp | 검색 관련 step | 검색/열기/증거수집 step 동등성 | `search_pivot_*` 설정 |

이론 3.3: 기능적 보상은 참조 정책을 "허용 집합의 확률질량이 큰 정책 집합" 으로 KL-사영한 것이라, 허용 집합 안팎의 조건부 분포(과제 무관 action 의 순위)를 보존한다 → OOD 보존의 근거.

### 1.4 결과 (Table 1, Base = Qwen3-30B-A3B-Thinking-2507, SFT 와 동일 데이터)

| 벤치 | Base | SFT | PivotRL | Δ vs SFT |
|---|---|---|---|---|
| τ²-Bench | 44.35 | 58.44 | 63.81 | +5.37 |
| SWE-Bench Verified | 19.07 | 37.40 | 32.67 | **−4.73** |
| Terminal-Bench | 5.42 | 13.75 | 20.00 | +6.25 |
| BrowseComp | 2.50 | 1.50 | 11.30 | +9.80 |

- SWE 만 SFT 에 진다. 검증기가 tool 이름만 보는 탓으로 읽는 것이 자연스럽고, Ultra 가 SWE 티처에서 PivotRL 뒤에 E2E SWE-RL 을 얹은 이유와 맞물린다(§2).
- OOD(IFBench 등): SFT 는 큰 하락, PivotRL 은 거의 보존(논문 Table 2, +10.04 pt 평균 우위).
- 절제: 피벗 필터·기능적 보상 둘 다 빼면 성능 하락. 무작위 피벗은 학습 중 배치 보상 분산이 급락.
- SWE E2E RL 대비: 같은 정확도까지 롤아웃 turn 4×↓, 벽시계 5.5×↓(같은 노드 수).

## 2. 처음 막힌 두 질문

**Q. 피벗은 특정 turn 인가?** 예. 궤적 안의 한 model-call 경계다. 같은 task 의 turn 3 은 버려지고 turn 11 만 남을 수 있다. 학습 샘플 = (기록, 전문가 action) 쌍 하나, 생성도 채점도 그 한 turn.

**Q. SWE·터미널에서 중간 turn 을 어떻게 채점하나?** 최종 성공으로 채점하지 않는다. SFT 시연이 정답 앵커고, 검증기는 "시연과 교환 가능한가" 만 본다. 그래서 프로파일링도 컨테이너를 끝까지 돌리지 않고, "어려움" 의 뜻은 task 난이도가 아니라 **"이 지점에서 현재 정책이 전문가와 같은 종류의 행동을 안정적으로 못 고르는가"** 다.

## 3. Nemotron 3 Ultra 에서의 사용 (기술 보고서)

| 티처 | 파이프라인 | 보고서 서술 |
|---|---|---|
| SWE | ① SFT(에이전틱 블렌드) → ② **PivotRL(단일 스텝 환경)** → ③ E2E SWE-RL(저장소 다중 턴, hidden test 이진 보상, 생성 192K·최대 200 turn) | 컨테이너 저장소를 fresh clone 처럼 재작성, git 이력 복원 명령 차단 |
| 터미널 | 최대 1시간 과제의 전문가 궤적 → **PivotRL 반복, 정확도 포화 시 재프로파일링** | — |
| 대화형 도구 | Super 와 같은 데이터·레시피로 PivotRL. Ultra 는 조기 종료 억제용으로 순차·의존 다단계 과제 추가 | — |

공개 레시피와의 대응(NeMo-RL main, 2026-09-22):

| 무엇 | 공개 여부 |
|---|---|
| `examples/nemo_gym/nemotron-3-ultra/{student_rlvr1,2, ifbench_teacher, rlhf_teacher, reasoning_teacher, swe_teacher, mopd}.yaml` + `ultra_launch.sh` | 있음 |
| `swe_teacher.yaml` | ③ E2E SWE-RL 만. `swe_agents` + SIF 컨테이너(SWE-Gym 206 · SWE-rebench-V2 7,610), GBS 512, 128 노드 |
| ② PivotRL 단계 설정, 터미널 티처 설정 | **없음**. NeMo-RL 2,290 파일 중 경로에 `pivot` 0개 |
| MOPD 티처 매핑 | `swe_pivot_*`, `droid_pivot_*`, `terminal_multi_harness_{opencode,agent006,codex}` → SWE 티처. 즉 피벗 데이터는 공개 레시피에서 **증류 환경**으로만 쓰인다 |
| PivotRL 레시피 PR | #4236 SpatialClaw PivotRL(Nano Omni), draft 미병합 |

## 4. 공개 자산 실사

### 4.1 NeMo Gym (main; 로컬 v0.6.0 3045a79 에도 전부 존재)

| 경로 | 내용 |
|---|---|
| `resources_servers/swe_pivot/` | SWE 피벗 검증기(binary/continuous 모드, 63 툴 이름 매핑, `test_verifier.py`). README 의 파이프라인 그림(`filter_difficult_patches.py` → `convert_to_minimax_format.py` → `judge_pivots.py` LLM-judge 다수결 → 피벗 ~6% turn)에 나오는 **스크립트 3종은 저장소에 없다** |
| `resources_servers/single_step_tool_use_with_argument_comparison/configs/{swe,search,droid}_pivot_*.yaml` | 단일 스텝 도구 호출 검증 설정 |
| `resources_servers/terminus_judge/` | 터미널 검증기. train/val 데이터 "미공개" 표기(§4.3 의 HF Terminal-Pivot-v1 이 그 공개판) |
| `.agents/skills/nemo-gym-pivot-datasets/` | 피벗 행 계약(`responses_create_params` + `expected_action`{function_call / function_call_batch / message} + `agent_ref`), 변환 참조 스크립트 5종, `validate_pivot_dataset.py`, 피벗 선별 지침(G≥8, 전부 성공/실패 제거, 데이터 풍부하면 쉬운 것부터 버림) |

### 4.2 NeMo-RL

§3 표. 단일 turn GRPO 는 기존 GRPO 설정에 `max_rollout_turns: 1` 과 Gym 피벗 환경을 붙이면 되므로 별도 알고리즘 코드는 필요 없다. 로컬 클론(`alpha/post-train`, 2026-08-24 기준)은 Ultra 예제 디렉토리가 없는 시점이므로 참고만 upstream 에서 한다.

### 4.3 HF 데이터셋 `nvidia/Nemotron-RL-Agentic-*-Pivot-v1` (cc-by-4.0)

| 데이터셋 | 실체 | 행 수 | 프로파일링 | 환경 |
|---|---|---|---|---|
| `SWE-Pivot-v1` (2026-06-27) | **E2E SWE-RL 과제 세트**. SWE-Gym·R2E-Gym 이슈를 OpenHands 환경 입력으로 재포맷. 필드 `responses_create_params, agent_ref` 뿐, `expected_action` 없음 | 6,436 | 해당 없음 | OpenHands swe_agents |
| `Terminal-Pivot-v1` (2026-08-28) | 진짜 피벗 행. ATCB 630 과제, GLM-5.1 + Terminus-2(Harbor) 성공 궤적(과제당 ≤5)의 **모든 유효 assistant turn**. `expected_answer` = Terminus-2 JSON. Ultra·Lightning 3.5 RL 에 사용 | 31,111 (2,716 궤적, 과제당 중앙값 45) | **없음** ("no filtering model") — 논문 용어로 random 후보 집합 | `terminus_judge` |
| `Function-Calling-Pivot-v1` (2026-03-11) | 전문가 도구 궤적의 각 step 을 행동 복제 문제로 | 8,458 | 미표기 | 단일 스텝 도구 호출 |
| `Conversational-Tool-Use-Pivot-v1` (2026-03-11) | τ² 형 대화형 도구. 필드에 `pass_rate, pass_rate_total, pass_rate_passed, qwen_235b_info` | 170,320 | **있음**(Qwen3-235B 기준 통과율 동봉) | 단일 스텝 도구 호출 |

`SFT_RL_DATASETS.md` §3.2 의 "RL-Agentic-SWE-Pivot(4.8G)" 는 이 표의 첫 행이다. 이름과 달리 피벗 데이터가 아니다.

### 4.4 여전히 없는 것

| 항목 | 대체 |
|---|---|
| Ultra 의 SWE 피벗 데이터(내부 GitLab `swe_pivot 0.0.1`)와 LLM-judge 선별 스크립트 | 우리 SFT 궤적에서 직접 추출·프로파일링(§5) |
| τ, G, 배치·lr 등 수치(HTML 수식 이미지) | PDF 재확인. G 는 8 로 시작 |
| 터미널 검증기의 LLM-judge 프롬프트 | `terminus_judge` 가 judge 호출 구조를 갖고 있으므로 프롬프트만 작성 |
| 재프로파일링 주기·기준 | "정확도 포화 시" 만 있음 → §5.4 게이트로 정의 |

## 5. alpha 적용 설계 (agentic SFT 완료 이후)

전제: 학습기 NeMo-RL(Megatron 백엔드 + alpha vLLM 플러그인)·Gym v0.6.0 준비 완료(사용자 확인 2026-09-22, 환경 상세는 `project_s/NEMO_RL_SETUP.md`, `gym/README.md`). 여기서는 데이터·검증기·루프·게이트만 다룬다.

### 5.1 루프

```
agentic SFT ckpt (π_ref 고정)
  │
  ├─ (A) SFT 원본 궤적 → turn 추출 → 피벗 후보 JSONL   [1회, 학습 무관]
  │
  ├─ (B) 프로파일링: 후보당 G=8 로컬 롤아웃 → 검증기 0/1 → μ, σ²
  │        σ²>0 ∧ μ<τ 만 피벗                              [vLLM 서빙 + Gym eval, GPU 학습 불필요]
  │
  ├─ (C) 단일 turn GRPO (NeMo-RL, max_rollout_turns=1, Gym 피벗 환경)
  │
  └─ (D) 검증 정확도 포화 → 현재 정책으로 (B) 재실행 → (C)   [Ultra 방식]
```

(B) 는 학습 인프라 없이도 먼저 돌릴 수 있는 **진단**이다. 분산 0 비율과 μ 분포만 봐도 우리 SFT 모델이 어느 결정 지점에서 흔들리는지, PivotRL 이 줄 신호량이 얼마인지 나온다.

### 5.2 우리 agent 버킷 → 검증기 매핑 (`SFT_RL_DATASETS.md` §2.11 의 18 멤버)

| 하위군 | 멤버 | 후보 turn | 검증기 | 준비 상태 |
|---|---|---|---|---|
| code(저장소 조작) | `swe_v3_tools_keepthink` `opencode_tools` `cuda_v1` `ntc_v1_swe` `ntc_v1_code` | 오류 아닌 tool-call | `swe_pivot` (툴 이름 매핑에 OpenCode 툴 `bash·glob·read·write` 포함 여부 확인) | Gym 로컬 존재 |
| terminal | `ntc_v1_{math,syn_easy,syn_medium,syn_mixed}` | Terminus-2 `commands` turn | `terminus_judge` (terminus_2 스키마) | Gym 로컬 존재. judge 모델·프롬프트 필요 |
| search | `agentic_v2_search` `search_ko_v1` `research_ko_v1` | 검색 호출 turn | `search_pivot_*` 설정 (`single_step_tool_use…`) | 설정 존재. 우리 로컬 BM25 툴 스키마로 인자 비교 규칙 조정 |
| tool-call | `agentic_v2_tc` `agentic_v2_ia` `kotool_v1_x2` `when2call_v1_x2` | 모든 assistant turn(툴콜·미호출 판단 포함) | `single_step_tool_use_with_argument_comparison` (`expected_action.type=message` 로 미호출 판단) | 그대로 사용. When2Call 은 미호출 라벨이 핵심 |
| usability·safety | `usab_v1` `safety_v2` | — | 해당 없음(단일 턴, 별도 RLVR) | 제외 |

HF `Terminal-Pivot-v1` 31k 는 우리 터미널 멤버(`ntc_v1_*` 는 같은 Nemotron-Terminal-Corpus 계열)와 원천이 겹칠 수 있다. 프로파일링 전 과제 ID 교집합을 먼저 본다. 겹치지 않으면 후보 풀 보강용.

### 5.3 데이터 전제

1. **토큰화 전 원본 JSONL 이 필요하다.** 피벗 행은 그 시점의 툴 스펙 전체와 대화 기록을 Responses 형식으로 다시 써야 하므로 bin/idx 에서는 복원 못 한다. 멤버별 원본 위치는 `study/sft_final_blend.md`·`sdg/*/README.md`.
2. **think 처리.** 프롬프트(기록)에는 이전 turn 의 think 를 SFT 렌더 규약(`INTERLEAVED_THINKING.md`, keepthink 여부)과 동일하게 넣고, 라벨(`expected_action`)에는 넣지 않는다. Gym 스킬의 "reasoning placement / label leakage" 지침과 같다.
3. **툴 이름 사전 검증.** 검증기가 우리 툴 이름을 모르면 전부 0 → 분산 0 → 후보 전멸. (B) 전에 멤버별 툴 이름 집합을 `swe_pivot` 매핑표와 대조한다.
4. **행 검증.** `validate_pivot_dataset.py` 통과를 (A) 의 게이트로.

### 5.4 게이트 (검증 규칙 준수)

| 단계 | 게이트 | 통과 기준(초기값, 실측 후 조정) |
|---|---|---|
| (A) | 행 검증기 PASS · 멤버별 후보 수·turn 깊이 히스토그램 기록 | 스킵 행 사유 전부 집계 |
| (B) | 멤버별 σ²=0 비율, μ 분포, 피벗 잔존율 | 잔존율 3~15% 범위(논문 SWE ~6%). 0% 나 >50% 면 검증기 오설정 의심 |
| (C) | 배치 보상 σ 추이(Figure 3 대응) · 검증 정확도 | σ 가 초기의 1/3 아래로 떨어지면 (D) 트리거 |
| 티어 벤치 | agentic SFT ckpt vs PivotRL ckpt **ON/OFF 대조**: TB-2·SWE·τ³(`SFT_BENCHMARKS.md` 스위트) | 도메인 내 상승 · T1(MMLU-Pro·GPQA·IFEval) 비하락(OOD 보존 주장 검증) |
| SWE | 논문 Table 1 의 SWE 역전 재현 여부 | 역전이면 Ultra 처럼 E2E SWE-RL 후속 단계 판단 |

### 5.5 열린 결정

- τ 값과 G. 논문 PDF 수치 확인 후 초기값 확정. G=8 시작.
- 터미널 judge 모델: fleet 의 상대역(gemma4) vs 별도. `terminus_judge` 는 judge 호출이 내장이라 모델 지정만 남는다.
- 재프로파일링 트리거를 검증 정확도로 볼지 배치 보상 σ 로 볼지(§5.4 는 σ 우선).
- 도메인 병렬 여부: 논문은 도메인별 분리 학습, Ultra 는 티처별 분리. alpha 는 단일 agentic 티처로 4 하위군을 한 블렌드에 넣을지, 하위군별로 돌려 MOPD 로 합칠지.

## 참고

- PivotRL: https://arxiv.org/abs/2603.21383
- Nemotron 3 Ultra Technical Report: https://arxiv.org/abs/2606.15007
- NeMo-RL Ultra 가이드: https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/models/nemotron/nemotron-3-ultra.md
- NeMo Gym `swe_pivot`: https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/swe_pivot
- NeMo Gym 피벗 데이터셋 스킬: https://github.com/NVIDIA-NeMo/Gym/tree/main/.agents/skills/nemo-gym-pivot-datasets
- HF: https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 외 3종
- 관련 문서: `docs/SFT_RL_DATASETS.md` §1·§2.11·§3, `study/nemo_gym_assessment.md`, `gym/README.md`, `project_s/NEMO_RL_SETUP.md`
