# 터미널 에이전트 SFT 데이터 — 공개 코퍼스 채택 정본 (2026-09-10)

Terminal-Bench 2.0 형(Terminus-2 스키마) SFT 데이터의 **현재 정본**. NVIDIA 공개 `nvidia/Nemotron-Terminal-Corpus` 를 우리 행 스키마·렌더
규약으로 변환해 128k bins 로 만들었다. 자체 합성 트랙(2026-09-07~10, 폐기)은 [`SYNTHESIS_ARCHIVE.md`](SYNTHESIS_ARCHIVE.md), 공부용 설명은
[`../../study/terminal_sdg_study.md`](../../study/terminal_sdg_study.md), 사고 서사는 `docs/KNOWN_ISSUES.md` 2026-09-10·09-09.

## 0. 한눈에

| 항목 | 값 |
|---|---|
| 원천 | `nvidia/Nemotron-Terminal-Corpus` 366,154행 (cc-by-4.0, 교사 DeepSeek-V3.2 · agent terminus-2, arXiv 2602.21193) |
| 변환 | 365,705행 (`/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Terminal-NTC-v1/`) — 파싱실패 교환 splice, null 말미 제거, 보존 렌더 |
| bins | `/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_terminal_pad16/ntc_v1_{code,math,swe,syn_easy,syn_medium,syn_mixed}` — 41,941 bins, 실토큰 5.48B / 학습 3.35B, 드롭 0 |
| 게이트 | `verify_sft_bins` PASS · `render_check` 6 멤버 clean(`<think>` 전 턴) |
| 상태 | **B안 블렌드(phase-2 교정 재실행, phase-2 세션 담당) 등록 대기** — §5 지침 |
| 원칙(사용자) | 전량 사용(품질 필터 미적용) · math 어댑터 포함 · 비율 설계 없음 · 자체 합성분 폐기 · 보존 렌더(`--keep-history-think`) |

## 1. 결정 이력

| 날짜 | 결정 | 근거·비고 |
|---|---|---|
| 09-07 | 자체 합성 착수 (교사 GLM-5.3-Flash, Harbor Terminus-2) | §2.9 조사가 공개 코퍼스를 놓침 (컬렉션 밖 릴리스). 기록은 아카이브 |
| 09-07 | **보존 렌더** — 모든 assistant 턴의 reasoning 을 남긴다 | Terminus 행은 tool-시나리오가 아니라 기본 렌더는 중간 턴 think 를 비움. 아카이브 §1.1 |
| 09-10 | **공개 코퍼스 채택, 합성 중단** | 규모 366k = Ultra 의 ~370K. 검증 후 채택 |
| 09-10 | **자체 합성 데이터 폐기** (`alpha-SFT-Terminal-v1`·bins 삭제, 원본 9.8 GB 보존) | 보강 용도로도 쓰지 않는다 |
| 09-10 | **코드:수학 비율 정책(7:3) 폐기** | 터미널 데이터가 없던 시절의 대체 정책 |
| 09-10 | **전량 365,705행** (완료 행만 아님) · **math 어댑터 포함** | 논문 소거 실험: 무필터 12.4% > 완료만 6.74% > 성공만 5.06% (TB-2) |

## 2. 공개 코퍼스의 실체

### 2.1 출처·구성
- HF: https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus · 형제 `Nemotron-Terminal-Synthetic-Tasks` · RL `Nemotron-RL-Agentic-Terminal-Pivot-v1` ·
  `Nemotron-Cascade-2-SFT-Data` 의 Terminal Agent 스플릿(~324k). 논문 https://arxiv.org/abs/2602.21193 (On Data Engineering for Scaling LLM Terminal Capabilities).
- 로컬 `/home/work/Datasets/LL_datasets/posttraining/SFT/Nemotron-Terminal-Corpus/` 29 parquet, 7.7 GB. 공개분 366,154 = 어댑터 226,313 + skill-based 139,841 (논문의 seed-based 124k 는 미공개).

| 분할 | 행 | 내용 |
|---|---:|---|
| dataset_adapters/code | 31,960 | 코드 문제를 터미널 과제로 재포맷 (solution.py) |
| dataset_adapters/math | 162,692 | 수학 문제 재포맷 (답 파일) |
| dataset_adapters/swe | 31,661 | SWE 과제 (리포 이슈) |
| synthetic_tasks/skill_based/easy | 44,809 | 9 스킬 × ≈5k (data_processing·querying·science, debugging, dependency, file_ops, sci_computing, security, sw_eng) |
| synthetic_tasks/skill_based/medium | 89,343 | 11 스킬 (model_training·system_administration 추가) |
| synthetic_tasks/skill_based/mixed | 5,689 | 6 스킬 × ≈1k |

- parquet 열: `conversations[{role,content}]`, `agent`(terminus-2), `model`(deepseek-ai/DeepSeek-V3.2), `task`, `episode`, `run_id`, `trial_name`, `enable_thinking`(True), `source`. **reward 열 없음.**
- 논문: 성공 여부로 거르지 않았고("retaining unsuccessful trajectories appears to provide valuable supervision"), 대부분 32,768 토큰 안에 든다.

### 2.2 스키마 실체 (변환이 다뤄야 했던 것)
| 관찰 | 값 | 처리 |
|---|---|---|
| system 역할 없음 — 첫 user 턴 = Terminus 시스템 프롬프트 2,836자 + `\n\nTask Description:\n…` | 프리픽스 md5 **fa616539** = `convert/swe_v3_terminus_system_prompt.txt` 와 바이트 일치 (133,531/133,531) | system 역할로 분리 |
| reasoning 이 assistant content 에 인라인 `<think>…</think>` | 사고 보유 턴 99.8% | `reasoning_content` 로 분리 |
| **파싱 실패 턴**: think 만 있고 content 가 빈 assistant 턴 + 뒤따르는 `Previous response had parsing errors: … No valid JSON` user 턴 | 표본 행의 **64%**, 전부 중간 턴 | (무효 assistant, 파싱오류 user) 쌍 **splice** (§3) |
| 마지막 assistant 가 JSON `null` (에피소드 끊김) | 행의 8.6% | 그 턴 + 직전 user 턴 제거 |
| 완료 선언(`task_complete:true`) 없이 끝남 | 행의 30% (code 37 · math 9 · swe 8 · easy 20 · medium 75 · mixed 87%) | 보존, `metadata.completed=false`·`quality_flags=["not_completed"]` |
| `Previous response had warnings:` (유효 JSON + 개행 누락 경고) | 소수 | 보존 (프로토콜의 일부) |
| system md5 가 TB-2 평가 harness(Harbor terminus-2, **7665e733**)와 다름 | 전 행 | 평가 프롬프트 계열 데이터는 없음 → TB-2 before/after 로 프롬프트 과적합 여부 확인 |

## 3. 변환 규칙과 재현

변환기 `convert/nemotron_terminal_to_rows.py` (parquet → 행 스키마 jsonl + MANIFEST). 행 스키마는 SWE-v3 Terminus 와 동일:
`{uuid, license, messages:[system, user, assistant(+reasoning_content), user, …], metadata:{task, trial, harness, teacher, split, source_file, assistant_turns, reasoning_turns, system_md5, enable_thinking, spliced_turns, null_tail, completed, quality_flags}}`.

- **splice**: 파싱 오류에서는 명령이 실행되지 않아 터미널 상태가 그대로이므로, 실패 턴과 오류 메시지를 빼면 앞 user(터미널 출력) → 다음 assistant(재시도, 유효 JSON) 로 이어져 대화가 일관하다.
  빈 응답을 학습 목표로 두지 않기 위한 조치. **논문 레시피와의 차이** — 논문의 최고 성적은 이 턴을 그대로 둔 데이터에서 나왔다. `--keep-parse-errors` 로 원본 보존.
- 드롭(전량 449): json_invalid 314(무효 턴 뒤가 파싱오류 메시지가 아닌 경우) · think_residual 126 · injection 6(특수토큰 리터럴) · no_assistant 3. canary 0.
- 품질 필터(`convert/filter_rows.py`)는 **학습 데이터에 적용하지 않는다**(사용자 결정·논문 무필터). 코퍼스에 맞게 보정한 판(연속 반복·코드 과제 한정 no_edit·분할별 dup 키)은 통과율 98.8% 이며
  `alpha-SFT-Terminal-NTC-v1/filtered/` 에 참고용으로 있다.

```bash
cd examples/alpha/sdg/terminal/convert
# 1) 변환 (12 워커 18분) → alpha-SFT-Terminal-NTC-v1/{ntc_v1.jsonl, MANIFEST.json}
python3 nemotron_terminal_to_rows.py /home/work/Datasets/LL_datasets/posttraining/SFT/Nemotron-Terminal-Corpus \
  --out /home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Terminal-NTC-v1 --tag ntc_v1 --workers 12 --keep-not-completed
# 2) 분할별 jsonl + 128k bins 6 멤버 + verify + render (main1 48 워커 13분) — /home/work/vidsearch/tools/glm53/ntc_bins_build.sh 와 동일
python3 ../../../../../toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py --input <split>.jsonl --tokenizer ../../../tokenizer_v5 \
  --output-prefix <bins>/<member>/data --seq-length 131072 --pad-doc-multiple 16 --workers 48 --keep-history-think
python3 ../../../../../toolkits/sft_data_preprocessing/verify_sft_bins.py --tree <6멤버 트리> --seq-length 131072
python3 ../../../../../toolkits/sft_data_preprocessing/render_check.py --member <bins>/<member> --docs 0,-1 --tokenizer ../../../tokenizer_v5 --write
# 3) 분포 대조
python3 dataset_stats.py --jsonl <a.jsonl> --jsonl <b.jsonl> --tokenizer ../../../tokenizer_v5 --out REFERENCE_STATS.md
```

## 4. 산출물

| 멤버 | 행 | bins | 실토큰 | 학습 토큰 | 실토큰 비 |
|---|---:|---:|---:|---:|---:|
| ntc_v1_code | 31,927 | 3,704 | 484.6M | 328.7M | 8.8% |
| ntc_v1_math | 162,662 | 15,431 | 2,019.5M | 1,349.0M | 36.8% |
| ntc_v1_swe | 31,421 | 6,332 | 827.3M | 511.2M | 15.1% |
| ntc_v1_syn_easy | 44,798 | 4,116 | 537.7M | 293.6M | 9.8% |
| ntc_v1_syn_medium | 89,216 | 11,835 | 1,543.2M | 830.4M | 28.2% |
| ntc_v1_syn_mixed | 5,681 | 523 | 68.3M | 36.0M | 1.2% |
| **합계** | **365,705** | **41,941** | **5,480.7M** | **3,349.9M** | 100% |

각 멤버 디렉터리: `data_text_document.{bin,idx}`, `data.stats.json`, `RENDER_CHECK.md`. 128k 초과 드롭 0 (행 토큰 max 92k).
데이터 디렉터리 `alpha-SFT-Terminal-NTC-v1/`: `ntc_v1.jsonl`(20.6 GB) · `MANIFEST.json`(파일별 입력/출력/드롭/splice) · `by_split/*.jsonl`(멤버 입력) · `filtered/`(참고).

## 5. 블렌드 등록 지침 (B안)

- 6 멤버를 **실토큰 비례 가중**으로 넣으면 코퍼스 단일 멤버와 동치 — 사용자 결정이 "구성 그대로" 이므로 기본. 분할 조정이 필요하면 재빌드 없이 가중만 바꾼다.
- 예산: 5.48B 라 phase-2 규모(12.6B)의 블렌드에서는 **ep ≤ 1**. `gen_phase2_blend.py --add ntc_v1_code … --ep <m>=<e>` 로 넣고 `--solve-iters`.
- 멤버 수 50+ → 프리셋에 **`mid-level-dataset-surplus: 0.05`** 필수 (valid 크기 Σceil 부풀림, `KNOWN_ISSUES` 09-09). 프리셋은 평면 YAML 이라 전체 복제.
- 렌더는 `--keep-history-think` 로 이미 구웠다(변환기 무관). 템플릿 불변.
- 투입 전 sub1 2-iter 스모크: `scripts/sub1_jit595_smoke.sh <training> <data>` (sudo 불필요).
- 평가: TB-2 before/after (Terminus-2 `interleaved_thinking=true`). 코퍼스 프롬프트(fa616539)와 평가 프롬프트(7665e733)가 다르므로 형식 준수율을 별도로 본다.

## 6. 대조 기준 (`REFERENCE_STATS.md`, tokenizer_v5)

| 지표 | SWE-v3 Terminus 3,542행 | 공개 코퍼스 (필터 후 19,361행 표본) | (폐기) 합성 v1 8,596행 |
|---|---|---|---|
| assistant 턴/행 median (p90 / max) | 34 (66 / 232) | 7 (12 / 30) | 5 (7 / 20) |
| reasoning 보유 턴 | 39.2% | 99.8% | 99.99% |
| reasoning 토큰/턴 median (p90) | 44 (283) | 306 (1,517) | 286 (1,839) |
| 행 토큰 median (p90 / max) · 128k 초과 | 30k (60k / 204k) · 7 | 13.1k (26k / 92k) · 0 | 5.1k (15k / 89k) · 0 |
| commands/턴 | 1.2 | 2.9 | 1.1 |
| 마지막 턴 task_complete | 98.8% | 72% | 100% |

## 7. 교훈

1. **조사는 컬렉션 밖까지** — 조직 datasets 탭 전체와 arXiv 데이터 절. 컬렉션·이름 패턴만 보다가 3일을 합성에 썼다 (`rules/sft-data.md` 에 규칙화).
2. **"고품질 필터링" 공개 데이터도 harness 흔적을 담는다** — 파싱 실패 턴은 원 실행에서는 무해한 재시도지만 SFT 목표로는 유해. 품질은 원천 기준이 아니라 우리 학습 목표 기준으로 다시 본다.
3. **품질 신호는 생성자 습관에 민감** — GLM 용 repeat_loop 가 DeepSeek 의 `cd /app` 을 루프로 오인. 새 원천에는 플래그된 표본을 직접 보고 재보정.
4. **선별 기준은 문서로 확인한 뒤 서술** — "미완료도 테스트 통과" 는 추정이었고 논문은 무필터였다. 사용자 질문으로 드러나 정정.
5. **인자는 존재≠적용** — surplus 인자가 제공자에서 끊겨 있었다. 로그의 args 덤프로 적용 여부를 확인한다.
