# 트랙 D — 검색 에이전트 SFT 셋 (D1 `search_ko_v1` 다중 홉 · D2 `research_ko_v1` 딥 리서치)

**근거**: NVIDIA Agentic-v2 `search`(5,968행 = Wikidata 4~8홉 영어 질문 + `web-search` 도구(Tavily 결과) + 정답 일치 채택, Ultra 2.0 epoch retain)의 **한국어 대응물**(영어판은 이미 블렌드에 있음).
D2 딥 리서치는 공개 대응물이 없어 한국어 코퍼스 위에서 1차로 만든다.
**검색 백엔드**: 유료 API 전부 폐기(Tavily 종량제·Google CSE·Vertex AI Search, 사용자 결정 2026-09-12) → **로컬 BM25 색인**(한국어 위키백과 덤프 527k 문서 + NIKL 2024 기사 974k → passage 3.43M, `bm25s`, 어절+한글 2-gram).
결과는 **Tavily 형식**으로 반환해 Agentic-v2·평가 하니스(`eval_sft/search_agent_eval.py`)와 도구 스키마·결과 형식이 같다. 한계: 실제 웹 잡음 없음, 코퍼스 범위(위키·2024 기사) 밖은 검색 불가 — 문서에 명시.

## 구성 요소
| 스크립트 | 역할 |
|---|---|
| `build_corpus.py` | 덤프·기사 → passage(≤700자, 제목을 모든 passage 앞에; NIKL 제목은 헤드라인으로) `out/corpus/passages_v2.jsonl` |
| `build_index.py` / `search_server.py` | bm25s 색인 → HTTP `/search`(Tavily 형식; NIKL 은 url 없음·매체·날짜 노출 — 지어낸 URL 학습 방지) |
| `wd_chain.py` | Wikidata 랜덤워크 3~6홉(한국어 라벨·kowiki 문서 보유 개체만, **단일값 속성만**, 일반 속성(국가·통화·수도·공용어·행정구역 등)은 마지막 홉 금지·연쇄당 ≤1). API 8 rps·429 백오프·캐시 |
| `wd_question.py` | 연쇄 → GLM(저효율)이 한국어 질문 1개로 자연어화(중간 개체·정답 은닉, 누출 검사) |
| `d2_seeds.py` | NIKL 기사 주제 → 조사형 요청(기간·관점·출처 요구 포함) |
| `agent_run.py` | 에이전트 루프(DSV4 정책, `web-search` 도구, D1: `최종 답:` 정규화·별칭 일치만 채택 / D2: 검색 ≥5) → Agentic-v2 형 행(system+user+assistant(tool_calls, reasoning)+tool…) |
| `judge_d2.py` | 보고서 심판(GLM): 주장 10개 지지 여부·구조·허위 출처. 채택 지지율 ≥0.8·허위 ≤1. 심판에는 수집 passage 전문(120k자)을 준다 |
| `out/d_loop.sh` | 본생성 루프(20분): 질문 작성 → D1 → D2 → 심판, 목표 D1 4k·D2 1k |

## P0 (2026-09-12)
- D1 50문항: 정답 일치 **34(68%)**, 검색 2~21회(중앙값 4), 턴당 사고 ≈660자. 오답은 상위 개념 답·음역 차이가 대부분. 정답이 뻔한 연쇄(마지막 홉 국가/통화 등, 기존 연쇄의 75%)를 발견해 빌더·질문기에 금지 규칙 추가.
- D2 20건: 검색 11~16회, 보고서 6~7천 자, 번호 인용. 첫 심판 2/20 → 원인은 심판 입력 절단(발췌 300자·24k) → 전문·120k자로 확대 후 지지율 중앙값 0.9, 기준 0.8 에서 19/20.
- 검색 품질: BM25 라 개체명 질의에서 목표 문서가 1위가 아닐 때가 있어 v2 색인에서 제목을 모든 passage 에 포함. 모델은 영어 질의가 실패하면 한국어로 전환하는 행동을 보인다.

## 실행
```
python3 search_server.py --index out/index/bm25_v2 --port 8600 &
D1_TARGET=4000 D2_TARGET=1000 bash out/d_loop.sh
```
