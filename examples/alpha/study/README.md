# 스터디 노트 (study/)

분산 학습·최적화 개념을 **바닥부터 공부하기 위한** 한글 자료 모음입니다.
프로젝트 결정 기록(`../docs/`)과는 성격이 다릅니다 — 여기는 *배우기 위한* 문서, `docs/`는
*결정·측정을 남기는* 문서입니다.

## 목록

| 문서 | 주제 | 한 줄 |
|---|---|---|
| [gradient_reduce.md](gradient_reduce.md) | Gradient reduction | accumulation(로컬 합) ≠ reduction(GPU 간 평균, all-reduce); reduction은 data parallelism에서 생기며 멀티노드 통신 비용을 결정 |

## 연계 문서 (docs/ — 결정·측정 기록)

- [`../docs/THROUGHPUT_INVESTIGATION.md`](../docs/THROUGHPUT_INVESTIGATION.md) — alpha throughput 최적화 전수조사 + 멀티노드 스케일링 분석 (스터디 노트가 설명하는 개념의 *실측 적용*)
