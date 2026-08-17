# 스터디 노트 (study/)

분산 학습·최적화 개념을 **바닥부터 공부하기 위한** 한글 자료 모음입니다.
프로젝트 결정 기록(`../docs/`)과는 성격이 다릅니다 — 여기는 *배우기 위한* 문서, `docs/`는
*결정·측정을 남기는* 문서입니다.

## 목록

| 문서 | 주제 | 한 줄 |
|---|---|---|
| [gradient_reduce.md](gradient_reduce.md) | Gradient reduction | accumulation(로컬 합) ≠ reduction(GPU 간 평균, all-reduce); reduction은 data parallelism에서 생기며 멀티노드 통신 비용을 결정 |

## 검증·규명 기록 (study/ 내 실측 문서)

| 문서 | 한 줄 |
|---|---|
| [diloco_pilot.md](diloco_pilot.md) | DiLoCo 2노드 파일럿 전체 실측·검증 기록 (2026-07) |
| [2node_project_report.md](2node_project_report.md) | 2노드 프로젝트 종합 보고 |
| [mirror_loss_aliasing.md](mirror_loss_aliasing.md) | 노드 간 거울상 loss 시소 규명 — 짝/홀 샤딩 × blend 세차운동, `DILOCO_SHARD_BLOCK` 수정 (2026-08-17). 재현: [mirror_loss_repro.py](mirror_loss_repro.py) |
| [netbench/](netbench/) | 노드 간 네트워크 실측 (IB 부재 확인) |

## 연계 문서 (docs/ — 결정·측정 기록)

- [`../docs/THROUGHPUT_INVESTIGATION.md`](../docs/THROUGHPUT_INVESTIGATION.md) — alpha throughput 최적화 전수조사 + 멀티노드 스케일링 분석 (스터디 노트가 설명하는 개념의 *실측 적용*)
