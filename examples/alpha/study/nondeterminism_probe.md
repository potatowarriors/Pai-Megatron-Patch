# alpha 풀모델 실행 간 비결정성 — 원천 규명 (2026-08-22)

**결론: 원천은 TE fused attention의 backward 하나다** (cuDNN FusedAttention,
기본 비결정 알고리즘). 검사한 다른 모든 커널 — dense matmul, embedding
scatter-add, causal_conv1d, **fla GDN**, TE RMSNorm — 은 반복 실행에서
비트 동일했다. 재현: `study/nondeterminism_probe.py`.

## 배경 — 왜 조사했나

Muon chunked offload A/B(MUON_OFFLOAD_BACKPORT.md S3)에서 오프로드 ON/OFF가
iter 3부터 갈라졌는데, **동일 구성 OFF↔OFF 재실행도 같은 지점에서 같은 규모로
갈라졌다** (analysis_24L mock 4런: 자기 산포 평균 상대 |Δ| 2.7e-3, iter 1–2만
전 런 bitwise 일치). 즉 bit-identical은 풀모델에서 성립 불가능한 판정 기준이고,
그 원인을 커널 단위로 특정할 필요가 있었다.

## 방법

같은 입력으로 같은 커널을 같은 프로세스에서 4~5회 실행해 gradient를 **비트
비교**(bf16→int16 뷰). atomicAdd 누적은 SM 스케줄링에 따라 덧셈 순서가 바뀌므로
(부동소수 비결합성) 비결정 커널은 이것만으로 드러난다. 비교는 전 쌍(pairwise) —
첫 실행이 autotune으로 혼자 특이할 수 있어서다.

## 결과 (H100, TE 2.9, NGC 25.03 스택)

| 커널 (backward) | 판정 | 비고 |
|---|---|---|
| dense matmul (대조군) | 결정적 | |
| embedding scatter-add (vocab 163,968) | 결정적 | 현 PyTorch는 정렬 기반 결정 경로 |
| causal_conv1d (GDN short conv) | 결정적 | dweight 포함 |
| **fla chunk_gated_delta_rule** (GDN 본체) | **결정적** | 4회 전 쌍 비트 동일 |
| TE RMSNorm (dgamma 행 누적) | 결정적 | |
| **TE fused attention** (GQA 16/2 · d256 · causal · S4096 = alpha 형상) | **비결정** | 매 실행 쌍 diff 2~5e-4 = 해당 크기에서 bf16 1-ulp |

### iter 1–2가 동일해 보인 이유

1-ulp 규모의 wgrad 차이는 fp32 master의 저비트에만 쌓이고, bf16 파라미터
캐스팅이 반올림으로 흡수한다. Muon NS 5회 반복 + momentum이 증폭한 뒤(iter 3)
bf16 경계를 넘어 loss에 가시화된다. 어텐션 6개 레이어만으로 전 모델 발산에 충분.

### `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`은 해결책이 아니다 (실측)

플래그를 꺼도 여전히 비결정 — 백엔드/엔진만 바뀌고, 오히려 더 큰 크기
스케일(0.3~5.3, dv 쪽 1-ulp)의 차이가 났으며 run2가 단독 특이(엔진 재선택
추정)했다. 진짜 결정론이 필요하면 결정적 attention 백엔드 고정 +
torch deterministic까지 필요하나, 학습엔 불필요하다.

## 함정 2건 (첫 프로브의 오판 — 재발 방지)

1. **비물리 합성 입력이 fla를 무고하게 만들었다**: 정규화 안 된 k, sigmoid 전
   스케일의 beta로 NaN grad가 발생 → `nan != nan`이라 torch.equal이 False →
   "NON-DET" 오판. 물리적 입력 범위(정규화 k, beta∈(0,1), g=log-sigmoid 스케일)로
   재검하니 4회 비트 동일.
2. **파이썬 `max(0.0, nan)`은 0.0을 반환한다**: maxdiff 리포트가 NaN을 침묵시켜
   "비트 불일치인데 diff 0"이라는 모순 출력으로 이어졌다. NaN 개수를 별도
   보고하고 diff는 `nan_to_num` 후 취한다.

## 실무 함의

- **A/B 등가 판정은 "동일 구성 재실행의 자기 산포 포락선 이내"로 한다.**
  실측 포락선: analysis_24L 20-iter에서 평균 상대 |Δ| ~2.7e-3, max ~1e-2.
  적용 사례: 오프로드 A/B(교차 편차 ≤ 자기 산포 → PASS), THD+CP 풀스택
  CP{1,2,4} 등가(1.2e-4 = 포락선의 1/20 → PASS).
- **비트 검증이 필요한 주장은 결정론적 유닛으로 내린다**: 단순 MLP + dist_muon
  골든(5스텝 torch.equal), mixer 단독 CP 등가(diff 0.0) 등 — 어텐션이 경로에
  없으면 풀 결정성이 나온다.
- resume 검증에는 "재개 첫 iter loss 정확 일치"가 유효하다 — 비결정은 momentum이
  개입하는 optimizer step 커널 경로에서 유입되므로, 같은 파라미터를 로드한 첫
  forward는 결정적이다.
- 과거 기록과의 정합: STAGE2_CURRICULUM_LOG의 "실행 간 비결정성" 관찰,
  LC 게이트 CP 검증이 |Δ|≤6e-5(forward-only는 훨씬 조임) 기준을 쓴 것 모두
  이 원천 하나로 설명된다.
