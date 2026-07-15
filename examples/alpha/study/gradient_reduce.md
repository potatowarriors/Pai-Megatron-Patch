# Gradient Reduction 완전 이해 — 분산 학습 스터디 노트

"gradient reduce"가 무엇이고, 왜 생기며, 어떻게 멀티노드 학습 속도를 결정하는 통신 비용이 되는지를
바닥부터 설명합니다. alpha(15B GDN+MoE 하이브리드, Muon, EP=8) 기준이지만 개념 자체는 일반적입니다.
연계 문서: [`../docs/THROUGHPUT_INVESTIGATION.md`](../docs/THROUGHPUT_INVESTIGATION.md) §5.

---

## 0. 한 문장 멘탈 모델

한 번의 학습 **step** = `forward → backward → [gradient 동기화] → optimizer.step()`.

이 `[gradient 동기화]` 안에 **사람들이 늘 헷갈리는 두 가지 다른 연산**이 숨어 있습니다:

| | **Gradient accumulation (누적)** | **Gradient reduction (all-reduce)** |
|---|---|---|
| 무엇을 가로질러? | **microbatch** (시간축, GPU 1개 안에서) | **data-parallel GPU 복제본** (공간축) |
| 연산 | 로컬 버퍼에 **합(sum)** | GPU 간 **평균(average)** |
| 통신? | **없음** (완전 로컬) | **있음** — 이게 바로 네트워크 비용 |
| 왜 생기나? | **메모리** (배치가 안 들어감) | **data parallelism** (GPU마다 다른 데이터) |

> **당신 질문에 대한 직답:** gradient *reduce*는 gradient *accumulation*에서 생기는 게 **아닙니다.**
> accumulation은 microbatch를 합치는 **로컬·무통신** 연산이고, reduction은 **data parallelism** 때문에
> 생기는 **GPU 간 평균**입니다. 둘은 같은 버퍼(`main_grad`)를 공유하기 때문에 하나처럼 느껴질 뿐,
> 서로 없어도 성립합니다 — **GPU 1개 + accumulation = reduce 없음**; **GPU 여러 개 + microbatch 1개 =
> 여전히 reduce 있음**.

아래에서 한 겹씩 쌓아 올리겠습니다.

---

## 1. 가장 단순한 경우: GPU 1개, 배치 1개 (accumulation·reduction 둘 다 없음)

```
   batch ──► forward ──► loss ──► backward ──► gradient g ──► optimizer.step(g)
```

`g = ∂loss/∂weights`. optimizer가 `g`로 weight를 살짝 움직입니다. 끝. 버퍼도, 통신도 없습니다.

---

## 2. Gradient ACCUMULATION 추가 (여전히 GPU 1개) — 메모리 트릭

배치가 너무 커서 GPU 메모리에 안 들어간다고 합시다. **microbatch**로 쪼개 하나씩 처리하면서,
gradient를 버퍼에 **합산**한 뒤 업데이트합니다:

```
   main_grad = 0                          # 모델 param 크기만 한 버퍼
   for mb in microbatches:                # 예: 64개
       g_mb = backward(forward(mb))
       main_grad += g_mb                  # ← ACCUMULATION (로컬 합, 네트워크 없음)
   optimizer.step(main_grad / N_microbatches)
```

- 큰 배치 하나와 **수학적으로 동일**하지만 메모리 예산 안에서 돕니다.
- **완전히 GPU 1개 안의 로컬 연산**입니다. **여기서 통신은 전혀 없습니다.**
- alpha가 이걸 합니다: GBS 1536, micro-batch 3, EP=8 → `1536/(8×3) = 64`개 microbatch를 전부
  로컬 `main_grad`에 합산.

**이게 "gradient accumulation"입니다. 네트워크 트래픽이 0인 점에 주목하세요.**

---

## 3. DATA PARALLELISM 추가 (여러 GPU) — *여기서* reduction이 탄생

이제 모델을 **N개 GPU**에 올립니다. 각 GPU는 동일한 weight 복사본을 갖지만 **서로 다른 데이터**를
처리합니다. 그래서 backward 후 **각 GPU의 gradient가 다릅니다**:

```
   GPU 0 (데이터 조각 A) ──► g₀
   GPU 1 (데이터 조각 B) ──► g₁
   GPU 2 (데이터 조각 C) ──► g₂
   GPU 3 (데이터 조각 D) ──► g₃
```

하지만 네 GPU가 동일한 복제본으로 남으려면 **똑같은** 업데이트를 적용해야 합니다. 올바른 업데이트는
*전체* 배치의 gradient = per-GPU gradient들의 **평균**을 씁니다:

```
   g_avg = (g₀ + g₁ + g₂ + g₃) / 4
```

`g_avg`를 계산해 **모든** GPU에 한 부씩 나눠주는 것 — 이게 바로 **all-reduce**입니다:

```
        g₀   g₁   g₂   g₃
          \   |   |   /
           all-reduce            ← GPU 간 통신 ("gradient reduction")
          /   |   |   \
        모든 GPU가 g_avg 보유
   → 모든 GPU가 동일한 optimizer step 적용
```

**이 all-reduce가 "gradient reduce" 과정입니다. data parallelism 때문에 생깁니다** — GPU마다 다른
데이터로 다른 gradient를 냈으니 조율해야 하는 것이죠. **microbatch accumulation을 썼는지 여부와는
무관합니다.**

- GPU 1개 (§2): accumulation 있음, reduce **없음**.
- GPU 여러 개, 각자 microbatch 1개: **reduce 있음**, accumulation 없음.
- alpha: **둘 다** — 64개 microbatch를 로컬 누적한 *뒤*, 그 결과를 DP 간 all-reduce.

---

## 4. 실제 스텝에서 둘이 어떻게 맞물리나

```
   ── 각 GPU에서 병렬로 ──────────────────────────────────────────────
   main_grad = 0
   for mb in 64 microbatches:
       main_grad += backward(forward(mb))       # (§2) 로컬 accumulation, 통신 없음
   ── 그 다음, 모든 DP GPU를 가로질러 ────────────────────────────────
   all_reduce(main_grad)                         # (§3) 네트워크: 복제본 간 평균
   optimizer.step(main_grad)                     # 어디서나 동일한 업데이트
```

둘은 **버퍼 하나(`main_grad`)를 공유**합니다: accumulation이 로컬로 *채우고*, reduction이 GPU 간에
*동기화*합니다. 이 공유 버퍼 때문에 헷갈리기 쉽지만, 비용이 다른 별개 연산입니다 (accumulation =
연산/메모리; reduction = 네트워크).

---

## 5. all-reduce가 실제로 데이터를 옮기는 법 (ring)

왜 reduction의 네트워크 비용이 **모델 크기**에 비례할까요?

순진한 all-reduce("모두가 GPU 0에 보내고, GPU 0이 합쳐서 다시 뿌림")는 GPU 0에서 병목이 생깁니다.
표준 **ring all-reduce**는 그걸 피합니다: gradient를 N조각으로 쪼개 **reduce-scatter**(각 GPU가 한
조각의 합을 소유)를 하고, **all-gather**(각 GPU가 완성된 조각을 방송)를 이어 합니다. 결과적으로 각
GPU가 보내고 받는 양 ≈ `2·(N-1)/N × (gradient 크기)` — **큰 N에서 N과 무관하고, gradient(≈모델)
크기에 비례**합니다.

> **핵심:** reduction은 매 step **대략 모델 하나 분량의 gradient**를 네트워크로 옮깁니다. 그 볼륨을
> 링크 대역폭으로 나눈 게 통신 시간입니다. alpha는 그 볼륨이 **~16B params** (§8) → dtype에 따라
> 32~64 GB (§7).

만나게 될 두 변종:
- **Full all-reduce** (일반 DDP, 그리고 alpha의 **Muon**이 쓰는 방식): 모든 GPU가 전체 평균
  gradient를 갖게 됨. Muon은 distributed optimizer가 아니라 전체 gradient가 로컬에 필요.
- **Reduce-scatter + all-gather 분리** (ZeRO / "distributed optimizer"): 각 GPU가 gradient·optimizer
  상태의 *조각*만 보유. alpha는 이걸 **안 씀** (Muon 비호환).

---

## 6. reduction을 backward와 overlap — 왜 때론 "공짜"인가

backward는 gradient를 **레이어 단위**로, 마지막 레이어부터 첫 레이어 순으로 만들어냅니다. backward가
다 끝날 때까지 기다릴 필요가 없습니다: 한 레이어의 gradient가 나오자마자, 앞쪽 레이어들을 계산하는
동안 그 레이어를 **버킷 단위**로 all-reduce하기 시작하면 됩니다.

```
   backward:   [L48 grad][L47 grad][L46 grad] ........... [L1 grad]
   all-reduce:        └─reduce L48─┘└─reduce L47─┘ ....... └reduce L1┘   (compute와 겹침)
                                                              ▲ 이 꼬리만 노출됨
```

- 네트워크가 backward가 gradient를 만드는 속도보다 **빠르게** 버킷을 밀어낼 수 있으면
  (`통신시간 < backward시간`), reduction이 거의 완전히 숨음 → near-linear 스케일링.
- 링크가 **느리면** (`통신시간 > backward시간`), reduction이 backward를 넘쳐 **노출**됨 → step 시간에
  그대로 더해짐 → 나쁜 스케일링.

**이 하나의 비교 — `all-reduce 시간` vs `backward window` — 이 멀티노드 인터커넥트 이야기의 전부입니다.**
Megatron은 `--overlap-grad-reduce`로 이 overlap을 켭니다 (alpha 기본 ON).

---

## 7. reduction의 dtype: fp32 vs bf16

`main_grad` 버퍼에는 **dtype**이 있고, 그게 두 가지를 동시에 결정합니다:
1. 로컬 accumulation(64 microbatch 합)의 **정밀도**, 그리고
2. all-reduce가 옮기는 **바이트 수**.

| dtype | param당 바이트 | accumulation 정밀도 | all-reduce 볼륨 |
|---|---|---|---|
| **fp32** (alpha 기본) | 4 | 높음 (다항 합에 안전) | 2배 |
| **bf16** (`--grad-reduce-in-bf16`) | 2 | 상대오차 ~0.4% | **½** |

alpha는 **기본이 fp32**입니다 (Megatron이 bf16 학습 시 `accumulate_allreduce_grads_in_fp32`를
자동으로 켬 — `arguments.py:803-809` — `--grad-reduce-in-bf16`을 명시하지 않는 한). bf16으로 바꾸면
**네트워크 트래픽이 절반**이 되고, 이게 느린 인터커넥트 멀티노드의 핵심 레버입니다 (throughput 문서 §5).
우리는 (analysis_24L mock, 64-microbatch 스트레스) bf16 reduction이 alpha에서 **수치적으로 안전**함을
검증했습니다 — zero-mean 노이즈, 발산 없음. **Muon이 gradient의 *크기(magnitude)*를 버리기**(방향만
orthogonalize) 때문에 bf16의 크기 반올림이 씻겨나가는 덕입니다.

---

## 8. MoE / Expert-Parallel 반전 (alpha 고유)

일반 data parallelism에선 모든 GPU가 *전체* 모델을 갖고 있어서, reduction을 **계층적으로** 할 수
있습니다: 먼저 노드 안에서 빠른 NVLink로 합치고, 노드 간에는 한 부만 보냅니다.

alpha는 **Expert-Parallel (EP=8)**입니다: 192개 expert가 *샤딩*되어 각 GPU가 **서로 다른** 24개를
갖습니다. 그래서 expert gradient는 **노드 내에서 reduce가 불가능**합니다 (합칠 게 없음 — 각 GPU의
expert가 유일). 모든 expert gradient가 **다른 노드의 data-parallel 쌍둥이**로 건너가야 합니다:

```
   Dense params (attention, GDN, embeddings)  ── 복제됨   ──► NVLink로 먼저 reduce, 노드 간 1번 건넘
   Expert params (192 experts, EP-sharded)    ── 전부 유일 ──► 모든 shard가 노드 경계를 건너야 함
```

alpha (2노드): ~1.53B dense (1번 건넘) + **14.5B expert (전부 건넘)** ≈ **16B params가 매 step
건넘** → **~64 GB (fp32) / ~32 GB (bf16)**. 이것이 alpha의 노드 간 비용이 "매 step 거의 전체
gradient"에 가까운 이유이고, fp32→bf16 절반이 그토록 중요한 이유입니다.

**철칙:** **EP 그룹은 노드 하나 안에** 두세요 (그 자체 all-to-all이 microbatch마다 발생 — NVLink 전용).
추가 노드는 data-parallel 복제로만 쓰고, EP를 느린 링크로 절대 쪼개지 마세요.

---

## 9. Megatron 코드 어디에 있나 (자습용 지도)

| 개념 | 파일 |
|---|---|
| grad 버퍼, accumulation, 버킷 all-reduce, overlap | `megatron/core/distributed/param_and_grad_buffer.py`, `distributed_data_parallel.py` |
| `grad_reduce_in_fp32` 플래그 (버퍼 dtype) | `megatron/core/distributed/distributed_data_parallel_config.py:11` |
| bf16의 fp32 자동 활성 + `--grad-reduce-in-bf16` | `megatron/training/arguments.py:803-809`, `:2731` |
| overlap 토글 | `--overlap-grad-reduce` (alpha 설정: `training/stage*.yaml`) |
| `main_grad`을 소비하는 곳 | optimizer (`megatron/core/optimizer/`, Muon은 `optimizer/muon.py`) |

한 step을 추적해보세요: `main_grad`(accumulation 대상) 검색 → `start_grad_sync` / `finish_grad_sync`
(버킷 all-reduce) → optimizer가 `param.main_grad`를 읽는 곳.

---

## 10. 종합 — alpha 한 스텝, 주석 달기

```
  GBS 1536, micro-batch 3, EP=8, 2노드 (DP=2)
  ┌─ GPU마다 ─────────────────────────────────────────────────────────────────┐
  │ main_grad(fp32) = 0                                                          │
  │ for mb in 64 microbatches:                                                  │
  │     main_grad += backward(forward(mb))     # ACCUMULATION (로컬, ~48초)      │
  │                                            #   ↑ 여기서 all-reduce가 overlap  │
  └─────────────────────────────────────────────────────────────────────────────┘
  ── 2노드를 가로질러 ──────────────────────────────────────────────────────────
  all_reduce(main_grad)   # REDUCTION: ~16B params → 64 GB fp32 / 32 GB bf16
                          #   노드 간 링크(~0.9 GB/s 실측) 통과
                          #   fp32 → ~71초 (> 48초 backward → 노출 → ~1.5×)
                          #   bf16 → ~36초 (< 48초 backward → 숨음 → ~1.9×)
  optimizer.step(main_grad)   # Muon: orthogonalize + 업데이트 (모든 GPU 동일)
```

> **⚠ 정정 (2026-07-13, H100×2 실측):** 위 그림의 "여기서 all-reduce가 overlap" 주석은 틀렸다.
> Megatron은 **마지막 microbatch에서만** grad sync를 시작할 수 있다 — 누적이 끝나기 전엔 reduce할
> 합이 없기 때문 (`schedules.py:630`의 no_sync 래핑 + `param_and_grad_buffer.py:511`의
> `is_last_microbatch` 게이트). 따라서 overlap 창은 microbatch 1개의 backward(~0.7초)뿐이고,
> 노출 통신은 step당 상수다: 실측 fp32 ~78초 / bf16 ~46초 (GBS 3072·6144에서 동일).
> 결과: 2노드 = 1노드 대비 **fp32 0.89× / bf16 1.15× (GBS 3072), bf16 1.45× (GBS 6144)**.
> bf16의 "36초 < 48초 → 숨음 → 1.9×" 결론은 성립하지 않는다. 실측 표와 원인 분석은
> [`../docs/THROUGHPUT_INVESTIGATION.md`](../docs/THROUGHPUT_INVESTIGATION.md) §5 MEASURED 블록 참조.

---

## 11. 스터디 체크리스트

- [ ] *단일* GPU + accumulation 실행이 왜 all-reduce를 **안** 하는지 한 호흡에 설명하기.
- [ ] 2-GPU가 각자 microbatch **1개**만 써도 왜 all-reduce가 필요한지 설명하기.
- [ ] ring all-reduce를 스케치하고, 비용이 왜 모델 크기 정도(× N 아님)인지 논증하기.
- [ ] reduction이 "공짜"가 되는 조건 (`통신시간 < backward시간`) 말하기.
- [ ] alpha의 expert gradient가 왜 노드 내 reduce가 안 되는지 (EP 샤딩) 설명하기.
- [ ] alpha의 노드 간 볼륨을 fp32 vs bf16으로 계산하고 H100×2 speedup 예측하기.

**더 읽을거리:** NVIDIA Megatron-LM DDP 문서; ring all-reduce 원조 글 (Baidu, 2017); ZeRO 논문
(reduce-scatter/all-gather 샤딩); alpha의 구체적 멀티노드 수치는 `../docs/THROUGHPUT_INVESTIGATION.md` §5.
