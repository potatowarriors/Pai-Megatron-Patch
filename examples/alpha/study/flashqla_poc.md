# FlashQLA PoC — GDN 커널 교체 검증 (2026-08-18)

FlashQLA(QwenLM, TileLang 기반 GDN 커널)를 alpha의 fla Triton 커널 대체 후보로 검증한다.
배경 검토 보고는 세션 기록 참조. 핵심 동기: LC-A/LC-B에서 gdn-cp head-split(CP=8이면
rank당 4헤드 × 전체 시퀀스)이 만드는 few-head 저활용 구간이 FlashQLA AutoCP의 설계 타깃.

## 상태

| 항목 | 결과 |
|---|---|
| GPU 적합성 | H100 = SM90, 지원 대상 ✅ |
| 격리 설치 | `/home/work/vidsearch/envs/flashqla_poc/pylibs` ✅ (시스템/유저 site 미오염) |
| torch 2.7.0a0+nv25.03 호환 | 전체 임포트 체인 성공 ✅ (`torch>=2.8` 핀은 우회 — 하단 참조) |
| `use_qk_l2norm_in_kernel=True` | FlashQLA 지원 확인 ✅ (autograd 래퍼 내 l2norm_fwd/bwd) |
| API | fla 0.4.x와 인자 완전 호환 + GVA 네이티브 지원 (k 16H/v 32H 직접 — repeat_interleave 불필요) |
| bwd (SM90) | 구현 완비 (SM120만 fwd-only) |
| **GPU 런타임 검증** | **미완 — 8×H100 전부 P3 학습 점유 중. 유휴 창구 대기** |

## 설치 재현

```bash
POC=/home/work/vidsearch/envs/flashqla_poc
pip download tilelang==0.1.9 apache-tvm-ffi==0.1.9 --no-deps -d wheels
pip install --no-deps --target $POC/pylibs wheels/*.whl
pip install --no-deps --target $POC/pylibs "z3-solver<4.15.5,>=4.13.0" torch-c-dlpack-ext
git clone https://github.com/QwenLM/FlashQLA && QLA_VERSION_SUFFIX=+poc \
  pip install --no-deps --no-build-isolation --target $POC/pylibs ./FlashQLA
```

- `--no-deps` 필수: setup.py의 `torch>=2.8`이 NGC torch(2.7.0a0, PEP440상 pre-release)를
  거부하지만, 코드가 실제 쓰는 torch API는 전부 2.7에 존재함을 확인했다
  (mark_dynamic, amp.custom_fwd/bwd, compiler.disable, cuda 기본, profiler 뿐).
  tilelang 자체의 Linux torch 요구는 버전 무제한.
- `torch_c_dlpack_ext`는 torch24~29 각각의 사전 빌드 .so 동봉 → torch27 바이너리 사용, ABI 문제 없음.
- **금지**: 어떤 단계에서도 `--user` 설치 또는 torch 재설치 금지 (NGC_ENV_REBUILD.md 오염 사례).

## 벤치마크 실행 (유휴 H100 확보 시)

```bash
cd /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch
PYTHONPATH=/home/work/vidsearch/envs/flashqla_poc/pylibs \
  python examples/alpha/study/flashqla_bench.py --device cuda:0
```

- 스크립트가 GPU 사용중이면 스스로 중단한다(>2GiB 점유 시). `--force`로만 무시 가능.
- 시나리오: `p3_today`(4K, b3), `lc_a_cp{1,4,8}`(32K), `lc_b_cp8`(128K), `corr_small`.
  CP 시나리오는 gdn-cp head-split 후 rank-local 형상(전체 seq × heads/cp)을 재현.
- 백엔드 변형: fla/qla × GQA-expanded(현행)·GVA-native, `auto_cp=False` 절연 측정.
- 정확도: fp32 oracle(`torch_chunk_gated_delta_rule`, Megatron-LM-251125) 대비 fwd+grad
  오차를 fla와 qla 각각 측정. AutoCP 최악 조건인 `g≈0`(무감쇠) 케이스 포함.
- 첫 호출 시간에 TileLang JIT 컴파일이 포함되므로 `first_call_s`로 별도 기록된다.

## 판정 기준 (제안)

1. 정확도: qla_vs_oracle 오차가 fla_vs_oracle 오차와 동일 자릿수 (bf16 노이즈 수준).
2. `g≈0` 케이스에서 qla_vs_fla fwd 오차 급증 없음 (AutoCP warmup 근사 안전성).
3. 성능: lc_a_cp8 / lc_b_cp8에서 fwd+bwd 유의미 개선 (블로그 주장: fwd 2–3×, bwd 2×).
4. 통과 시: `gated_deltanet.py:355` 호출부에 `ALPHA_GDN_BACKEND=flashqla` 가드 스왑 구현,
   LC-A 진입(stage 경계)에 채택. GVA-native 호출로 repeat_interleave 제거는 별도 커밋.

## 미해결 리스크

- TileLang JIT 산출물 캐시 위치/재컴파일 비용 — 16 rank 동시 첫 호출 시 컴파일 폭주 여부 확인 필요.
- `enable_fwd_cp_cache=True` 기본값의 메모리 영향 (bench의 peak_gb로 측정).
- varlen(cu_seqlens)은 batch=1 제약 — THD packing 미채택 상태라 당장 무관.

## 벤치 실행 결과 (2026-08-22, 게이트 §2 — main1 H100 단일 GPU)

결과 JSON: `flashqla_bench_results_20260822_141132.json`. (실행 참고: 251125
torch-oracle이 신형 fla의 `l2norm(dim=)` 시그니처를 가정 → 벤치 스크립트에
pure-torch l2norm 주입으로 해소.)

**판정 1·2 (정확도) — ✅ 통과**: qla_vs_oracle이 fla_vs_oracle과 전 항목 동일
자릿수(일반 조건 fwd mean 3.4e-4 vs 3.8e-4 — qla가 근소 우위). `g≈0` 최악
조건에서도 급증 없음 — 최악 grad 텐서 max err fla 130.3 vs **qla 68.2**
(조건 자체의 난이도이며 qla 고유 열화 아님).

**판정 3 (성능) — 조건부: LC-A 형상에서는 fla 유지, 채택 보류**:

| 시나리오 (rank-local 형상) | fla 최속 f+b | qla 최속 f+b | 승자 |
|---|---|---|---|
| p3_today (4K, full heads) | 3.42ms | **2.48ms** | qla 1.4× |
| lc_a_cp1 (32K, full heads) | 8.03ms | **6.42ms** | qla 1.25× |
| **lc_a_cp4 (32K, heads/4)** | **5.30ms** | 8.35ms | **fla 1.6×** |
| lc_a_cp8 (32K, heads/8) | **4.30ms** | 8.36ms | **fla 1.9×** |
| lc_b_cp8 (128K, heads/8) | 13.16ms | **6.01ms** | qla 2.2× |

패턴: TileLang(qla) 커널은 **CP head-split로 head 수가 줄면 backward 병렬성이
고갈**(8.4ms대 바닥 고정)되고, 시퀀스가 충분히 길면(128K) 회복한다. Triton(fla)은
head-split에 강함. **LC-A 채택 기준선이 CP4이므로 LC-A는 fla 유지가 정답** —
가드 스왑(§계획)은 보류. 재평가 시점: 128K 메모리 문제가 풀려 LC-B 후반이
128K 형상으로 가는 국면(그때 qla 2.2×는 유의미). 메모리는 전 시나리오에서
qla가 우위(lc_b_cp8 2.41 vs 3.14GB)였음을 부기.
