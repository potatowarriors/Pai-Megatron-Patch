# Muon chunked optimizer-state offload — 표적 백포트 로그 (2026-08-22 착수)

**목적**: upstream PR #6244(merged 08-18, `dev`)의 chunked optimizer-state/master-weight
offload를 251125 스냅샷에 표적 백포트 → **128K@CP8의 Muon 바닥짐(~45GB) 해소**
(배경·산술: `gdn_cp_port.md` 분석노트 2). 브랜치: `feature/muon-chunked-offload`.
사용자 지시: **단계별로 하나씩 검증하며 진행**.

## Stage 0 — 정찰 (✅ 08-22)

- PR 구성: 신규 자기완결 모듈 `cpu_offloading/chunked_optimizer_state_offload.py`
  (1,013줄, 의존 = torch + `log_single_rank`뿐) + 통합 훅(optimizer.py +258,
  optimizer_config +86, layer_wise +87, arguments +82, training +60, checkpointing +26).
  PR이 삭제하는 구 오프로더는 우리 스냅샷에 없음 → 순수 추가.
- 전제 검증 ✅: eopt 0.4.0a0 Muon 상태 = `momentum_buffer` 단일 fp32 full-size
  (upstream 계약과 일치); 우리 layer_wise도 자식을 Float16 래핑(`fp32_from_float16_groups`).
- 드리프트: layer_wise 314줄(ours) vs 1081(upstream), optimizer 1421 vs 2054 —
  **훅 그대로 적용 불가 → 신규 모듈은 원본 이식, 통합부는 우리 구조에 재작성.**
- 추출 교훈: PR diff에서 "+"만 취하는 추출은 **수정 파일에선 컨텍스트 라인이 소실**됨
  — 머지된 `dev` 브랜치에서 파일 직접 fetch가 정본.

## Stage 1a — 코어 모듈 이식 (✅ 08-22)

- `chunked_optimizer_state_offload.py`를 dev 원본 그대로 설치, import 검증.
- PR 테스트(dev, 1,258줄)를 `tests/unit_tests/test_chunked_offload_s1.py`로 설치
  (S2 의존 임포트만 try-가드). **순수 모듈 테스트 15종 PASSED** (1 GPU):
  스테이징 뷰 정렬·전송 스트림 순서·청크 플래닝·상태 로딩/스키마 교체·계획 우선순위.
- 잔여 11종은 `OptimizerConfig` 신규 필드 + `MegatronOptimizer` subset-step API 요구
  → S2 전제.

## Stage 2 — 통합 플러밍 (✅ 08-22)

- **optimizer.py 13훅** 이식: import + `MegatronOptimizer` 클래스 기본값(ChainedOptimizer가
  base `__init__`을 안 부르므로 클래스 속성 필수) + 라이프사이클 API 12종
  (`enable_chunked_optimizer_state_offload` ~ `start_param_sync_for_bucket_group_subset`)
  + MixedPrecision `prepare_grads` 멱등 prefetch + step 분기(`offloader.step()`) +
  `_copy_{main,model}_params` ensure/assert 가드 + Float16 sharded_state_dict
  init/sync + `load_state_dict_without_device_cast` 분기 + FP32Optimizer step 가드 +
  ChainedOptimizer 스트림 공유·10종 팬아웃·`_before_child_step` 훅.
- **optimizer_config.py**: 신규 필드 3종(`chunked_optimizer_state_offload`,
  `optimizer_state_offload_chunk_size_mb`, `optimizer_state_offload_fraction`) +
  최소 `__post_init__` 검증(chunk≥0, fraction∈[0,1], `optimizer_cpu_offload` 배타).
  upstream의 muon-모드 매트릭스 검증은 **의도적 미이식** — 이 스냅샷은
  `muon.py:366`이 `config.optimizer='adam'`으로 재기록해 config 이름 기반 검증이
  불가능, S3 통합 지점(layer_wise)에서 타입 가드로 대체.
- **distrib_optimizer.py 7훅(D1~D7)**: `__init__` 말미 enable 배선(master 분리 조건 =
  `shard_fp32_from_float16_groups` OR precision-aware `master_weights`; state_dtypes =
  precision-aware면 (exp_avg, exp_avg_sq) dtype, 아니면 fp32×2) + load/sharded_state_dict
  오프로드 분기 + **DistOpt가 override하는 복사 3메서드에 ensure/assert 재삽입**
  (base 가드를 우회하므로 — `test_chunk_plan_and_initial_master_offload`의
  `reload_model_params()` 후 `param.is_cuda` 실패로 검출). upstream mxfp8 gather-drain
  블록은 미사용이라 스킵.
- **검증 결과**:
  - `test_chunked_offload_s1.py`: **31 passed / 21 skipped / 0 failed** — 골든
    `test_training_matches_non_offloaded_optimizer`(5스텝 학습 = 비오프로드
    레퍼런스와 파라미터 일치) 포함 기능 그룹 전체 통과. skip 21 = 멀티랭크 1 +
    비이식 9개 함수(deprecated alias 2 · upstream 전용 config 매트릭스 2 ·
    arguments.py 훅 대기 5 — 각각 사유를 `@pytest.mark.skip`으로 파일에 명기).
  - 무회귀: `test_muon_optimizer.py` 20종 + `test_step_batch_size_schedule.py` 8종
    **28 passed** (오프로드 코드 존재·비활성 상태).
- 교훈: 앵커 치환 스크립트는 `count==1` assert로 원자성 확보 — 동일 docstring이
  4개 클래스에 존재(`init_state_fn`)해 클래스 고유 문맥으로 앵커를 확장해야 했음.

## Stage 3 — layer_wise/arguments/training 통합 (✅ 08-22)

- **layer_wise_optimizer.py** (✅): `__init__`에 enable 블록 — Muon 자식(`type is
  TensorParallelMuon`)만 오프로드, **Adam 자식은 GPU 상주로 스킵** (upstream은 Adam을
  형제 DistOpt로 라우팅해 비-Muon 자식을 reject — 우리 dist_muon 체인은 [muon, adam*]
  단일 layer_wise라 의도적 분기; INFO 로그로 스킵 수 명시). master =
  `fp32_from_float16_groups`, state_dtypes=(fp32,), deferred lifecycle =
  state_prefetch_to_step만 (fp8 param gather 없음). 오버라이드 3종:
  `prefetch_optimizer_state_for_gradient_finalization`(전 자식 master + 관리 자식 첫
  state 청크), `_before_child_step`(idx 0에서 관리 자식 state prefetch),
  `_managed_optimizer_state_offload_child_indices`. 순환 import 회피: muon.py가 이
  모듈을 로드하므로 TensorParallelMuon은 지연 import.
- **arguments.py** (✅): 플래그 3종 + validate_args 블록(chunk 0 경고 · muon 단독
  거부("requires the LayerWise") · dist_muon/DistOpt 외 경로 거부 · optim save/load 시
  torch_dist 요구 · async_save 거부 · full_iteration CUDA graph 거부). deprecated
  `--offload-optimizer-states` alias는 미이식(구 오프로더가 이 스냅샷에 없음).
- **training.py train_step** (✅): finalize_model_grads 래퍼(중복 래핑 방지 attr 스탬프,
  prefetch가 grad finalization과 오버랩) + rerun 루프 진입부 offload_for_forward
  (pre-forward param-sync/mxfp8 지연 분기는 충실 이식하되 우리 스택에선 항상 비활성).
- **checkpointing.py** (✅): save_checkpoint 진입 안전 불변식(async/비-torch_dist 저장
  런타임 거부) + layer_wise torch-format 저장의 `no_save_optim` 가드.
- **검증 (유닛)**: 전 스위트 **74 passed / 20 skipped / 0 failed** —
  `test_chunked_offload_s3_layerwise.py` 신규 8종(층위 mock 6 + GPU 2: dist_muon 실체인
  Muon-자식-단독 활성/step 후 momentum·master CPU 상주, **5스텝 골든 `torch.equal`
  bit-identical**) + s1의 training_args 5종 해제·적응(7케이스) + muon 20/step-GBS 8 무회귀.
- **analysis_24L mock ON/OFF A/B** (✅ 08-22, 4×H100 EP=4/DP=4, 16 iter, chunk 256MB):
  - **메모리: max-alloc 47.40 → 32.68 GB (−14.7 GB, −31%)** — Muon 자식의 momentum+master
    오프로드 실효 확인.
  - **오버헤드: iter당 6.3–6.9s → 6.8–7.3s (~+5–7%)**.
  - **등가성 — 판정 기준 교정**: 최초 계획한 bit-identical은 **풀모델에선 성립 불가능한
    기준**이었다. 동일 구성 OFF↔OFF 재실행도 iter 3부터 분기(평균 상대 |Δ| 2.7e-3,
    iter 1–2는 4개 런 전부 bitwise 일치 — 비결정성이 momentum≠0인 두 번째 step부터
    유입; GDN triton/MoE 경로의 알려진 실행 간 비결정성, STAGE2_CURRICULUM_LOG 참조).
    4런 6쌍 비교(iters 3–16): **ON↔OFF 교차 편차(1.7–2.7e-3)가 OFF↔OFF·ON↔ON
    자기 산포(2.7e-3/3.3e-3)와 동급 이하** → 계통 편향 없음 = PASS. 오프로더 수학
    자체의 정확성은 결정론적 모델의 유닛 골든(5스텝 `torch.equal`)이 담보.
  - **QK-Clip×offload 계약**: clip_qk는 `param.main_param.data`(fp32 master)에 오프로더
    수명주기 밖 쓰기를 한다. chunked step 종료 시 master는 GPU 상주이고 D2H는 다음
    iter의 offload_for_forward에서 일어나므로 현행 시퀀스에서 clip 쓰기는 보존되지만,
    train_step에 `ensure_master_weights_for_param_sync()`를 clip 직전에 명시해 향후
    수명주기 변경(지연 master offload 등)에도 안전하게 고정 (상주 시 no-op). LC preset은
    qk-clip 제거 확정이라 실전 경로엔 이 상호작용 자체가 없음.
## Stage 4 — 체크포인트 round-trip (✅ 08-22)

- **유닛 (신규 `tests/unit_tests/dist_checkpointing/test_chunked_offload_s4.py`, 3종)**:
  GPT 모델 dist_muon(진짜 layer_wise 경로)로 OFF→ON / ON→OFF / ON→ON 스위치 전부
  save→load→save 후 `load_plain_tensors` **비트 동일** + dest-ON에서 로드 후 CPU
  canonical(master·momentum 전부 CPU) 보존 확인.
  - 발견: 기존 `test_layer_wise_optimizer_save_load`는 헬퍼가
    `layer_wise_distributed_optimizer=False` 기본값으로 호출해 **이름과 달리 비-layer_wise
    체인을 테스트**하고 있었음 — `utils.setup_model_and_optimizer`에 `muon_layer_wise`
    opt-in kwarg 추가(기본 False, 기존 테스트 무변화)로 우회. 기존 save_load[True-1-1]
    무회귀 확인.
- **풀모델 (analysis_24L mock, 신규 preset `configs/training/profile_optsave.yaml`)**:
  - S 런: 오프로드 ON + iter 8 optimizer-state 저장(81GB torch_dist) 성공 —
    `synchronize_for_checkpoint` 경로 실증.
  - R-OFF: ON-저장 ckpt에서 오프로드 OFF 재개 — **재개 첫 iter(9) loss 2.339895E+00
    원본과 정확히 일치**, 이후 노이즈 범위.
  - R-ON: 동일 ckpt에서 ON 재개(`load_state_dict_without_device_cast` 경로) —
    역시 iter 9 정확 일치, max-alloc 32.7GB로 오프로드 유지.
  - 스크래치 ckpt(~324GB)는 검증 후 삭제(로그 보존).

## Stage 5~6 (계획)

- S5: **표적 검증 — 128K@CP8** (gdn-cp 브랜치와 결합 필요: CP는 그 브랜치에만 있음
  → S5는 gdn-cp 머지 후 rebase 시점에) max-alloc ~47GB 예상 확인. LC preset은
  qk-clip 제거이므로 clip×offload 상호작용 없음. chunk 크기·fraction 튜닝은 이때.
- S6: vendored patch 최종화 + gdn_cp_port.md 128K 경로 갱신 + megatron_patch/CLAUDE.md
  Muon 호환표("CPU Offloading ❌" → chunked offload 지원) 갱신.

## 규약

- 서브모듈 수정 커밋마다 `backends/submodule_patches/Megatron-LM-251125.patch` 재생성
  (`git -C <sub> add -A && git -C <sub> diff --cached a6d86a6da --binary > patch && reset`).
- QGKV-split 로컬 패치(muon.py)와의 충돌면: 이번 표적은 optimizer/layer_wise 계층 —
  muon.py 자체는 무수정 유지 예정, S2에서 재확인.
