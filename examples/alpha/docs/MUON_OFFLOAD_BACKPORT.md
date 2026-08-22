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

## Stage 3~6 (계획)

- S3: layer_wise 통합(+87 패턴 재작성: TensorParallelMuon 타입 가드, fp32_from_float16
  master 추출, enable 호출; fp8/overlap 분기는 우리 스택 미사용이라 생략) + arguments/
  training 최소 훅 → analysis_24L mock ON/OFF A/B **bit-identical + 메모리·오버헤드 실측**.
- S4: 체크포인트 round-trip (오프로드 ON 저장 ↔ OFF 재개 양방향 bit-identical).
- S5: **표적 검증 — 128K@CP8** (gdn-cp 브랜치와 결합 필요: CP는 그 브랜치에만 있음
  → S5는 gdn-cp 머지 후 rebase 시점에) max-alloc ~47GB 예상 확인.
- S6: vendored patch 최종화 + gdn_cp_port.md 128K 경로 갱신.

## 규약

- 서브모듈 수정 커밋마다 `backends/submodule_patches/Megatron-LM-251125.patch` 재생성
  (`git -C <sub> add -A && git -C <sub> diff --cached a6d86a6da --binary > patch && reset`).
- QGKV-split 로컬 패치(muon.py)와의 충돌면: 이번 표적은 optimizer/layer_wise 계층 —
  muon.py 자체는 무수정 유지 예정, S2에서 재확인.
