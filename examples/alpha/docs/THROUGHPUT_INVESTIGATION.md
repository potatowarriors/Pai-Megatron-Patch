# Alpha Training Throughput — Full Investigation (2026-06/07)

Canonical record of the alpha_v2 throughput investigation before Stage-2. Every claim
here is A/B-measured on the H100×4 analysis node (`analysis_24L`, EP=4, GBS=1536, MBS=3,
mock data) unless noted; the per-token fractions transfer to prod 8×H100 EP=8. Baseline
= `CUDA_DEVICE_MAX_CONNECTIONS=8` = **171.3 TFLOP/s/GPU**. Companion figures live in this
dir (`alpha_comm_compute.png`, `alpha_timeline.png`, `fp8_critical_path.png`,
`fp8_compute_lane.png`). See also memories `alpha-throughput-levers-swept`, `fp8-compute-rejected`.

---

## 1. Executive summary

- **alpha is near its single-node throughput optimum.** Of every lever tried, only two help,
  both cheap: `CUDA_DEVICE_MAX_CONNECTIONS=32` (**+3.4%**, bit-identical) and removing one
  redundant GDN `.contiguous()` (**+1.6%**, needs loss-eq check). Stacked = **+4.3%** (≈2.6
  days on a 60-day run).
- **The big levers are all blocked by the regime** (small model / single node / GDN-hybrid):
  fp8 −10.8%, recompute-drop −44.5%, offload −13.9%, MBS=6 −22.7%, comm-overlap not
  supported by MambaModel, DeepEP broken build. See §3.
- **"alpha 170 vs Qwen3-30B 277 TFLOP/s" is mostly a metric artifact** — alpha does ~⅓ the
  FLOPs/token (linear GatedDeltaNet + FFN 512 + 15B). tokens/sec is competitive. §4.
- **Real headroom (~10-15%) needs a regime change**: bigger GEMMs (→fp8 viable), Blackwell
  (→mxfp8+fusion), or multi-node with a fast interconnect. Over **10 GbE**, multi-node DP
  scaling is *not* worthwhile — the gradient all-reduce dominates. §5.

---

## 2. Measured baseline — where the time goes (nsys, hardware kernel-time)

At production GBS the GPU kernel-work splits **72% computation / 28% communication**
(compute = 2.5× comm). Within compute: **GEMM 48%** (= 35% of all work), elementwise/copy
21%, MoE permute 12%, Mamba/GDN 10%, norm 4%, attn 3%. Comm is dominated by the **EP
all-to-all (86% of comm, 80% EXPOSED** ≈ 21% of all work, on the critical path).

- alpha is **compute-DOMINANT**, not "comm-bound" (an earlier loose framing). See `alpha_comm_compute.png`.
- The real timeline (`alpha_timeline.png`) shows compute nearly fills the lane; comm fires in
  bursts that mostly overlap, with ~16% exposed.
- **⚠ nsys 'idle' (~24%) is an instrumentation artifact** — it is ~constant at GBS=96 and
  GBS=768 (so it is not per-step host overhead), and non-nsys throughput jumps +29% at
  GBS=1536. Trust kernel-times, not the nsys idle.

---

## 3. Levers tested — full results

| Lever | Result | Verdict |
|---|---:|---|
| `CUDA_DEVICE_MAX_CONNECTIONS` 8→16→32 | 171.3→173.6→**177.2** | ✅ **+3.4%**, bit-identical, adopt |
| GDN L298 `.contiguous()` removal (`gated_deltanet.py:298`) | **174.0** | ✅ **+1.6%**, needs loss-eq validation |
| stacked (conn=32 + GDN L298) | **178.7** | ✅ **+4.3%** |
| fp8 blockwise / tensorwise | 152.8 / 111.9 | ❌ −10.8% / −16% (small GEMM below break-even) |
| recompute-drop `moe` (→ forces MBS 3→1) | 95.1 | ❌ **−44.5%** (MBS 3→1 = −54% ≫ recompute gain +21%) |
| fine-grained offload `moe_act`/`expert_fc1` | 147.5 | ❌ −13.9% (PCIe exposed, worse than recompute) |
| MBS=6 | 132.5 | ❌ −22.7% (**MBS=3 is the PEAK**: MBS 1/3/6 = 78/171/132) |
| cuDNN RMSNorm (`NVTE_NORM_*_USE_CUDNN=1`) | 171.7 | ❌ +0.2% neutral (norm already TE-fused, 4% of compute) |
| `--overlap-moe-expert-parallel-comm` | crash | ❌ BLOCKED — MambaModel has no `build_schedule_plan` (see §6) |
| DeepEP (`--moe-token-dispatcher-type flex`) | import error | ❌ broken build (`ncclTeamWorld` vs bundled NCCL 2.26.5) + prior single-node loss |
| FSDP (torch-fsdp2 / megatron-fsdp) | — | ❌ asserted-out with EP / incompatible with dist_muon |
| GDN kernel fusion ("JIT fusion 2-5%") | — | ❌ phantom — Pai GDN already MORE fused than upstream (Triton gated-norm + in-kernel L2norm) |
| CUDA graphs / shared-expert-overlap / Muon-batched-NS / tp-comm-overlap | — | ❌ rejected earlier (QK-Clip host-sync / cross-stream barriers / GBS-invariant / TP=1 no-op) |

---

## 4. Why fp8 fails, and why TFLOP/s misleads

**fp8 (small GEMM).** fp8 GEMM *is* faster (−7%) but the compute lane = all non-comm GPU
kernels, and fp8's cast_transpose/pad kernels run there too: GEMM −887 ms but +4,150 ms
quant tax → net **+13% LONGER compute lane** (`fp8_compute_lane.png`). An FFN-width sweep
(num_experts×FFN held constant) confirmed the break-even: 192×512 = −10.8%, 96×1024 = −7.7%,
48×2048 = −0.5%. alpha's FFN=512 is below break-even; DSV3's hidden 7168/FFN 2048 is above.
Fused-quant kernels (NVIDIA CuTe GroupGemm+Quantize) would remove the tax but need TE≥2.15 /
Megatron≥26.04 / Blackwell (mxfp8 is Blackwell-only) — re-test fp8 only in that regime.

**TFLOP/s artifact.** alpha = **8.2 GFLOP/token** (24L) vs Qwen3-30B **23 GFLOP/token** — ~⅓,
because GatedDeltaNet is LINEAR (O(n)) attention (Megatron counts it via cheap
`mamba_layer_flops`) + FFN 512 (<768) + 15B (<30B). TFLOP/s rewards FLOP-density, so it is
structurally low for a linear-attention hybrid. **tokens/sec is the fair metric** (alpha
~21k/GPU at 24L, competitive/faster). Real efficiency gap after de-biasing ≈ 10-15%, from
architecture (GDN MFU) + stack (NeMo/DeepEP/TE-25.09 vs Pai/251125) — mostly not cheaply
capturable, and the NeMo stack cannot run GDN.

---

## 5. Multi-node scaling & interconnect

Single node is at its optimum, so more throughput ⇒ more nodes. But the interconnect decides
whether that scales.

**Data-parallel scaling needs a gradient all-reduce every step.** Precise for alpha (from
baseline_48L config; verified against the logged 3.26 GB/rank param count):
- Params ~16B = **14.5B expert (EP=8 sharded) + 1.53B dense.** Expert grads CANNOT be
  intra-node-reduced (each GPU holds different experts) → all 14.5B cross to the DP-twin node;
  dense (1.53B) intra-node-reduces then crosses once. **≈16B params cross per step.**
- **alpha all-reduces grads in FP32 by default** (arguments.py:803-809 auto-enables
  `accumulate_allreduce_grads_in_fp32` for bf16 training; alpha doesn't set `--grad-reduce-in-bf16`).
  → **~64 GB/step (fp32)**, or **~32 GB with `--grad-reduce-in-bf16`.**

**Measured interconnect:** **7.24 Gbit/s ≈ 0.9 GB/s** (iperf, on A100 nodes — VERIFY on the
actual H100 nodes with `nccl-tests all_reduce_perf`; iperf ≥ NCCL busbw). H100 backward window ≈ 48 s.

| grad dtype | inter-node vol | all-reduce @0.9 GB/s | weak scaling (GBS→3072) | strong (GBS=1536) |
|---|---:|---:|---:|---:|
| **FP32 (default)** | ~64 GB | ~71 s (> 48 s → exposed) | **~1.5×** | **~0.9× (slower than 1 node!)** |
| **bf16 (`--grad-reduce-in-bf16`)** | ~32 GB | ~36 s (< 48 s → **hides**) | **~1.9× (near-linear)** | **~1.5×** |
| for reference: 100 GbE / IB (~11–25 GB/s) | either | ~1.5–3 s | trivially hides, near-linear | near-linear |

**THE multi-node lever = `--grad-reduce-in-bf16`** (halves inter-node volume). It flips the
all-reduce from *exposed* (fp32 71 s > 48 s backward) to *hidden* (bf16 36 s < 48 s) — that single
dtype choice is the whole difference between ~1.5× and ~1.9× on a 0.9 GB/s link.

**Numerical safety — VALIDATED (2026-07-01, analysis_24L mock, 64 microbatch = prod stress, same seed):**
fp32 vs bf16 grad-reduce A/B — bf16 is SAFE: iter-1/2 loss bit-identical (clean A/B), 0 NaN/crash,
max |Δloss| 1.6e-2 (~0.3%, reconverges to 9e-5 by iter15), max |Δgnorm| 1.7% (only at tiny gnorm).
Perturbation is **zero-mean noise** (sign oscillates, no bias), non-diverging through a harsh 12→0.05
loss collapse — consistent with **Muon washing out magnitude noise via spectral normalization.**
Remaining gate (needs Stage-1 ckpt + real data): real-data resume ~few-hundred-step loss-curve A/B.

**Hard rules for multi-node:**
1. **EP=8 must stay within a node.** MoE all-to-all is per-microbatch per-MoE-layer (24 × 64 ≈
   1,500 A2As/step) — NVLink-only. Add nodes as DP replicas; never span EP across the slow link.
2. **Weak scaling (GBS→3072)**, not strong (GBS=1536 halves the backward window → all-reduce
   exposed → ~0.9–1.5×). GBS=3072 is already in the step-batch schedule.
3. **Faster GPUs hurt slow-interconnect scaling.** H100's ~48 s backward vs A100's ~120 s makes the
   same all-reduce relatively more exposed — an A100 "looks fine" does NOT transfer to H100.
4. Apply `--grad-reduce-in-bf16` at the **Stage boundary** (dtype isn't checkpoint state → resume-safe,
   no loss discontinuity).

**Bottom line:** default fp32 grad-reduce → H100×2 only ~1.5×. **With `--grad-reduce-in-bf16`
(validated numerically safe) + weak scaling + EP-within-node, H100×2 reaches ~1.8–1.9× (adequate)
IF the H100 interconnect gives ≥~0.7 GB/s all-reduce busbw** — measure it before committing. 100 GbE
(ideally RoCE) or IB makes it trivially near-linear.

**⚠ MEASURED on the H100×2 Backend.AI cluster (2026-07-13, mock, 4-iter steady state) — supersedes
the projections above:**

| run | config | s/iter | tokens/s | vs 1 node |
|---|---|---:|---:|---:|
| A | 1-node, GBS 1536 | ~62 | 101.5k | 1.00× |
| B | 2-node, GBS 3072, fp32 reduce | ~140 | 89.9k | **0.89× (slower than 1 node)** |
| C | 2-node, GBS 3072, **bf16 reduce** | ~108 | 116.5k | **1.15×** |
| D | C + `CUDA_DEVICE_MAX_CONNECTIONS=32` + `NCCL_SOCKET_NTHREADS=4` | ~109 | ~115k | 1.15× (no gain) |
| E | 2-node, GBS 6144, bf16 reduce | ~171 | 146.9k | **1.45×** |

Why the ~1.9× projection was wrong: the grad all-reduce can only START during the LAST microbatch's
backward — grads must finish accumulating first (`schedules.py:630` wraps all but the last microbatch
in `no_sync`; `param_and_grad_buffer.py:511` gates bucket dispatch on `is_last_microbatch`). The
overlap window is therefore ONE microbatch's backward (~0.7 s), not the full 48 s accumulation loop
assumed above. Exposed comm is ~constant per step: measured **78 s (fp32) / 46–47 s (bf16)** —
identical at GBS 3072 and 6144 (effective inter-node bw ~0.7 GB/s during training vs 1.07 GB/s in a
bare 16-GPU nccl-tests all-reduce). Scaling improves with GBS only because compute grows around the
fixed comm term. This cluster has NO InfiniBand (all HCAs admin-Disabled; 9.1 Gbit/s TCP VXLAN
overlay is the hard ceiling) and NCCL socket-thread tuning gained nothing.

---

## 6. Why `--overlap-moe-expert-parallel-comm` is blocked (and the cost to unblock)

It is the one lever aimed squarely at the ~16% exposed EP A2A, but it requires
`model.build_schedule_plan` (defined only on `GPTModel`). The schedule plan
(`model_chunk_schedule_plan.py`) decomposes **each layer into attn→post_attn→moe_dispatch→
mlp→moe_combine** — i.e. it assumes every layer is a GPT transformer layer with attention +
MoE. alpha's hybrid has **three single-function layer types** (GDN-mixer M / attn-mixer * /
MoE-mlp -) that do not fit. There is **zero mamba/hybrid awareness** in the schedule infra.

Implementing it for MambaModel = **multi-week, high-risk**: a new hybrid schedule-plan class +
per-layer-type nodes + MambaStack forward re-wiring + handling QK-Clip's host-side max-logit
stash (the same host-sync that broke CUDA graphs) and the GDN Triton kernels, plus exact
loss-equality validation. Estimated gain ~5–8% (hide part of the exposed A2A on the 24 MoE
layers). **ROI is negative before Stage-2** (≈3 weeks + run-corruption risk vs ~5 days saved).
Roadmap: wait for upstream to add hybrid schedule support, then it is a one-flag win.

---

## 7. Recommendations for the 2-month Stage-2 run

1. **Adopt now (free, safe):** `CUDA_DEVICE_MAX_CONNECTIONS=32` → **+3.4%** (≈2 days). `train.sh`
   already honors the env override.
2. **Validate then adopt:** GDN L298 `.contiguous()` removal → **+1.6%**. Gate on a 1-iter
   real-data loss-byte-equality A/B vs baseline (mock showed +1.6%, NaN 0).
3. **Do not pursue:** fp8, recompute-drop, offload, higher MBS, DeepEP rebuild, or hand-porting
   comm-overlap — all measured negative or blocked in this regime.
4. **For real scaling (add nodes):** set **`--grad-reduce-in-bf16`** (validated safe §5; halves
   inter-node volume 64→32 GB), use **weak scaling** (GBS→3072), keep **EP=8 within-node**, and
   apply at a stage boundary. Then H100×2 ≈ 1.8–1.9× *if* the interconnect gives ≥~0.7 GB/s
   all-reduce busbw (measured 7.24 Gbit/s on A100 — verify on H100 with nccl-tests). 100 GbE/IB
   makes it trivially near-linear; ≤10 GbE without bf16-reduce is only ~1.5× (not worth it).
5. **Re-open fp8** only when on Blackwell or TE≥2.15/Megatron≥26.04 (fused quant removes the tax).
