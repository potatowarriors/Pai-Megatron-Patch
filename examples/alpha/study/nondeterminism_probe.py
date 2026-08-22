"""alpha 스택 실행 간 비결정성 — 커널 단위 실증 프로브.

같은 입력으로 같은 커널을 같은 프로세스에서 N회 반복 실행해 gradient를
비트 비교한다. atomicAdd 기반 누적은 SM 스케줄링에 따라 덧셈 순서가 바뀌므로
(부동소수 비결합성) 비결정 커널은 반복 실행만으로 비트 차이가 드러난다.

결과·해석 정본: study/nondeterminism_probe.md (2026-08-22).

사용:
    CUDA_VISIBLE_DEVICES=0 NVIDIA_PYTORCH_VERSION=25.06 python study/nondeterminism_probe.py
    # attention만 결정론 플래그 반사실 비교:
    NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 python study/nondeterminism_probe.py --attn-only

주의(첫 작성 시 실제로 밟은 함정 두 가지):
  - fla에 비물리 입력(정규화 안 된 k, sigmoid 전 스케일의 beta)을 주면 NaN grad가
    나오고, nan != nan 이라 torch.equal이 False가 되어 "비결정"으로 오판된다.
    게다가 파이썬 max(0.0, nan)은 0.0을 반환해 maxdiff 리포트마저 침묵한다.
    → 입력을 물리적 범위로 만들고, NaN 개수를 별도 보고하고, 비트 비교는
    int 뷰로 한다.
  - 비교는 run1-vs-나머지만이 아니라 전 쌍(pairwise)으로 한다 — 첫 실행이
    autotune 등으로 혼자 특이할 수 있다.
"""

import argparse
import os

import torch


def pairwise_report(name, outs, names=None):
    """전 쌍 비트 비교 + NaN-안전 maxdiff 보고. 반환: 전 쌍 비트 동일 여부."""
    all_same = True
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            for k, (a, b) in enumerate(zip(outs[i], outs[j])):
                if a.dtype == torch.bfloat16:
                    bits_ne = (a.view(torch.int16) != b.view(torch.int16)).sum().item()
                elif a.dtype == torch.float32:
                    bits_ne = (a.view(torch.int32) != b.view(torch.int32)).sum().item()
                else:
                    bits_ne = int(not torch.equal(a, b))
                if bits_ne:
                    all_same = False
                    md = (a.float() - b.float()).abs().nan_to_num(0).max().item()
                    label = names[k] if names else f"t{k}"
                    print(
                        f"  {name} run{i+1} vs run{j+1} {label}: "
                        f"bit-diff={bits_ne} maxdiff={md:.3e}",
                        flush=True,
                    )
    nan_counts = [int(torch.isnan(t).sum()) for t in outs[0]]
    print(
        f"{name:42s} {'DETERMINISTIC' if all_same else 'NON-DET'} "
        f"(nan={nan_counts})",
        flush=True,
    )
    return all_same


def run_n(fn, n=4):
    outs = []
    for _ in range(n):
        torch.cuda.synchronize()
        outs.append(fn())
        torch.cuda.synchronize()
    return outs


def probe_attention():
    """TE fused attention bwd — alpha attention 형상 (GQA 16/2, d256, causal, S4096)."""
    import transformer_engine.pytorch as te

    S, B, Hq, Hkv, Dh = 4096, 1, 16, 2, 256
    attn = te.DotProductAttention(
        num_attention_heads=Hq,
        kv_channels=Dh,
        num_gqa_groups=Hkv,
        attn_mask_type="causal",
        qkv_format="sbhd",
    )
    q = torch.randn(S, B, Hq, Dh, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(S, B, Hkv, Dh, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(S, B, Hkv, Dh, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(S, B, Hq * Dh, device="cuda", dtype=torch.bfloat16)

    def step():
        for t in (q, k, v):
            t.grad = None
        attn(q, k, v).backward(g)
        return tuple(t.grad.clone() for t in (q, k, v))

    env = os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "<unset>")
    pairwise_report(
        f"TE fused attention bwd (NVTE_ALLOW_NONDET={env})",
        run_n(step, 5),
        names=("dq", "dk", "dv"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)

    if args.attn_only:
        probe_attention()
        return

    # ── 대조군: dense matmul bwd (결정적이어야 프로브가 유효) ──────────────
    a = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(2048, 8192, device="cuda", dtype=torch.bfloat16)
    g0 = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)

    def mm():
        a.grad = None
        (a @ b).backward(g0)
        return (a.grad.clone(),)

    pairwise_report("control: dense matmul bwd", run_n(mm))

    # ── embedding bwd (scatter-add) — alpha vocab 규모 ─────────────────────
    V, H, N = 163968, 2048, 4096 * 3
    emb = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    idx = torch.randint(0, V, (N,), device="cuda")
    ge = torch.randn(N, H, device="cuda", dtype=torch.bfloat16)

    def emb_bwd():
        emb.grad = None
        torch.nn.functional.embedding(idx, emb).backward(ge)
        return (emb.grad.clone(),)

    pairwise_report("embedding bwd (scatter-add)", run_n(emb_bwd))

    # ── causal_conv1d bwd (GDN short conv) ─────────────────────────────────
    try:
        from causal_conv1d import causal_conv1d_fn

        B2, D, L = 2, 4096, 4096
        x = torch.randn(B2, D, L, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(D, 4, device="cuda", dtype=torch.float32, requires_grad=True)
        gc = torch.randn(B2, D, L, device="cuda", dtype=torch.bfloat16)

        def conv_bwd():
            x.grad = None
            w.grad = None
            causal_conv1d_fn(x, w, None, activation="silu").backward(gc)
            return (x.grad.clone(), w.grad.clone())

        pairwise_report("causal_conv1d bwd", run_n(conv_bwd), names=("dx", "dweight"))
    except ImportError as e:
        print(f"causal_conv1d skipped: {e}")

    # ── fla chunk_gated_delta_rule bwd — 물리적 입력 범위 필수 (모듈 docstring) ──
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        B3, T, Hh, K = 1, 4096, 8, 128
        q = torch.randn(B3, T, Hh, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = (
            torch.nn.functional.normalize(
                torch.randn(B3, T, Hh, K, device="cuda", dtype=torch.float32), dim=-1
            )
            .bfloat16()
            .requires_grad_(True)
        )
        v = torch.randn(B3, T, Hh, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gg = (-torch.rand(B3, T, Hh, device="cuda", dtype=torch.float32) * 0.1).requires_grad_(True)
        beta = (
            torch.rand(B3, T, Hh, device="cuda", dtype=torch.bfloat16) * 0.9 + 0.05
        ).requires_grad_(True)
        go = torch.randn(B3, T, Hh, K, device="cuda", dtype=torch.bfloat16)

        def gdn_bwd():
            for t in (q, k, v, gg, beta):
                t.grad = None
            out, _ = chunk_gated_delta_rule(q, k, v, g=gg, beta=beta, output_final_state=False)
            out.backward(go)
            return tuple(t.grad.clone() for t in (q, k, v, gg, beta))

        pairwise_report(
            "fla chunk_gated_delta_rule bwd",
            run_n(gdn_bwd),
            names=("dq", "dk", "dv", "dg", "dbeta"),
        )
    except ImportError as e:
        print(f"fla skipped: {e}")

    # ── TE RMSNorm bwd (dgamma 행 누적) ────────────────────────────────────
    try:
        import transformer_engine.pytorch as te

        norm = te.RMSNorm(2048, eps=1e-6).cuda().bfloat16()
        xn = torch.randn(4096 * 3, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gn = torch.randn(4096 * 3, 2048, device="cuda", dtype=torch.bfloat16)

        def norm_bwd():
            xn.grad = None
            norm.weight.grad = None
            norm(xn).backward(gn)
            return (xn.grad.clone(), norm.weight.grad.clone())

        pairwise_report("TE RMSNorm bwd", run_n(norm_bwd), names=("dx", "dgamma"))
    except ImportError as e:
        print(f"TE RMSNorm skipped: {e}")

    # ── 주범: TE fused attention bwd ───────────────────────────────────────
    probe_attention()


if __name__ == "__main__":
    main()
