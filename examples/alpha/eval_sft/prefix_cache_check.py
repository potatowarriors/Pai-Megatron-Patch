#!/usr/bin/env python3
"""하이브리드 prefix caching 정확도·적중 게이트 (2026-09-17).

vLLM 0.25.1 의 GDN prefix caching(`--mamba-cache-mode align`)은 "experimental" 이다. fleet 에 켜기 전에 두 가지를 잰다.

  A. 정확도 (**teacher-forced**): OFF 서버에서 greedy 로 뽑은 토큰열 G 를 양쪽 서버에 똑같이 강제하고, 위치 k 마다
     "다음 토큰" 분포(top-5 logprob)를 대조한다. free-running greedy 비교는 근소한 동률이 한 번 뒤집히면 이후가 전부
     달라져 캐시 오류와 실행 간 비결정(배치 구성·커널 선택)을 구분하지 못한다 — 1차 시도에서 캐시와 무관한 단발
     요청도 3번째 토큰에서 갈렸다(Δlogprob 0.022). 프롬프트는 **토큰 id 로 직접** 보낸다(/v1/completions, prompt=ids)
     → 양쪽이 비트 동일한 입력을 받고, ON 서버는 긴 접두사를 캐시에서 이어 계산한다(검증하려는 경로).
     판정(refit_verifier 기준 준용, **캐시를 실제로 타는 turn2_long_cached 에만 적용**): 위치별 top-1 일치 ≥ 14/16,
     OFF 가 고른 토큰의 mean(exp|Δlogprob|) < 1.05, max|Δlogprob| < 0.5. short·turn1_long 은 캐시를 안 타는 참조값이다 —
     2026-09-17 실측에서 short 는 OFF↔ON 1.090 / ON↔ON 대조군 1.082 로 캐시와 무관하게 기준을 넘었고(짧은 개방형 답변의
     고엔트로피 위치 + 서버 간 배치·커널 비결정), 긴 케이스는 turn1 1.026 / turn2(캐시) 1.040 / 대조군 turn2 1.045 로
     캐시가 추가 오차를 만들지 않았다.
  B. 적중: ON 서버 `/metrics` 의 prefix_cache_hits_total·prompt_tokens_cached_total 증분으로 2턴 요청이 실제로 적중했는지.
     판정: 2턴 cached 토큰 ≥ (1턴 프롬프트 토큰 − 2×block).

대조군: `--off-url` 에 **다른 ON 서버**를 주면 캐시 ON 끼리의 산포(실행 간 비결정 바닥)를 같은 잣대로 잰다.

사용 (서버를 각각 띄운 뒤):
  python3 eval_sft/prefix_cache_check.py --off-url http://localhost:8011/v1 --on-url http://localhost:8012/v1 \
      --hf-dir <hfmodel> [--prompt-tokens 60000] [--gen 64] [--positions 16] [--block 544] --out results.json
  --off-url 을 생략하면 B(적중)만 잰다. exit 0 = PASS.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path


def post(url: str, path: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metrics(url: str) -> dict[str, float]:
    base = url.rstrip("/")
    base = base[: -len("/v1")] if base.endswith("/v1") else base
    with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
        txt = r.read().decode()
    out: dict[str, float] = {}
    for l in txt.splitlines():
        m = re.match(r"^(vllm:[a-zA-Z_]+)(\{[^}]*\})?\s+([-+0-9.eE]+)", l)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def build_long_text(tok, n_tokens: int, seed_paths: list[Path]) -> str:
    """리포 소스·문서를 이어 붙여 n_tokens 토큰짜리 긴 문맥을 만든다."""
    buf: list[str] = []
    total = 0
    i = 0
    while total < n_tokens and i < 2000:
        p = seed_paths[i % len(seed_paths)]; i += 1
        try:
            t = p.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        if t.strip():
            buf.append(f"\n\n### FILE {p.name} (part {i})\n" + t)
            total = len(tok(''.join(buf)).input_ids)
    return tok.decode(tok(''.join(buf)).input_ids[:n_tokens])


def _tid(s: str) -> int:
    return int(s.split(":", 1)[1]) if s.startswith("token_id:") else -1


def greedy_ids(url: str, ids: list[int], n: int) -> tuple[list[int], list[float]]:
    r = post(url, "/completions", {"model": "alpha", "prompt": ids, "max_tokens": n, "temperature": 0.0, "top_p": 1.0,
                                   "seed": 0, "logprobs": 1, "return_tokens_as_token_ids": True,
                                   "skip_special_tokens": False})
    lp = r["choices"][0]["logprobs"]
    return [_tid(t) for t in lp["tokens"]], list(lp["token_logprobs"])


def next_dist(url: str, ids: list[int]) -> dict[int, float]:
    r = post(url, "/completions", {"model": "alpha", "prompt": ids, "max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
                                   "seed": 0, "logprobs": 5, "return_tokens_as_token_ids": True,
                                   "skip_special_tokens": False})
    top = r["choices"][0]["logprobs"]["top_logprobs"][0]
    return {_tid(k): v for k, v in top.items()}


def forced_compare(url_a: str, url_b: str, prefix: list[int], forced: list[int], positions: int) -> dict:
    """위치 k = 0..len(forced) 균등 분할 지점에서 다음 토큰 분포를 대조."""
    ks = sorted({int(round(i * len(forced) / max(1, positions - 1))) for i in range(positions)})
    ks = [k for k in ks if k <= len(forced)]
    top1 = 0; ratios = []; maxd = 0.0; miss = 0; rows = []
    for k in ks:
        ids = prefix + forced[:k]
        da = next_dist(url_a, ids); db = next_dist(url_b, ids)
        a1 = max(da, key=da.get); b1 = max(db, key=db.get)
        top1 += (a1 == b1)
        ref = forced[k] if k < len(forced) else a1   # OFF 가 고른 토큰 (마지막 위치는 A 의 top-1)
        if ref in da and ref in db:
            d = abs(da[ref] - db[ref]); ratios.append(math.exp(d)); maxd = max(maxd, d)
        else:
            miss += 1
        rows.append({"k": k, "top1_a": a1, "top1_b": b1, "lp_a": da.get(ref), "lp_b": db.get(ref)})
    n = len(ks)
    out = {"positions": n, "top1_agree": top1, "ref_missing_in_top5": miss,
           "mean_exp_abs_dlogprob": (sum(ratios) / len(ratios)) if ratios else float("nan"),
           "max_abs_dlogprob": maxd, "rows": rows}
    out["pass"] = bool(top1 >= n - 2 and miss == 0 and out["mean_exp_abs_dlogprob"] < 1.05 and maxd < 0.5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--on-url", required=True)
    ap.add_argument("--off-url", default=None)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=60000)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--positions", type=int, default=16)
    ap.add_argument("--block", type=int, default=544)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf_dir, trust_remote_code=True)
    repo = Path(__file__).resolve().parents[3]
    seeds = sorted((repo / "megatron_patch").rglob("*.py"))[:400] + sorted((repo / "examples/alpha/docs").glob("*.md"))
    long_text = build_long_text(tok, a.prompt_tokens, seeds)

    def render(msgs):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)

    sys_msg = {"role": "system", "content": "You are a careful engineer. Answer concisely."}
    short = render([sys_msg, {"role": "user", "content": "Explain in three sentences why prefix caching helps multi-turn agents."}])
    turn1_msgs = [sys_msg, {"role": "user", "content": long_text + "\n\nSummarize what the last file above does in two sentences."}]
    turn1 = render(turn1_msgs)
    n_long = len(turn1)
    print(f"[pc] 긴 문맥 프롬프트 {n_long} 토큰", flush=True)
    result: dict = {"prompt_tokens_long": n_long, "block": a.block, "cases": {}, "hit": {}}

    # ---- ON 서버: 1턴(캐시 생성) → 2턴(캐시 적중 경로), 적중 계측 ----
    m0 = metrics(a.on_url)
    g1_on, _ = greedy_ids(a.on_url, turn1, a.gen)
    m1 = metrics(a.on_url)
    turn2_msgs = turn1_msgs + [{"role": "assistant", "content": tok.decode(g1_on, skip_special_tokens=True)},
                               {"role": "user", "content": "Now list three concrete risks in that file, one line each."}]
    turn2 = render(turn2_msgs)
    g2_on, _ = greedy_ids(a.on_url, turn2, a.gen)
    m2 = metrics(a.on_url)
    q2 = m2.get("vllm:prefix_cache_queries_total", 0) - m1.get("vllm:prefix_cache_queries_total", 0)
    h2 = m2.get("vllm:prefix_cache_hits_total", 0) - m1.get("vllm:prefix_cache_hits_total", 0)
    c2 = m2.get("vllm:prompt_tokens_cached_total", 0) - m1.get("vllm:prompt_tokens_cached_total", 0)
    h1 = m1.get("vllm:prefix_cache_hits_total", 0) - m0.get("vllm:prefix_cache_hits_total", 0)
    hit_ok = c2 >= (n_long - 2 * a.block)
    result["hit"] = {"turn1_hits": h1, "turn2_queries": q2, "turn2_hits": h2, "turn2_cached_tokens": c2,
                     "required": n_long - 2 * a.block, "pass": bool(hit_ok)}
    print(f"[pc] 적중: 1턴 hits {h1:.0f} · 2턴 hits {h2:.0f}/{q2:.0f} cached_tokens {c2:.0f} (요구 ≥ {n_long - 2*a.block}) → {'PASS' if hit_ok else 'FAIL'}", flush=True)

    ok = hit_ok
    if a.off_url:
        # OFF 서버의 greedy 토큰열을 강제 접두사로 쓴다 (turn2 는 캐시된 접두사 위에서 이어 계산되는 경로).
        for name, prefix in (("short", short), ("turn1_long", turn1), ("turn2_long_cached", turn2)):
            forced, _ = greedy_ids(a.off_url, prefix, a.gen)
            cmp = forced_compare(a.off_url, a.on_url, prefix, forced, a.positions)
            gated = name == "turn2_long_cached"
            cmp["gated"] = gated
            result["cases"][name] = cmp
            if gated:
                ok = ok and cmp["pass"]
            tag = ("PASS" if cmp["pass"] else "FAIL") if gated else ("ref-ok" if cmp["pass"] else "ref-over")
            print(f"[pc] {name:18s} top1 {cmp['top1_agree']}/{cmp['positions']} mean_exp|Δlp|={cmp['mean_exp_abs_dlogprob']:.4f} "
                  f"max|Δlp|={cmp['max_abs_dlogprob']:.3f} miss={cmp['ref_missing_in_top5']} → {tag}", flush=True)
    result["pass"] = bool(ok)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"[pc] 판정: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
