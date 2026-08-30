"""변환된 HF 체크포인트에 `generation_config.json` 을 쓰고 eos 정합성을 검사한다.

`docs/SFT_BENCHMARKS.md` §7 게이트 **G1**. 2026-08-30 사고의 재발 방지 장치다.

배경: MG→HF 변환기는 `config.json`(모델 골격)과 tokenizer 만 남기고
`generation_config.json` 을 만들지 않았다. `config.json` 의 `eos_token_id` 는
사전학습 관례대로 `<|endoftext|>`(0) 에 머무는데, SFT 챗 템플릿이 턴을 끝내는 토큰은
`<|im_end|>`(3) 이다. 그래서 서버가 턴 종료를 인식하지 못해 max_tokens 까지 계속
생성했고, 벤치 전 항목이 무효가 됐다 (`docs/KNOWN_ISSUES.md` 2026-08-30).

이 스크립트는 tokenizer 에서 챗 종료 토큰을 **찾아서** eos 집합에 넣는다. 하드코딩된
id 를 믿지 않는다 — tokenizer 가 바뀌면 검사도 같이 바뀌어야 하기 때문이다.

사용:
    python3 emit_generation_config.py <HF_DIR>            # 생성 + 검사
    python3 emit_generation_config.py <HF_DIR> --check    # 검사만 (변경 없음)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# SFT 챗 템플릿이 어시스턴트 턴을 끝낼 때 쓰는 토큰 후보. 앞쪽이 우선.
CHAT_END_CANDIDATES = ("<|im_end|>", "<|endoftext|>")
# 사전학습 문서 구분자 — 항상 eos 에 남겨 둔다(안전망).
DOC_EOD = "<|endoftext|>"

DEFAULTS = {"do_sample": True, "temperature": 1.0, "top_p": 0.95}


def _added_tokens(hf_dir: Path) -> dict[str, int]:
    """tokenizer_config.json 의 added_tokens_decoder → {content: id}."""
    tc = hf_dir / "tokenizer_config.json"
    if not tc.exists():
        return {}
    data = json.loads(tc.read_text())
    out = {}
    for tid, meta in (data.get("added_tokens_decoder") or {}).items():
        content = meta.get("content")
        if content:
            out[content] = int(tid)
    return out


def _pad_id(tokens: dict[str, int], fallback: int | None) -> int | None:
    for name in ("<|pad|>", "<pad>"):
        if name in tokens:
            return tokens[name]
    return fallback


def build(hf_dir: Path) -> tuple[dict, list[str]]:
    """generation_config 내용과 문제 목록을 만든다."""
    problems: list[str] = []
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        return {}, [f"config.json 없음: {cfg_path}"]

    cfg = json.loads(cfg_path.read_text())
    tokens = _added_tokens(hf_dir)
    if not tokens:
        problems.append("tokenizer_config.json 의 added_tokens_decoder 를 읽지 못함")

    eos: list[int] = []
    chat_end = None
    for name in CHAT_END_CANDIDATES:
        if name in tokens:
            chat_end = name
            eos.append(tokens[name])
            break
    if chat_end is None:
        problems.append(
            f"챗 종료 토큰을 찾지 못함 (후보 {CHAT_END_CANDIDATES}). "
            "tokenizer 가 바뀌었다면 CHAT_END_CANDIDATES 를 갱신할 것."
        )

    if DOC_EOD in tokens and tokens[DOC_EOD] not in eos:
        eos.append(tokens[DOC_EOD])

    # config.json 의 기존 eos 도 흡수 (누락 방지)
    legacy = cfg.get("eos_token_id")
    for v in (legacy if isinstance(legacy, list) else [legacy]):
        if isinstance(v, int) and v not in eos:
            eos.append(v)

    if not eos:
        problems.append("eos_token_id 를 하나도 만들지 못함")

    gen = {"eos_token_id": eos, **DEFAULTS}
    pad = _pad_id(tokens, cfg.get("pad_token_id"))
    if pad is not None:
        gen["pad_token_id"] = pad
    return gen, problems


def check(hf_dir: Path) -> list[str]:
    """G1 판정: generation_config.json 이 있고 챗 종료 토큰을 eos 에 담고 있는가."""
    problems: list[str] = []
    gc_path = hf_dir / "generation_config.json"
    if not gc_path.exists():
        return ["generation_config.json 없음 — 서버가 턴 종료를 인식하지 못한다"]

    try:
        gen = json.loads(gc_path.read_text())
    except Exception as e:  # noqa: BLE001
        return [f"generation_config.json 파싱 실패: {e}"]

    eos = gen.get("eos_token_id")
    eos = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
    tokens = _added_tokens(hf_dir)

    chat_end = next((n for n in CHAT_END_CANDIDATES if n in tokens), None)
    if chat_end is None:
        problems.append(f"tokenizer 에서 챗 종료 토큰을 찾지 못함 (후보 {CHAT_END_CANDIDATES})")
    elif tokens[chat_end] not in eos:
        problems.append(
            f"eos_token_id={eos} 에 챗 종료 토큰 {chat_end}(id {tokens[chat_end]}) 가 없다 "
            "— 모델이 턴을 끝내도 서버가 멈추지 않는다"
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description="G1 게이트: HF ckpt 의 eos 정합성")
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("--check", action="store_true", help="검사만 하고 쓰지 않는다")
    a = ap.parse_args()

    hf_dir: Path = a.hf_dir
    if not hf_dir.is_dir():
        print(f"❌ 디렉토리 없음: {hf_dir}", file=sys.stderr)
        return 1

    if not a.check:
        gen, problems = build(hf_dir)
        if problems:
            for p in problems:
                print(f"❌ {p}", file=sys.stderr)
            return 1
        (hf_dir / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
        print(f"✅ generation_config.json 생성: eos_token_id={gen['eos_token_id']}")

    problems = check(hf_dir)
    if problems:
        for p in problems:
            print(f"❌ G1 실패: {p}", file=sys.stderr)
        return 1

    gen = json.loads((hf_dir / "generation_config.json").read_text())
    tokens = _added_tokens(hf_dir)
    named = {v: k for k, v in tokens.items()}
    pretty = ", ".join(f"{i}={named.get(i, '?')}" for i in gen["eos_token_id"])
    print(f"✅ G1 통과 — eos: {pretty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
