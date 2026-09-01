"""HF 체크포인트의 롱컨텍스트 서빙 **프로파일**을 만든다 — 가중치는 symlink, config.json 만 교체.

배경 (2026-09-01, `docs/SFT_BENCHMARKS.md`): vLLM 은 `max_model_len` 이 config 에서 유도한
상한을 넘으면 기동을 거절한다. 현재 변환본은 `max_position_embeddings=262144`,
`rope_scaling=null` 이라 상한이 262,144 다. 512K 를 서빙하려면 config 를 바꿔야 하는데,
변환 산출물을 제자리에서 고치면 그 디렉토리로 낸 벤치 이력과 정합이 깨진다. 그래서
**별도 디렉토리에 프로파일**을 만든다: safetensors·tokenizer·modeling 파일은 절대경로
symlink, `config.json` 만 새로 쓴다. 디스크 비용 0, 원본 불변.

프로파일 종류:
  ext   `max_position_embeddings` 만 올림, `rope_scaling: null` — **순수 외삽**. 393K 절벽(전 깊이
        0%, `study/lc_b_final_eval.md`)의 대조군.
  yarn  `rope_scaling = {rope_type: yarn, factor, original_max_position_embeddings}` — 저주파
        rotary dim 을 factor 배 보간. vLLM 은 `original × factor` 를 상한으로 유도하고
        (`vllm/config/model.py`), HF 는 `max_position_embeddings / original` 로 factor 를 재계산한다.
        그래서 **factor × original == max_pos** 를 강제한다 — 셋이 어긋나면 두 스택이 다른
        스케일을 쓴다.

검증 (2026-09-01 CPU 프로브, main1): transformers 5.16.1(서빙 venv)이 `rope_scaling` 을
`rope_parameters` 로 표준화하고, 플러그인 로드 후 vLLM 0.25.1 `ModelConfig` 가 yarn s=2
프로파일을 524288 로 수용(원본 config 는 거절). transformers 4.57(시스템)의 HF rotary 는
램프 dim 14~21 이 1.0→0.56 으로 보간되고 저주파 dim 은 ×0.5 — 계산값과 일치.

사용:
  python3 tools/set_long_context_config.py <HF_DIR> --out <DIR> --max-pos 524288                          # ext
  python3 tools/set_long_context_config.py <HF_DIR> --out <DIR> --max-pos 524288 --yarn-factor 2           # yarn, original = 262144
  python3 tools/set_long_context_config.py <HF_DIR> --out <DIR> --max-pos 524288 --yarn-factor 4 --yarn-original 131072
  python3 tools/set_long_context_config.py --check <DIR>            # 검사만 (유도 상한·정합·G1)
  python3 tools/set_long_context_config.py --check <DIR> --hf-probe # + HF rotary 실제 보간 확인 (transformers 필요)

G1: 원본에 `generation_config.json` 이 없으면(사전학습 변환본) `emit_generation_config.build` 로
프로파일 안에 생성한다 — 원본은 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emit_generation_config as egc  # noqa: E402

MARKER = ".lc_profile.json"


def _derived_max(cfg: dict) -> tuple[int, str]:
    """vLLM 이 config 에서 유도하는 max_model_len 상한과 근거 문자열."""
    mp = int(cfg["max_position_embeddings"])
    rs = cfg.get("rope_scaling")
    if not rs:
        return mp, f"max_position_embeddings={mp} (rope_scaling null)"
    if rs.get("rope_type", rs.get("type")) != "yarn":
        return mp, f"max_position_embeddings={mp} (rope_type={rs.get('rope_type')}: 이 도구는 yarn 만 다룬다)"
    orig = int(rs["original_max_position_embeddings"])
    fac = float(rs["factor"])
    return int(orig * fac), f"yarn original={orig} × factor={fac:g}"


def build(src: Path, out: Path, max_pos: int, factor: float | None, original: int | None,
          force: bool) -> int:
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        print(f"❌ config.json 없음: {src}", file=sys.stderr)
        return 1
    if out.exists():
        if not force:
            print(f"❌ 이미 있음: {out} (덮어쓰려면 --force; 이 도구가 만든 프로파일만 지운다)", file=sys.stderr)
            return 1
        if not (out / MARKER).exists():
            print(f"❌ {out} 은 이 도구가 만든 프로파일이 아니다 — 지우지 않는다", file=sys.stderr)
            return 1
        shutil.rmtree(out)

    cfg = json.loads(cfg_path.read_text())
    rope_scaling = None
    if factor is not None:
        if original is None:
            if max_pos % 1 or (max_pos / factor) != int(max_pos / factor):
                print(f"❌ max_pos {max_pos} 가 factor {factor:g} 로 나누어떨어지지 않는다", file=sys.stderr)
                return 1
            original = int(max_pos / factor)
        if not math.isclose(original * factor, max_pos, rel_tol=0, abs_tol=0.5):
            print(f"❌ factor × original = {original * factor:g} ≠ max_pos {max_pos} — vLLM 상한과 HF factor 가 어긋난다",
                  file=sys.stderr)
            return 1
        rope_scaling = {
            "rope_type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": int(original),
        }

    out.mkdir(parents=True)
    linked = 0
    for entry in sorted(src.iterdir()):
        if entry.name in ("config.json", MARKER):
            continue
        os.symlink(entry.resolve(), out / entry.name)
        linked += 1

    old_mp, old_rs = cfg.get("max_position_embeddings"), cfg.get("rope_scaling")
    cfg["max_position_embeddings"] = int(max_pos)
    cfg["rope_scaling"] = rope_scaling
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")

    gen_note = "symlink"
    if not (src / "generation_config.json").exists():
        gen, problems = egc.build(out)
        if problems:
            for p in problems:
                print(f"   ⚠️ generation_config: {p}")
        (out / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")
        gen_note = f"생성 (원본에 없음) eos={gen.get('eos_token_id')}"

    (out / MARKER).write_text(json.dumps({
        "src": str(src.resolve()),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_position_embeddings": {"from": old_mp, "to": int(max_pos)},
        "rope_scaling": {"from": old_rs, "to": rope_scaling},
        "generation_config": gen_note,
    }, indent=2, ensure_ascii=False) + "\n")

    derived, why = _derived_max(cfg)
    print(f"✅ 프로파일 생성: {out}")
    print(f"   src={src}  symlink {linked}개  generation_config={gen_note}")
    print(f"   max_position_embeddings {old_mp} → {max_pos}; rope_scaling {old_rs} → {rope_scaling}")
    print(f"   vLLM 유도 상한 = {derived} ({why}) → --max-model-len ≤ {derived}")
    return check(out, hf_probe=False)


def check(out: Path, hf_probe: bool) -> int:
    cfg_path = out / "config.json"
    if not cfg_path.exists():
        print(f"❌ config.json 없음: {out}", file=sys.stderr)
        return 1
    cfg = json.loads(cfg_path.read_text())
    problems: list[str] = []
    mp = int(cfg["max_position_embeddings"])
    derived, why = _derived_max(cfg)
    rs = cfg.get("rope_scaling")
    if rs and derived != mp:
        problems.append(f"factor × original = {derived} ≠ max_position_embeddings {mp}")
    if rs and rs.get("rope_type", rs.get("type")) == "yarn" and "original_max_position_embeddings" not in rs:
        problems.append("yarn 인데 original_max_position_embeddings 가 없다 — HF 는 max_position_embeddings 를 원본으로 오인한다")
    g1 = egc.check(out)
    problems += [f"G1: {p}" for p in g1]

    print(f"── 프로파일 검사: {out}")
    print(f"   max_position_embeddings={mp}  rope_scaling={rs}")
    print(f"   vLLM 유도 상한 = {derived} ({why})")
    if (out / MARKER).exists():
        meta = json.loads((out / MARKER).read_text())
        print(f"   src={meta.get('src')}  created={meta.get('created')}")
    print(f"   G1 {'PASS' if not g1 else 'FAIL'}")

    if hf_probe:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from transformers import AutoConfig
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            hf_cfg = AutoConfig.from_pretrained(str(out), trust_remote_code=True)
            Rot = get_class_from_dynamic_module("modeling_alpha.AlphaRotaryEmbedding", str(out))
            rot = Rot(hf_cfg)
            base_cfg = AutoConfig.from_pretrained(str(out), trust_remote_code=True)
            base_cfg.rope_scaling = None
            base = Rot(base_cfg)
            ratio = (rot.inv_freq / base.inv_freq).tolist()
            print(f"   HF rotary: rope_type={rot.rope_type} attention_scaling={float(rot.attention_scaling):.4f} "
                  f"inv_freq 비율 min={min(ratio):.3f} max={max(ratio):.3f} (dims={len(ratio)})")
            if rs and rs.get("rope_type") == "yarn":
                exp = 1.0 / float(rs["factor"])
                if not math.isclose(min(ratio), exp, abs_tol=1e-3):
                    problems.append(f"HF 저주파 보간 비율 {min(ratio):.3f} ≠ 1/factor {exp:.3f}")
                if rot.rope_type != "yarn":
                    problems.append(f"HF rotary rope_type={rot.rope_type} (yarn 이어야 함)")
        except Exception as e:  # noqa: BLE001
            problems.append(f"HF 프로브 실패: {type(e).__name__}: {e}")

    for p in problems:
        print(f"   ❌ {p}")
    print("   → " + ("PASS" if not problems else "FAIL"))
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="HF ckpt 롱컨텍스트 서빙 프로파일 (symlink + config.json)")
    ap.add_argument("src", nargs="?", type=Path, help="원본 HF 디렉토리")
    ap.add_argument("--out", type=Path, help="프로파일 디렉토리 (새로 만든다)")
    ap.add_argument("--max-pos", type=int, default=524288, help="max_position_embeddings (기본 524288)")
    ap.add_argument("--yarn-factor", type=float, help="YaRN factor. 없으면 ext(순수 외삽) 프로파일")
    ap.add_argument("--yarn-original", type=int, help="YaRN original_max_position_embeddings (기본 max_pos/factor)")
    ap.add_argument("--force", action="store_true", help="이 도구가 만든 기존 프로파일을 지우고 다시 만든다")
    ap.add_argument("--check", type=Path, metavar="DIR", help="검사만")
    ap.add_argument("--hf-probe", action="store_true", help="--check 에서 HF rotary 를 실제로 만들어 보간 확인")
    a = ap.parse_args()

    if a.check:
        return check(a.check, hf_probe=a.hf_probe)
    if not a.src or not a.out:
        ap.error("<HF_DIR> 와 --out 이 필요하다 (또는 --check DIR)")
    if a.yarn_original is not None and a.yarn_factor is None:
        ap.error("--yarn-original 은 --yarn-factor 와 함께 쓴다")
    return build(a.src, a.out, a.max_pos, a.yarn_factor, a.yarn_original, a.force)


if __name__ == "__main__":
    sys.exit(main())
