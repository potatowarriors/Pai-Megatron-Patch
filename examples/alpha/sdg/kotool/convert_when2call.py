#!/usr/bin/env python3
"""convert_when2call.py — nvidia/When2Call `train/when2call_train_sft.jsonl`(15,000행, cc-by-4.0) → 우리 tool 시나리오 행.

When2Call SFT 행은 전부 **미호출** 예시다(2026-09-13 실측: 되묻기 7,110 · 불가/한계 안내 7,540 · 기타 350, tool_calls 0건, 도구 0개 행 2,223).
도구 스키마는 BFCL 형(`"type": "dict"`, `"str, optional"`, `List[int]`)이라 JSON Schema 형(object/string/integer/number/boolean/array)으로 정규화해
Agentic-v2·kotool_v1 과 같은 `{"type":"function","function":{...}}` 선언으로 맞춘다. reasoning 없음 → no-think 규약(빈 <think></think>).
용도: phase-2 회귀 "도구 선언 시 유령 호출"(KNOWN_ISSUES 09-09) 교정 — tool-calling 버킷 안에서 미호출:호출 = 2:1 근사.
사용: python3 convert_when2call.py --inp <sft.jsonl> --out out/when2call/when2call_v1.jsonl
"""
import argparse, collections, hashlib, json, re

TYPE_MAP = {"dict": "object", "str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number", "number": "number",
            "bool": "boolean", "boolean": "boolean", "list": "array", "array": "array", "tuple": "array", "any": None, "object": "object"}

def norm_type(t):
    """'str, optional' → string ; 'List[int]' → array(items integer) ; 'dict' → object. 반환 (type, items|None)."""
    if not isinstance(t, str): return None, None
    t = t.split(",")[0].strip()
    m = re.fullmatch(r"(?:List|list|Tuple|tuple)\[(.+)\]", t)
    if m:
        it, _ = norm_type(m.group(1)); return "array", ({"type": it} if it else None)
    return TYPE_MAP.get(t, TYPE_MAP.get(t.lower())), None

def norm_schema(node):
    if isinstance(node, list): return [norm_schema(x) for x in node]
    if not isinstance(node, dict): return node
    out = {}
    for k, v in node.items():
        if k == "type":
            ty, items = norm_type(v)
            if ty: out["type"] = ty
            if items and "items" not in node: out["items"] = items
        else: out[k] = norm_schema(v)
    return out

def classify(text):
    t = text.strip()
    if t.endswith("?"): return "clarify"
    if re.search(r"\b(unable|can't|cannot|apolog|sorry|don't have|do not have|not able|no tool|not possible)\b", t, re.I): return "infeasible"
    return "direct"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    st = collections.Counter(); seen = set()
    with open(a.out, "w") as fo:
        for l in open(a.inp):
            r = json.loads(l); msgs = r["messages"]
            if [m["role"] for m in msgs] != ["user", "assistant"] or msgs[1].get("tool_calls"): st["drop_shape"] += 1; continue
            tools = []
            for t in r.get("tools") or []:
                try: t = json.loads(t) if isinstance(t, str) else t
                except Exception: st["drop_tool_json"] += 1; tools = None; break
                if not isinstance(t, dict) or not t.get("name"): st["drop_tool_shape"] += 1; tools = None; break
                fn = {"name": t["name"], "description": t.get("description") or "", "parameters": norm_schema(t.get("parameters") or {"type": "object", "properties": {}})}
                if fn["parameters"].get("type") is None: fn["parameters"]["type"] = "object"
                tools.append({"type": "function", "function": fn})
            if tools is None: continue
            key = hashlib.sha1((msgs[0]["content"] + "\x00" + msgs[1]["content"] + "\x00" + json.dumps(tools, sort_keys=True)).encode()).hexdigest()[:24]
            if key in seen: st["drop_dup"] += 1; continue
            seen.add(key)
            if any(s in (msgs[0]["content"] + msgs[1]["content"]) for s in ("<think>", "</think>", "<tool_call>", "<tool_response>", "<|endoftext|>", "<|im_end|>")): st["drop_special"] += 1; continue
            case = classify(msgs[1]["content"]); st["case_" + case] += 1; st["tools_" + ("0" if not tools else "n")] += 1
            row = {"messages": [{"role": "user", "content": msgs[0]["content"]}, {"role": "assistant", "content": msgs[1]["content"]}],
                   "tools": tools, "uuid": f"when2call_v1:{key}",
                   "metadata": {"pipeline": "when2call_v1", "case": case, "n_tools": len(tools), "think": False, "license": "cc-by-4.0", "source": "nvidia/When2Call train_sft"}}
            fo.write(json.dumps(row, ensure_ascii=False) + "\n"); st["kept"] += 1
    print("DONE", dict(st))

if __name__ == "__main__":
    main()
