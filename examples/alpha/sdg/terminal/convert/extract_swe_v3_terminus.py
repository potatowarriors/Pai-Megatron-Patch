#!/usr/bin/env python3
"""extract_swe_v3_terminus.py — SWE-v3 의 Terminus 행(≈3.5k)을 골라 jsonl 로 뽑는다 (보존 렌더 재변환용).

사용자 결정(2026-09-07): 합성 데이터와 SWE-v3 Terminus 행을 같은 규약(--keep-history-think)으로 변환한다. 기존
`swe_v3_keepthink` 멤버는 Terminus 행을 기본 렌더(중간 턴 think 비움)로 구웠으므로, 이 3.5k 행만 별도 멤버
`swe_v3_terminus_keephist` 로 다시 굽는다. 판별: system 프롬프트에 "solving command-line tasks in a Linux environment".

사용: python3 extract_swe_v3_terminus.py --out <dir>   → <dir>/swe_v3_terminus.jsonl + MANIFEST.json
"""
import argparse, glob, hashlib, json, os, time
from collections import Counter

SRC = "/home/work/Datasets/LL_datasets/posttraining/SFT/Nemotron-SFT-SWE-v3/data/*.parquet"
KEY = "solving command-line tasks in a Linux environment"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    import pyarrow.parquet as pq
    os.makedirs(a.out, exist_ok=True)
    n_in = n_out = 0; sys_md5 = Counter(); rc_turns = asst_turns = 0; t0 = time.time()
    with open(os.path.join(a.out, "swe_v3_terminus.jsonl"), "w") as f:
        for p in sorted(glob.glob(SRC)):
            t = pq.read_table(p, columns=["uuid", "license", "messages"])
            for row in t.to_pylist():
                n_in += 1
                ms = row["messages"] or []
                if not ms or ms[0].get("role") != "system" or KEY not in (ms[0].get("content") or ""):
                    continue
                msgs = []
                for m in ms:
                    d = {"role": m["role"], "content": m.get("content") or ""}
                    if m.get("reasoning_content"):
                        d["reasoning_content"] = m["reasoning_content"]; rc_turns += 1
                    if m.get("tool_calls"):
                        d["tool_calls"] = m["tool_calls"]
                    msgs.append(d)
                asst_turns += sum(1 for m in msgs if m["role"] == "assistant")
                sys_md5[hashlib.md5(msgs[0]["content"].encode()).hexdigest()[:8]] += 1
                f.write(json.dumps({"uuid": row["uuid"], "license": row["license"], "messages": msgs,
                                    "metadata": {"task": "swe-v3-terminus", "source": "Nemotron-SFT-SWE-v3"}}, ensure_ascii=False) + "\n")
                n_out += 1
    man = {"source": SRC, "rows_scanned": n_in, "rows": n_out, "assistant_turns": asst_turns, "reasoning_turns": rc_turns,
           "system_prompt_md5": dict(sys_md5),
           "conversion_note": "build_alpha_sft_idxmap.py --keep-history-think --seq-length 131072 --pad-doc-multiple 16 (멤버 swe_v3_terminus_keephist)"}
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[swe_v3_terminus] scanned={n_in:,} rows={n_out:,} asst_turns={asst_turns:,} reasoning_turns={rc_turns:,} md5={dict(sys_md5)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
