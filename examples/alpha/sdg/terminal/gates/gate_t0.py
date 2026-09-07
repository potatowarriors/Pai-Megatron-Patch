#!/usr/bin/env python3
"""gate_t0.py — G-T0: GLM-5.3-Flash 교사 엔드포인트 준비 게이트 (터미널 SDG 트랙 P0).

무엇을 재는가 (README §게이트):
  T0-1 models      /v1/models 에 glm53-flash 가 있다
  T0-2 reasoning   reasoning 파서(glm45)가 reasoning_content 와 content 를 분리한다
  T0-3 tools       tools+tool_choice=auto 요청이 수용되고 tool 파서(glm47)가 tool_calls 를 만든다
                   (Harbor/litellm 이 tool_choice=auto 를 보내므로 400 이면 전부 실패 — A1/A4 상당)
  T0-4 terminus    Terminus-2 형 시스템 프롬프트에 대해 응답이 JSON 으로 파싱된다 (n회, 유효율)
  T0-5 throughput  동시 N 요청 출력 tok/s, 단일 스트림 tok/s
  T0-6 gpu         GPU 별 메모리 사용량 (nvidia-smi)
  T0-7 effort      reasoning_effort low/max 가 reasoning 길이를 바꾸는가 (정보용)

사용 (sub1, venv 불필요 — 표준 라이브러리만):
  python3 gate_t0.py --base http://localhost:8300/v1 --wait 3600 --out /home/work/vidsearch/tools/glm53/gate_t0.json
종료 코드: 0 = T0-1~T0-4 전부 PASS, 1 = 실패, 3 = 준비 대기 초과
"""
import argparse, json, re, subprocess, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

MODEL = "glm53-flash"

TERMINUS_SYSTEM = (
    "You are an AI assistant tasked with solving command-line tasks in a Linux environment. "
    "You will be given a task description and the output from previously executed commands. "
    "Your goal is to solve the task by providing batches of shell commands.\n\n"
    "Format your response as JSON with the following structure:\n\n"
    '{\n  "analysis": "Analyze the current state based on the terminal output provided.",\n'
    '  "plan": "Describe your plan for the next steps.",\n'
    '  "commands": [\n    {\n      "keystrokes": "ls -la\\n",\n      "duration": 0.1\n    }\n  ],\n'
    '  "task_complete": true\n}\n\n'
    "Required fields: analysis, plan, commands. Optional: task_complete (defaults to false).\n"
    "Extra text before or after the JSON will generate warnings but be tolerated. The JSON must be valid.\n\n"
    "Task Description:\nCreate a file /app/hello.txt containing exactly the line `hello world`, "
    "then print its md5sum.\n\nCurrent terminal state:\nroot@a1b2c3:/app# \n"
)


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


def chat(base, messages, max_tokens=1024, temperature=1.0, top_p=0.95, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p}
    body.update(extra)
    t0 = time.time()
    r = post(base, "/chat/completions", body)
    return r, time.time() - t0


def reasoning_of(m):
    """vLLM nightly(0.28.1rc1)는 `reasoning`, 구판은 `reasoning_content` — 둘 다 받는다."""
    return m.get("reasoning_content") or m.get("reasoning") or ""


def wait_ready(base, wait_s):
    t0 = time.time()
    while time.time() - t0 < wait_s:
        try:
            ids = [m["id"] for m in get(base, "/models")["data"]]
            if MODEL in ids:
                return True
        except Exception:
            pass
        time.sleep(15)
    return False


def extract_json(text):
    """Terminus-2 json 파서와 같은 관용: 앞뒤 잡텍스트 허용, 첫 '{' ~ 마지막 '}'."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8300/v1")
    ap.add_argument("--wait", type=int, default=0, help="준비 대기 초 (0 = 대기 안 함)")
    ap.add_argument("--n-terminus", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    res, ok = {}, True

    if a.wait and not wait_ready(base, a.wait):
        print("[T0] ❌ 준비 대기 초과"); sys.exit(3)

    # T0-1
    try:
        ids = [m["id"] for m in get(base, "/models")["data"]]
        res["t0_1_models"] = {"pass": MODEL in ids, "ids": ids}
    except Exception as e:  # noqa: BLE001
        res["t0_1_models"] = {"pass": False, "error": str(e)}
    ok &= res["t0_1_models"]["pass"]
    print(f"[T0-1] models {'PASS' if res['t0_1_models']['pass'] else 'FAIL'} {res['t0_1_models']}")

    # T0-2 reasoning split
    try:
        r, dt = chat(base, [{"role": "user", "content": "What is 17*23? Think it through, then answer."}],
                     max_tokens=2048, temperature=0.0)
        m = r["choices"][0]["message"]
        rc, c = reasoning_of(m), (m.get("content") or "")
        p = bool(rc.strip()) and bool(c.strip()) and "<think>" not in c
        res["t0_2_reasoning"] = {"pass": p, "reasoning_chars": len(rc), "content_chars": len(c),
                                 "field": "reasoning_content" if m.get("reasoning_content") else "reasoning",
                                 "content_head": c[:120], "sec": round(dt, 1),
                                 "usage": r.get("usage")}
    except Exception as e:  # noqa: BLE001
        res["t0_2_reasoning"] = {"pass": False, "error": str(e)}
    ok &= res["t0_2_reasoning"]["pass"]
    print(f"[T0-2] reasoning split {'PASS' if res['t0_2_reasoning']['pass'] else 'FAIL'} "
          f"{ {k: v for k, v in res['t0_2_reasoning'].items() if k != 'usage'} }")

    # T0-3 tools
    tools = [{"type": "function", "function": {"name": "bash", "description": "Run a bash command",
              "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                             "required": ["command"]}}}]
    try:
        r, dt = chat(base, [{"role": "user", "content": "List the files in /tmp. Use the bash tool."}],
                     max_tokens=2048, temperature=0.0, tools=tools, tool_choice="auto")
        m = r["choices"][0]["message"]
        tc = m.get("tool_calls") or []
        names = [t.get("function", {}).get("name") for t in tc]
        p = bool(tc) and names[0] == "bash"
        res["t0_3_tools"] = {"pass": p, "tool_calls": names, "finish": r["choices"][0].get("finish_reason"),
                             "content_head": (m.get("content") or "")[:120], "sec": round(dt, 1)}
    except urllib.error.HTTPError as e:
        res["t0_3_tools"] = {"pass": False, "error": f"HTTP {e.code}: {e.read()[:300]!r}"}
    except Exception as e:  # noqa: BLE001
        res["t0_3_tools"] = {"pass": False, "error": str(e)}
    ok &= res["t0_3_tools"]["pass"]
    print(f"[T0-3] tools {'PASS' if res['t0_3_tools']['pass'] else 'FAIL'} {res['t0_3_tools']}")

    # T0-4 terminus format validity (Terminus 실행 조건: temp 1.0 / top_p 0.95)
    def one(_):
        try:
            r, dt = chat(base, [{"role": "user", "content": TERMINUS_SYSTEM}], max_tokens=8192)
            m = r["choices"][0]["message"]
            j = extract_json(m.get("content") or "")
            valid = j is not None and all(k in j for k in ("analysis", "plan", "commands")) \
                and isinstance(j["commands"], list) and all("keystrokes" in c for c in j["commands"])
            return {"valid": valid, "reasoning_chars": len(reasoning_of(m)),
                    "completion_tokens": r["usage"]["completion_tokens"],
                    "prompt_tokens": r["usage"]["prompt_tokens"], "sec": round(dt, 1),
                    "content_head": (m.get("content") or "")[:100]}
        except Exception as e:  # noqa: BLE001
            return {"valid": False, "error": str(e)}
    with ThreadPoolExecutor(a.n_terminus) as ex:
        outs = list(ex.map(one, range(a.n_terminus)))
    nv = sum(o["valid"] for o in outs)
    p = nv >= max(1, round(0.95 * a.n_terminus) - (1 if a.n_terminus <= 10 else 0))
    res["t0_4_terminus"] = {"pass": p, "valid": nv, "n": a.n_terminus,
                            "completion_tokens": [o.get("completion_tokens") for o in outs],
                            "reasoning_chars": [o.get("reasoning_chars") for o in outs],
                            "samples": outs[:2]}
    ok &= p
    print(f"[T0-4] terminus json {'PASS' if p else 'FAIL'} {nv}/{a.n_terminus} valid; "
          f"completion_tokens={res['t0_4_terminus']['completion_tokens']}")

    # T0-5 throughput
    prompt = [{"role": "user", "content": "Explain in detail how TCP congestion control works, "
               "covering slow start, congestion avoidance, fast retransmit and fast recovery."}]
    try:
        r, dt = chat(base, prompt, max_tokens=512, temperature=0.0)
        single = r["usage"]["completion_tokens"] / dt
        def many(_):
            r, _dt = chat(base, prompt, max_tokens=1024)
            return r["usage"]["completion_tokens"]
        t0 = time.time()
        with ThreadPoolExecutor(a.concurrency) as ex:
            toks = list(ex.map(many, range(a.concurrency)))
        wall = time.time() - t0
        res["t0_5_throughput"] = {"single_stream_tok_s": round(single, 1),
                                  "concurrency": a.concurrency, "aggregate_out_tok_s": round(sum(toks) / wall, 1),
                                  "wall_s": round(wall, 1), "out_tokens": sum(toks)}
    except Exception as e:  # noqa: BLE001
        res["t0_5_throughput"] = {"error": str(e)}
    print(f"[T0-5] throughput {res['t0_5_throughput']}")

    # T0-6 gpu memory
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        gpus = [[int(x) for x in line.split(",")] for line in q.strip().splitlines()]
        res["t0_6_gpu_mem_mib"] = {f"gpu{i}": used for i, used, _tot in gpus}
    except Exception as e:  # noqa: BLE001
        res["t0_6_gpu_mem_mib"] = {"error": str(e)}
    print(f"[T0-6] gpu mem MiB {res['t0_6_gpu_mem_mib']}")

    # T0-7 reasoning_effort (정보용): 같은 프롬프트, low vs max
    eff = {}
    for level in ("low", "max"):
        try:
            r, dt = chat(base, [{"role": "user", "content": "Prove that the square root of 2 is irrational."}],
                         max_tokens=8192, temperature=0.0, reasoning_effort=level)
            m = r["choices"][0]["message"]
            eff[level] = {"reasoning_chars": len(reasoning_of(m)),
                          "completion_tokens": r["usage"]["completion_tokens"], "sec": round(dt, 1)}
        except Exception as e:  # noqa: BLE001
            eff[level] = {"error": str(e)[:200]}
    res["t0_7_effort"] = eff
    print(f"[T0-7] reasoning_effort {eff}")

    res["pass"] = bool(ok)
    res["base"] = base
    res["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[T0] → {a.out}")
    print(f"[T0] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
