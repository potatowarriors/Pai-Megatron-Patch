#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""SWE-v3 도구 선언 사이드카 생성 — `tools` 컬럼이 없는 Nemotron-SFT-SWE-v3 에 하니스별 도구 스키마를 복원한다.

배경 (docs/KNOWN_ISSUES.md 2026-09-09 ①): SWE-v3 행 키는 messages·uuid·license 뿐이라 구조화 tool_call 을 쓰는
232k 행(97.5%)이 `# Tools` 블록 없이 렌더됐다 — 배포(mini-swe-agent·OpenHands·TB-2)는 항상 tools 를 선언한다.

전수 스캔 실측 (2026-09-09, 237,970행 / 19 패밀리 = system 프롬프트 앞 200자 md5):
  - 실제 하니스 6 패밀리(OpenHands 85.7k·SWE-agent 44.3k+3.3k·mini-swe-agent 20.1k·opencode 16.1k·Codex CLI 2.9k)는
    도구 집합이 세션마다 고정 → 패밀리 **합집합**을 선언 (declare="union").
  - 합성 하니스 11 패밀리(≈50k)는 같은 프롬프트 아래 행마다 도구 이름이 별칭으로 바뀐다
    (bash_exec/run_bash/shell/shell_run…, code_editor/file_editor/text_editor…, read/read_file/view_file/cat_file…).
    → 그 행이 호출한 별칭만 선언 (declare="called") — 행이 쓴 별칭이 곧 그 세션의 도구 집합.
  - Terminus 2 패밀리(5.9k)는 tool_call 이 없어 대상 아님(변환기 inject_tools 가 호출 없는 행은 건드리지 않음).

스키마 복원 규칙:
  - 클래스 판정 = 인자 서명(호출의 ≥5% 에 나타난 키) + 이름 힌트. 설명문·파라미터 설명은 실제 하니스 스키마에서 가져온다:
    OpenHands 4종(SWE-v2 swe.jsonl 의 tools), opencode 10종(OpenCode-v1 의 tools). SWE-agent(bash/submit)·Codex CLI
    (shell_command/apply_patch/update_plan/read_file/grep_files/list_dir)는 공개 하니스 정의를 요약해 적는다.
  - properties = 클래스 표준 파라미터 ∪ 관측 키(호출의 ≥2%; 환각 키 `security_isk` 류 제외). 타입은 관측 다수값
    (bool→boolean, int→integer, float→number, list→array, dict→object, str→string).
  - required = 클래스 required ∩ 관측 빈도 ≥99% 키.
  - 클래스 미판정 도구는 이름·관측 인자만으로 최소 스키마(설명 "Tool `<name>`.") — 요약에 unclassified 로 기록.

산출: <out>/alpha_tools_sidecar.json ({families, by_name}) + alpha_tools_sidecar.summary.md.
검증: --validate N 이면 표본 N 행에 대해 inject_tools 를 적용해 "호출된 도구 ⊆ 선언된 도구" 를 확인한다.

사용:
  python3 build_swe_v3_tools_sidecar.py --swe-v3 $SFT/Nemotron-SFT-SWE-v3/data \\
      --openhands $SFT/Nemotron-SFT-SWE-v2/data/swe.jsonl --opencode $SFT/Nemotron-SFT-OpenCode-v1 \\
      --out $SFT/Nemotron-SFT-SWE-v3 --workers 32 --validate 2000
"""
import argparse, glob, hashlib, json, os, random, re, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_alpha_sft_idxmap import _json_type, _sidecar_system_key, inject_tools, normalize_row  # noqa: E402

MAJOR_FRAC = 0.05      # 서명 판정에 쓰는 키 빈도
PROP_FRAC = 0.02       # properties 에 넣는 관측 키 빈도
REQ_FRAC = 0.99        # required 판정 빈도
UNION_MAX_TOOLS = 10   # 이 이하 도구 수 + 최빈 도구집합 ≥20% 행 → 실제 하니스(union)
UNION_TOP_SHARE = 0.20

PY2JSON = {"bool": "boolean", "int": "integer", "float": "number", "list": "array", "dict": "object",
           "str": "string", "NoneType": "string"}


# ---------------------------------------------------------------------------
# 1. 전수 스캔 (패밀리 × 도구 × 인자)
# ---------------------------------------------------------------------------
def _scan_file(f):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(f)
    fam = {}
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg).to_pylist():
            msgs = r["messages"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            sysc = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
            k = _sidecar_system_key(sysc)
            d = fam.setdefault(k, {"rows": 0, "rows_tc": 0, "prefix": (sysc or "")[:160],
                                   "tools": {}, "toolsets": Counter()})
            d["rows"] += 1
            names = set()
            for m in msgs:
                if m["role"] != "assistant":
                    continue
                tcs = m.get("tool_calls")
                if isinstance(tcs, str):
                    try:
                        tcs = json.loads(tcs)
                    except Exception:
                        tcs = None
                for t in tcs or []:
                    fn = t.get("function", t)
                    name = fn.get("name")
                    if not name:
                        continue
                    a = fn.get("arguments")
                    if isinstance(a, str):
                        try:
                            a = json.loads(a)
                        except Exception:
                            a = None
                    names.add(name)
                    td = d["tools"].setdefault(name, {"calls": 0, "args": {}})
                    td["calls"] += 1
                    if isinstance(a, dict):
                        for ak, av in a.items():
                            td["args"].setdefault(ak, Counter())[type(av).__name__] += 1
            if names:
                d["rows_tc"] += 1
                d["toolsets"][tuple(sorted(names))] += 1
    return fam


def _merge(a, b):
    for k, d in b.items():
        if k not in a:
            a[k] = d
            continue
        e = a[k]
        e["rows"] += d["rows"]; e["rows_tc"] += d["rows_tc"]
        e["toolsets"].update(d["toolsets"])
        for name, td in d["tools"].items():
            et = e["tools"].setdefault(name, {"calls": 0, "args": {}})
            et["calls"] += td["calls"]
            for ak, cnt in td["args"].items():
                et["args"].setdefault(ak, Counter()).update(cnt)
    return a


def scan_swe_v3(data_dir, workers):
    fs = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    assert fs, f"parquet 없음: {data_dir}"
    with Pool(workers) as pool:
        parts = pool.map(_scan_file, fs)
    fam = {}
    for p in parts:
        _merge(fam, p)
    return fam


# ---------------------------------------------------------------------------
# 2. 실제 하니스 스키마 (설명문 원천)
# ---------------------------------------------------------------------------
def _seek_rows(path, n, seed=1):
    rng = random.Random(seed)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        for _ in range(n):
            f.seek(rng.randrange(size - 1)); f.readline(); line = f.readline()
            if line.strip():
                yield json.loads(line)


def load_canonical(openhands_jsonl, opencode_dir):
    """{('openhands', name): function-dict, ('opencode', name): function-dict}"""
    canon = {}
    for r in _seek_rows(openhands_jsonl, 40):
        for t in r.get("tools") or []:
            fn = t.get("function", t)
            canon.setdefault(("openhands", fn["name"]), fn)
    for sub in ("bash_only_tool", "general", "agent_skills"):
        p = os.path.join(opencode_dir, sub, "data.jsonl")
        if not os.path.exists(p):
            continue
        for r in _seek_rows(p, 25, seed=3):
            for t in r.get("tools") or []:
                name = t.get("id") or t.get("name")
                schema = (t.get("inputSchema") or {}).get("jsonSchema") or t.get("parameters") or {}
                if name and ("opencode", name) not in canon:
                    canon[("opencode", name)] = {"name": name, "description": t.get("description", ""),
                                                 "parameters": schema}
    return canon


def _cdesc(canon, lib, name, fallback=""):
    fn = canon.get((lib, name))
    return (fn.get("description") if fn else None) or fallback


def _cparam(canon, lib, name, pname):
    fn = canon.get((lib, name))
    props = ((fn or {}).get("parameters") or {}).get("properties") or {}
    return props.get(pname) or {}


# 클래스 표: (판정 함수, 설명문, 파라미터 설명, required)
def build_classes(canon):
    C = {}

    def add(key, match, desc, params, required):
        C[key] = {"match": match, "desc": desc, "params": params, "required": required}

    oh_bash = _cparam(canon, "openhands", "execute_bash", "command").get("description", "The bash command to execute.")
    add("shell_openhands",
        lambda n, keys: "command" in keys and "path" not in keys and "workdir" not in keys
        and ("security_risk" in keys or "timeout" in keys or "is_input" in keys),
        _cdesc(canon, "openhands", "execute_bash", "Execute a bash command in the terminal within a persistent shell session."),
        {"command": {"type": "string", "description": oh_bash},
         "is_input": {"type": "string", "enum": ["true", "false"],
                      "description": _cparam(canon, "openhands", "execute_bash", "is_input").get("description", "If True, the command is an input to the running process.")},
         "timeout": {"type": "number", "description": _cparam(canon, "openhands", "execute_bash", "timeout").get("description", "Optional hard timeout in seconds.")},
         "security_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"],
                           "description": "The LLM's assessment of the safety risk of this action. LOW: read-only or reversible; MEDIUM: modifies files or state in the workspace; HIGH: destructive, irreversible or reaches outside the workspace."}},
        ["command"])
    add("shell_sweagent",
        lambda n, keys: keys <= {"command"} and n in ("bash",),
        "Run a bash command in the persistent shell session and return its output (stdout and stderr). "
        "The working directory and environment persist between calls. Avoid interactive commands.",
        {"command": {"type": "string", "description": "The bash command to execute."}},
        ["command"])
    add("shell_codex",
        lambda n, keys: "command" in keys and "workdir" in keys,
        "Runs a shell command and returns its output. Use `workdir` to choose the working directory instead of `cd`. "
        "Set `is_input` to send text to a still-running interactive process. Long-running commands are cut off at `timeout_ms`.",
        {"command": {"type": "string", "description": "The shell command to execute."},
         "workdir": {"type": "string", "description": "The working directory to run the command in (absolute path)."},
         "timeout_ms": {"type": "integer", "description": "Optional timeout in milliseconds for the command."},
         "is_input": {"type": "boolean", "description": "If true, `command` is sent as input to the running process instead of being executed as a new command."},
         "login": {"type": "boolean", "description": "If true, run the command in a login shell so that profile files are sourced."}},
        ["command"])
    add("str_replace_editor",
        lambda n, keys: "command" in keys and "path" in keys,
        _cdesc(canon, "openhands", "str_replace_editor", "Custom editing tool for viewing, creating and editing files in plain-text format."),
        {p: dict(_cparam(canon, "openhands", "str_replace_editor", p)) for p in
         ("command", "path", "file_text", "old_str", "new_str", "insert_line", "view_range")}
        | {"security_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"],
                             "description": "The LLM's assessment of the safety risk of this action."}},
        ["command", "path"])
    add("think", lambda n, keys: "thought" in keys,
        _cdesc(canon, "openhands", "think", "Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought."),
        {"thought": {"type": "string", "description": "The thought to log."}}, ["thought"])
    add("finish", lambda n, keys: "message" in keys and n in ("finish", "complete", "done", "submit_result"),
        _cdesc(canon, "openhands", "finish", "Signals the completion of the current task or conversation."),
        {"message": {"type": "string", "description": "Final message to send to the user"}}, ["message"])
    add("submit", lambda n, keys: not keys and n == "submit",
        "Submit the current state of the repository as the final solution. Call this once the task is complete; "
        "the diff of the working tree is collected as your answer.", {}, [])
    add("task_tracker", lambda n, keys: "command" in keys and "task_list" in keys,
        "Track and manage a task list for the current session. Use `plan` to create or update the list and `view` to display it. "
        "Each task has an id, title, status (todo, in_progress, done) and optional notes.",
        {"command": {"type": "string", "enum": ["view", "plan"], "description": "The command to run: `view` shows the current list, `plan` creates or updates it."},
         "task_list": {"type": "array", "description": "Required for `plan`: the full task list, each item `{id, title, status, notes}`."}},
        ["command"])
    rd = _cdesc(canon, "opencode", "read", "Reads a file from the local filesystem.")
    add("read_file", lambda n, keys: "file_path" in keys and ("offset" in keys or "limit" in keys) and "old_string" not in keys and "content" not in keys,
        rd.replace("filePath", "file_path"),
        {"file_path": {"type": "string", "description": "The absolute path to the file to read"},
         "offset": {"type": "integer", "description": "The line number to start reading from (0-based)"},
         "limit": {"type": "integer", "description": "The number of lines to read (defaults to 2000)"}},
        ["file_path"])
    ed = _cdesc(canon, "opencode", "edit", "Performs exact string replacements in files.")
    add("edit_file", lambda n, keys: "file_path" in keys and "old_string" in keys,
        ed.replace("filePath", "file_path").replace("oldString", "old_string").replace("newString", "new_string"),
        {"file_path": {"type": "string", "description": "The absolute path to the file to modify"},
         "old_string": {"type": "string", "description": "The text to replace"},
         "new_string": {"type": "string", "description": "The text to replace it with (must be different from old_string)"},
         "replace_all": {"type": "boolean", "description": "Replace all occurrences of old_string (default false)"}},
        ["file_path", "old_string", "new_string"])
    wr = _cdesc(canon, "opencode", "write", "Writes a file to the local filesystem.")
    add("write_file", lambda n, keys: "file_path" in keys and "content" in keys,
        wr.replace("filePath", "file_path"),
        {"file_path": {"type": "string", "description": "The absolute path to the file to write (must be absolute, not relative)"},
         "content": {"type": "string", "description": "The content to write to the file"}},
        ["file_path", "content"])
    gr = _cdesc(canon, "opencode", "grep", "Fast content search tool that searches file contents using regular expressions.")
    add("grep", lambda n, keys: "pattern" in keys and ("include" in keys or re.search(r"grep|search|find_in|text|code", n)),
        gr,
        {"pattern": {"type": "string", "description": "The regex pattern to search for in file contents"},
         "path": {"type": "string", "description": "The directory to search in. Defaults to the current working directory."},
         "include": {"type": "string", "description": 'File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}")'},
         "limit": {"type": "integer", "description": "Maximum number of matches to return."}},
        ["pattern"])
    gl = _cdesc(canon, "opencode", "glob", "Fast file pattern matching tool that works with any codebase size.")
    add("glob", lambda n, keys: "pattern" in keys and "include" not in keys,
        gl,
        {"pattern": {"type": "string", "description": "The glob pattern to match files against"},
         "path": {"type": "string", "description": "The directory to search in. If not specified, the current working directory will be used."}},
        ["pattern"])
    add("list_dir", lambda n, keys: ("dir_path" in keys or keys <= {"path", "ignore", "limit", "offset"}) and re.search(r"list|^ls|dir", n),
        "List the files and directories at the given path (non-recursive). Directory entries are suffixed with `/`.",
        {"path": {"type": "string", "description": "The absolute path of the directory to list."},
         "dir_path": {"type": "string", "description": "The absolute path of the directory to list."},
         "offset": {"type": "integer", "description": "Number of entries to skip."},
         "limit": {"type": "integer", "description": "Maximum number of entries to return."},
         "ignore": {"type": "array", "description": "List of glob patterns to ignore."}},
        [])
    add("todo_write", lambda n, keys: "todos" in keys,
        _cdesc(canon, "opencode", "todowrite", "Use this tool to create and manage a structured task list for your current coding session."),
        {"todos": {"type": "array", "description": "The updated todo list"}}, ["todos"])
    add("todo_read", lambda n, keys: not keys and re.search(r"todo_read|read_todo", n),
        _cdesc(canon, "opencode", "todoread", "Use this tool to read your todo list"), {}, [])
    add("apply_patch", lambda n, keys: "input" in keys,
        "Apply a patch to files in the workspace. The patch uses the Codex patch format: a block starting with "
        "`*** Begin Patch` and ending with `*** End Patch`, containing `*** Add File:`, `*** Update File:` and "
        "`*** Delete File:` sections with `@@` hunks whose lines are prefixed by ` `, `+` or `-`.",
        {"input": {"type": "string", "description": "The entire patch text in the Codex patch format."}}, ["input"])
    add("update_plan", lambda n, keys: "plan" in keys,
        "Update the current task plan. Provide the full list of steps with their status so the user can follow progress; "
        "at most one step should be `in_progress` at a time.",
        {"plan": {"type": "array", "description": "The list of plan steps, each `{step, status}` with status one of pending, in_progress, completed."},
         "explanation": {"type": "string", "description": "Optional short explanation of why the plan changed."}},
        ["plan"])
    return C


def classify(name, arg_freq, classes):
    """arg_freq: {key: fraction of calls}. 특이 순서로 검사한다."""
    keys = {k for k, f in arg_freq.items() if f >= MAJOR_FRAC}
    order = ("submit", "todo_read", "think", "finish", "task_tracker", "str_replace_editor", "shell_codex",
             "shell_openhands", "shell_sweagent", "edit_file", "write_file", "read_file", "todo_write",
             "apply_patch", "update_plan", "list_dir", "grep", "glob")
    for c in order:
        if classes[c]["match"](name, keys):
            return c
    return None


def build_schema(name, tool_stats, classes, canon):
    calls = tool_stats["calls"]
    arg_freq = {k: sum(c.values()) / calls for k, c in tool_stats["args"].items()}
    cls = classify(name, arg_freq, classes)
    props = {}
    required = []
    if cls:
        spec = classes[cls]
        for p, s in spec["params"].items():
            if p in arg_freq and arg_freq[p] >= PROP_FRAC or p in spec["required"]:
                props[p] = dict(s)
        for p in spec["required"]:
            if p in props and arg_freq.get(p, 0) >= REQ_FRAC:
                required.append(p)
        desc = spec["desc"]
    else:
        desc = f"Tool `{name}`."
    for k, c in tool_stats["args"].items():
        f = sum(c.values()) / calls
        if f >= PROP_FRAC and k not in props:
            props[k] = {"type": PY2JSON.get(c.most_common(1)[0][0], "string"),
                        "description": f"Parameter `{k}`."}
    # 관측 다수 타입이 표에 적힌 타입과 다르면 관측을 따른다 (모델이 실제로 내는 값이 정답)
    for k, s in props.items():
        c = tool_stats["args"].get(k)
        if c:
            obs = PY2JSON.get(c.most_common(1)[0][0], "string")
            if obs != s.get("type") and not (obs == "integer" and s.get("type") == "number"):
                s["type"] = obs
                s.pop("enum", None) if obs != "string" else None
    schema = {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}
    return schema, cls


# ---------------------------------------------------------------------------
# 3. 사이드카 조립 + 검증
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swe-v3", required=True, help="Nemotron-SFT-SWE-v3/data (parquet 디렉터리)")
    ap.add_argument("--openhands", required=True, help="Nemotron-SFT-SWE-v2/data/swe.jsonl (OpenHands 실제 tools)")
    ap.add_argument("--opencode", required=True, help="Nemotron-SFT-OpenCode-v1 (opencode 실제 tools)")
    ap.add_argument("--out", required=True, help="출력 디렉터리 (alpha_tools_sidecar.json, .summary.md)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--validate", type=int, default=2000, help="표본 행 수로 '호출 ⊆ 선언' 검증 (0 이면 생략)")
    a = ap.parse_args()

    t0 = time.time()
    fam = scan_swe_v3(a.swe_v3, a.workers)
    canon = load_canonical(a.openhands, a.opencode)
    classes = build_classes(canon)
    print(f"[scan] families={len(fam)} rows={sum(d['rows'] for d in fam.values()):,} canonical={len(canon)} ({time.time()-t0:.0f}s)")

    families = {}
    by_name_best = {}     # name -> (calls, schema)
    summary = []
    unclassified = Counter()
    for k, d in sorted(fam.items(), key=lambda x: -x[1]["rows"]):
        if not d["tools"]:
            continue
        top_share = d["toolsets"].most_common(1)[0][1] / max(1, d["rows_tc"])
        declare = "union" if (len(d["tools"]) <= UNION_MAX_TOOLS and top_share >= UNION_TOP_SHARE) else "called"
        tools = {}
        cls_map = {}
        for name, ts in sorted(d["tools"].items(), key=lambda x: -x[1]["calls"]):
            schema, cls = build_schema(name, ts, classes, canon)
            tools[name] = schema
            cls_map[name] = cls or "-"
            if cls is None:
                unclassified[name] += ts["calls"]
            if ts["calls"] > by_name_best.get(name, (0, None))[0]:
                by_name_best[name] = (ts["calls"], schema)
        families[k] = {"declare": declare, "rows": d["rows"], "rows_tc": d["rows_tc"],
                       "prefix": d["prefix"], "tools": tools}
        summary.append((k, d["rows"], d["rows_tc"], declare, round(top_share, 3), cls_map))
    sidecar = {"generated": time.strftime("%Y-%m-%d %H:%M"), "source": a.swe_v3,
               "rule": {"major_frac": MAJOR_FRAC, "prop_frac": PROP_FRAC, "req_frac": REQ_FRAC,
                        "union_max_tools": UNION_MAX_TOOLS, "union_top_share": UNION_TOP_SHARE},
               "families": families, "by_name": {n: s for n, (_, s) in by_name_best.items()}}
    os.makedirs(a.out, exist_ok=True)
    out_json = os.path.join(a.out, "alpha_tools_sidecar.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=1)

    # 검증: 표본 행에 inject_tools 적용 → 호출 ⊆ 선언, 선언 name/parameters 비어있지 않음
    val = {}
    if a.validate:
        import pyarrow.parquet as pq
        fs = sorted(glob.glob(os.path.join(a.swe_v3, "*.parquet")))
        rng = random.Random(20260909)
        rows = []
        for f in rng.sample(fs, min(8, len(fs))):
            tbl = pq.ParquetFile(f).read_row_group(0).to_pylist()
            rows += rng.sample(tbl, min(a.validate // 8, len(tbl)))
        info = Counter()
        bad = 0
        for r in rows:
            norm, why = normalize_row(r, tools_sidecar=sidecar, info=info)
            if norm is None:
                info[f"drop_{why}"] += 1
                continue
            called = {tc["function"]["name"] for m in norm["messages"] for tc in (m.get("tool_calls") or [])}
            declared = {t["function"]["name"] for t in (norm["tools"] or [])}
            if not called <= declared:
                bad += 1
            for t in norm["tools"] or []:
                fn = t["function"]
                assert fn["name"] and isinstance(fn["parameters"], dict), fn
        val = {"rows": len(rows), "uncovered_rows": bad, **{k: int(v) for k, v in info.items()}}
        print(f"[validate] {val}")
        assert bad == 0, "호출된 도구가 선언되지 않은 행이 있다"

    lines = [f"# SWE-v3 도구 선언 사이드카 요약 — {sidecar['generated']}",
             f"패밀리 {len(families)} (tool_call 있는 것만) · by_name {len(sidecar['by_name'])} · 검증 {val}", "",
             "| family | rows | tc rows | declare | top toolset share | tools (class) |", "|---|---|---|---|---|---|"]
    for k, rows_, tc, declare, ts, cm in summary:
        tl = ", ".join(f"{n}({c})" for n, c in cm.items())
        lines.append(f"| {k} | {rows_:,} | {tc:,} | {declare} | {ts} | {tl} |")
    if unclassified:
        lines += ["", "미판정(최소 스키마): " + ", ".join(f"{n}×{c}" for n, c in unclassified.most_common())]
    with open(os.path.join(a.out, "alpha_tools_sidecar.summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    print(f"[done] {out_json} ({time.time()-t0:.0f}s) unclassified={dict(unclassified)}")


if __name__ == "__main__":
    main()
