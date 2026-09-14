"""Unit tests for examples/alpha/eval_sft/tau_proxy.py (τ-bench think 분리·재부착 프록시).

Run from the repo root:
    python3 -m pytest tests/test_tau_proxy.py -v
No GPU, no network beyond loopback: a fake upstream HTTP server stands in for the fleet.
"""
from __future__ import annotations

import http.server
import json
import socketserver
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples/alpha/eval_sft"))

import tau_proxy as tp  # noqa: E402


# ── pure functions ───────────────────────────────────────────────────────────

def test_split_think_forms():
    r = tp.split_think("reason</think>\n\nanswer")
    assert r == {"think": "reason", "sep": "\n\n", "answer": "answer", "unclosed": False}
    r = tp.split_think("<think>\nreason</think>answer")
    assert r["think"] == "reason" and r["sep"] == "" and r["answer"] == "answer"
    r = tp.split_think("<think>\nstill thinking")
    assert r["unclosed"] and r["answer"] == "" and r["think"] is None
    r = tp.split_think("plain")
    assert r["think"] is None and not r["unclosed"] and r["answer"] == "plain"
    assert tp.split_think(None)["think"] is None


def test_canon_invariance():
    a = {"role": "assistant", "content": "<think>\nx</think>\n\nok",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "f", "arguments": '{"b": 2, "a": "1"}'}}]}
    b = {"role": "assistant", "content": "ok", "reasoning_content": "x",
         "tool_calls": [{"id": "call_999", "type": "function",
                         "function": {"name": "f", "arguments": '{"a":"1","b":2}'}}]}
    assert tp.canon(a) == tp.canon(b)
    assert tp.step(tp.H0, a) == tp.step(tp.H0, b)
    # dict arguments (already parsed) hash like the string form
    c = {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": {"b": 2, "a": "1"}}}]}
    d = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": '{"a": "1", "b": 2}'}}]}
    assert tp.canon(c) == tp.canon(d)
    # visible text differs -> different key
    e = dict(b, content="not ok")
    assert tp.step(tp.H0, b) != tp.step(tp.H0, e)


# ── fake upstream ────────────────────────────────────────────────────────────

class FakeUpstream:
    """Returns canned chat completions; records every request body it received."""

    def __init__(self):
        self.requests: list[dict] = []
        self.raw: list[bytes] = []
        self.queue: list[dict] = []          # per-request overrides, consumed FIFO
        self.default = {"content": "d</think>\ndefault", "tool_calls": None, "finish_reason": "stop"}
        self.lock = threading.Lock()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                with outer.lock:
                    outer.raw.append(raw)
                    try:
                        outer.requests.append(json.loads(raw))
                    except Exception:  # noqa: BLE001
                        outer.requests.append({"_raw": raw.decode(errors="replace")})
                    spec = outer.queue.pop(0) if outer.queue else outer.default
                msg = {"role": "assistant", "content": spec.get("content")}
                if spec.get("tool_calls"):
                    msg["tool_calls"] = spec["tool_calls"]
                if spec.get("reasoning") is not None:          # reasoning 파서가 켜진 fleet 흉내 (vLLM 0.25.1 필드명)
                    msg["reasoning"] = spec["reasoning"]
                body = json.dumps({"id": "x", "object": "chat.completion", "model": "alpha",
                                   "choices": [{"index": 0, "message": msg,
                                                "finish_reason": spec.get("finish_reason", "stop")}],
                                   "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                body = b'{"object":"list","data":[{"id":"alpha"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.srv = tp.Threaded(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def start_proxy(upstream_port, **kw):
    proxy = tp.Proxy(f"http://127.0.0.1:{upstream_port}", flush_every=0, **kw)
    srv = tp.Threaded(("127.0.0.1", 0), tp.make_handler(proxy))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return proxy, srv.server_address[1]


def post(port, body, path="/v1/chat/completions"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read())


SYS = {"role": "system", "content": "policy"}
GREET = {"role": "assistant", "content": "Hi! How can I help you today?"}
USER1 = {"role": "user", "content": "I want to change my flight."}
TC = [{"id": "call_1", "type": "function", "function": {"name": "get_user", "arguments": '{"user_id": "u1"}'}}]
TC_RESENT = [{"id": "call_XYZ", "type": "function", "function": {"name": "get_user", "arguments": '{"user_id":"u1"}'}}]
TOOL = {"role": "tool", "tool_call_id": "call_XYZ", "content": '{"name": "A"}'}


@pytest.fixture
def stack():
    up = FakeUpstream()
    proxy, port = start_proxy(up.port)
    return up, proxy, port


def test_strip_and_reattach_roundtrip(stack):
    up, proxy, port = stack
    # turn 1: tool call with think
    up.queue.append({"content": "reason A</think>\n\n", "tool_calls": TC, "finish_reason": "tool_calls"})
    _, r1 = post(port, {"model": "alpha", "messages": [SYS, GREET, USER1], "seed": 123, "tools": []})
    m1 = r1["choices"][0]["message"]
    assert m1["content"] is None                      # nothing visible -> null (tool turn)
    assert m1["reasoning_content"] == "reason A"
    assert m1["tool_calls"] == TC
    sent = up.requests[-1]
    assert "seed" not in sent and sent["skip_special_tokens"] is False
    s = proxy.snapshot()
    assert s["think_stripped"] == 1 and s["tool_calls"] == 1 and s["mixed_content_and_tools"] == 0
    assert s["miss_first_assistant"] == 1 and s["miss"] == 0 and s["seed_stripped"] == 1 and s["sst_forced"] == 1

    # turn 2: harness re-sends the visible turn (different call id, re-serialised args) + tool result
    asst1 = {"role": "assistant", "content": None, "tool_calls": TC_RESENT}
    up.queue.append({"content": "reason B</think>\nSure, here is your info."})
    _, r2 = post(port, {"model": "alpha", "messages": [SYS, GREET, USER1, asst1, TOOL]})
    assert r2["choices"][0]["message"]["content"] == "Sure, here is your info."
    sent = up.requests[-1]["messages"]
    assert sent[3]["content"] == "<think>\nreason A</think>\n\n"    # byte-identical original restored
    assert sent[3]["tool_calls"] == TC_RESENT                        # tool_calls untouched
    assert sent[1]["content"] == GREET["content"]                     # greeting untouched
    s = proxy.snapshot()
    assert s["reinlined"] == 1 and s["miss"] == 0 and s["miss_first_assistant"] == 2

    # turn 3: both historical assistant turns restored
    asst2 = {"role": "assistant", "content": "Sure, here is your info."}
    up.queue.append({"content": "reason C</think>Anything else?"})
    _, r3 = post(port, {"model": "alpha", "messages": [SYS, GREET, USER1, asst1, TOOL, asst2,
                                                       {"role": "user", "content": "thanks"}]})
    sent = up.requests[-1]["messages"]
    assert sent[3]["content"] == "<think>\nreason A</think>\n\n"
    assert sent[5]["content"] == "<think>\nreason B</think>\nSure, here is your info."
    assert r3["choices"][0]["message"]["content"] == "Anything else?"
    s = proxy.snapshot()
    assert s["reinlined"] == 3 and s["miss"] == 0 and s["miss_rate"] == 0.0


def test_two_trials_same_prefix_do_not_cross(stack):
    up, proxy, port = stack
    tc_b = [{"id": "c", "type": "function", "function": {"name": "get_user", "arguments": '{"user_id": "u2"}'}}]
    up.queue.append({"content": "think-1</think>", "tool_calls": TC, "finish_reason": "tool_calls"})
    up.queue.append({"content": "think-2</think>", "tool_calls": tc_b, "finish_reason": "tool_calls"})
    post(port, {"messages": [SYS, GREET, USER1]})
    post(port, {"messages": [SYS, GREET, USER1]})
    up.queue.append({"content": "x</think>a"}); up.queue.append({"content": "y</think>b"})
    post(port, {"messages": [SYS, GREET, USER1, {"role": "assistant", "content": None, "tool_calls": TC}, TOOL]})
    post(port, {"messages": [SYS, GREET, USER1, {"role": "assistant", "content": None, "tool_calls": tc_b}, TOOL]})
    assert up.requests[-2]["messages"][3]["content"] == "<think>\nthink-1</think>"
    assert up.requests[-1]["messages"][3]["content"] == "<think>\nthink-2</think>"
    assert proxy.snapshot()["miss"] == 0


def test_mixed_text_and_tool_turn_is_kept_and_counted(stack):
    up, proxy, port = stack
    up.queue.append({"content": "r</think>\nLet me look that up.", "tool_calls": TC, "finish_reason": "tool_calls"})
    _, r = post(port, {"messages": [SYS, GREET, USER1]})
    assert r["choices"][0]["message"]["content"] == "Let me look that up."
    assert proxy.snapshot()["mixed_content_and_tools"] == 1
    # re-sent with the visible preamble -> restored with think in front
    up.queue.append({"content": "s</think>done"})
    post(port, {"messages": [SYS, GREET, USER1, {"role": "assistant", "content": "Let me look that up.", "tool_calls": TC}, TOOL]})
    assert up.requests[-1]["messages"][3]["content"] == "<think>\nr</think>\nLet me look that up."


def test_reasoning_content_field_is_inlined(stack):
    up, proxy, port = stack
    post(port, {"messages": [SYS, {"role": "assistant", "content": "hello", "reasoning_content": "why"}, USER1]})
    sent = up.requests[-1]["messages"]
    assert sent[1]["content"] == "<think>\nwhy</think>hello" and "reasoning_content" not in sent[1]
    assert proxy.snapshot()["reasoning_field_inlined"] == 1


def test_unclosed_think_and_length(stack):
    up, proxy, port = stack
    up.queue.append({"content": "<think>\nnever closes", "finish_reason": "length"})
    _, r = post(port, {"messages": [SYS, GREET, USER1]})
    assert r["choices"][0]["message"]["content"] == ""
    s = proxy.snapshot()
    assert s["think_unclosed"] == 1 and s["finish_length"] == 1 and s["cache_entries"] == 0


def test_think_absent_passthrough(stack):
    up, proxy, port = stack
    up.queue.append({"content": "no think here"})
    _, r = post(port, {"messages": [SYS, GREET, USER1]})
    assert r["choices"][0]["message"]["content"] == "no think here"
    assert proxy.snapshot()["think_absent"] == 1


def test_no_reattach_counts_but_does_not_modify():
    up = FakeUpstream()
    proxy, port = start_proxy(up.port, reattach=False)
    up.queue.append({"content": "t</think>", "tool_calls": TC, "finish_reason": "tool_calls"})
    post(port, {"messages": [SYS, GREET, USER1]})
    up.queue.append({"content": "u</think>ok"})
    post(port, {"messages": [SYS, GREET, USER1, {"role": "assistant", "content": None, "tool_calls": TC}, TOOL]})
    assert up.requests[-1]["messages"][3]["content"] is None
    s = proxy.snapshot()
    assert s["reinlined"] == 1 and s["miss"] == 0 and s["reattach"] is False


def test_keep_seed():
    up = FakeUpstream()
    _, port = start_proxy(up.port, keep_seed=True)
    post(port, {"messages": [SYS, USER1], "seed": 7})
    assert up.requests[-1]["seed"] == 7


def test_stream_and_other_paths_pass_through(stack):
    up, proxy, port = stack
    body = {"messages": [SYS, USER1], "stream": True, "seed": 5}
    post(port, body)
    assert json.loads(up.raw[-1]) == body                 # untouched bytes
    st, models = post(port, {}, path="/v1/models") if False else (None, None)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 200
    s = proxy.snapshot()
    assert s["passthrough"] == 2 and s["requests"] == 0


def test_stats_endpoint_and_flush(stack, tmp_path):
    up, proxy, port = stack
    proxy.stats_file = str(tmp_path / "stats.json")
    up.queue.append({"content": "a</think>b"})
    post(port, {"messages": [SYS, USER1]})
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=10) as r:
        d = json.loads(r.read())
    assert d["requests"] == 1 and d["think_stripped"] == 1
    proxy.flush()
    assert json.load(open(proxy.stats_file))["requests"] == 1


def test_eviction_bounds():
    up = FakeUpstream()
    proxy, port = start_proxy(up.port, max_entries=2)
    for i in range(3):
        up.queue.append({"content": f"t{i}</think>a{i}"})
        post(port, {"messages": [SYS, {"role": "user", "content": f"q{i}"}]})
    assert proxy.snapshot()["cache_entries"] == 2


def test_concurrency_counters_add_up(stack):
    up, proxy, port = stack
    errors = []

    def worker(k):
        try:
            for j in range(20):
                _, r = post(port, {"messages": [SYS, {"role": "user", "content": f"{k}-{j}"}]})
                assert r["choices"][0]["message"]["content"] == "default"
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(16)]
    [t.start() for t in ts]; [t.join(60) for t in ts]
    assert not errors
    s = proxy.snapshot()
    assert s["requests"] == 320 and s["think_stripped"] == 320 and s["cache_entries"] == 320


def test_reasoning_field_path_text_and_tool_turns(stack):
    """reasoning 파서가 켜진 fleet: think 는 `reasoning` 필드, content 는 답변만 → 캐시·복원은 텍스트 경로와 동일."""
    up, proxy, port = stack
    up.queue.append({"content": None, "reasoning": "need lookup", "tool_calls": TC, "finish_reason": "tool_calls"})
    _, r1 = post(port, {"messages": [SYS, GREET, USER1]})
    m1 = r1["choices"][0]["message"]
    assert m1["content"] is None and m1["reasoning_content"] == "need lookup" and m1["tool_calls"] == TC
    up.queue.append({"content": "Here you go.", "reasoning": "answer now"})
    _, r2 = post(port, {"messages": [SYS, GREET, USER1, {"role": "assistant", "content": None, "tool_calls": TC_RESENT}, TOOL]})
    assert r2["choices"][0]["message"]["content"] == "Here you go."
    assert up.requests[-1]["messages"][3]["content"] == "<think>\nneed lookup</think>"
    up.queue.append({"content": "bye", "reasoning": "z"})
    post(port, {"messages": [SYS, GREET, USER1, {"role": "assistant", "content": None, "tool_calls": TC_RESENT}, TOOL,
                             {"role": "assistant", "content": "Here you go."}, {"role": "user", "content": "thanks"}]})
    sent = up.requests[-1]["messages"]
    assert sent[3]["content"] == "<think>\nneed lookup</think>" and sent[5]["content"] == "<think>\nanswer now</think>Here you go."
    s = proxy.snapshot()
    assert s["think_from_field"] == 3 and s["think_stripped"] == 0 and s["miss"] == 0 and s["reinlined"] == 3


def test_reasoning_field_unclosed_not_cached(stack):
    up, proxy, port = stack
    up.queue.append({"content": "", "reasoning": "partial thinking", "finish_reason": "length"})
    _, r = post(port, {"messages": [SYS, GREET, USER1]})
    assert r["choices"][0]["message"]["content"] == ""
    s = proxy.snapshot()
    assert s["think_unclosed"] == 1 and s["think_from_field"] == 0 and s["cache_entries"] == 0


def test_double_close_splits_at_last_marker(stack):
    up, proxy, port = stack
    up.queue.append({"content": "a</think>b</think>c"})
    _, r = post(port, {"messages": [SYS, GREET, USER1]})
    assert r["choices"][0]["message"]["content"] == "c"          # 상대역에는 마지막 답변부만
    assert r["choices"][0]["message"]["reasoning_content"] == "a</think>b"
    up.queue.append({"content": "d</think>e"})
    post(port, {"messages": [SYS, GREET, USER1, {"role": "assistant", "content": "c"}, {"role": "user", "content": "?"}]})
    assert up.requests[-1]["messages"][3]["content"] == "<think>\na</think>b</think>c"   # 원문 그대로 복원


def test_no_greeting_counts_first_assistant_miss_as_miss():
    """합성 인사가 없는 하니스(--no-greeting): 첫 assistant 의 miss 는 miss 로, hit 은 reinlined 로."""
    up = FakeUpstream()
    proxy, port = start_proxy(up.port, greeting=False)
    # 캐시에 없는 첫 assistant(이 프록시가 생성하지 않은 턴) → miss
    up.queue.append({"content": "t</think>ok"})
    post(port, {"messages": [SYS, {"role": "assistant", "content": "unknown turn"}, USER1]})
    s = proxy.snapshot()
    assert s["miss"] == 1 and s["miss_first_assistant"] == 0 and s["greeting"] is False
    # 프록시가 만든 턴은 첫 assistant 여도 정상 복원
    up.queue.append({"content": "u</think>next"})
    post(port, {"messages": [SYS, USER1, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "?"}]})
    assert up.requests[-1]["messages"][2]["content"] == "<think>\nt</think>ok"
    s = proxy.snapshot()
    assert s["reinlined"] == 1 and s["miss"] == 1
