"""lb_proxy — 세션 고정 라우팅·최소 부하 배정·연결 실패 재배정 (2026-09-17)."""
from __future__ import annotations

import http.server
import json
import socketserver
import sys
import threading
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples/alpha/eval_sft"))

import lb_proxy as L  # noqa: E402


def _chat(system, user, model="alpha", extra=None):
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    msgs += extra or []
    return json.dumps({"model": model, "messages": msgs}).encode()


# ---------------------------------------------------------------- session_key
def test_key_same_session_across_turns():
    t1 = _chat("sys", "task A")
    t3 = _chat("sys", "task A", extra=[{"role": "assistant", "content": "x"}, {"role": "user", "content": "obs"}])
    assert L.session_key(t1) == L.session_key(t3)


def test_key_differs_by_first_user_and_system():
    assert L.session_key(_chat("sys", "task A")) != L.session_key(_chat("sys", "task B"))
    assert L.session_key(_chat("sys1", "task A")) != L.session_key(_chat("sys2", "task A"))


def test_key_handles_content_parts_and_missing_system():
    parts = [{"type": "text", "text": "task A"}]
    k1 = L.session_key(json.dumps({"model": "alpha", "messages": [{"role": "user", "content": parts}]}).encode())
    k2 = L.session_key(json.dumps({"model": "alpha", "messages": [{"role": "user", "content": parts}]}).encode())
    assert k1 == k2 and k1 is not None


def test_key_none_for_non_chat():
    assert L.session_key(None) is None
    assert L.session_key(b"not json") is None
    assert L.session_key(json.dumps({"model": "alpha", "prompt": "x"}).encode()) is None
    assert L.session_key(json.dumps({"model": "alpha", "messages": [{"role": "system", "content": "s"}]}).encode()) is None


# ---------------------------------------------------------------- Router
def test_router_sticky_and_least_loaded():
    r = L.Router([8000, 8001, 8002])
    a = r.pick("A")          # 처음 → 최소 부하(전부 0, 라운드로빈 첫 항목)
    b = r.pick("B")          # A 가 잡고 있으니 다른 백엔드
    assert a != b
    assert r.pick("A") == a  # 고정
    assert r.pick("A") == a
    snap = r.snapshot()
    assert snap["sessions"] == 2 and snap["sticky_hit"] == 2 and snap["sticky_new"] == 2
    assert snap["backends"][a] == 3 and snap["backends"][b] == 1


def test_router_unkeyed_goes_least_loaded():
    r = L.Router([8000, 8001])
    p1 = r.pick(None)
    p2 = r.pick(None)
    assert {p1, p2} == {8000, 8001}
    r.done(p1)
    assert r.pick(None) == p1  # 이제 p1 이 덜 바쁘다


def test_router_no_sticky_flag():
    r = L.Router([8000, 8001], sticky=False)
    a = r.pick("A")
    b = r.pick("A")
    assert a != b and r.snapshot()["sessions"] == 0


def test_router_reassign_moves_session():
    r = L.Router([8000, 8001])
    a = r.pick("A")
    r.done(a)
    b = r.reassign("A", a)
    assert b != a and r.pick("A") == b
    r.done(b); r.done(b)
    assert all(v == 0 for v in r.snapshot()["backends"].values())


def test_router_session_lru_bound():
    r = L.Router([8000], max_sessions=2)
    for k in ("A", "B", "C"):
        r.done(r.pick(k))
    assert r.snapshot()["sessions"] == 2


# ---------------------------------------------------------------- end-to-end (스레드 백엔드 2개)
class _Echo(http.server.BaseHTTPRequestHandler):
    port = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        data = json.dumps({"served_by": self.server.server_address[1]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        data = b'{"object":"list"}'
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


class _Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _start(handler, port=0):
    s = _Srv(("127.0.0.1", port), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def _post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_e2e_sticky_and_failover():
    b1, b2 = _start(_Echo), _start(_Echo)
    p1, p2 = b1.server_address[1], b2.server_address[1]
    router = L.Router([p1, p2])
    proxy = _start(L.make_handler(router))
    pp = proxy.server_address[1]
    try:
        first = _post(pp, _chat("sys", "task A"))["served_by"]
        assert all(_post(pp, _chat("sys", "task A", extra=[{"role": "user", "content": f"t{i}"}]))["served_by"] == first
                   for i in range(5))
        other = _post(pp, _chat("sys", "task B"))["served_by"]
        assert other != first
        # 고정 백엔드 사망 → 재배정 후 응답
        dead = b1 if first == p1 else b2
        dead.shutdown(); dead.server_close()   # 소켓까지 닫아야 connection refused 가 난다
        got = _post(pp, _chat("sys", "task A"))["served_by"]
        assert got != first and got in (p1, p2)
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{pp}/lb/stats", timeout=5).read())
        assert st["reassigned"] == 1 and st["sessions"] == 2
        assert all(v == 0 for v in st["backends"].values())
    finally:
        for s_ in (proxy, b1, b2):
            try:
                s_.shutdown(); s_.server_close()
            except Exception:  # noqa: BLE001
                pass
