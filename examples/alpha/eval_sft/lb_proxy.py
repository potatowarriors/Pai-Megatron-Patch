"""단일 GPU vLLM 서버 N개를 하나의 OpenAI 엔드포인트로 묶는 프록시 — **세션 고정(sticky) + 최소 부하 배정**.

vLLM data-parallel 이 이 환경(CUDA13 compat + NCCL)에서 munmap 크래시로 불가 → 검증된 단일 GPU 서버(DP1) N개를
띄우고 이 프록시가 요청을 분배한다. 표준 라이브러리만 사용 (aiohttp 불요).

라우팅 (2026-09-17, 사용자 결정 — 하이브리드 prefix caching 과 함께):
  - 채팅 요청은 **세션 키**(system 메시지 + 첫 user 메시지 해시)로 백엔드를 고정한다. 에이전틱 궤적(SWE 평균 119턴,
    프롬프트 5.7만 토큰)은 턴마다 같은 접두사를 다시 보내므로, 같은 GPU 로 가야 vLLM prefix cache 가 살아난다.
    라운드로빈이던 때는 N 대 중 1/N 만 적중할 수 있었다.
  - 처음 보는 세션은 **in-flight 최소** 백엔드에 배정한다(동률이면 라운드로빈). 단발 요청(lm_eval)은 세션이 매번
    새로우므로 사실상 최소 부하 분산이 된다.
  - 고정 백엔드로의 연결 자체가 실패하면(프로세스 사망) 다른 백엔드로 재배정해 1회 재시도한다. HTTP 오류(4xx/5xx 응답)는
    그대로 전달한다 — 모델 응답이므로.
  - GET /lb/stats 로 배정 상태를 본다(백엔드별 in-flight·세션 수·sticky 적중/신규).

사용: python3 eval_sft/lb_proxy.py --port 8100 --backends 8000,8001,...,8007 [--no-sticky] [--max-sessions 200000]
lm_eval 는 http://localhost:8100/v1 로 붙는다.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import socketserver
import threading
import urllib.error
import urllib.request
from collections import OrderedDict


def session_key(body: bytes | None) -> str | None:
    """채팅 요청 본문에서 세션 키를 뽑는다. 채팅이 아니거나 user 메시지가 없으면 None(비고정)."""
    if not body:
        return None
    try:
        obj = json.loads(body)
    except Exception:  # noqa: BLE001
        return None
    msgs = obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(msgs, list):
        return None
    system = None
    first_user = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system" and system is None:
            system = m.get("content")
        elif role == "user" and first_user is None:
            first_user = m.get("content")
        if first_user is not None:
            break
    if first_user is None:
        return None

    def norm(c):
        return c if isinstance(c, str) else json.dumps(c, sort_keys=True, ensure_ascii=False)

    h = hashlib.sha1()
    h.update(str(obj.get("model", "")).encode())
    h.update(b"\x00")
    h.update(norm(system or "").encode())
    h.update(b"\x00")
    h.update(norm(first_user).encode())
    return h.hexdigest()


class Router:
    """백엔드 선택. 스레드 안전. pick() 으로 받은 포트는 반드시 done() 으로 돌려준다."""

    def __init__(self, backends: list[int], sticky: bool = True, max_sessions: int = 200_000):
        self.backends = list(backends)
        self.sticky = sticky
        self.max_sessions = max_sessions
        self.inflight = {p: 0 for p in self.backends}
        self.sessions: OrderedDict[str, int] = OrderedDict()
        self.rr_idx = 0   # 동률 타이브레이크용 회전 시작점
        self.lock = threading.Lock()
        self.stats = {"sticky_hit": 0, "sticky_new": 0, "unkeyed": 0, "reassigned": 0}

    def _least_loaded(self) -> int:
        # 동률이면 회전 시작점부터 처음 만나는 백엔드 — 시작 직후 전부 0 일 때 한 백엔드에 몰리지 않게.
        n = len(self.backends)
        start = self.rr_idx
        self.rr_idx = (self.rr_idx + 1) % n
        best = None
        for i in range(n):
            p = self.backends[(start + i) % n]
            if best is None or self.inflight[p] < self.inflight[best]:
                best = p
        return best

    def pick(self, key: str | None) -> int:
        with self.lock:
            if key is None or not self.sticky:
                self.stats["unkeyed"] += 1
                port = self._least_loaded()
            elif key in self.sessions:
                self.stats["sticky_hit"] += 1
                port = self.sessions[key]
                self.sessions.move_to_end(key)
            else:
                self.stats["sticky_new"] += 1
                port = self._least_loaded()
                self.sessions[key] = port
                while len(self.sessions) > self.max_sessions:
                    self.sessions.popitem(last=False)
            self.inflight[port] += 1
            return port

    def reassign(self, key: str | None, bad_port: int) -> int:
        """고정 백엔드 연결 실패 → 다른 백엔드로. 호출 전 bad_port 는 done() 돼 있어야 한다."""
        with self.lock:
            self.stats["reassigned"] += 1
            candidates = [p for p in self.backends if p != bad_port] or self.backends
            port = min(candidates, key=lambda p: self.inflight[p])
            if key is not None and self.sticky:
                self.sessions[key] = port
                self.sessions.move_to_end(key)
            self.inflight[port] += 1
            return port

    def done(self, port: int) -> None:
        with self.lock:
            self.inflight[port] = max(0, self.inflight[port] - 1)

    def snapshot(self) -> dict:
        with self.lock:
            return {"backends": dict(self.inflight), "sessions": len(self.sessions),
                    "sticky": self.sticky, **self.stats}


def make_handler(router: Router):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # noqa: D102
            pass

        def _forward(self, port: int, body: bytes | None):
            url = f"http://127.0.0.1:{port}{self.path}"
            hdr = {k: v for k, v in self.headers.items() if k.lower() != "host"}
            req = urllib.request.Request(url, data=body, headers=hdr, method=self.command)
            with urllib.request.urlopen(req, timeout=1800) as r:
                return r.status, r.headers.get("Content-Type", "application/json"), r.read()

        def _proxy(self, body: bytes | None = None):
            if self.command == "GET" and self.path == "/lb/stats":
                data = json.dumps(router.snapshot()).encode()
                self._reply(200, "application/json", data)
                return
            key = session_key(body) if self.command == "POST" else None
            port = router.pick(key)
            try:
                try:
                    code, ctype, data = self._forward(port, body)
                except urllib.error.HTTPError as e:
                    code, ctype, data = e.code, e.headers.get("Content-Type", "application/json"), e.read()
                except urllib.error.URLError as e:
                    # 연결 수준 실패(백엔드 사망·거부) → 다른 백엔드로 1회 재시도.
                    router.done(port)
                    port = router.reassign(key, port)
                    try:
                        code, ctype, data = self._forward(port, body)
                    except urllib.error.HTTPError as e2:
                        code, ctype, data = e2.code, e2.headers.get("Content-Type", "application/json"), e2.read()
                    except Exception as e2:  # noqa: BLE001
                        code, ctype, data = 502, "application/json", f'{{"error":{{"message":"proxy: {e2} (after {e})"}}}}'.encode()
                except Exception as e:  # noqa: BLE001
                    code, ctype, data = 502, "application/json", f'{{"error":{{"message":"proxy: {e}"}}}}'.encode()
            finally:
                router.done(port)
            self._reply(code, ctype, data)

        def _reply(self, code: int, ctype: str, data: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            self._proxy()

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            self._proxy(self.rfile.read(n) if n else None)

    return H


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--backends", required=True, help="comma-separated backend ports")
    ap.add_argument("--no-sticky", action="store_true", help="세션 고정 끄기(최소 부하 분산만)")
    ap.add_argument("--max-sessions", type=int, default=200_000)
    a = ap.parse_args()
    ports = [int(p) for p in a.backends.split(",")]
    router = Router(ports, sticky=not a.no_sticky, max_sessions=a.max_sessions)
    srv = Threaded(("0.0.0.0", a.port), make_handler(router))
    print(f"[lb] :{a.port} -> {ports} sticky={router.sticky}", flush=True)
    srv.serve_forever()
