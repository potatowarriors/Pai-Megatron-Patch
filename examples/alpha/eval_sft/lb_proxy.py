"""단일 GPU vLLM 서버 N개를 하나의 OpenAI 엔드포인트로 묶는 라운드로빈 프록시.

vLLM data-parallel 이 이 환경(CUDA13 compat + NCCL)에서 munmap 크래시로 불가 →
검증된 단일 GPU 서버(DP1) N개를 띄우고 이 프록시가 요청을 분배한다.
표준 라이브러리만 사용 (aiohttp 불요).

사용: python3 eval_sft/lb_proxy.py --port 8100 --backends 8000,8001,...,8007
lm_eval 는 http://localhost:8100/v1 로 붙는다.
"""
import argparse, http.server, socketserver, threading, urllib.request, urllib.error, itertools

def make_handler(backends):
    rr = itertools.cycle(backends)
    lock = threading.Lock()
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def _pick(self):
            with lock: return next(rr)
        def _proxy(self, body=None):
            port = self._pick()
            url = f"http://127.0.0.1:{port}{self.path}"
            hdr = {k: v for k, v in self.headers.items() if k.lower() != "host"}
            req = urllib.request.Request(url, data=body, headers=hdr, method=self.command)
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    data = r.read(); code = r.status; ctype = r.headers.get("Content-Type","application/json")
            except urllib.error.HTTPError as e:
                data = e.read(); code = e.code; ctype = e.headers.get("Content-Type","application/json")
            except Exception as e:  # noqa: BLE001
                data = f'{{"error":{{"message":"proxy: {e}"}}}}'.encode(); code = 502; ctype="application/json"
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self): self._proxy()
        def do_POST(self):
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
    a = ap.parse_args()
    ports = [int(p) for p in a.backends.split(",")]
    srv = Threaded(("0.0.0.0", a.port), make_handler(ports))
    print(f"[lb] :{a.port} -> {ports}", flush=True)
    srv.serve_forever()
