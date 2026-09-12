#!/usr/bin/env python3
"""search_server.py — 로컬 BM25 검색 HTTP 서버. 결과를 **Tavily 형식**(NVIDIA Agentic-v2 search·우리 평가 하니스와 동일)으로 반환.
  POST /search  {"query": str, "max_results": 10}  →  {"query", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url","title","content","score","raw_content"}], "response_time"}
  같은 문서(doc_id 앞부분)의 passage 는 상위 1개만 남겨 다양성 확보. content = passage 본문(≤ 700자), raw_content = null.
사용: python3 search_server.py --index out/index/bm25 --port 8600 [--threads 32]
"""
import argparse, json, time, threading, re, sys, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import bm25s
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from build_index import tokenize

class S:
    idx = None; docs = None; lock = threading.Lock()

def search(query, k):
    toks = tokenize(query)
    if not toks: return []
    with S.lock:
        ids, scores = S.idx.retrieve([toks], k=min(k * 4, 60))
    out, seen = [], set()
    for i, sc in zip(ids[0].tolist(), scores[0].tolist()):
        d = S.docs[i]; base = d["doc_id"].split("#")[0]
        if base in seen: continue
        seen.add(base); out.append({"url": d["url"], "title": d["title"], "content": d["text"], "score": round(float(sc), 4), "raw_content": None})
        if len(out) >= k: break
    return out

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/health": self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
        self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        t0 = time.time(); q = str(body.get("query", ""))[:500]; k = int(body.get("max_results", 10))
        res = {"query": q, "follow_up_questions": None, "answer": None, "images": [], "results": search(q, k), "response_time": round(time.time() - t0, 3)}
        data = json.dumps(res, ensure_ascii=False).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--index", required=True); ap.add_argument("--port", type=int, default=8600); a = ap.parse_args()
    t0 = time.time(); S.idx = bm25s.BM25.load(a.index); S.docs = [json.loads(l) for l in open(os.path.join(a.index, "docs.jsonl"))]
    print(f"loaded {len(S.docs)} passages in {time.time()-t0:.0f}s; serving :{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()

if __name__ == "__main__":
    main()
