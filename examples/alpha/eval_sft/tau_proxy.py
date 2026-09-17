#!/usr/bin/env python3
"""tau_proxy.py — τ-bench 하니스(tau2-bench)와 alpha fleet 사이의 think 분리·재부착 프록시.

왜 필요한가 (docs/SFT_BENCHMARKS.md §3.13):
  1. 에이전틱 fleet 는 reasoning 파서 없이 뜬다(G2: `</think>` 가 content 에 살아남아야 한다).
     그래서 응답 content 는 `{think}</think>{answer}` 다. 그대로 두면 상대역(user simulator)과
     COMMUNICATE 채점기가 think 를 "agent 의 발화"로 읽는다.
  2. tau2-bench 는 응답의 content·tool_calls 만 저장하고 reasoning_content 는 읽지도 재전송하지도
     않는다. 규칙 5(INTERLEAVED_THINKING §7: 하니스가 reasoning 을 재전달해야 think 가 보존)를
     하니스 수정 없이 지키려면 프록시가 (a) 응답에서 think 를 떼고 (b) 다음 요청의 히스토리에
     원문을 복원해야 한다.
  3. 템플릿(tokenizer_v5/chat_template.jinja:119)은 히스토리 assistant content 에 `<think>…</think>`
     가 인라인이면 그대로 렌더한다 → 복원 = content 원문을 되돌려 놓는 것으로 충분하다.

키 = 체인 해시. h0 = sha256("tau_proxy_v1"), h_i = sha256(h_{i-1} ‖ canon(m_i)).
canon 은 think 유무·tool-call id·인자 직렬화 공백/키순서에 불변이라 응답 시점과 다음 요청 시점에
같은 키가 나온다. assistant 의 키는 자기 자신(보이는 답·도구호출)까지 포함하므로 같은 과제의
동시 trial(동일 system·인사·첫 user 턴)이 충돌하지 않는다. 과제별 상태 없음, 요청당 O(n).

한계: `stream:true` 는 손대지 않고 통과(tau2 는 스트리밍 안 씀). choices[0] 만 처리(n=1).
첫 assistant(tau2 의 합성 인사 "Hi! How can I help you today?")는 생성된 적이 없어 캐시에 없다 —
`miss_first_assistant` 로 따로 센다. 합성 인사가 없는 하니스는 `--no-greeting`(첫 턴 miss 도 miss).

사용: python3 eval_sft/tau_proxy.py --port 8110 --upstream http://127.0.0.1:8100 \
        --stats-file results/<run>/tau_raw/proxy_stats.json --dump-dir results/<run>/tau_raw
      GET /stats → 카운터 JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import signal
import socketserver
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
H0 = hashlib.sha256(b"tau_proxy_v1").digest()

COUNTERS = (
    # 요청 쪽
    # reinlined = 캐시 적중(복원 가능 턴). restored = 실제로 content 에 think 를 넣은 **총수**(캐시 복원 + 필드 인라인).
    # --no-reattach 는 reinlined 만 오르고 restored=0. 필드를 재전송하는 하니스(mini-swe-agent)는 캐시 대신 필드 경로로
    # 복원되므로 reinlined=0 이어도 restored=reasoning_field_inlined 가 된다 — "복원됐는가"는 restored 하나로 판정.
    "requests", "reinlined", "restored", "miss", "miss_first_assistant", "seed_stripped", "sst_forced",
    "reasoning_field_inlined", "reasoning_field_dropped",
    # 응답 쪽
    "think_stripped", "think_from_field", "think_absent", "think_unclosed", "think_unclosed_stop",
    "tool_calls", "mixed_content_and_tools",
    "finish_length", "upstream_errors", "resp_parse_error", "passthrough",
)


# ── think 분리 / 정규화 ──────────────────────────────────────────────────────

def split_think(text):
    """`{think}</think>{sep}{answer}` → dict(think, sep, answer, unclosed).

    **마지막** `</think>` 에서 가른다(runners/gen_common.split_think 와 같은 규약) — 모델이 한 턴에 `</think>` 를
    두 번 내는 경우(2026-09-14 스모크 실측)에도 상대역에는 마지막 답변부만 보인다. 복원은 원문 전체라 무관.
    선행 `<think>`(+개행 1개)는 think 에서 뺀다. `</think>` 가 없고 `<think>` 만 있으면 unclosed.
    둘 다 없으면 think=None, answer=text.
    """
    if not isinstance(text, str):
        return {"think": None, "sep": "", "answer": text, "unclosed": False}
    i = text.rfind(THINK_CLOSE)
    if i < 0:
        if THINK_OPEN in text:
            return {"think": None, "sep": "", "answer": "", "unclosed": True}
        return {"think": None, "sep": "", "answer": text, "unclosed": False}
    head = text[:i]
    j = head.find(THINK_OPEN)
    if j >= 0:
        head = head[j + len(THINK_OPEN):]
    if head.startswith("\n"):
        head = head[1:]
    rest = text[i + len(THINK_CLOSE):]
    k = 0
    while k < len(rest) and rest[k] in " \t\r\n":
        k += 1
    return {"think": head, "sep": rest[:k], "answer": rest[k:], "unclosed": False}


def visible(content):
    """assistant content 의 '보이는' 부분 — think 인라인 여부에 불변."""
    if content is None:
        return ""
    if isinstance(content, str):
        return split_think(content)["answer"].strip()
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def canon_args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:  # noqa: BLE001
            return a.strip()
    if isinstance(a, (dict, list)):
        return json.dumps(a, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "" if a is None else str(a)


def canon(m):
    """메시지 → 해시 입력. tool-call id·reasoning_content 는 제외."""
    role = m.get("role") or ""
    if role == "assistant":
        tcs = []
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") if isinstance(tc, dict) and isinstance(tc.get("function"), dict) else tc
            fn = fn or {}
            tcs.append([fn.get("name"), canon_args(fn.get("arguments"))])
        return ["assistant", visible(m.get("content")), tcs]
    c = m.get("content")
    cs = c.strip() if isinstance(c, str) else json.dumps(c, sort_keys=True, ensure_ascii=False)
    return [role, cs]


def step(h, m):
    return hashlib.sha256(h + json.dumps(canon(m), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).digest()


# ── 프록시 상태 ──────────────────────────────────────────────────────────────

class Proxy:
    def __init__(self, upstream, reattach=True, keep_seed=False, max_entries=20000, max_bytes=1 << 30,
                 stats_file=None, dump_dir=None, flush_every=50, timeout=1800, greeting=True):
        self.upstream = upstream.rstrip("/")
        self.reattach = reattach
        # greeting=True: 요청의 첫 assistant 가 캐시 miss 면 tau2 합성 인사로 보고 miss_first_assistant 로 센다.
        # 합성 인사가 없는 하니스(mini-swe-agent 등)는 --no-greeting → 첫 턴 miss 도 miss 로 집계(miss_rate 과소 방지).
        self.greeting = greeting
        self.keep_seed = keep_seed
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.stats_file = stats_file
        self.dump_dir = dump_dir
        self.flush_every = flush_every
        self.timeout = timeout
        self.lock = threading.Lock()
        self.cache: OrderedDict[bytes, str] = OrderedDict()
        self.cache_bytes = 0
        self.stats = {k: 0 for k in COUNTERS}
        # 복원 통계의 분모·분자를 **턴 단위**로도 센다 (2026-09-17). 요청마다 이력 전체가 다시 오므로 per-request 카운터는
        # 같은 턴을 요청 수만큼 반복 집계한다. 깨진 턴이 궤적 앞쪽에 있으면 miss 가 그만큼 부풀고, 반대로 필드로 복원되는
        # 턴은 `reinlined` 가 아니라 `reasoning_field_inlined` 에 쌓여 miss_rate 분모에서 빠졌다(TB-2 09-17: 32.5% 로 무효 오판,
        # 전체 턴 기준 0.7%). 체인 해시 앞 12바이트를 집합에 넣어 distinct 턴 수를 구한다(상한 max_entries × 8).
        self._turns_restored: set = set()
        self._turns_missed: set = set()
        self.miss_samples: list = []
        self.started = time.time()
        self._last_dump = 0.0
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    # 카운터
    def inc(self, k, n=1):
        with self.lock:
            self.stats[k] += n

    def snapshot(self):
        with self.lock:
            d = dict(self.stats)
            d["cache_entries"] = len(self.cache)
            d["cache_bytes"] = self.cache_bytes
        d["reattach"] = self.reattach
        d["greeting"] = self.greeting
        d["uptime_s"] = round(time.time() - self.started, 1)
        hit = d["reinlined"]; miss = d["miss"]
        # 캐시 경로만 본 옛 비율은 참고용으로 남긴다. 정본 miss_rate 는 복원이 필요했던 **모든** 이력 턴(캐시 적중 + 필드
        # 인라인 + miss) 대비 miss 다 — 하니스가 reasoning 필드를 되돌려 보내면(harbor) 그 턴들도 분모에 들어가야 한다.
        d["miss_rate_cache_path"] = (miss / (hit + miss)) if (hit + miss) else 0.0
        lookups = hit + d["reasoning_field_inlined"] + miss
        d["miss_rate"] = (miss / lookups) if lookups else 0.0
        with self.lock:
            d["restored_turns"] = len(self._turns_restored)
            d["miss_turns"] = len(self._turns_missed)
            d["miss_samples"] = list(self.miss_samples)
        tt = d["restored_turns"] + d["miss_turns"]
        d["miss_turn_rate"] = (d["miss_turns"] / tt) if tt else 0.0
        return d

    def flush(self):
        if not self.stats_file:
            return
        tmp = self.stats_file + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(self.stats_file)), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(self.snapshot(), f, indent=2)
        os.replace(tmp, self.stats_file)

    # 캐시
    def cache_put(self, key, val):
        with self.lock:
            old = self.cache.pop(key, None)
            if old is not None:
                self.cache_bytes -= len(old)
            self.cache[key] = val
            self.cache_bytes += len(val)
            while self.cache and (len(self.cache) > self.max_entries or self.cache_bytes > self.max_bytes):
                _, v = self.cache.popitem(last=False)
                self.cache_bytes -= len(v)

    def cache_get(self, key):
        with self.lock:
            v = self.cache.get(key)
            if v is not None:
                self.cache.move_to_end(key)
            return v

    def _note_turn(self, h, restored, msg=None, idx=None):
        """distinct 턴 집계. 같은 턴(체인 해시)은 몇 번 다시 와도 한 번만 센다. miss 는 앞 8건을 표본으로 남긴다."""
        k = h[:12]
        with self.lock:
            target = self._turns_restored if restored else self._turns_missed
            if k in target or len(target) >= self.max_entries * 8:
                return
            target.add(k)
            if not restored and len(self.miss_samples) < 8 and msg is not None:
                c = msg.get("content")
                cs = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                self.miss_samples.append({"assistant_idx": idx, "content_len": len(cs or ""), "content_head": (cs or "")[:120],
                                          "has_tool_calls": bool(msg.get("tool_calls")), "keys": sorted(msg.keys())})

    # 요청 처리: 히스토리 복원 + 파라미터 정합. 반환 = 마지막 메시지까지의 체인 해시
    def on_request(self, body):
        self.inc("requests")
        msgs = body.get("messages") or []
        h = H0
        seen_assistant = False
        n_asst = 0
        for m in msgs:
            if not isinstance(m, dict):
                continue
            is_asst = m.get("role") == "assistant"
            if is_asst:
                n_asst += 1
                # 다른 하니스(mini-swe-agent/litellm)가 reasoning_content 를 이력에 실어 보내면:
                #   reattach ON  → 인라인으로 정규화(템플릿 119행 경로로 통일)
                #   reattach OFF → 필드를 떼어 버린다(reasoning_field_dropped). 남겨 두면 서버/템플릿이 그 필드로 추론을
                #                  렌더해 strip 이 성립하지 않는다 (2026-09-14 SWE 실측: strip 이 restore 와 구분 안 됨).
                rc = m.pop("reasoning_content", None)
                c = m.get("content")
                field_inlined = False
                if isinstance(rc, str) and rc.strip() and (c is None or isinstance(c, str)) and THINK_CLOSE not in (c or ""):
                    if self.reattach:
                        m["content"] = THINK_OPEN + "\n" + rc + THINK_CLOSE + (c or "")
                        self.inc("reasoning_field_inlined")
                        self.inc("restored")
                        field_inlined = True
                    else:
                        self.inc("reasoning_field_dropped")
            h = step(h, m)
            if is_asst:
                c = m.get("content")
                has_think = isinstance(c, str) and THINK_CLOSE in c
                if field_inlined:
                    self._note_turn(h, restored=True)
                if not has_think:
                    blk = self.cache_get(h)
                    if blk is None:
                        first = self.greeting and not seen_assistant
                        self.inc("miss_first_assistant" if first else "miss")
                        if not first:
                            self._note_turn(h, restored=False, msg=m, idx=n_asst)
                    else:
                        self.inc("reinlined")
                        self._note_turn(h, restored=True)
                        if self.reattach:
                            m["content"] = blk
                            self.inc("restored")
                seen_assistant = True
        if not self.keep_seed and "seed" in body:
            body.pop("seed", None)
            self.inc("seed_stripped")
        if body.get("skip_special_tokens") is None:
            body["skip_special_tokens"] = False
            self.inc("sst_forced")
        # 렌더 검사용 덤프 (assistant ≥3 인 최신 요청, 5초 스로틀)
        if self.dump_dir and n_asst >= 3 and time.time() - self._last_dump > 5:
            self._last_dump = time.time()
            try:
                tmp = os.path.join(self.dump_dir, "last_request.json.tmp")
                with open(tmp, "w") as f:
                    json.dump(body, f, ensure_ascii=False, indent=1)
                os.replace(tmp, os.path.join(self.dump_dir, "last_request.json"))
            except Exception:  # noqa: BLE001
                pass
        with self.lock:
            n = self.stats["requests"]
        if self.flush_every and n % self.flush_every == 0:
            try:
                self.flush()
            except Exception:  # noqa: BLE001
                pass
        return h

    # 응답 처리: think 분리 + 캐시
    def on_response(self, h_req, resp):
        choices = resp.get("choices") or []
        if not choices:
            return
        ch = choices[0]
        msg = ch.get("message") or {}
        tcs = msg.get("tool_calls") or []
        c = msg.get("content")
        # reasoning 파서가 켜진 fleet(τ 규약): think 는 `reasoning`(vLLM 0.25.1) / `reasoning_content` 필드, content 는 답변만.
        # vLLM 0.25.1 parser engine 은 tool 파서가 켜지면 </think> 토큰을 터미널로 소비하므로 reasoning 파서 없이는
        # content 가 think+답변이 마커 없이 붙어 나온다(2026-09-14 실측) → τ fleet 는 반드시 reasoning 파서와 함께 뜬다.
        field = msg.get("reasoning_content") or msg.get("reasoning")
        st = split_think(c) if isinstance(c, str) else None
        if (st is None or st["think"] is None) and isinstance(field, str) and field.strip():
            answer = (c or "").strip() if isinstance(c, str) else ""
            if ch.get("finish_reason") == "length" and not answer and not tcs:
                msg["content"] = ""
                self.inc("think_unclosed")
            else:
                if not answer and not tcs:
                    # 모델이 </think> 를 닫지 않고 답을 쓴 채 종료(finish=stop): 파서가 전부 reasoning 으로 분류해 답변이 빈다.
                    # 복원하면 이력에 `<think>{답}</think>`+빈 답변이 남아 모델이 그 패턴을 베끼며 루프로 굳을 수 있다
                    # (2026-09-14 TB-2 실측: restore 385/425 vs strip 38/262 스텝). 동작은 유지하고 여기서 센다.
                    self.inc("think_unclosed_stop")
                full = THINK_OPEN + "\n" + field + THINK_CLOSE + (c if isinstance(c, str) else "")
                new_content = answer if answer else (None if tcs else "")
                reply = {"role": "assistant", "content": new_content, "tool_calls": tcs}
                self.cache_put(step(h_req, reply), full)
                msg["content"] = new_content
                msg["reasoning_content"] = field
                self.inc("think_from_field")
        elif st is not None and st["think"] is not None:
            answer = st["answer"].strip()
            full = c if c.lstrip().startswith(THINK_OPEN) else THINK_OPEN + "\n" + c
            new_content = answer if answer else (None if tcs else "")
            reply = {"role": "assistant", "content": new_content, "tool_calls": tcs}
            self.cache_put(step(h_req, reply), full)
            msg["content"] = new_content
            msg["reasoning_content"] = st["think"]
            self.inc("think_stripped")
        elif st is not None and st["unclosed"]:
            msg["content"] = None if tcs else ""
            self.inc("think_unclosed")
        else:
            self.inc("think_absent")
        if tcs:
            self.inc("tool_calls")
            if isinstance(msg.get("content"), str) and msg["content"].strip():
                self.inc("mixed_content_and_tools")
        if ch.get("finish_reason") == "length":
            self.inc("finish_length")

    # upstream 호출
    def forward(self, path, method, headers, body):
        url = self.upstream + path
        hdr = {k: v for k, v in headers.items() if k.lower() not in ("host", "content-length")}
        req = urllib.request.Request(url, data=body, headers=hdr, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read(), r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("Content-Type", "application/json")
        except Exception as e:  # noqa: BLE001
            return 502, json.dumps({"error": {"message": f"tau_proxy upstream: {e}"}}).encode(), "application/json"


def make_handler(proxy: Proxy):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # noqa: D401
            pass

        def _send(self, code, data, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/stats":
                self._send(200, json.dumps(proxy.snapshot(), indent=1).encode())
                return
            proxy.inc("passthrough")
            self._send(*proxy.forward(self.path, "GET", self.headers, None))

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b""
            if self.path.rstrip("/").endswith("/chat/completions"):
                try:
                    body = json.loads(raw)
                except Exception:  # noqa: BLE001
                    body = None
                if isinstance(body, dict) and not body.get("stream"):
                    h = proxy.on_request(body)
                    code, data, ctype = proxy.forward(self.path, "POST", self.headers,
                                                      json.dumps(body, ensure_ascii=False).encode("utf-8"))
                    if code == 200:
                        try:
                            resp = json.loads(data)
                            proxy.on_response(h, resp)
                            data = json.dumps(resp, ensure_ascii=False).encode("utf-8")
                        except Exception:  # noqa: BLE001
                            proxy.inc("resp_parse_error")
                    else:
                        proxy.inc("upstream_errors")
                    self._send(code, data, ctype)
                    return
            proxy.inc("passthrough")
            self._send(*proxy.forward(self.path, "POST", self.headers, raw))

    return H


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # 기본 backlog(5)는 동시 연결 16개에서 SYN 큐가 넘쳐 Connection reset 이 난다 (유닛 테스트 실측)
    request_queue_size = 256


def build(argv=None):
    ap = argparse.ArgumentParser(description="τ-bench think 분리·재부착 프록시")
    ap.add_argument("--port", type=int, default=8110)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--upstream", default="http://127.0.0.1:8100")
    ap.add_argument("--stats-file", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--no-reattach", action="store_true", help="히스토리 복원을 끈다(카운트는 유지) — ON/OFF differential 용")
    ap.add_argument("--keep-seed", action="store_true", help="요청의 seed 를 제거하지 않는다")
    ap.add_argument("--no-greeting", action="store_true",
                    help="첫 assistant 의 miss 를 miss_first_assistant 가 아니라 miss 로 센다 — 합성 인사가 없는 하니스(mini-swe-agent 등)용")
    ap.add_argument("--max-entries", type=int, default=20000)
    ap.add_argument("--max-bytes", type=int, default=1 << 30)
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args(argv)
    proxy = Proxy(a.upstream, reattach=not a.no_reattach, keep_seed=a.keep_seed, max_entries=a.max_entries,
                  max_bytes=a.max_bytes, stats_file=a.stats_file, dump_dir=a.dump_dir,
                  flush_every=a.flush_every, timeout=a.timeout, greeting=not a.no_greeting)
    srv = Threaded((a.host, a.port), make_handler(proxy))
    return proxy, srv, a


if __name__ == "__main__":
    proxy, srv, a = build()

    def _stop(*_):
        try:
            proxy.flush()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print(f"[tau_proxy] :{srv.server_address[1]} -> {a.upstream} reattach={not a.no_reattach} "
          f"stats={a.stats_file} dump={a.dump_dir}", flush=True)
    srv.serve_forever()
