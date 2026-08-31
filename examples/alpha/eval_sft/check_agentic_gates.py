"""에이전틱(SWE·Terminal) 투입 전 게이트 — `docs/SFT_BENCHMARKS.md` §7 의 에이전틱 판.

T1 의 G1~G3(`check_gates.py`)에 더해, 에이전틱 레인은 전제가 셋 더 있다. 하나라도
깨지면 에이전트가 매 스텝 실패하고 결과는 0점이 되는데, 그 0점은 "모델이 못 푼다"와
구분되지 않는다 — 2026-08-30 SWE 0/20 · Terminal 0/10 이 정확히 그 상태였다.

| # | 게이트 | 깨지면 |
|---|---|---|
| A1 | 엔드포인트가 `tool_choice=auto` 수용 | mini-swe-agent litellm 요청이 전부 HTTP 400 |
| A4 | 파서가 모델 형식을 **실제로 파싱** | `tool_calls: null` → 에이전트가 `RepeatedFormatError` 로 즉시 종료 |
| A2 | 컨테이너→fleet 역터널 생존 | 하니스가 모델에 닿지 못함 |
| A3 | 디스크 여유 | 태스크 이미지 pull 중 중단 |

A1·A4 는 서빙 시 `TOOLS=1 TOOL_PARSER=qwen3_xml` 로 해결한다. T1 용 fleet 는 이 플래그 없이
뜨므로 **에이전틱 전에 fleet 를 재기동**해야 한다. 파서는 모델이 배운 형식에 맞춰야 하며
(alpha 는 XML `<function=…><parameter=…>`), 잘못된 파서는 A1 을 통과하고 A4 에서 걸린다.

사용:
    python3 check_agentic_gates.py --base-url http://localhost:8100/v1 [--min-disk-gb 300]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.error
import urllib.request

SSH_CONFIG = "/home/work/vidsearch/.ssh-keys/config"
CONTAINER = "alpha-eval"
TUNNEL_PORT = 8199


def _post(base_url: str, body: dict, timeout: int = 300):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return False, e.read()[:300].decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def gate_a1(base_url: str) -> tuple[bool, str]:
    """tool_choice=auto 수용 — mini-swe-agent 의 litellm 이 이 형태로 보낸다."""
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {"type": "object",
                           "properties": {"cmd": {"type": "string"}},
                           "required": ["cmd"]},
        },
    }]
    ok, res = _post(base_url, {
        "model": "alpha",
        "messages": [{"role": "user", "content": "Say hi in 3 words."}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 256,
        "seed": None, "skip_special_tokens": False,
        "tools": tools, "tool_choice": "auto",
    })
    if ok:
        return True, "tool_choice=auto 수용"
    return False, (f"{res} — 서빙에 TOOLS=1 이 필요하다: "
                   f"TOOL_PARSER=qwen3_xml TOOLS=1 GPUS=... bash eval_sft/serve_fleet.sh <ckpt> 106496 <N> 8100")


def gate_a4(base_url: str) -> tuple[bool, str]:
    """파서가 모델 출력을 **실제로 파싱**하는가.

    A1 은 파서가 *등록됐는지*만 본다. 그 파서가 우리 모델이 배운 형식과 *맞는지*는 보지
    않는다 — 2026-08-30 에 hermes(JSON 본문)로 띄운 채 A1 을 통과했고, 모델이 내는
    XML(`<function=…><parameter=…>`)이 파싱되지 않아 `tool_calls: null` 이 됐다.
    에이전트는 "No tool calls found" 를 반복하다 RepeatedFormatError 로 죽는다.

    여기서는 실제로 도구를 쓰게 만들고 `tool_calls` 가 채워지는지 확인한다.
    """
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command.",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string", "description": "the command"}},
                           "required": ["command"]},
        },
    }]
    ok, res = _post(base_url, {
        "model": "alpha",
        "messages": [{"role": "user",
                      "content": "List the files in the current directory. Use the bash tool."}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 2048,
        "seed": None, "skip_special_tokens": False,
        "tools": tools, "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=600)
    if not ok:
        return False, f"요청 실패: {res}"

    ch = res["choices"][0]
    tc = ch["message"].get("tool_calls")
    if tc:
        fn = tc[0].get("function", {})
        return True, f"tool_calls 파싱 OK — {fn.get('name')}({str(fn.get('arguments'))[:60]})"

    raw = (ch["message"].get("content") or "")[:200]
    hint = ""
    if "<function=" in raw or "<parameter=" in raw:
        hint = ("  ← 모델은 XML 형식을 내고 있다. 파서를 `qwen3_xml` 로 바꿀 것: "
                "TOOL_PARSER=qwen3_xml TOOLS=1 bash eval_sft/serve_fleet.sh …")
    return False, f"tool_calls 가 비어 있다. content={raw!r}{hint}"


def _ssh(cmd: str, timeout: int = 40) -> tuple[int, str]:
    p = subprocess.run(
        ["ssh", "-F", SSH_CONFIG, "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", CONTAINER, cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return p.returncode, (p.stdout or p.stderr).strip()


def gate_a2() -> tuple[bool, str]:
    """컨테이너 안에서 본 역터널(:8199)이 fleet 에 닿는가."""
    rc, out = _ssh(f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 6 "
                   f"http://localhost:{TUNNEL_PORT}/v1/models")
    if rc != 0:
        return False, f"컨테이너 접속 실패: {out[:160]}"
    if out.strip() != "200":
        return False, (f"역터널 응답 {out.strip() or '없음'} — 기동: "
                       f"ssh sub1 'bash /home/work/vidsearch/tools/start_swe_tunnel.sh'")
    return True, f"컨테이너 :{TUNNEL_PORT} → fleet 200"


def gate_a3(min_gb: int) -> tuple[bool, str]:
    """태스크 이미지 pull 여유.

    **회수 가능량을 함께 본다.** `df` 의 여유만 보면 실제보다 적게 보인다 — build cache 가
    수십 GB 를 쥐고 있고 `docker builder prune` 으로 즉시 돌려받을 수 있기 때문이다
    (2026-08-31 실측: 여유 589GB + 회수 가능 31.8GB).
    sweb.eval 이미지(498개 479.7GB)는 회수 대상이 아니다 — 다음 실행에서 재사용된다.
    """
    rc, out = _ssh("df -BG --output=avail /var/lib/docker | tail -1")
    if rc != 0:
        return False, f"디스크 조회 실패: {out[:160]}"
    try:
        avail = int(out.strip().rstrip("G"))
    except ValueError:
        return False, f"디스크 파싱 실패: {out[:80]}"

    # 회수 가능량 (build cache) — 실패해도 게이트를 막지 않는다
    rec = 0
    rc2, out2 = _ssh("docker system df --format '{{.Type}}|{{.Reclaimable}}' 2>/dev/null")
    if rc2 == 0:
        for line in out2.splitlines():
            if line.startswith("Build Cache"):
                m = re.search(r"([\d.]+)\s*(GB|MB)", line)
                if m:
                    rec = int(float(m.group(1)) * (1 if m.group(2) == "GB" else 0.001))

    total = avail + rec
    extra = f" + 회수가능 {rec}GB" if rec else ""
    if total < min_gb:
        return False, (f"여유 {avail}GB{extra} < 요구 {min_gb}GB — "
                       "정리: bash eval_sft/docker_gc.sh")
    if avail < min_gb <= total:
        return True, (f"여유 {avail}GB{extra} = {total}GB (요구 {min_gb}GB) — "
                      "여유가 빠듯하다. bash eval_sft/docker_gc.sh 권장")
    return True, f"여유 {avail}GB{extra} (요구 {min_gb}GB)"


def main() -> int:
    ap = argparse.ArgumentParser(description="에이전틱 투입 전 게이트 A1~A4")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--min-disk-gb", type=int, default=300)
    a = ap.parse_args()

    results = {}
    print("── A1: tool_choice=auto 수용 " + "─" * 34)
    ok, msg = gate_a1(a.base_url)
    print(f"   {msg}\n   → {'PASS' if ok else 'FAIL'}\n")
    results["A1"] = ok

    print("── A2: 컨테이너 역터널 " + "─" * 39)
    ok, msg = gate_a2()
    print(f"   {msg}\n   → {'PASS' if ok else 'FAIL'}\n")
    results["A2"] = ok

    print("── A3: 컨테이너 디스크 여유 " + "─" * 34)
    ok, msg = gate_a3(a.min_disk_gb)
    print(f"   {msg}\n   → {'PASS' if ok else 'FAIL'}\n")
    results["A3"] = ok

    print("── A4: 파서가 모델 형식을 실제로 파싱 " + "─" * 24)
    ok, msg = gate_a4(a.base_url)
    print(f"   {msg}\n   → {'PASS' if ok else 'FAIL'}\n")
    results["A4"] = ok

    bad = [k for k, v in results.items() if not v]
    if bad:
        print(f"❌ 게이트 실패: {', '.join(sorted(bad))} — 에이전틱을 돌리지 말 것. "
              "이 상태의 0점은 모델 실패와 구분되지 않는다.")
        return 1
    print("✅ 에이전틱 게이트 통과 (A1·A2·A3·A4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
