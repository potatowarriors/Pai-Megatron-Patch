"""에이전틱(SWE·Terminal) 투입 전 게이트 — `docs/SFT_BENCHMARKS.md` §7 의 에이전틱 판.

T1 의 G1~G3(`check_gates.py`)에 더해, 에이전틱 레인은 전제가 셋 더 있다. 하나라도
깨지면 에이전트가 매 스텝 실패하고 결과는 0점이 되는데, 그 0점은 "모델이 못 푼다"와
구분되지 않는다 — 2026-08-30 SWE 0/20 · Terminal 0/10 이 정확히 그 상태였다.

| # | 게이트 | 깨지면 |
|---|---|---|
| A1 | 엔드포인트가 `tool_choice=auto` 수용 | mini-swe-agent litellm 요청이 전부 HTTP 400 |
| A2 | 컨테이너→fleet 역터널 생존 | 하니스가 모델에 닿지 못함 |
| A3 | 디스크 여유 | 태스크 이미지 pull 중 중단 |

A1 은 서빙 시 `TOOLS=1`(=`--enable-auto-tool-choice --tool-call-parser hermes`)로 해결한다.
T1 용 fleet 는 이 플래그 없이 뜨므로 **에이전틱 전에 fleet 를 재기동**해야 한다.

사용:
    python3 check_agentic_gates.py --base-url http://localhost:8100/v1 [--min-disk-gb 300]
"""

from __future__ import annotations

import argparse
import json
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
                   f"TOOLS=1 GPUS=... bash eval_sft/serve_fleet.sh <ckpt> 40960 <N> 8100")


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
    """태스크 이미지 pull 여유. swebench 5.x 는 인스턴스당 1~4GB 를 Docker Hub 에서 받는다."""
    rc, out = _ssh("df -BG --output=avail /var/lib/docker | tail -1")
    if rc != 0:
        return False, f"디스크 조회 실패: {out[:160]}"
    try:
        avail = int(out.strip().rstrip("G"))
    except ValueError:
        return False, f"디스크 파싱 실패: {out[:80]}"
    if avail < min_gb:
        return False, f"여유 {avail}GB < 요구 {min_gb}GB — 이미지 정리 필요 (docker image prune)"
    return True, f"여유 {avail}GB (요구 {min_gb}GB)"


def main() -> int:
    ap = argparse.ArgumentParser(description="에이전틱 투입 전 게이트 A1~A3")
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

    bad = [k for k, v in results.items() if not v]
    if bad:
        print(f"❌ 게이트 실패: {', '.join(sorted(bad))} — 에이전틱을 돌리지 말 것. "
              "이 상태의 0점은 모델 실패와 구분되지 않는다.")
        return 1
    print("✅ 에이전틱 게이트 통과 (A1·A2·A3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
