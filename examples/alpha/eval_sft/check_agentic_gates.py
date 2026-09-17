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
    python3 check_agentic_gates.py --base-url ... --skip-container   # τ-bench: A1·A4 만 (docker 불요)
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


# ---------------------------------------------------------------- A4
# 2026-09-17: 1회 표본으로 판정하다 오판했다. iter600(최종 런) fleet 에서 모델이 도구를 부르지 않고
# "어떤 명령을 실행할지 알려달라" 고 되물은 표본 하나로 A4 FAIL → 에이전틱 전체가 건너뛰어졌다. 같은 fleet 의
# A5 는 도구호출 2/2 파싱 PASS 였으니 파서는 멀쩡했다. 이 게이트가 막아야 할 것은 **파서 불일치**(모델이 XML 을
# 냈는데 tool_calls 가 빈 것)이지 모델이 한 번 도구를 안 쓴 것이 아니다. A5 와 같은 구조로 바꾼다:
# 관측(파싱 성공 / XML 미파싱)이 나올 때까지 최대 A4_MAX_ATTEMPTS 회 뽑고, 미호출은 "경로 미관측" 으로 센다.
# 전부 미호출이면 통과로 쓰지 않는다(검증 규칙) — 단 메시지로 파서 문제가 아님을 밝힌다.
A4_MAX_ATTEMPTS = 8


def judge_a4(message: dict) -> tuple[str, str]:
    """도구호출 응답 하나를 판정한다 (순수 함수 — 단위 테스트 대상).

    반환 = (판정, 사유). parsed = 통과, xml_unparsed = 파서 불일치(실패), no_call = 경로 미관측(모델이 도구를 안 불렀다).
    """
    tc = message.get("tool_calls")
    if tc:
        fn = tc[0].get("function", {})
        return "parsed", f"tool_calls 파싱 OK — {fn.get('name')}({str(fn.get('arguments'))[:60]})"
    raw = message.get("content") or ""
    if "<function=" in raw or "<parameter=" in raw:
        return "xml_unparsed", (f"모델은 XML 형식을 냈는데 tool_calls 가 비었다: {raw[:120]!r} — 파서를 `qwen3_xml` 로 바꿀 것: "
                                "TOOL_PARSER=qwen3_xml TOOLS=1 bash eval_sft/serve_fleet.sh …")
    return "no_call", f"도구 미호출(평문 답변): {raw[:120]!r}"


def gate_a4(base_url: str) -> tuple[bool, str]:
    """파서가 모델 출력을 **실제로 파싱**하는가.

    A1 은 파서가 *등록됐는지*만 본다. 그 파서가 우리 모델이 배운 형식과 *맞는지*는 보지
    않는다 — 2026-08-30 에 hermes(JSON 본문)로 띄운 채 A1 을 통과했고, 모델이 내는
    XML(`<function=…><parameter=…>`)이 파싱되지 않아 `tool_calls: null` 이 됐다.
    에이전트는 "No tool calls found" 를 반복하다 RepeatedFormatError 로 죽는다.

    여기서는 실제로 도구를 쓰게 만들고 `tool_calls` 가 채워지는지 확인한다. 생성 파라미터는
    태스크 조건(temp 1.0)이라 모델이 도구를 안 부르는 표본이 섞인다 — 그건 관측 실패이지 파서
    실패가 아니므로 관측이 나올 때까지 다시 뽑는다(A4_MAX_ATTEMPTS).

    thinking 모드: 먼저 OFF(사고 처리와 분리해 파서만 본다)로 뽑고, OFF 에서 한 번도 도구를 안 부르면
    **평가 조건인 ON** 으로 다시 뽑는다. 2026-09-17 iter600(최종 런): OFF 8/8 미호출("bash 도구를 이 환경에서
    쓸 수 없다") 인데 ON 은 4/4 호출·파싱 — 템플릿은 두 모드에 tools 를 똑같이 렌더하므로 OFF 모드의 모델
    행동 차이다. 하니스(SWE·TB-2·τ³)는 전부 ON 으로 돌므로 파서는 ON 에서 확인하면 되고, OFF 미호출은
    경고로 남긴다(모델 발견 — STATUS·SFT_BENCHMARKS §3.10).
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
    body = {
        "model": "alpha",
        "messages": [{"role": "user",
                      "content": "List the files in the current directory. Use the bash tool."}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 2048,
        "seed": None, "skip_special_tokens": False,
        "tools": tools, "tool_choice": "auto",
    }
    no_call: dict[str, list[str]] = {"off": [], "on": []}
    for mode in ("off", "on"):
        b = dict(body)
        if mode == "off":
            b["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            b["max_tokens"] = 4096   # ON 은 사고가 앞에 붙는다
        for _ in range(A4_MAX_ATTEMPTS):
            ok, res = _post(base_url, b, timeout=600)
            if not ok:
                return False, f"요청 실패: {res}"
            v, why = judge_a4(res["choices"][0]["message"])
            if v == "parsed":
                note = ""
                if mode == "on":
                    note = (f" ⚠️ thinking OFF 에서는 {A4_MAX_ATTEMPTS}회 모두 미호출(마지막: {no_call['off'][-1]}) — "
                            "파서는 ON 에서 확인됨. OFF 모드 도구 거부는 모델 발견으로 기록할 것")
                elif no_call["off"]:
                    note = f" (미호출 {len(no_call['off'])}회 뒤 관측)"
                return True, f"[thinking {mode}] {why}{note}"
            if v == "xml_unparsed":
                return False, f"[thinking {mode}] {why}"
            no_call[mode].append(why)
    return False, (f"OFF·ON 각 {A4_MAX_ATTEMPTS}회 모두 도구 미호출 — content 에 XML 이 없으니 파서 문제가 아니라 모델이 이 지시에 "
                   f"도구를 안 쓴다. 경로 미관측이라 통과로 쓰지 않는다. 마지막(ON): {no_call['on'][-1]}")


# ---------------------------------------------------------------- A5
# **think + 도구호출 동시 경로** (2026-09-14, `KNOWN_ISSUES.md` 09-14).
#
# vLLM 0.25.1 의 도구 파서(qwen3_xml)는 reasoning 파서 없이 켜면 THINK_END 를 소비하고 마커만 떨어뜨린다.
# 보존된 SWE 궤적 전수(iter1800 74,833턴 · iter1500 74,003턴)에서 도구호출 턴의 `</think>` 보존이 **0건**
# 이었고, 챗 템플릿은 마커 없는 이전 턴을 `<think></think>` + 추론문(답변 취급)으로 재렌더했다.
#
# 기존 게이트는 이 경로를 보지 않았다. G2 는 도구 없는 T1 fleet 에서만 `</think>` 를 보고, A4 는
# `enable_thinking: False` 로 도구 파싱만 본다. A5 는 **thinking ON + tools 선언** 으로 실제 호출을 시키고,
# 그 턴에서 추론이 분리(reasoning 필드)되거나 보존(content 의 `</think>`)되는지 본다.
#
# 결함 fleet 의 통과율은 0/74,009, 정상 fleet 의 실패 패턴은 원리상 나오지 않는다(thinking ON 이면
# 출력이 think 안에서 시작해 `</think>` 로 나와야 하고, reasoning 파서가 그것을 필드로 옮긴다). 두 분포가
# 완전히 갈리므로 소수 표본으로 충분하다 — 결론 난 관측 2개가 일치하면 멈추고, 갈리면 3개째로 다수결.
A5_MAX_ATTEMPTS = 8
A5_CONCLUSIVE = 3


def judge_a5(message: dict) -> tuple[str, str]:
    """도구호출 턴 하나를 판정한다 (순수 함수 — 단위 테스트 대상).

    반환 = (판정, 사유). 판정은 pass · fail · no_tool_call · no_reasoning 중 하나이며,
    pass·fail 만 결론이고 나머지는 경로를 관측하지 못한 것이다.
    """
    tc = message.get("tool_calls") or []
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not tc:
        return "no_tool_call", "도구를 부르지 않아 동시 경로를 관측하지 못함"
    if reasoning.strip():
        if "</think>" in content:
            return "pass", f"reasoning 필드 분리({len(reasoning)}자) — 단 content 에 </think> 잔존"
        return "pass", f"reasoning 필드 분리({len(reasoning)}자), content 에 마커 없음"
    if "</think>" in content:
        return "pass", "content 에 </think> 보존 (reasoning 파서 없이 인라인)"
    if content.strip():
        return "fail", ("결함 — 도구호출 턴인데 reasoning 필드가 비고 content 에 </think> 가 없다. "
                        "추론이 답변에 붙어 이력이 <think></think>+추론문으로 재렌더된다")
    return "no_reasoning", "도구호출만 있고 추론·content 가 모두 비어 판정 불가"


def gate_a5(base_url: str) -> tuple[bool, str]:
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
    body = {
        "model": "alpha",
        "messages": [{"role": "user", "content":
                      "Find which Python file in the current directory defines a function named main. "
                      "Think about the right command first, then use the bash tool."}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 4096,
        "seed": None, "skip_special_tokens": False,
        "tools": tools, "tool_choice": "auto",
        # enable_thinking 을 넣지 않는다 — 기본값(ON)이 평가 조건이다. A4 와 다른 점이 이것이다.
    }
    verdicts: list[tuple[str, str]] = []
    seen = {"no_tool_call": 0, "no_reasoning": 0}
    for _ in range(A5_MAX_ATTEMPTS):
        ok, res = _post(base_url, body, timeout=600)
        if not ok:
            return False, f"요청 실패: {res}"
        v, why = judge_a5(res["choices"][0]["message"])
        if v in seen:
            seen[v] += 1
            continue
        verdicts.append((v, why))
        passes = sum(1 for x, _ in verdicts if x == "pass")
        fails = len(verdicts) - passes
        if passes >= 2 or fails >= 2 or len(verdicts) >= A5_CONCLUSIVE:
            break
    if not verdicts:
        # 경로를 한 번도 못 봤다 — 통과로 쓰지 않는다 (검증 규칙: 미실행을 통과처럼 쓰지 않는다).
        return False, (f"{A5_MAX_ATTEMPTS}회 중 결론 난 도구호출 턴 0 "
                       f"(무호출 {seen['no_tool_call']} · 무추론 {seen['no_reasoning']}) — 경로 미관측")
    passes = sum(1 for x, _ in verdicts if x == "pass")
    fails = len(verdicts) - passes
    tally = f"결론 {len(verdicts)}건 pass {passes} / fail {fails}"
    if fails > passes:
        why = next(w for x, w in verdicts if x == "fail")
        return False, (f"{tally}. {why}. 수정: 에이전틱 fleet 를 REASONING_PARSER=nemotron_v3 로 띄울 것 "
                       "(`SFT_BENCHMARKS.md` §3.14)")
    return True, f"{tally}. {next(w for x, w in verdicts if x == 'pass')}"


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
    ap = argparse.ArgumentParser(description="에이전틱 투입 전 게이트 A1~A5")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--min-disk-gb", type=int, default=300)
    ap.add_argument("--skip-container", action="store_true",
                    help="A2·A3(컨테이너 역터널·디스크) 생략 — docker 가 필요 없는 하니스(τ-bench) 용")
    ap.add_argument("--tool-path", choices=("required", "report"), default="required",
                    help="A5(think+도구호출 동시 경로) 처리. required = 실패 시 차단(기본). "
                         "report = 결과만 출력 — 네이티브 tools 를 안 쓰는 하니스(TB-1·TB-2) 용")
    a = ap.parse_args()

    results = {}
    print("── A1: tool_choice=auto 수용 " + "─" * 34)
    ok, msg = gate_a1(a.base_url)
    print(f"   {msg}\n   → {'PASS' if ok else 'FAIL'}\n")
    results["A1"] = ok

    if a.skip_container:
        print("── A2·A3: 생략 (--skip-container)\n")
    else:
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

    print("── A5: think + 도구호출 동시 경로 " + "─" * 28)
    ok, msg = gate_a5(a.base_url)
    if a.tool_path == "required":
        print(f"   {msg}\n   → {'PASS' if ok else 'FAIL'}\n")
        results["A5"] = ok
    else:
        print(f"   {msg}\n   → {'PASS' if ok else 'FAIL'} (report — 이 하니스는 네이티브 tools 를 쓰지 않아 차단하지 않음)\n")

    bad = [k for k, v in results.items() if not v]
    if bad:
        print(f"❌ 게이트 실패: {', '.join(sorted(bad))} — 에이전틱을 돌리지 말 것. "
              "이 상태의 0점은 모델 실패와 구분되지 않는다.")
        return 1
    print("✅ 에이전틱 게이트 통과 (" + "·".join(results) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
