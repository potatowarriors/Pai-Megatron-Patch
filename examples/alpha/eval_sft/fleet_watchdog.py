#!/usr/bin/env python3
"""fleet 백엔드 정지 감시·자동 재기동 (2026-09-17).

왜: W=96 + prefix caching(align) 첫 실행에서 :8003(GPU 3) EngineCore 가 기동 2.5분 뒤 멈췄다 — CPU 106% 스핀, GPU util 100%
인데 전력 126 W, 요청 12건이 30분 프록시 타임아웃까지 걸림(GPU 7 09-15 정지와 같은 서명). 세션 고정 라우팅은 죽은 백엔드에
세션을 못 박아 두므로, 정지를 빨리 끊고 재기동해야 한다(프록시가 연결 거부를 보면 세션을 재배정한다). 컨테이너에는 ptrace
권한이 없어 스택을 못 잡는다 — 재발 위치(같은 GPU 인가)를 기록해 하드웨어/소프트웨어를 가른다.

판정: `/metrics` 의 num_requests_running > 0 인데 generation_tokens_total 이 STALL_S(기본 120 s) 동안 그대로 → 정지.
      `/metrics` 자체가 DEAD_S(기본 60 s) 동안 불통 → 사망. 둘 다 같은 argv·env 로 재기동한다(/proc 에서 읽음 — 단계별 플래그
      차이를 몰라도 된다).
종료: lb_proxy 가 EXIT_S(기본 90 s) 동안 없으면(stop_fleet 로 단계 종료) 끝난다.
기록: fleet_logs/watchdog.log + watchdog_events.jsonl (시각·포트·GPU·사유·마지막 통계·nvidia-smi 한 줄).

사용: python3 eval_sft/fleet_watchdog.py --ports 8000,...,8006 [--stall-s 120] [--dead-s 60] [--poll-s 15]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LOGD = Path("/home/work/vidsearch/tools/fleet_logs")


def log(msg: str) -> None:
    line = f"[watchdog {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOGD / "watchdog.log", "a") as f:
        f.write(line + "\n")


def metrics(port: int) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            txt = r.read().decode()
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, float] = {}
    for l in txt.splitlines():
        m = re.match(r"^(vllm:[a-zA-Z_]+)(\{[^}]*\})?\s+([-+0-9.eE]+)", l)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def find_server(port: int) -> int | None:
    """`vllm serve … --port <port>` 의 API 서버 pid."""
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            argv = Path(f"/proc/{p}/cmdline").read_bytes().split(b"\0")
        except Exception:  # noqa: BLE001
            continue
        if any(b"alpha_serve_venv/bin/vllm" in a for a in argv) and b"serve" in argv and b"--port" in argv:
            i = argv.index(b"--port")
            if i + 1 < len(argv) and argv[i + 1] == str(port).encode():
                return int(p)
    return None


def proc_argv_env(pid: int) -> tuple[list[str], dict[str, str]]:
    argv = [a.decode() for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
    env = {}
    for kv in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" in kv:
            k, v = kv.split(b"=", 1)
            env[k.decode()] = v.decode(errors="replace")
    return argv, env


def lb_alive() -> bool:
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            if b"lb_proxy.py" in Path(f"/proc/{p}/cmdline").read_bytes():
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def kill_tree(pid: int) -> None:
    try:
        pg = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pg, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(15):
        time.sleep(2)
        if find_server_pid_alive(pid) is False:
            break
    try:
        os.killpg(pg, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # 고아 spawn 워커(같은 CUDA_VISIBLE_DEVICES, alpha_serve_venv) 정리 — stop_fleet.sh 와 같은 규칙
    time.sleep(2)


def find_server_pid_alive(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


def kill_orphans(cvd: str) -> int:
    n = 0
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            env = Path(f"/proc/{p}/environ").read_bytes()
            cmd = Path(f"/proc/{p}/cmdline").read_bytes()
        except Exception:  # noqa: BLE001
            continue
        if b"alpha_serve_venv" in env and f"CUDA_VISIBLE_DEVICES={cvd}".encode() in env and b"spawn_main" in cmd:
            try:
                os.kill(int(p), signal.SIGKILL); n += 1
            except Exception:  # noqa: BLE001
                pass
    return n


def nvsmi(gpu: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                               "--format=csv,noheader", "-i", gpu], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


def restart(port: int, reason: str, last: dict | None) -> None:
    pid = find_server(port)
    if pid is None:
        log(f":{port} 서버 프로세스 없음 — 재기동할 argv 를 모른다. 건너뜀 ({reason})")
        return
    argv, env = proc_argv_env(pid)
    cvd = env.get("CUDA_VISIBLE_DEVICES", "?")
    event = {"time": time.strftime("%F %T"), "port": port, "gpu": cvd, "reason": reason, "pid": pid,
             "last": {k: last.get(k) for k in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                                                "vllm:generation_tokens_total", "vllm:kv_cache_usage_perc")} if last else None,
             "nvidia_smi": nvsmi(cvd) if cvd != "?" else None}
    log(f":{port} GPU {cvd} 재기동 — {reason}. nvidia-smi[{event['nvidia_smi']}] last={event['last']}")
    with open(LOGD / "watchdog_events.jsonl", "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    logf = LOGD / f"serve_{port}.log"
    if logf.exists():
        logf.rename(LOGD / f"serve_{port}.stall_{time.strftime('%Y%m%d_%H%M%S')}.log")
    kill_tree(pid)
    n = kill_orphans(cvd)
    if n:
        log(f":{port} 고아 워커 {n}개 정리")
    time.sleep(3)
    with open(logf, "ab") as out:
        subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, cwd=env.get("PWD", "/"))
    log(f":{port} 재기동 명령 실행 (argv {len(argv)} 토큰, CUDA_VISIBLE_DEVICES={cvd})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True)
    ap.add_argument("--stall-s", type=int, default=120)
    ap.add_argument("--dead-s", type=int, default=60)
    ap.add_argument("--poll-s", type=int, default=15)
    ap.add_argument("--exit-s", type=int, default=90)
    ap.add_argument("--grace-s", type=int, default=600, help="재기동 직후 이 시간 동안은 사망 판정 유예(모델 적재)")
    a = ap.parse_args()
    ports = [int(p) for p in a.ports.split(",")]
    LOGD.mkdir(parents=True, exist_ok=True)
    state = {p: {"gen": None, "since": time.time(), "dead_since": None, "restarted": 0.0} for p in ports}
    lb_missing_since = None
    log(f"감시 시작 ports={ports} stall={a.stall_s}s dead={a.dead_s}s poll={a.poll_s}s")
    while True:
        now = time.time()
        if lb_alive():
            lb_missing_since = None
        else:
            lb_missing_since = lb_missing_since or now
            if now - lb_missing_since > a.exit_s:
                log("lb_proxy 없음 — 단계 종료로 보고 감시 끝")
                return 0
        for p in ports:
            st = state[p]
            m = metrics(p)
            if m is None:
                if now - st["restarted"] < a.grace_s:
                    continue
                st["dead_since"] = st["dead_since"] or now
                if now - st["dead_since"] > a.dead_s:
                    restart(p, f"/metrics 불통 {a.dead_s}s+", None)
                    st.update(gen=None, since=now, dead_since=None, restarted=now)
                continue
            st["dead_since"] = None
            gen = m.get("vllm:generation_tokens_total", 0.0)
            running = m.get("vllm:num_requests_running", 0.0)
            if st["gen"] is None or gen != st["gen"] or running == 0:
                st["gen"] = gen; st["since"] = now
                continue
            if now - st["since"] > a.stall_s:
                restart(p, f"running={running:.0f} 인데 생성 토큰 {a.stall_s}s 정지", m)
                st.update(gen=None, since=now, dead_since=None, restarted=now)
        time.sleep(a.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
