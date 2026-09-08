"""common.py — Harbor 과제 디렉토리 작성기 + 교사 호출 + 공용 유틸 (터미널 SDG P2).

Harbor 과제 계약 (terminal-bench@2.0 과제 실측, 2026-09-07):
  <task>/task.toml                 version, [metadata], [verifier] timeout_sec, [agent] timeout_sec,
                                   [environment] build_timeout_sec · cpus · memory · storage (docker_image 없으면 Dockerfile 빌드)
  <task>/instruction.md            에이전트에게 주는 과제 설명 (Terminus-2 템플릿의 {instruction})
  <task>/environment/Dockerfile    과제 컨테이너 (WORKDIR /app 필수 — test.sh 가 PWD=/ 이면 실패)
  <task>/tests/test.sh             검증 스크립트: /tests/ 에 복사돼 실행, 보상을 /logs/verifier/reward.txt 에 0/1 로 기록
  <task>/solution/solve.sh         oracle 에이전트가 실행하는 정답 스크립트 (과제 유효성 검증용)

이미지 정책: 과제 수천 개가 같은 베이스 레이어(BASE_DOCKERFILE)를 공유하고 과제 파일만 마지막 COPY —
docker 레이어 캐시로 빌드가 수 초, 디스크는 과제당 수백 KB.
"""
import hashlib, json, os, re, time, urllib.request

TEACHER_BASE = os.environ.get("TEACHER_BASE", "http://localhost:8300/v1")
TEACHER_MODEL = "glm53-flash"

# 학습 데이터 오염 방지: 합성 과제에는 이 문자열을 넣지 않는다 (TB-2 canary 와 구분). 변환기가 canary 를 검출해 드롭한다.
BASE_DOCKERFILE = """FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \\
    bash coreutils findutils grep sed gawk curl wget git jq tree less vim-tiny nano procps \\
    build-essential make unzip zip tar gzip bzip2 xz-utils sqlite3 tmux \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest==8.4.1 numpy==2.3.3 pyyaml==6.0.2 requests==2.32.4 pandas==2.3.2
WORKDIR /app
"""


def slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:n].rstrip("-") or "task"


def task_toml(category: str, difficulty: str, agent_timeout: int, verifier_timeout: int = 600,
              cpus: int = 1, memory: str = "2G", tags=None, author="alpha-sdg") -> str:
    tags = tags or []
    return f'''version = "1.0"

[metadata]
author_name = "{author}"
author_email = "alpha-sdg@local"
difficulty = "{difficulty}"
category = "{category}"
tags = {json.dumps(tags)}
source = "alpha-terminal-sdg"

[verifier]
timeout_sec = {float(verifier_timeout)}

[agent]
timeout_sec = {float(agent_timeout)}

[environment]
build_timeout_sec = 900.0
cpus = {cpus}
memory = "{memory}"
storage = "10G"
'''


def write_task(root: str, name: str, instruction: str, files: dict, test_sh: str, test_files: dict,
               solve_sh: str, category: str, difficulty: str = "medium", agent_timeout: int = 1200,
               dockerfile_extra: str = "", tags=None, meta: dict | None = None) -> str:
    """과제 디렉토리 생성. files = 컨테이너 /app 에 놓을 파일들 {상대경로: 내용}, test_files = /tests 에 놓을 파일들."""
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "environment", "app"), exist_ok=True)
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    os.makedirs(os.path.join(d, "solution"), exist_ok=True)
    with open(os.path.join(d, "task.toml"), "w") as f:
        f.write(task_toml(category, difficulty, agent_timeout, tags=tags))
    with open(os.path.join(d, "instruction.md"), "w") as f:
        f.write(instruction.rstrip() + "\n")
    for rel, content in files.items():
        p = os.path.join(d, "environment", "app", rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    with open(os.path.join(d, "environment", "Dockerfile"), "w") as f:
        f.write(BASE_DOCKERFILE + (dockerfile_extra.rstrip() + "\n" if dockerfile_extra.strip() else "")
                + ("COPY app/ /app/\n" if files else ""))
    with open(os.path.join(d, "tests", "test.sh"), "w") as f:
        f.write(test_sh if test_sh.startswith("#!") else "#!/bin/bash\n" + test_sh)
    os.chmod(os.path.join(d, "tests", "test.sh"), 0o755)
    for rel, content in test_files.items():
        p = os.path.join(d, "tests", rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    with open(os.path.join(d, "solution", "solve.sh"), "w") as f:
        f.write(solve_sh if solve_sh.startswith("#!") else "#!/bin/bash\n" + solve_sh)
    os.chmod(os.path.join(d, "solution", "solve.sh"), 0o755)
    if meta:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
    return d


def teacher(messages, max_tokens=8192, temperature=1.0, top_p=0.95, reasoning_effort="high",
            timeout=900, retries=3, **extra):
    """OpenAI 호환 chat 호출. 반환 (content, reasoning, usage)."""
    body = {"model": TEACHER_MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "reasoning_effort": reasoning_effort}
    body.update(extra)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(TEACHER_BASE + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.load(r)
            m = j["choices"][0]["message"]
            return (m.get("content") or ""), (m.get("reasoning") or m.get("reasoning_content") or ""), j.get("usage") or {}
        except Exception as e:  # noqa: BLE001
            last = e; time.sleep(2 + 3 * i)
    raise RuntimeError(f"teacher call failed: {last}")


def extract_json_block(text: str):
    """응답에서 첫 JSON 객체를 뽑는다 (```json 펜스 허용)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cand = m.group(1) if m else None
    if cand is None:
        i, j = text.find("{"), text.rfind("}")
        cand = text[i:j + 1] if i >= 0 and j > i else None
    if cand is None:
        return None
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None


def extract_code_block(text: str, lang: str = "python"):
    m = re.search(r"```" + lang + r"\s*\n(.*?)```", text, re.S) or re.search(r"```\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else None


def short_hash(s: str, n: int = 8) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:n]
