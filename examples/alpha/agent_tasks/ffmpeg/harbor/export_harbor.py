"""Export pilot tasks as a Harbor (Terminal-Bench 2 format) local dataset.

Per task:  task.toml · instruction.md · environment/Dockerfile · solution/solve.sh · tests/
  environment/  inputs are generated at image build time with the same media.py
  tests/        uploaded by Harbor only at verification, so the agent never sees the reference.
                run_verify.py regenerates inputs + reference outputs under /ref_root (the agent
                may have touched /work/in) and scores /work against them.
  solution/     the reference commands — `harbor run -a oracle` must score 50/50.

  python3 harbor/export_harbor.py --out <dir>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tasks_pilot import TASKS  # noqa: E402

BASE_IMAGE = "alpha-ffmpeg-base:1"

# The Terminus-2 system prompt is English; the request itself stays in the user's language.
CONTEXT = """

---
작업 환경: 현재 디렉터리는 /work 이고 모든 경로는 여기 기준이야. 입력 파일은 in/ 아래에 있고 결과는 out/ 아래에 저장해.
in/ 의 원본 파일은 수정하거나 지우지 마. ffmpeg, ffprobe, python3 를 쓸 수 있어.
"""

TASK_TOML = """version = "1.0"

[metadata]
author_name = "alpha"
difficulty = "{difficulty}"
category = "{category}"
tags = ["ffmpeg", "level-{level}"]

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 600.0

[environment]
build_timeout_sec = 600.0
cpus = 2
memory = "4G"
storage = "10G"
"""

DOCKERFILE = f"""FROM {BASE_IMAGE}
WORKDIR /work
COPY media.py env.py task.json /opt/ffb/
RUN cd /opt/ffb && python3 -c "import json; from env import make_inputs; make_inputs(json.load(open('task.json')), '/work')" \\
    && rm -rf /opt/ffb
"""

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/run_verify.py > /logs/verifier/verify.log 2>&1
[ -f /logs/verifier/reward.txt ] || echo 0 > /logs/verifier/reward.txt
cat /logs/verifier/verify.log
"""

RUN_VERIFY = '''import json
import sys

sys.path.insert(0, "/tests")
from checks import verify
from env import prepare

task = json.load(open("/tests/task.json"))
_, refs = prepare(task, "/ref_root")
result = verify(task, "/work", refs)
json.dump(result, open("/logs/verifier/details.json", "w"), ensure_ascii=False, indent=1)
open("/logs/verifier/reward.txt", "w").write("1" if result["passed"] else "0")
for r in result["results"]:
    print("PASS" if r["ok"] else "FAIL", r["type"], r["file"], "-", r["detail"])
'''


def export(task, out):
    d = out / task["id"].replace("_", "-")
    for sub in ("environment", "solution", "tests"):
        (d / sub).mkdir(parents=True)
    (d / "task.toml").write_text(TASK_TOML.format(
        difficulty={1: "easy", 2: "medium", 3: "hard"}[task["level"]], category=task["category"], level=task["level"]))
    (d / "instruction.md").write_text(task["instruction"] + CONTEXT, encoding="utf-8")
    # the image only needs the input spec — the reference must not be baked into it
    public = {"id": task["id"], "inputs": task["inputs"]}
    (d / "environment" / "task.json").write_text(json.dumps(public, ensure_ascii=False))
    (d / "environment" / "Dockerfile").write_text(DOCKERFILE)
    for f in ("media.py", "env.py"):
        shutil.copy(HERE.parent / f, d / "environment" / f)
    for f in ("media.py", "env.py", "checks.py"):
        shutil.copy(HERE.parent / f, d / "tests" / f)
    (d / "tests" / "task.json").write_text(json.dumps(task, ensure_ascii=False))
    (d / "tests" / "run_verify.py").write_text(RUN_VERIFY)
    (d / "tests" / "test.sh").write_text(TEST_SH)
    (d / "solution" / "solve.sh").write_text("#!/bin/bash\nset -e\ncd /work\n" + "\n".join(task["reference"]) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for t in TASKS:
        export(t, out)
    print(f"{len(TASKS)} tasks -> {out}")
