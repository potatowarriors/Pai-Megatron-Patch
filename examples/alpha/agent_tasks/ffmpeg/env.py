"""Task workspace: build inputs, build the reference output, run shell commands.

Layout under <root>:
  work/in/   inputs (what the agent sees)      work/out/  agent output
  ref/in  -> symlink to work/in                 ref/out/   reference output
  ref_v1/, ref_v2/ ...                          outputs of the task's `variants`
The reference lives outside work/ so an agent with a shell cannot read it.
"""
import os
import subprocess
from pathlib import Path

from media import make_input


def run_commands(cmds, cwd, timeout=300):
    """Run shell commands in order; stop at the first failure. Returns (ok, log)."""
    log = []
    for cmd in cmds:
        try:
            proc = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            log.append({"cmd": cmd, "rc": None, "stderr": f"timeout after {timeout}s"})
            return False, log
        log.append({"cmd": cmd, "rc": proc.returncode, "stderr": proc.stderr[-1000:]})
        if proc.returncode != 0:
            return False, log
    return True, log


def make_inputs(task, work):
    work = Path(work)
    (work / "in").mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(exist_ok=True)
    for spec in task["inputs"]:
        make_input(spec, work / spec["path"])


def prepare(task, root):
    """Create inputs and reference outputs. Returns (workdir, refdirs); refdirs[0] is the
    canonical reference, the rest are the task's `variants`."""
    root = Path(root)
    work = root / "work"
    make_inputs(task, work)
    refs = []
    for i, cmds in enumerate([task["reference"]] + task.get("variants", [])):
        ref = root / ("ref" if i == 0 else f"ref_v{i}")
        (ref / "out").mkdir(parents=True)
        os.symlink(work / "in", ref / "in")
        ok, log = run_commands(cmds, ref)
        if not ok:
            raise RuntimeError(f"{task['id']}: reference #{i} failed: {log[-1]}")
        refs.append(ref)
    return work, refs


def reset_output(work):
    out = Path(work) / "out"
    for p in out.iterdir():
        p.unlink()
