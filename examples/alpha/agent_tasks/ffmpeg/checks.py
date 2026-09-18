"""Deterministic output checks for ffmpeg agent tasks.

No LLM judge: every check reads the produced file with ffprobe/ffmpeg and
compares against a declared value or against the reference output. A task
passes only if all of its checks pass.
"""
import glob
import json
import re
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

MIN_PSNR = 25.0  # calibrated: correct >= 36.9 dB even at crf 30, wrong <= 18.0 dB (README "검증기 보정")


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path):
    out = _run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    if out.returncode != 0:
        raise ValueError(f"ffprobe failed: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout)


def _streams(info, kind):
    return [s for s in info["streams"] if s["codec_type"] == kind]


def _fps(stream):
    rate = stream.get("avg_frame_rate", "0/0")
    return float(Fraction(rate)) if not rate.endswith("/0") else 0.0


def _expect(errors, name, got, want):
    if got != want:
        errors.append(f"{name}: got {got!r}, want {want!r}")


def check_format(path, ref, c):
    name = probe(path)["format"]["format_name"]
    return (c["contains"] in name, f"format_name={name}")


def check_vstream(path, ref, c):
    vs = _streams(probe(path), "video")
    errors = []
    _expect(errors, "video stream count", len(vs), c.get("count", 1))
    if vs and not errors:
        s = vs[0]
        for key, field in (("codec", "codec_name"), ("width", "width"), ("height", "height"),
                           ("pix_fmt", "pix_fmt"), ("profile", "profile")):
            if key in c:
                _expect(errors, key, s.get(field), c[key])
        if "width_any" in c and s.get("width") not in c["width_any"]:
            errors.append(f"width: got {s.get('width')}, want one of {c['width_any']}")
        if "height_any" in c and s.get("height") not in c["height_any"]:
            errors.append(f"height: got {s.get('height')}, want one of {c['height_any']}")
        if "level" in c:
            _expect(errors, "level", s.get("level"), c["level"])
        if "sar" in c and s.get("sample_aspect_ratio", "1:1") != c["sar"]:
            errors.append(f"sample_aspect_ratio: got {s.get('sample_aspect_ratio')}, want {c['sar']}")
        if "rotation" in c:
            rot = [int(d["rotation"]) for d in s.get("side_data_list", []) if "rotation" in d] or [0]
            if abs(rot[0]) != abs(c["rotation"]):
                errors.append(f"rotation: got {rot[0]}, want ±{abs(c['rotation'])}")
        if "frames" in c:
            n = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                      "stream=nb_read_frames", "-of", "csv=p=0", str(path)]).stdout.strip()
            _expect(errors, "frame count", int(n or 0), c["frames"])
        if "fps" in c and abs(_fps(s) - c["fps"]) > c.get("fps_tol", 0.1):
            errors.append(f"fps: got {_fps(s):.3f}, want {c['fps']}")
        if c.get("no_rotation"):
            rot = [d.get("rotation") for d in s.get("side_data_list", []) if d.get("rotation")]
            if rot:
                errors.append(f"rotation side data present: {rot}")
    return (not errors, "; ".join(errors) or "ok")


def check_astream(path, ref, c):
    auds = _streams(probe(path), "audio")
    errors = []
    _expect(errors, "audio stream count", len(auds), c.get("count", 1))
    if auds and not errors:
        s = auds[c.get("index", 0)]
        if "codec" in c:
            _expect(errors, "codec", s.get("codec_name"), c["codec"])
        if "channels" in c:
            _expect(errors, "channels", s.get("channels"), c["channels"])
        if "sample_rate" in c:
            _expect(errors, "sample_rate", int(s.get("sample_rate", 0)), c["sample_rate"])
    return (not errors, "; ".join(errors) or "ok")


def check_duration(path, ref, c):
    dur = float(probe(path)["format"]["duration"])
    return (abs(dur - c["value"]) <= c.get("tol", 0.15), f"duration={dur:.3f}, want {c['value']}±{c.get('tol', 0.15)}")


def check_size_max(path, ref, c):
    size = Path(path).stat().st_size
    return (size <= c["bytes"], f"size={size}, max {c['bytes']}")


def check_bitrate_max(path, ref, c):
    rate = int(probe(path)["format"]["bit_rate"])
    return (rate <= c["bps"], f"bit_rate={rate}, max {c['bps']}")


def _psnr(cand, ref, shift_cand, shift_ref, region=None):
    rv = _streams(probe(ref), "video")[0]
    w, h, fps = rv["width"], rv["height"], _fps(rv)
    still = fps == 0 or str(ref).lower().endswith((".png", ".jpg", ".jpeg"))
    # region: crop expression "w:h:x:y" — a small edit (subtitle, logo) barely moves whole-frame PSNR
    crop = f",crop={region}" if region else ""
    def chain(shift):
        if still:
            return f"scale={w}:{h}{crop}"
        return f"scale={w}:{h}{crop},fps={fps},trim=start_frame={shift},setpts=N/({fps}*TB)"
    graph = f"[0:v]{chain(shift_cand)}[a];[1:v]{chain(shift_ref)}[b];[a][b]psnr"
    out = _run(["ffmpeg", "-hide_banner", "-i", str(cand), "-i", str(ref), "-lavfi", graph, "-f", "null", "-"])
    m = re.search(r"average:(inf|[0-9.]+)", out.stderr)
    if not m:
        raise ValueError(f"psnr not measurable: {out.stderr.strip()[-300:]}")
    return float("inf") if m.group(1) == "inf" else float(m.group(1))


def check_frames_match(path, ref, c):
    # valid solutions can differ by one frame at the boundary (-r vs fps filter, -ss placement),
    # so take the best of a +-1 frame alignment
    # exact: the task is about frame-exact selection, so an off-by-one must not be forgiven
    shifts = ((0, 0),) if c.get("exact") else ((0, 0), (1, 0), (0, 1))
    best = max(_psnr(path, r, a, b, c.get("region")) for r in c["_refs"] if r.is_file() for a, b in shifts)
    need = c.get("min_psnr", MIN_PSNR)
    return (best >= need, f"psnr={best:.2f} dB vs reference, need >= {need}")


def _band_energy(path, stream_index=0, channel=None):
    # channel: measure one channel only (0 = left); default is the mono downmix
    mix = ["-af", f"pan=mono|c0=c{channel}"] if channel is not None else ["-ac", "1"]
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:a:{stream_index}", *mix,
                          "-ar", "8000", "-f", "s16le", "-"], capture_output=True)
    pcm = np.frombuffer(out.stdout, dtype=np.int16).astype(np.float64)
    if pcm.size < 8000:
        raise ValueError("audio shorter than 1 s or undecodable")
    spec = np.abs(np.fft.rfft(pcm)) ** 2
    freqs = np.fft.rfftfreq(pcm.size, 1 / 8000)
    total = spec[freqs > 50].sum() or 1.0
    return lambda hz: spec[(freqs > hz - 15) & (freqs < hz + 15)].sum() / total


def check_audio_tones(path, ref, c):
    energy = _band_energy(path, c.get("index", 0), c.get("channel"))
    errors = [f"{hz} Hz missing (share {energy(hz):.3f})" for hz in c.get("present", []) if energy(hz) < 0.05]
    errors += [f"{hz} Hz still present (share {energy(hz):.3f})" for hz in c.get("absent", []) if energy(hz) > 0.01]
    return (not errors, "; ".join(errors) or "ok")


def _mean_volume(path):
    out = _run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-vn", "-f", "null", "-"])
    m = re.search(r"mean_volume: (-?[0-9.]+|-inf) dB", out.stderr)
    if not m:
        raise ValueError("no audio to measure")
    return float(m.group(1))


def check_mean_volume(path, ref, c):
    got = _mean_volume(path)
    if "max_db" in c:
        return (got <= c["max_db"], f"mean_volume={got} dB, max {c['max_db']}")
    want = _mean_volume(ref)
    return (abs(got - want) <= c.get("tol", 1.0), f"mean_volume={got} dB, reference {want} dB")


def check_loudness(path, ref, c):
    out = _run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "ebur128=framelog=quiet", "-vn", "-f", "null", "-"])
    m = re.findall(r"I:\s+(-?[0-9.]+) LUFS", out.stderr)
    if not m:
        raise ValueError("loudness not measurable")
    got = float(m[-1])
    return (abs(got - c["lufs"]) <= c.get("tol", 1.5), f"integrated={got} LUFS, want {c['lufs']}±{c.get('tol', 1.5)}")


def check_faststart(path, ref, c):
    order = []
    with open(path, "rb") as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            size, kind = struct.unpack(">I4s", head)
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0] - 8
            order.append(kind.decode("latin1"))
            if size < 8:
                break
            f.seek(size - 8, 1)
    ok = "moov" in order and "mdat" in order and order.index("moov") < order.index("mdat")
    return (ok, f"atom order={order}")


def check_cfr(path, ref, c):
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                "-of", "csv=p=0", str(path)])
    pts = sorted(float(x) for x in out.stdout.split() if x not in ("N/A", ""))
    deltas = np.diff(pts)
    spread = float(deltas.max() - deltas.min())
    return (spread < 0.002, f"frame interval spread={spread * 1000:.2f} ms over {len(pts)} frames")


def _keyframes(path):
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey", "-show_entries",
                "frame=pts_time", "-of", "csv=p=0", str(path)])
    return [float(x.strip(",")) for x in out.stdout.split() if x.strip(",")]


def check_gop(path, ref, c):
    """Keyframes exactly every `seconds` and nowhere else (fixed GOP for adaptive streaming)."""
    keys, step = _keyframes(path), c["seconds"]
    dur = float(_streams(probe(path), "video")[0].get("duration") or probe(path)["format"]["duration"])
    want = [i * step for i in range(int((dur - 0.05) // step) + 1)]  # a single keyframe at 0 is not "every 2 s"
    ok = len(keys) == len(want) and all(abs(k - w) <= 0.02 for k, w in zip(keys, want))
    return (ok, f"keyframes at {[round(k, 2) for k in keys][:8]}, want {want}")


def check_silence_intervals(path, ref, c):
    """Silences of >= 0.5 s found in the OUTPUT must equal `expected` ([] = none left)."""
    out = _run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "silencedetect=noise=-40dB:d=0.5", "-vn", "-f", "null", "-"])
    got = list(zip(map(float, re.findall(r"silence_start: (-?[0-9.]+)", out.stderr)),
                   map(float, re.findall(r"silence_end: ([0-9.]+)", out.stderr))))
    want, tol = [tuple(x) for x in c["expected"]], c.get("tol", 0.15)
    ok = len(got) == len(want) and all(abs(g[0] - w[0]) <= tol and abs(g[1] - w[1]) <= tol for g, w in zip(got, want))
    return (ok, f"silences {got}, want {want}±{tol}")


def check_json_records(path, ref, c):
    """JSON list of objects matched by `key`; numbers compared with tolerance, the rest exactly."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    key, tol = c["key"], c.get("tol", 0.1)
    got = {str(d[key]): d for d in data}
    errors = []
    for want in c["expected"]:
        g = got.get(str(want[key]))
        if g is None:
            errors.append(f"{want[key]}: missing")
            continue
        for k, v in want.items():
            same = abs(float(g.get(k, "nan")) - v) <= tol if isinstance(v, (int, float)) and not isinstance(v, bool) \
                else g.get(k) == v
            if not same:
                errors.append(f"{want[key]}.{k}: got {g.get(k)!r}, want {v!r}")
    if len(got) != len(c["expected"]):
        errors.append(f"{len(got)} records, want {len(c['expected'])}")
    return (not errors, "; ".join(errors) or "ok")


def check_json_numbers(path, ref, c):
    got = sorted(float(x) for x in json.loads(Path(path).read_text(encoding="utf-8")))
    want, tol = sorted(c["expected"]), c.get("tol", 0.05)
    ok = len(got) == len(want) and all(abs(g - w) <= tol for g, w in zip(got, want))
    return (ok, f"got {got}, want {want}±{tol}")


def check_file_count(path, ref, c):
    found = sorted(glob.glob(str(Path(c["_workdir"]) / c["glob"])))
    return (len(found) == c["count"], f"{len(found)} files match {c['glob']}, want {c['count']}")


def check_text_contains(path, ref, c):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    missing = [s for s in c["substrings"] if s not in text]
    return (not missing, f"missing: {missing}" if missing else "ok")


def check_json_intervals(path, ref, c):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    got = sorted((float(d["start"]), float(d["end"])) for d in data)
    want, tol = sorted(map(tuple, c["expected"])), c.get("tol", 0.2)
    ok = len(got) == len(want) and all(abs(g[0] - w[0]) <= tol and abs(g[1] - w[1]) <= tol for g, w in zip(got, want))
    return (ok, f"got {got}, want {want}±{tol}")


CHECKS = {name[len("check_"):]: fn for name, fn in globals().items() if name.startswith("check_")}


def verify(task, workdir, refdirs):
    """Run every check of `task` against refdirs (canonical first, then variants).
    Returns {"passed": bool, "results": [...]}."""
    results = []
    for c in task["checks"]:
        rel = c.get("file", task["output"])
        path, refs = Path(workdir) / rel, [Path(d) / rel for d in refdirs]
        try:
            if c["type"] != "file_count" and not path.is_file():
                ok, detail = False, f"{rel} not found"
            else:
                ok, detail = CHECKS[c["type"]](path, refs[0], {**c, "_workdir": str(workdir), "_refs": refs})
        except Exception as e:  # unreadable or malformed output is a task failure, not a harness crash
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append({"type": c["type"], "file": rel, "ok": bool(ok), "detail": detail})
    return {"passed": all(r["ok"] for r in results), "results": results}
