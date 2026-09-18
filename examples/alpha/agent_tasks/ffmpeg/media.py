"""Synthetic input media for ffmpeg agent tasks.

Every input is generated from lavfi sources, so tasks carry no real footage
(no copyright or data-leak concern) and stay small enough for an 8-core box.
testsrc2 changes every frame, which is what lets the frame-compare check
catch timing mistakes (see README "검증기 보정").
"""
import subprocess
from pathlib import Path

FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]


def run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{proc.stderr[-2000:]}")
    return proc


def _video_src(spec):
    w, h, fps, dur = spec.get("w", 320), spec.get("h", 240), spec.get("fps", 30), spec.get("dur", 8)
    pattern = spec.get("pattern", "testsrc2")
    # content: [cw, ch] -> the picture is cw x ch, centred on a black w x h canvas (baked-in letterbox/pillarbox)
    cw, ch = spec.get("content", [w, h])
    src = f"{pattern}=s={cw}x{ch}:r={fps}:d={dur}"
    chain = []
    if [cw, ch] != [w, h]:
        chain.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")
    # black: [start, end] or a list of them -> frames in that window are forced to black (blackdetect tasks)
    if "black" in spec:
        windows = spec["black"] if isinstance(spec["black"][0], list) else [spec["black"]]
        for a, b in windows:
            chain.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,{a},{b})'")
    # sar: non-square pixels, e.g. "4/3" (anamorphic sources)
    if "sar" in spec:
        chain.append(f"setsar={spec['sar']}")
    # busy: temporal noise makes the clip expensive to encode (size/bitrate tasks)
    if spec.get("busy"):
        chain.append("noise=alls=40:allf=t")
    # vfr: drop an irregular subset of frames and keep their timestamps
    if spec.get("vfr"):
        chain.append("select='not(eq(mod(n,5),1)+eq(mod(n,7),3))'")
    return src + ("," + ",".join(chain) if chain else "")


def _audio_src(a, dur):
    hz, vol, sr = a.get("hz", 440), a.get("volume", 0.5), a.get("sr", 48000)
    dur = a.get("dur", dur)  # an audio track may outlast (or fall short of) the picture
    if "hz_right" in a:
        # different tone per channel (channel-mapping tasks)
        return (f"sine=f={hz}:r={sr}:d={dur},volume={vol}[l];sine=f={a['hz_right']}:r={sr}:d={dur},volume={vol}[r];"
                f"[l][r]join=inputs=2:channel_layout=stereo[out0]")
    src = f"sine=f={hz}:r={sr}:d={dur},volume={vol}"
    # silence: list of [start, end] windows muted (silencedetect tasks)
    for s, e in a.get("silence", []):
        src += f",volume=0:enable='between(t,{s},{e})'"
    ch = a.get("channels", 2)
    layout = "mono" if ch == 1 else "stereo"
    return src + f",aformat=channel_layouts={layout}"


def make_video(spec, path):
    """spec keys: w h fps dur pattern vcodec pix_fmt crf gop audio(list) black busy vfr
    rotation subs(list of [start,end,text]) container is taken from the file suffix."""
    dur = spec.get("dur", 8)
    cmd = FFMPEG + ["-f", "lavfi", "-i", _video_src(spec)]
    audios = spec.get("audio", [{}])
    for a in audios:
        cmd += ["-f", "lavfi", "-i", _audio_src(a, dur)]
    # subs: one cue list, or sub_tracks: [{"lang": .., "cues": [..]}, ...] for several embedded tracks
    tracks = spec.get("sub_tracks") or ([{"cues": spec["subs"]}] if spec.get("subs") else [])
    srts = []
    for i, tr in enumerate(tracks):
        srt = Path(path).with_suffix(f".embed{i}.srt")
        write_srt(tr["cues"], srt)
        srts.append(srt)
        cmd += ["-i", str(srt)]
    cmd += ["-map", "0:v"]
    for i in range(len(audios)):
        cmd += ["-map", f"{i + 1}:a"]
    for i, tr in enumerate(tracks):
        cmd += ["-map", f"{len(audios) + 1 + i}:s"]
        if "lang" in tr:
            cmd += [f"-metadata:s:s:{i}", f"language={tr['lang']}"]
    if tracks:
        cmd += ["-c:s", "srt" if str(path).endswith(".mkv") else "mov_text"]
    cmd += ["-c:v", spec.get("vcodec", "libx264"), "-pix_fmt", spec.get("pix_fmt", "yuv420p"),
            "-crf", str(spec.get("crf", 18))]
    if spec.get("vcodec") == "libx265":
        cmd += ["-x265-params", "log-level=error"]
    if "gop" in spec:
        cmd += ["-g", str(spec["gop"])]
    if spec.get("vfr"):
        cmd += ["-fps_mode", "vfr"]
    if audios:
        cmd += ["-c:a", spec.get("acodec", "aac")]
        for i, a in enumerate(audios):
            if "lang" in a:
                cmd += [f"-metadata:s:a:{i}", f"language={a['lang']}"]
    cmd += ["-t", str(max([dur] + [a.get("dur", dur) for a in audios])), str(path)]
    run(cmd)
    for srt in srts:
        srt.unlink()
    if "rotation" in spec:
        # display-matrix rotation is a container-level flag: set it with a stream copy
        tmp = Path(path).with_name("_rot_" + Path(path).name)
        Path(path).rename(tmp)
        run(FFMPEG + ["-display_rotation", str(spec["rotation"]), "-i", str(tmp), "-c", "copy", str(path)])
        tmp.unlink()


def make_audio(spec, path):
    run(FFMPEG + ["-f", "lavfi", "-i", _audio_src(spec, spec.get("dur", 8)), str(path)])


def make_image(spec, path):
    w, h, color = spec.get("w", 64), spec.get("h", 32), spec.get("color", "red")
    run(FFMPEG + ["-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:d=1", "-frames:v", "1", str(path)])


def _ts(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(cues, path):
    lines = []
    for i, (s, e, text) in enumerate(cues, 1):
        lines += [str(i), f"{_ts(s)} --> {_ts(e)}", text, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def make_input(spec, path):
    kind = spec["kind"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if kind == "video":
        make_video(spec, path)
    elif kind == "audio":
        make_audio(spec, path)
    elif kind == "image":
        make_image(spec, path)
    elif kind == "srt":
        write_srt(spec["cues"], path)
    else:
        raise ValueError(f"unknown input kind: {kind}")
