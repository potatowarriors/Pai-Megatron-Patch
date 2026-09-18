"""Pilot task set v0 (50 tasks) — for calibrating difficulty, NOT the frozen benchmark.

Levels: L1 single operation · L2 compound operation · L3 spec compliance,
input diagnosis and error recovery. Each task carries
  reference  the canonical solution (must PASS)
  variants   extra reference outputs: valid solutions whose frames legitimately differ
             (fps filter vs -r pick different source frames); frames_match takes the best
  alts       other valid solutions (must PASS — guards against false negatives)
  negatives  plausible wrong solutions (must FAIL — guards against false positives)
`python tasks_pilot.py` writes tasks/pilot_v0.jsonl.
"""
import json
from pathlib import Path

F = "ffmpeg -y -loglevel error"
X = "-c:v libx264 -pix_fmt yuv420p"


def vid(path="in/input.mp4", **kw):
    return {"kind": "video", "path": path, **kw}


def V(**kw):
    return {"type": "vstream", **kw}


def A(**kw):
    return {"type": "astream", **kw}


def D(value, tol=0.15, **kw):
    return {"type": "duration", "value": value, "tol": tol, **kw}


def M(**kw):
    return {"type": "frames_match", **kw}


TASKS = []


def task(id, level, category, instruction, output, reference, checks, inputs=None, alts=(), negatives=(), variants=()):
    TASKS.append({
        "id": id, "level": level, "category": category, "instruction": instruction,
        "inputs": inputs or [vid()], "output": output,
        "reference": reference if isinstance(reference, list) else [reference],
        "variants": [v if isinstance(v, list) else [v] for v in variants],
        "alts": [{"why": w, "cmds": c if isinstance(c, list) else [c]} for w, c in alts],
        "negatives": [{"why": w, "cmds": c if isinstance(c, list) else [c]} for w, c in negatives],
        "checks": checks,
    })


# ---------------------------------------------------------------- L1: single operation
task("l1_trim", 1, "cut",
     "in/input.mp4에서 2초 지점부터 3초 분량만 정확히 잘라서 out/clip.mp4로 저장해줘. 소리도 같이 있어야 해.",
     "out/clip.mp4", f"{F} -i in/input.mp4 -ss 2 -t 3 {X} -c:a aac out/clip.mp4",
     [D(3), V(), A(), M()],
     alts=[("input seeking + -to", f"{F} -ss 2 -to 5 -i in/input.mp4 {X} -crf 28 -c:a aac out/clip.mp4"),
           ("stream copy (mp4 edit list keeps the cut exact)", f"{F} -ss 2 -i in/input.mp4 -t 3 -c copy out/clip.mp4")],
     negatives=[("wrong start (0s)", f"{F} -i in/input.mp4 -ss 0 -t 3 {X} -c:a aac out/clip.mp4")])

task("l1_scale", 1, "resize",
     "in/input.mp4 해상도를 160x120으로 줄여서 out/small.mp4로 만들어줘.",
     "out/small.mp4", f"{F} -i in/input.mp4 -vf scale=160:120 {X} -c:a copy out/small.mp4",
     [V(width=160, height=120), D(8), A(), M()],
     negatives=[("wrong height", f"{F} -i in/input.mp4 -vf scale=160:90 {X} -c:a copy out/small.mp4")])

task("l1_webm", 1, "transcode",
     "in/input.mp4를 웹용 WebM(VP9 영상, Opus 오디오)으로 변환해서 out/web.webm으로 저장해줘.",
     "out/web.webm", f"{F} -i in/input.mp4 -c:v libvpx-vp9 -b:v 300k -c:a libopus out/web.webm",
     [{"type": "format", "contains": "webm"}, V(codec="vp9"), A(codec="opus"), D(8), M()],
     negatives=[("h264 in mkv renamed", [f"{F} -i in/input.mp4 -c copy out/web.mkv", "mv out/web.mkv out/web.webm"])])

task("l1_mp3", 1, "audio",
     "in/input.mp4에서 소리만 뽑아서 MP3 파일 out/audio.mp3로 저장해줘.",
     "out/audio.mp3", f"{F} -i in/input.mp4 -vn -c:a libmp3lame out/audio.mp3",
     [{"type": "format", "contains": "mp3"}, V(count=0), A(codec="mp3"), D(8, 0.2),
      {"type": "audio_tones", "present": [440]}])

task("l1_mute", 1, "audio",
     "in/input.mp4에서 오디오를 완전히 제거한 영상을 out/silent.mp4로 저장해줘.",
     "out/silent.mp4", f"{F} -i in/input.mp4 -an -c:v copy out/silent.mp4",
     [A(count=0), V(codec="h264"), D(8), M()],
     negatives=[("audio kept", f"{F} -i in/input.mp4 -c copy out/silent.mp4")])

task("l1_fps", 1, "framerate",
     "in/input.mp4는 30fps인데 15fps로 바꿔서 out/fps15.mp4로 저장해줘. 길이는 그대로여야 해.",
     "out/fps15.mp4", f"{F} -i in/input.mp4 -vf fps=15 {X} -c:a copy out/fps15.mp4",
     [V(fps=15), D(8), M()],
     variants=[f"{F} -i in/input.mp4 -r 15 {X} -c:a copy out/fps15.mp4"],
     alts=[("-r output option", f"{F} -i in/input.mp4 -r 15 {X} -crf 28 -c:a copy out/fps15.mp4")],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/fps15.mp4")])

task("l1_rotate", 1, "geometry",
     "in/input.mp4를 시계 방향으로 90도 돌려서 out/rotated.mp4로 저장해줘.",
     "out/rotated.mp4", f"{F} -i in/input.mp4 -vf transpose=1 {X} -c:a copy out/rotated.mp4",
     [V(width=240, height=320), D(8), M()],
     negatives=[("counter-clockwise", f"{F} -i in/input.mp4 -vf transpose=2 {X} -c:a copy out/rotated.mp4")])

task("l1_hflip", 1, "geometry",
     "in/input.mp4를 좌우 반전(거울상)해서 out/mirror.mp4로 저장해줘.",
     "out/mirror.mp4", f"{F} -i in/input.mp4 -vf hflip {X} -c:a copy out/mirror.mp4",
     [V(width=320, height=240), D(8), M()],
     negatives=[("vertical flip", f"{F} -i in/input.mp4 -vf vflip {X} -c:a copy out/mirror.mp4")])

task("l1_crop", 1, "geometry",
     "in/input.mp4 화면 정중앙을 160x120 크기로 잘라낸 영상을 out/crop.mp4로 저장해줘.",
     "out/crop.mp4", f"{F} -i in/input.mp4 -vf crop=160:120 {X} -c:a copy out/crop.mp4",
     [V(width=160, height=120), D(8), M()],
     negatives=[("top-left crop", f"{F} -i in/input.mp4 -vf crop=160:120:0:0 {X} -c:a copy out/crop.mp4")])

task("l1_gif", 1, "transcode",
     "in/input.mp4의 1초~3초 구간을 가로 160픽셀, 초당 10프레임 GIF로 만들어서 out/clip.gif로 저장해줘.",
     "out/clip.gif",
     f"{F} -ss 1 -t 2 -i in/input.mp4 -filter_complex \"fps=10,scale=160:-1,split[a][b];[a]palettegen[p];[b][p]paletteuse\" out/clip.gif",
     [{"type": "format", "contains": "gif"}, V(codec="gif", width=160, height=120, fps=10, fps_tol=0.5), D(2, 0.3),
      M(min_psnr=22)],
     variants=[f"{F} -ss 1 -t 2 -i in/input.mp4 -r 10 -vf scale=160:-1 out/clip.gif"],
     alts=[("no palette pass", f"{F} -ss 1 -t 2 -i in/input.mp4 -vf fps=10,scale=160:-1 out/clip.gif"),
           ("-r instead of fps filter", f"{F} -ss 1 -t 2 -i in/input.mp4 -r 10 -vf scale=160:-1 out/clip.gif")],
     negatives=[("wrong range (0-2s)", f"{F} -ss 0 -t 2 -i in/input.mp4 -vf fps=10,scale=160:-1 out/clip.gif")])

task("l1_thumb", 1, "still",
     "in/input.mp4의 3초 지점 화면을 PNG 이미지 out/thumb.png로 뽑아줘.",
     "out/thumb.png", f"{F} -ss 3 -i in/input.mp4 -frames:v 1 out/thumb.png",
     [V(codec="png", width=320, height=240), M()],
     negatives=[("first frame", f"{F} -i in/input.mp4 -frames:v 1 out/thumb.png")])

task("l1_mono", 1, "audio",
     "in/input.mp4의 스테레오 오디오를 모노로 바꿔서 out/mono.mp4로 저장해줘. 영상은 건드리지 마.",
     "out/mono.mp4", f"{F} -i in/input.mp4 -c:v copy -ac 1 out/mono.mp4",
     [A(channels=1), V(codec="h264"), D(8), M()],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/mono.mp4")])

task("l1_asr_wav", 1, "audio",
     "음성 인식에 넣을 거라 in/input.mp4의 오디오를 16kHz 모노 16비트 PCM WAV로 뽑아서 out/speech.wav로 저장해줘.",
     "out/speech.wav", f"{F} -i in/input.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le out/speech.wav",
     [{"type": "format", "contains": "wav"}, V(count=0), A(codec="pcm_s16le", channels=1, sample_rate=16000), D(8, 0.2)],
     negatives=[("sample rate kept", f"{F} -i in/input.mp4 -vn -ac 1 out/speech.wav")])

task("l1_hevc", 1, "transcode",
     "in/input.mp4를 H.265(HEVC)로 다시 인코딩해서 out/hevc.mp4로 저장해줘. 오디오는 유지해.",
     "out/hevc.mp4", f"{F} -i in/input.mp4 -c:v libx265 -x265-params log-level=error -c:a copy out/hevc.mp4",
     [V(codec="hevc"), A(), D(8), M()],
     negatives=[("still h264", f"{F} -i in/input.mp4 {X} -c:a copy out/hevc.mp4")])

task("l1_volume", 1, "audio",
     "in/input.mp4 소리가 너무 커. 볼륨을 절반(0.5배)으로 줄여서 out/quiet.mp4로 저장해줘. 영상은 그대로 둬.",
     "out/quiet.mp4", f"{F} -i in/input.mp4 -c:v copy -af volume=0.5 out/quiet.mp4",
     [{"type": "mean_volume", "tol": 1.0}, V(codec="h264"), D(8), M()],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/quiet.mp4")])

task("l1_speed2x", 1, "speed",
     "in/input.mp4를 2배속으로 만들어서 out/fast.mp4로 저장해줘. 소리도 같이 2배속이어야 해.",
     "out/fast.mp4",
     f"{F} -i in/input.mp4 -filter_complex \"[0:v]setpts=0.5*PTS[v];[0:a]atempo=2[a]\" -map \"[v]\" -map \"[a]\" {X} out/fast.mp4",
     [D(4, 0.2), V(), A(), M()],
     negatives=[("video only, audio left at 1x", f"{F} -i in/input.mp4 -vf setpts=0.5*PTS {X} -c:a copy out/fast.mp4")])

task("l1_remux", 1, "container",
     "in/input.mkv를 재인코딩 없이 컨테이너만 MP4로 바꿔서 out/remux.mp4로 저장해줘.",
     "out/remux.mp4", f"{F} -i in/input.mkv -c copy out/remux.mp4",
     [{"type": "format", "contains": "mp4"}, V(codec="h264"), A(codec="aac"), D(8), M(min_psnr=50)],
     inputs=[vid("in/input.mkv")],
     negatives=[("re-encoded", f"{F} -i in/input.mkv {X} -crf 35 -c:a aac out/remux.mp4")])

task("l1_pillarbox", 1, "geometry",
     "in/input.mp4는 4:3 화면이야. 비율을 유지한 채 480x270(16:9) 캔버스 가운데에 놓고 남는 좌우는 검은색으로 채워서 out/wide.mp4로 저장해줘.",
     "out/wide.mp4", f"{F} -i in/input.mp4 -vf scale=360:270,pad=480:270:60:0 {X} -c:a copy out/wide.mp4",
     [V(width=480, height=270), D(8), M()],
     alts=[("force_original_aspect_ratio",
            f"{F} -i in/input.mp4 -vf \"scale=480:270:force_original_aspect_ratio=decrease,pad=480:270:(ow-iw)/2:(oh-ih)/2\" {X} -c:a copy out/wide.mp4")],
     negatives=[("stretched", f"{F} -i in/input.mp4 -vf scale=480:270 {X} -c:a copy out/wide.mp4")])

task("l1_reverse", 1, "speed",
     "in/input.mp4를 거꾸로 재생되는 영상으로 만들어서 out/reverse.mp4로 저장해줘. 소리는 빼줘.",
     "out/reverse.mp4", f"{F} -i in/input.mp4 -vf reverse -an {X} out/reverse.mp4",
     [A(count=0), D(8), M()],
     negatives=[("not reversed", f"{F} -i in/input.mp4 -an {X} out/reverse.mp4")])

task("l1_bitrate", 1, "compress",
     "in/input.mp4 용량이 너무 커. 해상도와 길이는 그대로 두고 파일 전체 비트레이트가 400kbps 이하가 되게 다시 인코딩해서 out/light.mp4로 저장해줘.",
     "out/light.mp4", f"{F} -i in/input.mp4 {X} -b:v 250k -maxrate 300k -bufsize 600k -c:a aac -b:a 48k out/light.mp4",
     [{"type": "bitrate_max", "bps": 400000}, V(width=640, height=480), A(), D(6, 0.2)],
     inputs=[vid(w=640, h=480, dur=6, busy=True)],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/light.mp4")])

# ---------------------------------------------------------------- L2: compound operation
task("l2_trim_scale_fps", 2, "cut",
     "in/input.mp4의 1초~6초 구간을 잘라서 160x120, 15fps로 만든 미리보기 영상을 out/preview.mp4로 저장해줘.",
     "out/preview.mp4", f"{F} -ss 1 -t 5 -i in/input.mp4 -vf scale=160:120,fps=15 {X} -c:a aac out/preview.mp4",
     [D(5), V(width=160, height=120, fps=15), A(), M()],
     variants=[f"{F} -ss 1 -t 5 -i in/input.mp4 -vf scale=160:120 -r 15 {X} -c:a aac out/preview.mp4"],
     alts=[("-r instead of fps filter", f"{F} -ss 1 -t 5 -i in/input.mp4 -vf scale=160:120 -r 15 {X} -crf 28 -c:a aac out/preview.mp4")],
     negatives=[("range 0-5s", f"{F} -ss 0 -t 5 -i in/input.mp4 -vf scale=160:120,fps=15 {X} -c:a aac out/preview.mp4")])

_AB = [vid("in/a.mp4", dur=8), vid("in/b.mp4", dur=5, pattern="smptebars", audio=[{"hz": 880}])]
task("l2_concat", 2, "join",
     "in/a.mp4 뒤에 in/b.mp4를 이어 붙여서 out/joined.mp4로 저장해줘. 영상과 소리 모두 이어져야 해.",
     "out/joined.mp4",
     f"{F} -i in/a.mp4 -i in/b.mp4 -filter_complex \"[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {X} out/joined.mp4",
     [D(13, 0.3), V(width=320, height=240), A(), {"type": "audio_tones", "present": [440, 880]}, M()],
     inputs=_AB,
     alts=[("concat demuxer", ["printf \"file 'in/a.mp4'\\nfile 'in/b.mp4'\\n\" > list.txt",
                               f"{F} -f concat -safe 0 -i list.txt -c copy out/joined.mp4"])],
     negatives=[("reversed order",
                 f"{F} -i in/b.mp4 -i in/a.mp4 -filter_complex \"[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {X} out/joined.mp4")])

task("l2_concat_mixed", 2, "join",
     "in/a.mp4와 in/b.mp4는 해상도와 프레임레이트가 서로 달라. a 다음에 b 순서로 이어 붙여서 640x360, 30fps 영상 out/joined.mp4를 만들어줘. "
     "화면 비율은 유지하고 남는 부분은 검은색으로 채워.",
     "out/joined.mp4",
     f"{F} -i in/a.mp4 -i in/b.mp4 -filter_complex \"[0:v]scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v0];"
     f"[1:v]scale=640:360,setsar=1,fps=30[v1];[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {X} out/joined.mp4",
     [D(8, 0.3), V(width=640, height=360, fps=30), A(), M()],
     inputs=[vid("in/a.mp4", dur=4), vid("in/b.mp4", w=640, h=360, fps=25, dur=4, audio=[{"hz": 880}])],
     negatives=[("a stretched to 16:9",
                 f"{F} -i in/a.mp4 -i in/b.mp4 -filter_complex \"[0:v]scale=640:360,setsar=1,fps=30[v0];[1:v]scale=640:360,setsar=1,fps=30[v1];"
                 f"[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {X} out/joined.mp4")])

_SUB = {"kind": "srt", "path": "in/sub.srt", "cues": [[1, 4, "Opening scene"], [5, 7, "Second line here"]]}
task("l2_burn_subs", 2, "subtitle",
     "in/input.mp4에 in/sub.srt 자막을 화면에 구워 넣어서(하드섭) out/subbed.mp4로 저장해줘.",
     "out/subbed.mp4", f"{F} -i in/input.mp4 -vf subtitles=in/sub.srt {X} -c:a copy out/subbed.mp4",
     [V(width=320, height=240), D(8), A(), M(), M(region="iw:ih*0.3:0:ih*0.7", min_psnr=31)],
     inputs=[vid(), _SUB],
     alts=[("lower quality encode", f"{F} -i in/input.mp4 -vf subtitles=in/sub.srt {X} -crf 28 -c:a copy out/subbed.mp4")],
     negatives=[("no subtitles", f"{F} -i in/input.mp4 {X} -c:a copy out/subbed.mp4"),
                ("soft subtitle track only", f"{F} -i in/input.mp4 -i in/sub.srt -c copy -c:s mov_text out/subbed.mp4")])

_LOGO = {"kind": "image", "path": "in/logo.png", "w": 64, "h": 32, "color": "red"}
task("l2_logo", 2, "overlay",
     "in/input.mp4 오른쪽 위 구석에 in/logo.png 로고를 얹어줘. 위와 오른쪽 가장자리에서 각각 10픽셀 띄우고, out/branded.mp4로 저장해.",
     "out/branded.mp4", f"{F} -i in/input.mp4 -i in/logo.png -filter_complex \"overlay=W-w-10:10\" {X} -c:a copy out/branded.mp4",
     [V(width=320, height=240), D(8), A(), M(min_psnr=30)],
     inputs=[vid(), _LOGO],
     negatives=[("top-left", f"{F} -i in/input.mp4 -i in/logo.png -filter_complex \"overlay=10:10\" {X} -c:a copy out/branded.mp4"),
                ("no logo", f"{F} -i in/input.mp4 {X} -c:a copy out/branded.mp4")])

_MUSIC = {"kind": "audio", "path": "in/music.wav", "hz": 880, "dur": 10}
task("l2_replace_audio", 2, "audio",
     "in/input.mp4의 원래 소리는 버리고 in/music.wav를 새 오디오로 넣어줘. 영상 길이에 맞춰 끊고 out/scored.mp4로 저장해.",
     "out/scored.mp4", f"{F} -i in/input.mp4 -i in/music.wav -map 0:v -map 1:a -c:v copy -c:a aac -shortest out/scored.mp4",
     [D(8, 0.3), V(codec="h264"), A(), {"type": "audio_tones", "present": [880], "absent": [440]}, M()],
     inputs=[vid(), _MUSIC],
     negatives=[("original audio kept", f"{F} -i in/input.mp4 -i in/music.wav -c copy -map 0 out/scored.mp4")])

task("l2_mix_bgm", 2, "audio",
     "in/input.mp4의 원래 소리는 유지하면서 in/music.wav를 배경음악으로 깔아줘. 배경음악 볼륨은 30%로 하고 길이는 영상에 맞춰. out/mixed.mp4로 저장해.",
     "out/mixed.mp4",
     f"{F} -i in/input.mp4 -i in/music.wav -filter_complex \"[1:a]volume=0.3[b];[0:a][b]amix=inputs=2:duration=first:normalize=0[a]\" -map 0:v -map \"[a]\" -c:v copy -c:a aac out/mixed.mp4",
     [D(8, 0.3), A(), {"type": "audio_tones", "present": [440, 880]}, M()],
     inputs=[vid(), _MUSIC],
     negatives=[("music replaced the original", f"{F} -i in/input.mp4 -i in/music.wav -map 0:v -map 1:a -c:v copy -c:a aac -shortest out/mixed.mp4")])

task("l2_loudnorm", 2, "audio",
     "in/input.mp4 소리가 너무 작아. 통합 라우드니스를 -16 LUFS로 맞춰서 out/normalized.mp4로 저장해줘. 영상은 그대로.",
     "out/normalized.mp4", f"{F} -i in/input.mp4 -c:v copy -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 48000 out/normalized.mp4",
     [{"type": "loudness", "lufs": -16, "tol": 1.5}, V(codec="h264"), D(8), M()],
     inputs=[vid(audio=[{"volume": 0.02}])],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/normalized.mp4")])

_AB8 = [vid("in/a.mp4"), vid("in/b.mp4", pattern="smptebars", audio=[{"hz": 880}])]
task("l2_side_by_side", 2, "layout",
     "in/a.mp4를 왼쪽, in/b.mp4를 오른쪽에 나란히 놓은 비교 영상 out/compare.mp4를 만들어줘. 소리는 a 것만 써.",
     "out/compare.mp4",
     f"{F} -i in/a.mp4 -i in/b.mp4 -filter_complex \"[0:v][1:v]hstack=inputs=2[v]\" -map \"[v]\" -map 0:a {X} -c:a copy out/compare.mp4",
     [V(width=640, height=240), D(8), A(), {"type": "audio_tones", "present": [440], "absent": [880]}, M()],
     inputs=_AB8,
     negatives=[("left/right swapped",
                 f"{F} -i in/a.mp4 -i in/b.mp4 -filter_complex \"[1:v][0:v]hstack=inputs=2[v]\" -map \"[v]\" -map 0:a {X} -c:a copy out/compare.mp4")])

task("l2_fades", 2, "effect",
     "in/input.mp4 시작 1초는 페이드 인, 마지막 1초는 페이드 아웃을 넣어줘. 영상과 소리 둘 다. out/faded.mp4로 저장해.",
     "out/faded.mp4",
     f"{F} -i in/input.mp4 -vf \"fade=t=in:st=0:d=1,fade=t=out:st=7:d=1\" -af \"afade=t=in:st=0:d=1,afade=t=out:st=7:d=1\" {X} out/faded.mp4",
     [D(8), V(), A(), M()],
     negatives=[("no fade", f"{F} -i in/input.mp4 {X} -c:a aac out/faded.mp4"),
                ("fade in only", f"{F} -i in/input.mp4 -vf fade=t=in:st=0:d=1 {X} -c:a aac out/faded.mp4")])

task("l2_segments", 2, "cut",
     "in/input.mp4(12초)를 4초씩 세 조각으로 나눠서 out/part_000.mp4, out/part_001.mp4, out/part_002.mp4로 저장해줘. 각 조각 길이가 정확해야 해.",
     "out/part_001.mp4",
     f"{F} -i in/input.mp4 {X} -force_key_frames \"expr:gte(t,n_forced*4)\" -c:a aac -f segment -segment_time 4 -segment_time_delta 0.02 -reset_timestamps 1 out/part_%03d.mp4",
     [{"type": "file_count", "glob": "out/part_*.mp4", "count": 3},
      D(4, 0.3, file="out/part_000.mp4"), D(4, 0.3), D(4, 0.3, file="out/part_002.mp4"), M()],
     inputs=[vid(dur=12)],
     alts=[("three explicit trims", [f"{F} -ss {s} -t 4 -i in/input.mp4 {X} -c:a aac out/part_00{i}.mp4" for i, s in enumerate((0, 4, 8))])],
     negatives=[("forced keyframes but no segment_time_delta: the 4.0 s boundary is missed",
                 f"{F} -i in/input.mp4 {X} -force_key_frames \"expr:gte(t,n_forced*4)\" -c:a aac -f segment -segment_time 4 -reset_timestamps 1 out/part_%03d.mp4"),
                ("segment with stream copy (single keyframe)",
                 f"{F} -i in/input.mp4 -c copy -f segment -segment_time 4 -reset_timestamps 1 out/part_%03d.mp4")])

task("l2_stills", 2, "still",
     "in/input.mp4의 1초, 3초, 5초 지점 화면을 각각 out/shot_1.png, out/shot_3.png, out/shot_5.png로 저장해줘.",
     "out/shot_3.png", [f"{F} -ss {t} -i in/input.mp4 -frames:v 1 out/shot_{t}.png" for t in (1, 3, 5)],
     [M(file="out/shot_1.png"), M(), M(file="out/shot_5.png")],
     negatives=[("all from 1s", [f"{F} -ss 1 -i in/input.mp4 -frames:v 1 out/shot_{t}.png" for t in (1, 3, 5)])])

task("l2_audio_clip", 2, "audio",
     "in/input.mp4의 2초~6초 구간 소리만 AAC 96kbps M4A 파일 out/clip.m4a로 뽑아줘.",
     "out/clip.m4a", f"{F} -ss 2 -t 4 -i in/input.mp4 -vn -c:a aac -b:a 96k out/clip.m4a",
     [V(count=0), A(codec="aac"), D(4, 0.2), {"type": "bitrate_max", "bps": 130000}],
     negatives=[("whole audio", f"{F} -i in/input.mp4 -vn -c:a aac -b:a 96k out/clip.m4a")])

task("l2_pip", 2, "layout",
     "in/a.mp4를 본 화면으로 두고, in/b.mp4를 80x60으로 줄여 오른쪽 아래 구석에 작은 화면으로 넣어줘. 아래와 오른쪽에서 10픽셀씩 띄워. "
     "소리는 a 것만 쓰고 out/pip.mp4로 저장해.",
     "out/pip.mp4",
     f"{F} -i in/a.mp4 -i in/b.mp4 -filter_complex \"[1:v]scale=80:60[p];[0:v][p]overlay=W-w-10:H-h-10[v]\" -map \"[v]\" -map 0:a {X} -c:a copy out/pip.mp4",
     [V(width=320, height=240), D(8), A(), M(min_psnr=30)],
     inputs=_AB8,
     negatives=[("top-left",
                 f"{F} -i in/a.mp4 -i in/b.mp4 -filter_complex \"[1:v]scale=80:60[p];[0:v][p]overlay=10:10[v]\" -map \"[v]\" -map 0:a {X} -c:a copy out/pip.mp4")])

task("l2_square", 2, "geometry",
     "SNS 썸네일 영상용으로 in/input.mp4 가운데를 정사각형으로 잘라 128x128로 줄이고 소리는 뺀 out/square.mp4를 만들어줘.",
     "out/square.mp4", f"{F} -i in/input.mp4 -vf crop=240:240,scale=128:128 -an {X} out/square.mp4",
     [V(width=128, height=128), A(count=0), D(8), M()],
     negatives=[("squashed instead of cropped", f"{F} -i in/input.mp4 -vf scale=128:128 -an {X} out/square.mp4")])

task("l2_vertical", 2, "geometry",
     "가로 영상 in/input.mp4(640x360)의 가운데를 세로 9:16 비율로 잘라 180x320 쇼츠용 영상 out/vertical.mp4로 만들어줘. 소리는 유지해.",
     "out/vertical.mp4", f"{F} -i in/input.mp4 -vf crop=202:360,scale=180:320 {X} -c:a copy out/vertical.mp4",
     [V(width=180, height=320), A(), D(6), M()],
     inputs=[vid(w=640, h=360, dur=6)],
     alts=[("crop width 204", f"{F} -i in/input.mp4 -vf \"crop=204:360,scale=180:320\" {X} -c:a copy out/vertical.mp4")],
     negatives=[("squashed", f"{F} -i in/input.mp4 -vf scale=180:320 {X} -c:a copy out/vertical.mp4")])

task("l2_slowmo", 2, "speed",
     "in/input.mp4를 0.5배속 슬로모션으로 만들어줘. 소리도 같이 느려져야 해. out/slow.mp4로 저장해.",
     "out/slow.mp4",
     f"{F} -i in/input.mp4 -filter_complex \"[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]\" -map \"[v]\" -map \"[a]\" {X} out/slow.mp4",
     [D(8, 0.3), V(), A(), M()],
     inputs=[vid(dur=4)],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/slow.mp4")])

task("l2_prores", 2, "transcode",
     "편집용 마스터가 필요해. in/input.mp4를 ProRes 422 HQ 영상과 16비트 PCM 오디오의 MOV 파일 out/master.mov로 변환해줘.",
     "out/master.mov", f"{F} -i in/input.mp4 -c:v prores_ks -profile:v 3 -c:a pcm_s16le out/master.mov",
     [{"type": "format", "contains": "mov"}, V(codec="prores", profile="HQ"), A(codec="pcm_s16le"), D(8), M()],
     negatives=[("h264 in mov", f"{F} -i in/input.mp4 -c copy out/master.mov")])

# ---------------------------------------------------------------- L3: spec, diagnosis, recovery
task("l3_delivery_spec", 3, "spec",
     "in/source.mkv를 아래 납품 규격에 맞는 out/delivery.mp4로 변환해줘.\n"
     "- 영상: H.264 High 프로파일, yuv420p, 해상도 유지, 25fps\n- 오디오: AAC, 48kHz, 스테레오\n- 컨테이너: MP4, 웹 스트리밍용 faststart",
     "out/delivery.mp4",
     f"{F} -i in/source.mkv -c:v libx264 -profile:v high -pix_fmt yuv420p -r 25 -c:a aac -ar 48000 -ac 2 -movflags +faststart out/delivery.mp4",
     [V(codec="h264", profile="High", pix_fmt="yuv420p", width=640, height=480, fps=25),
      A(codec="aac", channels=2, sample_rate=48000), {"type": "faststart"}, D(6, 0.2), M()],
     inputs=[vid("in/source.mkv", w=640, h=480, dur=6, pix_fmt="yuv444p", audio=[{"channels": 1, "sr": 22050}])],
     negatives=[("naive transcode keeps 4:4:4, mono, 30fps", f"{F} -i in/source.mkv out/delivery.mp4"),
                ("no faststart", f"{F} -i in/source.mkv -c:v libx264 -profile:v high -pix_fmt yuv420p -r 25 -c:a aac -ar 48000 -ac 2 out/delivery.mp4")])

task("l3_add_silence", 3, "recovery",
     "in/input.mp4를 플랫폼에 올리려는데 오디오 트랙이 없으면 거부된대. 영상은 그대로 두고 48kHz 스테레오 무음 오디오 트랙을 넣어서 out/with_audio.mp4로 저장해줘.",
     "out/with_audio.mp4",
     f"{F} -i in/input.mp4 -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 -c:v copy -c:a aac -shortest out/with_audio.mp4",
     [A(codec="aac", channels=2, sample_rate=48000), {"type": "mean_volume", "max_db": -70}, V(codec="h264"), D(8, 0.3), M()],
     inputs=[vid(audio=[])],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/with_audio.mp4")])

task("l3_vfr_to_cfr", 3, "recovery",
     "in/input.mp4를 편집 프로그램에 넣으면 싱크가 밀려. 프레임 간격이 일정하지 않은 것 같아. 30fps 고정 프레임레이트로 바꿔서 out/cfr.mp4로 저장해줘.",
     "out/cfr.mp4", f"{F} -i in/input.mp4 -vf fps=30 {X} -c:a copy out/cfr.mp4",
     [{"type": "cfr"}, V(fps=30), A(), D(8, 0.2), M()],
     inputs=[vid(vfr=True)],
     alts=[("-fps_mode cfr -r 30", f"{F} -i in/input.mp4 -fps_mode cfr -r 30 {X} -c:a copy out/cfr.mp4")],
     negatives=[("stream copy keeps VFR", f"{F} -i in/input.mp4 -c copy out/cfr.mp4")])

task("l3_bake_rotation", 3, "recovery",
     "휴대폰으로 찍은 in/input.mp4가 어떤 플레이어에서는 옆으로 누워서 나와. 회전 메타데이터에 의존하지 않도록 픽셀 자체를 바로 세운 out/upright.mp4를 만들어줘.",
     "out/upright.mp4", f"{F} -i in/input.mp4 {X} -c:a copy out/upright.mp4",
     [V(width=240, height=320, no_rotation=True), A(), D(8), M()],
     inputs=[vid(rotation=90)],
     negatives=[("stream copy keeps the flag", f"{F} -i in/input.mp4 -c copy out/upright.mp4"),
                ("rotated twice", f"{F} -i in/input.mp4 -vf transpose=1 {X} -c:a copy out/upright.mp4")])

task("l3_trim_silence", 3, "analysis",
     "in/input.mp4는 앞과 뒤에 소리가 없는 구간이 있어. 소리가 나는 구간만 남기도록 앞뒤를 잘라서 out/trimmed.mp4로 저장해줘. 영상도 같은 구간으로 잘려야 해.",
     "out/trimmed.mp4", f"{F} -i in/input.mp4 -ss 2 -t 4.5 {X} -c:a aac out/trimmed.mp4",
     [D(4.5, 0.3), V(), A(), M()],
     inputs=[vid(audio=[{"silence": [[0, 2], [6.5, 8]]}])],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/trimmed.mp4"),
                ("only the head trimmed", f"{F} -i in/input.mp4 -ss 2 {X} -c:a aac out/trimmed.mp4")])

task("l3_silence_report", 3, "analysis",
     "in/input.mp4에서 0.5초 이상 이어지는 무음 구간을 전부 찾아서 out/silence.json에 "
     "[{\"start\": 초, \"end\": 초}, ...] 형식의 JSON 배열로 저장해줘.",
     "out/silence.json",
     "ffmpeg -hide_banner -i in/input.mp4 -af silencedetect=noise=-40dB:d=0.5 -f null - 2>&1 | python3 -c \""
     "import sys,re,json; t=sys.stdin.read(); s=re.findall(r'silence_start: ([0-9.]+)',t); e=re.findall(r'silence_end: ([0-9.]+)',t); "
     "json.dump([{'start':float(a),'end':float(b)} for a,b in zip(s,e)], open('out/silence.json','w'))\"",
     [{"type": "json_intervals", "expected": [[2, 3.5], [5, 6]], "tol": 0.2}],
     inputs=[vid(audio=[{"silence": [[2, 3.5], [5, 6]]}])],
     negatives=[("empty list", "echo '[]' > out/silence.json")])

task("l3_cut_black", 3, "analysis",
     "in/input.mp4 중간에 화면이 까맣게 나오는 구간이 있어. 그 구간을 영상과 소리 모두에서 들어내고 앞뒤를 이어 붙여서 out/clean.mp4로 저장해줘.",
     "out/clean.mp4",
     f"{F} -i in/input.mp4 -filter_complex \"[0:v]select='not(between(t,4,5.99))',setpts=N/FRAME_RATE/TB[v];"
     f"[0:a]aselect='not(between(t,4,5.99))',asetpts=N/SR/TB[a]\" -map \"[v]\" -map \"[a]\" {X} out/clean.mp4",
     [D(8, 0.3), V(), A(), M()],
     inputs=[vid(dur=10, black=[4, 5.99])],
     alts=[("trim + concat",
            f"{F} -i in/input.mp4 -filter_complex \"[0:v]trim=0:4,setpts=PTS-STARTPTS[v0];[0:a]atrim=0:4,asetpts=PTS-STARTPTS[a0];"
            f"[0:v]trim=6:10,setpts=PTS-STARTPTS[v1];[0:a]atrim=6:10,asetpts=PTS-STARTPTS[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]\" "
            f"-map \"[v]\" -map \"[a]\" {X} out/clean.mp4")],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/clean.mp4"),
                ("wrong window (3-5s)",
                 f"{F} -i in/input.mp4 -filter_complex \"[0:v]select='not(between(t,3,4.99))',setpts=N/FRAME_RATE/TB[v];"
                 f"[0:a]aselect='not(between(t,3,4.99))',asetpts=N/SR/TB[a]\" -map \"[v]\" -map \"[a]\" {X} out/clean.mp4")])

task("l3_pick_language", 3, "streams",
     "in/input.mkv에는 오디오 트랙이 여러 개 들어 있어. 영상은 그대로 두고 영어 오디오 트랙 하나만 남긴 out/english.mkv를 만들어줘.",
     "out/english.mkv", f"{F} -i in/input.mkv -map 0:v -map 0:a:m:language:eng -c copy out/english.mkv",
     [A(count=1), {"type": "audio_tones", "present": [880], "absent": [440]}, V(codec="h264"), D(8), M()],
     inputs=[vid("in/input.mkv", audio=[{"hz": 440, "lang": "kor"}, {"hz": 880, "lang": "eng"}])],
     negatives=[("default stream selection keeps the first track", f"{F} -i in/input.mkv -c copy out/english.mkv"),
                ("both tracks kept", f"{F} -i in/input.mkv -map 0 -c copy out/english.mkv")])

task("l3_extract_subs", 3, "streams",
     "in/input.mkv 안에 들어 있는 자막 트랙을 SRT 파일 out/subs.srt로 꺼내줘.",
     "out/subs.srt", f"{F} -i in/input.mkv -map 0:s:0 out/subs.srt",
     [{"type": "text_contains", "substrings": ["Opening scene", "Second line here", "-->"]}],
     inputs=[vid("in/input.mkv", subs=_SUB["cues"])])

_ABC = [vid("in/a.mp4", dur=4), vid("in/b.mp4", dur=4, pattern="smptebars"), vid("in/c.mp4", dur=4, pattern="pal75bars")]
task("l3_batch", 3, "batch",
     "in/ 폴더에 있는 mp4 파일 전부를 160x120으로 줄여서 같은 파일 이름으로 out/ 폴더에 저장해줘.",
     "out/b.mp4",
     "for f in in/*.mp4; do ffmpeg -y -loglevel error -i \"$f\" -vf scale=160:120 -c:v libx264 -pix_fmt yuv420p -c:a copy \"out/$(basename \"$f\")\"; done",
     [{"type": "file_count", "glob": "out/*.mp4", "count": 3}]
     + [c for n in "abc" for c in (V(width=160, height=120, file=f"out/{n}.mp4"), M(file=f"out/{n}.mp4"))],
     inputs=_ABC,
     negatives=[("only the first file", f"{F} -i in/a.mp4 -vf scale=160:120 {X} -c:a copy out/a.mp4")])

task("l3_target_size", 3, "compress",
     "in/input.mp4를 메신저로 보내야 하는데 1MB(1,000,000바이트)를 넘으면 안 돼. 해상도와 길이, 소리는 유지하면서 그 안에 들어오게 out/small.mp4로 만들어줘.",
     "out/small.mp4", f"{F} -i in/input.mp4 {X} -b:v 900k -maxrate 1000k -bufsize 1000k -c:a aac -b:a 64k out/small.mp4",
     [{"type": "size_max", "bytes": 1000000}, V(width=640, height=480), A(), D(6, 0.2)],
     inputs=[vid(w=640, h=480, dur=6, busy=True)],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/small.mp4")])

task("l3_odd_dims", 3, "recovery",
     "in/input.mkv를 일반 플레이어에서 재생되는 H.264 yuv420p MP4로 바꿔줘. 가로는 320픽셀로 하고 세로는 비율에 맞춰. out/playable.mp4로 저장해.",
     "out/playable.mp4", f"{F} -i in/input.mkv -vf scale=320:-2 {X} -c:a aac out/playable.mp4",
     [V(codec="h264", pix_fmt="yuv420p", width=320, height=240), A(), D(8), M()],
     inputs=[vid("in/input.mkv", w=321, h=241, pix_fmt="yuv444p")],
     negatives=[("4:4:4 kept", f"{F} -i in/input.mkv -c:v libx264 -c:a aac out/playable.mp4")])


if __name__ == "__main__":
    out = Path(__file__).parent / "tasks" / "pilot_v0.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in TASKS), encoding="utf-8")
    levels = {lv: sum(t["level"] == lv for t in TASKS) for lv in (1, 2, 3)}
    print(f"{len(TASKS)} tasks -> {out}  by level: {levels}")
