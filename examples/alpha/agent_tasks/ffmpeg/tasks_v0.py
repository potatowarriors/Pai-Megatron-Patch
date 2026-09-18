"""Task set v0 (50 tasks) — rebuilt after the Gemma4-12B pilot baseline (88.5%, README "baseline 기록").

The pilot showed that single and compound edits do not discriminate. Failures came from three places,
so v0 is built on those axes (id prefix):
  d_  input diagnosis     the request hides a property the agent must find with ffprobe
  a_  analyse, then edit  detect something in the media, then cut/crop/report exactly
  c_  numeric constraint  a spec that must be met exactly (size, GOP, frame count, layout)
plus 10 pilot tasks kept as regression anchors (l1_/l2_) and the 7 pilot tasks Gemma4 failed.
`python tasks_v0.py` writes tasks/v0.jsonl.
"""
import copy
import json
from pathlib import Path

import tasks_pilot as P
from tasks_pilot import A, D, F, M, V, X, _LOGO, task, vid

_pilot = {t["id"]: t for t in P.TASKS}
_n = len(P.TASKS)  # task() appends to P.TASKS; everything past _n is ours and is moved out below

ANCHORS = ["l1_trim", "l1_scale", "l1_webm", "l1_asr_wav", "l1_rotate",
           "l2_concat", "l2_burn_subs", "l2_logo", "l2_loudnorm", "l2_prores"]
PILOT_HARD = ["l1_bitrate", "l2_segments", "l3_bake_rotation", "l3_cut_black", "l3_trim_silence", "l3_silence_report"]

T = {"type": "audio_tones"}


# ---------------------------------------------------------------- d_: input diagnosis
for deg, (w, h) in ((270, (240, 320)), (180, (320, 240))):
    task(f"d_rot{deg}", 3, "diagnosis",
         "in/input.mp4가 플레이어마다 방향이 다르게 나와. 회전 메타데이터에 의존하지 않도록 픽셀 자체를 올바른 방향으로 만든 "
         "out/upright.mp4를 만들어줘. 보통 플레이어에서 지금 보이는 방향이 맞는 방향이야.",
         "out/upright.mp4", f"{F} -i in/input.mp4 {X} -c:a copy out/upright.mp4",
         [V(width=w, height=h, no_rotation=True), A(), D(8), M()],
         inputs=[vid(rotation=deg)],
         negatives=[("stream copy keeps the flag", f"{F} -i in/input.mp4 -c copy out/upright.mp4"),
                    ("manual rotation on top of autorotate", f"{F} -i in/input.mp4 -vf transpose=1 {X} -c:a copy out/upright.mp4")])

task("d_rot_meta_only", 3, "diagnosis",
     "in/input.mp4를 재인코딩 없이, 회전 메타데이터만 바꿔서 플레이어에서 시계 방향으로 90도 돌아가 보이게 한 out/rotated.mp4를 만들어줘.",
     "out/rotated.mp4", f"{F} -display_rotation -90 -i in/input.mp4 -c copy out/rotated.mp4",
     [V(width=320, height=240, rotation=90), A(), D(8), M(min_psnr=50)],
     negatives=[("counter-clockwise", f"{F} -display_rotation 90 -i in/input.mp4 -c copy out/rotated.mp4"),
                ("re-encoded pixels", f"{F} -i in/input.mp4 -vf transpose=1 {X} -c:a copy out/rotated.mp4")])

task("d_10bit", 3, "diagnosis",
     "in/input.mp4가 구형 셋톱박스에서 재생이 안 돼. 그 기기는 H.264 High 프로파일까지만 지원해. 거기서 재생되는 out/stb.mp4로 변환해줘. 해상도와 길이는 그대로.",
     "out/stb.mp4", f"{F} -i in/input.mp4 -c:v libx264 -profile:v high -pix_fmt yuv420p -c:a copy out/stb.mp4",
     [V(codec="h264", profile="High", pix_fmt="yuv420p", width=320, height=240), A(), D(8), M()],
     inputs=[vid(vcodec="libx265", pix_fmt="yuv420p10le")],
     negatives=[("naive h264 transcode stays 10-bit (High 10)", f"{F} -i in/input.mp4 -c:v libx264 -c:a copy out/stb.mp4")])

task("d_sar", 3, "diagnosis",
     "in/input.mp4가 어떤 플레이어에서는 정상인데 어떤 플레이어에서는 홀쭉하게 나와. 어디서든 같은 비율로 보이도록 정사각 픽셀 영상 out/square_px.mp4로 바꿔줘. 세로 해상도는 유지해.",
     "out/square_px.mp4", f"{F} -i in/input.mp4 -vf scale=426:240,setsar=1 {X} -c:a copy out/square_px.mp4",
     [V(width_any=[426, 428], height=240, sar="1:1"), A(), D(8), M()],
     inputs=[vid(sar="4/3")],
     alts=[("expression + even rounding", f"{F} -i in/input.mp4 -vf \"scale=trunc(iw*sar/2)*2:ih,setsar=1\" {X} -c:a copy out/square_px.mp4")],
     negatives=[("SAR flag reset without rescaling", f"{F} -i in/input.mp4 -vf setsar=1 {X} -c:a copy out/square_px.mp4")])

task("d_long_audio", 3, "diagnosis",
     "in/input.mp4를 올리면 끝부분에서 화면이 한참 멈춰 있다가 끝나. 원인을 찾아서 화면이 실제로 움직이는 길이에 맞게 정리한 out/fixed.mp4를 만들어줘.",
     "out/fixed.mp4", f"{F} -i in/input.mp4 -t 6 -c:v copy -c:a aac out/fixed.mp4",
     [D(6, 0.2), V(), A(), M()],
     inputs=[vid(dur=6, audio=[{"dur": 10}])],
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/fixed.mp4")])

_KE = [{"hz": 440, "lang": "kor"}, {"hz": 880, "lang": "eng"}]
task("d_audio_reorder", 3, "diagnosis",
     "in/input.mkv에는 오디오 트랙이 두 개 있어. 영어 트랙이 첫 번째(기본)로, 한국어 트랙이 두 번째로 오도록 순서를 바꾼 out/reordered.mkv를 재인코딩 없이 만들어줘.",
     "out/reordered.mkv",
     f"{F} -i in/input.mkv -map 0:v -map 0:a:m:language:eng -map 0:a:m:language:kor -c copy -disposition:a:0 default -disposition:a:1 0 out/reordered.mkv",
     [A(count=2), {**T, "index": 0, "present": [880], "absent": [440]}, {**T, "index": 1, "present": [440], "absent": [880]},
      V(codec="h264"), D(8), M(min_psnr=50)],
     inputs=[vid("in/input.mkv", audio=_KE)],
     negatives=[("unchanged order", f"{F} -i in/input.mkv -map 0 -c copy out/reordered.mkv"),
                ("english only", f"{F} -i in/input.mkv -map 0:v -map 0:a:1 -c copy out/reordered.mkv")])

_CUES = [[1, 4, "Opening scene"], [5, 7, "Second line here"]]
_SUBR = "iw:ih*0.3:0:ih*0.7"
task("d_burn_embedded", 3, "diagnosis",
     "in/input.mkv 안에 들어 있는 자막을 화면에 구워 넣은(하드섭) out/subbed.mp4를 만들어줘. 자막 파일은 따로 없어.",
     "out/subbed.mp4", f"{F} -i in/input.mkv -vf subtitles=in/input.mkv {X} -c:a copy -sn out/subbed.mp4",
     [V(width=320, height=240), D(8), A(), M(), M(region=_SUBR, min_psnr=31)],
     inputs=[vid("in/input.mkv", subs=_CUES)],
     alts=[("extract to srt first", [f"{F} -i in/input.mkv -map 0:s:0 subs.srt",
                                     f"{F} -i in/input.mkv -vf subtitles=subs.srt {X} -c:a copy -sn out/subbed.mp4"])],
     negatives=[("soft subtitles carried over, nothing burned", f"{F} -i in/input.mkv {X} -c:a copy -c:s mov_text out/subbed.mp4")])

_KTRACK = [[1, 4, "K"], [5, 7, "K"]]
_ETRACK = [[1, 4, "ENGLISH SUBTITLE LINE NUMBER ONE"], [5, 7, "ENGLISH SUBTITLE LINE NUMBER TWO"]]
task("d_burn_eng_track", 3, "diagnosis",
     "in/input.mkv에는 자막 트랙이 여러 개 있어. 그중 영어 자막을 화면에 구워 넣은 out/subbed_en.mp4를 만들어줘.",
     "out/subbed_en.mp4", f"{F} -i in/input.mkv -vf subtitles=in/input.mkv:si=1 {X} -c:a copy -sn out/subbed_en.mp4",
     [V(width=320, height=240), D(8), A(), M(), M(region=_SUBR, min_psnr=31)],
     inputs=[vid("in/input.mkv", sub_tracks=[{"lang": "kor", "cues": _KTRACK}, {"lang": "eng", "cues": _ETRACK}])],
     negatives=[("first track (korean) burned", f"{F} -i in/input.mkv -vf subtitles=in/input.mkv {X} -c:a copy -sn out/subbed_en.mp4")])

_NORM = "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"
_ANORM = "aresample=48000,aformat=channel_layouts=stereo"
task("d_mixed_concat", 3, "diagnosis",
     "in/a.mp4, in/b.mp4, in/c.mp4를 이 순서로 이어 붙여서 out/joined.mp4를 만들어줘. 결과는 640x360, 30fps, 48kHz 스테레오여야 해. "
     "화면 비율은 유지하고 남는 부분은 검은색으로 채워. 소리가 없는 클립 구간은 무음으로 둬.",
     "out/joined.mp4",
     f"{F} -i in/a.mp4 -i in/b.mp4 -i in/c.mp4 -f lavfi -t 3 -i anullsrc=channel_layout=stereo:sample_rate=48000 -filter_complex "
     f"\"[0:v]{_NORM}[v0];[1:v]{_NORM}[v1];[2:v]{_NORM}[v2];[0:a]{_ANORM}[a0];[1:a]{_ANORM}[a1];[3:a]{_ANORM}[a2];"
     f"[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {X} out/joined.mp4",
     [D(11, 0.3), V(width=640, height=360, fps=30), A(channels=2, sample_rate=48000), {**T, "present": [440, 880]},
      {"type": "silence_intervals", "expected": [[8, 11]], "tol": 0.3}, M()],
     inputs=[vid("in/a.mp4", dur=4), vid("in/b.mp4", w=640, h=360, fps=25, dur=4, audio=[{"hz": 880, "channels": 1, "sr": 22050}]),
             vid("in/c.mp4", dur=3, pattern="smptebars", audio=[])],
     negatives=[("video only, no audio", f"{F} -i in/a.mp4 -i in/b.mp4 -i in/c.mp4 -filter_complex "
                 f"\"[0:v]{_NORM}[v0];[1:v]{_NORM}[v1];[2:v]{_NORM}[v2];[v0][v1][v2]concat=n=3:v=1:a=0[v]\" -map \"[v]\" {X} out/joined.mp4")])

_LR = [vid(audio=[{"hz": 440, "hz_right": 880}])]
task("d_left_channel", 3, "diagnosis",
     "in/input.mp4는 왼쪽과 오른쪽 채널에 서로 다른 소리가 녹음됐어. 왼쪽 채널 소리만 양쪽 스피커에서 똑같이 나오게 한 out/left_only.mp4를 만들어줘. 스테레오는 유지하고 영상은 건드리지 마.",
     "out/left_only.mp4", f"{F} -i in/input.mp4 -c:v copy -af \"pan=stereo|c0=c0|c1=c0\" out/left_only.mp4",
     [A(channels=2), {**T, "channel": 0, "present": [440], "absent": [880]}, {**T, "channel": 1, "present": [440], "absent": [880]},
      V(codec="h264"), D(8), M()],
     inputs=_LR,
     negatives=[("mono downmix mixes both", f"{F} -i in/input.mp4 -c:v copy -ac 1 out/left_only.mp4"),
                ("unchanged", f"{F} -i in/input.mp4 -c copy out/left_only.mp4")])

task("d_swap_channels", 3, "diagnosis",
     "in/input.mp4는 녹음할 때 좌우 채널이 뒤바뀌었어. 좌우를 맞바꾼 out/swapped.mp4를 만들어줘. 영상은 건드리지 마.",
     "out/swapped.mp4", f"{F} -i in/input.mp4 -c:v copy -af \"pan=stereo|c0=c1|c1=c0\" out/swapped.mp4",
     [A(channels=2), {**T, "channel": 0, "present": [880], "absent": [440]}, {**T, "channel": 1, "present": [440], "absent": [880]},
      V(codec="h264"), D(8), M()],
     inputs=_LR,
     negatives=[("unchanged", f"{F} -i in/input.mp4 -c copy out/swapped.mp4")])

# ---------------------------------------------------------------- a_: analyse, then edit
def _drop(windows):
    expr = "+".join(f"between(t,{a},{b})" for a, b in windows)
    return (f"[0:v]select='not({expr})',setpts=N/FRAME_RATE/TB[v];[0:a]aselect='not({expr})',asetpts=N/SR/TB[a]")


task("a_cut_two_blacks", 3, "analysis",
     "in/input.mp4에는 화면이 까맣게 나오는 구간이 여러 군데 있어. 그 구간들을 영상과 소리 모두에서 전부 들어내고 남은 부분을 이어 붙여서 out/clean.mp4로 저장해줘.",
     "out/clean.mp4", f"{F} -i in/input.mp4 -filter_complex \"{_drop([(2, 2.99), (7, 8.49)])}\" -map \"[v]\" -map \"[a]\" {X} out/clean.mp4",
     [D(9.5, 0.3), V(), A(), M(min_psnr=21)],
     inputs=[vid(dur=12, black=[[2, 2.99], [7, 8.49]])],
     negatives=[("only the first black window removed",
                 f"{F} -i in/input.mp4 -filter_complex \"{_drop([(2, 2.99)])}\" -map \"[v]\" -map \"[a]\" {X} out/clean.mp4")])

task("a_split_at_black", 3, "analysis",
     "in/input.mp4는 중간의 검은 화면을 경계로 두 장면이 붙어 있어. 검은 구간은 버리고 앞 장면을 out/part_1.mp4, 뒤 장면을 out/part_2.mp4로 나눠 저장해줘. 경계가 정확해야 해.",
     "out/part_2.mp4", [f"{F} -i in/input.mp4 -t 4 {X} -c:a aac out/part_1.mp4", f"{F} -ss 6 -i in/input.mp4 {X} -c:a aac out/part_2.mp4"],
     [D(4, 0.2, file="out/part_1.mp4"), D(4, 0.2), M(file="out/part_1.mp4"), M()],
     inputs=[vid(dur=10, black=[4, 5.99])],
     negatives=[("split in the middle, black kept", [f"{F} -i in/input.mp4 -t 5 {X} -c:a aac out/part_1.mp4",
                                                     f"{F} -ss 5 -i in/input.mp4 {X} -c:a aac out/part_2.mp4"])])

task("a_jumpcut", 3, "analysis",
     "in/input.mp4에서 0.5초 이상 이어지는 무음 구간을 전부 찾아서, 그 구간을 영상과 소리 모두에서 잘라내고 이어 붙인 점프컷 영상 out/jumpcut.mp4를 만들어줘.",
     "out/jumpcut.mp4", f"{F} -i in/input.mp4 -filter_complex \"{_drop([(2, 3.499), (5, 5.999)])}\" -map \"[v]\" -map \"[a]\" {X} out/jumpcut.mp4",
     [D(5.5, 0.3), {"type": "silence_intervals", "expected": []}, V(), A(), M(min_psnr=21)],
     inputs=[vid(audio=[{"silence": [[2, 3.5], [5, 6]]}])],
     negatives=[("audio-only silenceremove, picture untouched then trimmed",
                 f"{F} -i in/input.mp4 -af silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB -t 5.5 {X} out/jumpcut.mp4")])

task("a_black_report", 3, "analysis",
     "in/input.mp4에서 화면이 까만 구간을 전부 찾아서 out/black.json에 [{\"start\": 초, \"end\": 초}, ...] 형식의 JSON 배열로 저장해줘.",
     "out/black.json",
     "ffmpeg -hide_banner -i in/input.mp4 -vf blackdetect=d=0.3 -an -f null - 2>&1 | python3 -c \""
     "import sys,re,json; t=sys.stdin.read(); m=re.findall(r'black_start:([0-9.]+) black_end:([0-9.]+)',t); "
     "json.dump([{'start':float(a),'end':float(b)} for a,b in m], open('out/black.json','w'))\"",
     [{"type": "json_intervals", "expected": [[2, 3], [7, 8.5]], "tol": 0.2}],
     inputs=[vid(dur=12, black=[[2, 2.99], [7, 8.49]])],
     negatives=[("first window only", "echo '[{\"start\": 2.0, \"end\": 3.0}]' > out/black.json")])

for name, content, crop, (w, h) in (("letterbox", [320, 180], "320:180:0:30", (320, 180)),
                                    ("pillarbox", [180, 240], "180:240:70:0", (180, 240))):
    task(f"a_crop_{name}", 3, "analysis",
         "in/input.mp4는 화면에 검은 띠가 같이 인코딩돼 있어. 검은 띠만 정확히 잘라내고 실제 화면은 한 픽셀도 잃지 않은 out/cropped.mp4를 만들어줘.",
         "out/cropped.mp4", f"{F} -i in/input.mp4 -vf crop={crop} {X} -c:a copy out/cropped.mp4",
         [V(width=w, height=h), A(), D(8), M()],
         inputs=[vid(content=content)],
         negatives=[("unchanged", f"{F} -i in/input.mp4 {X} -c:a copy out/cropped.mp4")])

task("a_first_last", 3, "analysis",
     "in/input.mp4의 맨 앞 2초와 맨 끝 2초만 이어 붙인 4초짜리 미리보기 out/teaser.mp4를 만들어줘. 소리도 같이.",
     "out/teaser.mp4",
     f"{F} -i in/input.mp4 -filter_complex \"[0:v]trim=0:2,setpts=PTS-STARTPTS[v0];[0:a]atrim=0:2,asetpts=PTS-STARTPTS[a0];"
     f"[0:v]trim=7:9,setpts=PTS-STARTPTS[v1];[0:a]atrim=7:9,asetpts=PTS-STARTPTS[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]\" "
     f"-map \"[v]\" -map \"[a]\" {X} out/teaser.mp4",
     [D(4, 0.2), V(), A(), M()],
     inputs=[vid(dur=9)],
     negatives=[("assumed an 8 s clip (6-8 s tail)",
                 f"{F} -i in/input.mp4 -filter_complex \"[0:v]trim=0:2,setpts=PTS-STARTPTS[v0];[0:a]atrim=0:2,asetpts=PTS-STARTPTS[a0];"
                 f"[0:v]trim=6:8,setpts=PTS-STARTPTS[v1];[0:a]atrim=6:8,asetpts=PTS-STARTPTS[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]\" "
                 f"-map \"[v]\" -map \"[a]\" {X} out/teaser.mp4")])

task("a_mute_bleep", 3, "analysis",
     "in/input.mp4의 3초부터 4초까지 1초 동안만 소리를 완전히 없앤 out/bleeped.mp4를 만들어줘. 영상과 나머지 소리는 그대로 둬.",
     "out/bleeped.mp4", f"{F} -i in/input.mp4 -c:v copy -af \"volume=0:enable='between(t,3,4)'\" out/bleeped.mp4",
     [{"type": "silence_intervals", "expected": [[3, 4]], "tol": 0.15}, V(codec="h264"), A(), D(8), M()],
     negatives=[("whole track muted", f"{F} -i in/input.mp4 -c:v copy -af volume=0 out/bleeped.mp4"),
                ("unchanged", f"{F} -i in/input.mp4 -c copy out/bleeped.mp4")])

task("a_keyframes_report", 3, "analysis",
     "in/input.mp4의 키프레임(I-frame) 시각을 전부 찾아서 out/keyframes.json에 초 단위 숫자 배열(예: [0.0, 2.5])로 저장해줘.",
     "out/keyframes.json",
     "ffprobe -v error -select_streams v:0 -skip_frame nokey -show_entries frame=pts_time -of csv=p=0 in/input.mp4 | python3 -c \""
     "import sys,json; json.dump([float(x.strip().strip(',')) for x in sys.stdin if x.strip().strip(',')], open('out/keyframes.json','w'))\"",
     [{"type": "json_numbers", "expected": [0, 1.5, 3, 4.5, 6, 7.5], "tol": 0.05}],
     inputs=[vid(gop=45)],
     negatives=[("assumed one keyframe per second", "echo '[0,1,2,3,4,5,6,7]' > out/keyframes.json")])

_ABC = [vid("in/a.mp4", dur=4), vid("in/b.mp4", w=640, h=360, fps=25, dur=6, pattern="smptebars"),
        vid("in/c.mp4", w=160, h=120, fps=15, dur=5, audio=[])]
task("a_media_report", 3, "analysis",
     "in/ 폴더의 mp4 파일을 전부 조사해서 out/report.json에 JSON 배열로 저장해줘. 파일마다 "
     "{\"file\": 파일이름, \"width\": 가로, \"height\": 세로, \"fps\": 프레임레이트, \"duration\": 초, \"has_audio\": true/false} 형식이야.",
     "out/report.json",
     "python3 -c \"import json,subprocess,glob,os\nrows=[]\nfor f in sorted(glob.glob('in/*.mp4')):\n"
     " d=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',f],capture_output=True,text=True).stdout)\n"
     " v=[s for s in d['streams'] if s['codec_type']=='video'][0]; n,m=v['avg_frame_rate'].split('/')\n"
     " rows.append({'file':os.path.basename(f),'width':v['width'],'height':v['height'],'fps':int(n)/int(m),"
     "'duration':float(d['format']['duration']),'has_audio':any(s['codec_type']=='audio' for s in d['streams'])})\n"
     "json.dump(rows,open('out/report.json','w'))\"",
     [{"type": "json_records", "key": "file", "tol": 0.15, "expected": [
         {"file": "a.mp4", "width": 320, "height": 240, "fps": 30, "duration": 4, "has_audio": True},
         {"file": "b.mp4", "width": 640, "height": 360, "fps": 25, "duration": 6, "has_audio": True},
         {"file": "c.mp4", "width": 160, "height": 120, "fps": 15, "duration": 5, "has_audio": False}]}],
     inputs=_ABC,
     negatives=[("has_audio assumed true", "echo '[{\"file\":\"a.mp4\",\"width\":320,\"height\":240,\"fps\":30,\"duration\":4,\"has_audio\":true},"
                 "{\"file\":\"b.mp4\",\"width\":640,\"height\":360,\"fps\":25,\"duration\":6,\"has_audio\":true},"
                 "{\"file\":\"c.mp4\",\"width\":160,\"height\":120,\"fps\":15,\"duration\":5,\"has_audio\":true}]' > out/report.json")])

_FOUR = [vid("in/a.mp4", dur=4), vid("in/b.mp4", dur=4, fps=25, pattern="smptebars"),
         vid("in/c.mp4", dur=4, pattern="pal75bars"), vid("in/d.mp4", dur=4, audio=[{"hz": 880, "channels": 1}])]
_FIXB = "{F} -i in/b.mp4 {fps} {X} -c:a copy out/b.mp4"
_FIXREST = ["cp in/a.mp4 out/a.mp4", "cp in/c.mp4 out/c.mp4", f"{F} -i in/d.mp4 -c:v copy -ac 2 out/d.mp4"]
task("a_fix_nonconforming", 3, "analysis",
     "in/ 폴더의 mp4 파일들을 납품해야 해. 규격은 H.264, 320x240, 30fps, AAC 스테레오야. 규격에 맞는 파일은 손대지 말고 그대로 out/에 복사하고, "
     "규격에 안 맞는 파일만 맞게 고쳐서 같은 이름으로 out/에 저장해줘.",
     "out/b.mp4", [_FIXB.format(F=F, X=X, fps="-vf fps=30")] + _FIXREST,
     [{"type": "file_count", "glob": "out/*.mp4", "count": 4}]
     + [c for n in "abcd" for c in (V(codec="h264", width=320, height=240, fps=30, file=f"out/{n}.mp4"),
                                    A(codec="aac", channels=2, file=f"out/{n}.mp4"))]
     + [M(min_psnr=50, file="out/a.mp4"), M(min_psnr=50, file="out/c.mp4"), M(), M(file="out/d.mp4")],
     inputs=_FOUR,
     variants=[[_FIXB.format(F=F, X=X, fps="-r 30")] + _FIXREST],
     negatives=[("everything copied as is", ["cp in/*.mp4 out/"]),
                ("everything re-encoded, conforming files too",
                 ["for f in in/*.mp4; do ffmpeg -y -loglevel error -i $f -vf fps=30 -c:v libx264 -crf 32 -pix_fmt yuv420p -ac 2 out/$(basename $f); done"])])

# ---------------------------------------------------------------- c_: numeric constraint
_BUSY = [vid(w=640, h=480, dur=6, busy=True)]
task("c_size_300k", 3, "constraint",
     "in/input.mp4를 메일에 첨부해야 하는데 300,000바이트를 넘으면 안 돼. 해상도와 길이, 소리는 유지하면서 그 안에 들어오는 out/small.mp4를 만들어줘.",
     "out/small.mp4", f"{F} -i in/input.mp4 {X} -b:v 250k -maxrate 280k -bufsize 280k -c:a aac -b:a 32k out/small.mp4",
     [{"type": "size_max", "bytes": 300000}, V(width=640, height=480), A(), D(6, 0.2)],
     inputs=_BUSY,
     negatives=[("video bitrate alone set to the whole budget", f"{F} -i in/input.mp4 {X} -b:v 400k -c:a aac out/small.mp4")])

task("c_gop_spec", 3, "constraint",
     "in/input.mp4를 적응형 스트리밍용으로 인코딩해줘. H.264, 25fps, 키프레임은 정확히 2초마다 하나씩이어야 하고 장면 전환 때문에 키프레임이 추가로 생기면 안 돼. "
     "오디오는 AAC 48kHz. out/abr.mp4로 저장해.",
     "out/abr.mp4", f"{F} -i in/input.mp4 -r 25 {X} -g 50 -keyint_min 50 -sc_threshold 0 -c:a aac -ar 48000 out/abr.mp4",
     [{"type": "gop", "seconds": 2}, V(codec="h264", fps=25), A(codec="aac", sample_rate=48000), D(8, 0.2)],
     alts=[("force_key_frames expression", f"{F} -i in/input.mp4 -r 25 {X} -force_key_frames \"expr:gte(t,n_forced*2)\" -sc_threshold 0 -c:a aac -ar 48000 out/abr.mp4")],
     negatives=[("-g 50 but fps left at 30", f"{F} -i in/input.mp4 {X} -g 50 -c:a aac -ar 48000 out/abr.mp4"),
                ("default gop", f"{F} -i in/input.mp4 -r 25 {X} -c:a aac -ar 48000 out/abr.mp4")])

task("c_hls", 3, "constraint",
     "in/input.mp4를 HLS VOD로 패키징해줘. 재생목록은 out/hls/index.m3u8, 세그먼트는 같은 폴더에 .ts 파일로, 세그먼트 하나는 정확히 2초여야 해.",
     "out/hls/index.m3u8",
     ["mkdir -p out/hls",
      f"{F} -i in/input.mp4 {X} -g 60 -keyint_min 60 -sc_threshold 0 -c:a aac -f hls -hls_time 2 -hls_playlist_type vod "
      "-hls_segment_filename out/hls/seg_%03d.ts out/hls/index.m3u8"],
     [{"type": "text_contains", "substrings": ["#EXTM3U", "#EXTINF", "#EXT-X-ENDLIST"]},
      {"type": "file_count", "glob": "out/hls/*.ts", "count": 4}, D(8, 0.3)],
     negatives=[("default gop: segments cannot be cut at 2 s",
                 ["mkdir -p out/hls", f"{F} -i in/input.mp4 {X} -c:a aac -f hls -hls_time 2 -hls_playlist_type vod "
                  "-hls_segment_filename out/hls/seg_%03d.ts out/hls/index.m3u8"])])

task("c_frames_exact", 3, "constraint",
     "in/input.mp4에서 프레임 번호 60번부터 149번까지(둘 다 포함, 0번부터 셈) 정확히 90프레임만 뽑은 out/frames.mp4를 만들어줘. 소리는 빼.",
     "out/frames.mp4", f"{F} -i in/input.mp4 -vf \"select='between(n,60,149)',setpts=N/FRAME_RATE/TB\" -an {X} out/frames.mp4",
     [V(frames=90), A(count=0), M(exact=True)],
     alts=[("time based: 2 s .. 5 s", f"{F} -ss 2 -i in/input.mp4 -frames:v 90 -an {X} out/frames.mp4")],
     negatives=[("off by one: 61..150", f"{F} -i in/input.mp4 -vf \"select='between(n,61,150)',setpts=N/FRAME_RATE/TB\" -an {X} -crf 10 out/frames.mp4"),
                ("89 frames", f"{F} -i in/input.mp4 -vf \"select='between(n,60,148)',setpts=N/FRAME_RATE/TB\" -an {X} out/frames.mp4")])

task("c_pad_10s", 3, "constraint",
     "in/input.mp4를 편성 슬롯에 맞게 정확히 10초로 늘려줘. 모자란 시간은 마지막 화면을 정지시켜 채우고 그동안 소리는 무음이어야 해. out/slot.mp4로 저장해.",
     "out/slot.mp4", f"{F} -i in/input.mp4 -vf tpad=stop_mode=clone:stop_duration=2 -af apad=pad_dur=2 {X} -t 10 out/slot.mp4",
     [D(10, 0.15), {"type": "silence_intervals", "expected": [[8, 10]], "tol": 0.25}, V(), A(), M()],
     negatives=[("black padding instead of a frozen frame", f"{F} -i in/input.mp4 -vf tpad=stop_mode=add:stop_duration=2:color=black -af apad=pad_dur=2 {X} -t 10 out/slot.mp4"),
                ("slowed down to 10 s", f"{F} -i in/input.mp4 -vf setpts=1.25*PTS -af atempo=0.8 {X} out/slot.mp4")])

task("c_sprite", 3, "constraint",
     "in/input.mp4의 0초, 1초, 2초, …, 7초 지점 화면 8장을 각각 80x60으로 줄여서 가로 4칸 × 세로 2칸(왼쪽 위부터 시간순)으로 붙인 "
     "썸네일 시트 out/sprite.jpg를 만들어줘.",
     "out/sprite.jpg", f"{F} -i in/input.mp4 -vf \"select='not(mod(n,30))',scale=80:60,tile=4x2\" -frames:v 1 -update 1 out/sprite.jpg",
     [V(width=320, height=120), M(min_psnr=24)],
     negatives=[("2x4 layout", f"{F} -i in/input.mp4 -vf \"select='not(mod(n,30))',scale=80:60,tile=2x4\" -frames:v 1 -update 1 out/sprite.jpg"),
                ("first 8 consecutive frames", f"{F} -i in/input.mp4 -vf \"scale=80:60,tile=4x2\" -frames:v 1 -update 1 out/sprite.jpg")])

task("c_watermark_half", 3, "constraint",
     "in/input.mp4 왼쪽 아래 구석에 in/logo.png를 50% 불투명도로 얹어줘. 왼쪽과 아래에서 각각 10픽셀 띄우고 out/marked.mp4로 저장해.",
     "out/marked.mp4",
     f"{F} -i in/input.mp4 -i in/logo.png -filter_complex \"[1:v]format=rgba,colorchannelmixer=aa=0.5[l];[0:v][l]overlay=10:H-h-10\" {X} -c:a copy out/marked.mp4",
     [V(width=320, height=240), A(), D(8), M(), M(region="84:52:0:188", min_psnr=30)],
     inputs=[vid(), _LOGO],
     negatives=[("fully opaque", f"{F} -i in/input.mp4 -i in/logo.png -filter_complex \"overlay=10:H-h-10\" {X} -c:a copy out/marked.mp4"),
                ("no logo", f"{F} -i in/input.mp4 {X} -c:a copy out/marked.mp4")])

task("c_xfade", 3, "constraint",
     "in/a.mp4에서 in/b.mp4로 넘어갈 때 1초 동안 크로스페이드(영상은 페이드 전환, 소리는 크로스페이드)가 들어간 out/mix.mp4를 만들어줘. "
     "겹치는 1초만큼 전체 길이는 줄어들어야 해.",
     "out/mix.mp4",
     f"{F} -i in/a.mp4 -i in/b.mp4 -filter_complex \"[0:v][1:v]xfade=transition=fade:duration=1:offset=7[v];[0:a][1:a]acrossfade=d=1[a]\" "
     f"-map \"[v]\" -map \"[a]\" {X} out/mix.mp4",
     [D(12, 0.3), V(), A(), {**T, "present": [440, 880]}, M()],
     inputs=[vid("in/a.mp4", dur=8), vid("in/b.mp4", dur=5, pattern="smptebars", audio=[{"hz": 880}])],
     negatives=[("hard cut, no transition",
                 f"{F} -i in/a.mp4 -i in/b.mp4 -filter_complex \"[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {X} out/mix.mp4")])

task("c_level_main", 3, "constraint",
     "in/input.mp4를 저사양 기기용으로 H.264 Main 프로파일, 레벨 3.1에 맞춰 out/main31.mp4로 인코딩해줘. 오디오는 유지해.",
     "out/main31.mp4", f"{F} -i in/input.mp4 -c:v libx264 -profile:v main -level:v 3.1 -pix_fmt yuv420p -c:a copy out/main31.mp4",
     [V(codec="h264", profile="Main", level=31), A(), D(8), M()],
     negatives=[("profile only", f"{F} -i in/input.mp4 -c:v libx264 -profile:v main -pix_fmt yuv420p -c:a copy out/main31.mp4")])

task("c_delivery_strict", 3, "constraint",
     "in/source.mkv를 아래 납품 규격에 맞는 out/delivery.mp4로 변환해줘.\n"
     "- 영상: H.264 High 프로파일(8비트 4:2:0), 해상도 유지, 25fps, 키프레임은 정확히 2초 간격\n"
     "- 오디오: AAC 48kHz 스테레오, 통합 라우드니스 -23 LUFS (±1)\n- 컨테이너: MP4 faststart, 파일 전체 비트레이트 800kbps 이하",
     "out/delivery.mp4",
     f"{F} -i in/source.mkv -c:v libx264 -profile:v high -pix_fmt yuv420p -r 25 -g 50 -keyint_min 50 -sc_threshold 0 -b:v 600k -maxrate 650k -bufsize 1300k "
     "-af loudnorm=I=-23:TP=-2:LRA=7 -c:a aac -ar 48000 -ac 2 -b:a 96k -movflags +faststart out/delivery.mp4",
     [V(codec="h264", profile="High", pix_fmt="yuv420p", width=640, height=480, fps=25), {"type": "gop", "seconds": 2},
      A(codec="aac", channels=2, sample_rate=48000), {"type": "loudness", "lufs": -23, "tol": 1.0}, {"type": "faststart"},
      {"type": "bitrate_max", "bps": 800000}, D(6, 0.2)],
     inputs=[vid("in/source.mkv", w=640, h=480, dur=6, busy=True, vcodec="libx265", pix_fmt="yuv420p10le",
                 audio=[{"channels": 1, "sr": 22050, "volume": 0.05}])],
     negatives=[("everything but the GOP",
                 f"{F} -i in/source.mkv -c:v libx264 -profile:v high -pix_fmt yuv420p -r 25 -b:v 600k -maxrate 650k -bufsize 1300k "
                 "-af loudnorm=I=-23:TP=-2:LRA=7 -c:a aac -ar 48000 -ac 2 -b:a 96k -movflags +faststart out/delivery.mp4")])

# ---------------------------------------------------------------- assemble
NEW = P.TASKS[_n:]
del P.TASKS[_n:]

_vertical = copy.deepcopy(_pilot["l2_vertical"])
_vertical["id"] = "c_vertical"
# the pilot wording let "crop a 180x320 window" pass as a reading; say that the full height must survive
_vertical["instruction"] = ("가로 영상 in/input.mp4(640x360)를 쇼츠용 세로 영상으로 바꿔줘. 세로 전체(360픽셀)는 살린 채 가운데를 9:16 비율로 잘라내고, "
                            "그 결과를 180x320으로 줄여서 out/vertical.mp4로 저장해. 소리는 유지해.")
_vertical["negatives"].append({"why": "180x320 window cropped without scaling (pilot failure)",
                               "cmds": [f"{F} -i in/input.mp4 -vf crop=180:320 {X} -c:a copy out/vertical.mp4"]})

TASKS = [copy.deepcopy(_pilot[i]) for i in ANCHORS + PILOT_HARD] + [_vertical] + NEW

if __name__ == "__main__":
    out = Path(__file__).parent / "tasks" / "v0.jsonl"
    out.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in TASKS), encoding="utf-8")
    groups = {}
    for t in TASKS:
        groups[t["id"].split("_")[0]] = groups.get(t["id"].split("_")[0], 0) + 1
    print(f"{len(TASKS)} tasks -> {out}  by prefix: {groups}")
