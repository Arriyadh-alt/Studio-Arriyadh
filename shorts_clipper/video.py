"""Cut a time range and convert it to 9:16 vertical video with ffmpeg."""

import json
import logging
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Literal

import cv2

from .facecam import FacecamOverlay
from .captions import FONT_DIR

WIDTH, HEIGHT = 1080, 1920
ExportPreset = Literal["preview", "standard", "full_hd"]
EXPORT_SIZES = {"preview": (360, 640), "standard": (720, 1280), "full_hd": (WIDTH, HEIGHT)}
EXPORT_SOURCE_HEIGHTS = {"preview": 720, "standard": 1440, "full_hd": 2160}
logger = logging.getLogger(__name__)

FILTERS = {
    # Default when face sampling finds no focal point.
    "crop": (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},setsar=1"
    ),
    # Keep the whole frame, centred over a blurred copy of itself.
    "blur": (
        f"split[bg][fg];"
        f"[bg]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},boxblur=20:5,eq=brightness=-0.12:saturation=0.35[bg];"
        f"[fg]crop=w='if(gt(iw/ih,2),trunc(ih*16/9/2)*2,iw)':h=ih,"
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    ),
}


def crop_left(source_width: float, crop_width: float, focal_x: float) -> float:
    return max(0, min(source_width - crop_width, source_width * focal_x - crop_width / 2))


def source_frame_rate(source: Path) -> str | None:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(source)],
                            check=True, capture_output=True, text=True)
    video = json.loads(result.stdout)["streams"][0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = Fraction(video.get(key, "0"))
            if rate > 0:
                return f"{rate.numerator}/{rate.denominator}"
        except (ValueError, ZeroDivisionError):
            continue
    return None


def professional_video_options(*, export_preset=None) -> list[str]:
    quality = (["-crf", "23" if export_preset == "preview" else "19"] if export_preset else
               ["-b:v", "16M", "-minrate", "16M", "-maxrate", "16M", "-bufsize", "16M", "-x264-params", "nal-hrd=cbr"])
    return [*quality, "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def hdr_to_sdr(source: Path) -> str:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                             "stream=color_transfer", "-of", "csv=p=0", str(source)],
                            check=True, capture_output=True, text=True)
    if result.stdout.strip() not in ("smpte2084", "arib-std-b67"):
        return ""
    return ("zscale=transfer=linear:npl=100,format=gbrpf32le,tonemap=mobius,"
            "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=limited,format=yuv420p")


def detect_focal_x(source: Path, start: float, end: float) -> float:
    if not source.is_file():
        return 0.5
    capture = cv2.VideoCapture(str(source))
    try:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        if not capture.isOpened() or detector.empty():
            return 0.5
        samples = []
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
            capture.set(cv2.CAP_PROP_POS_MSEC, (start + (end - start) * fraction) * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (640, max(1, round(height * 640 / width))))
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            faces = list(detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)))
            if not profile.empty():
                faces.extend(profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)))
                mirrored = profile.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
                faces.extend((640 - x - w, y, w, h) for x, y, w, h in mirrored)
            samples.append([((x + w / 2) / 640, (y + h / 2) / gray.shape[0], w * h) for x, y, w, h in faces])
        upper_faces = any(y < 0.5 for faces in samples for _, y, _ in faces)
        centers = []
        for faces in samples:
            candidates = [face for face in faces if not upper_faces or face[1] < 0.5]
            if candidates:
                centers.append(max(candidates, key=lambda face: face[2])[0])
        return float(median(centers)) if centers else 0.5
    except cv2.error:
        return 0.5
    finally:
        capture.release()


def source_size(source: Path) -> tuple[int, int] | None:
    capture = cv2.VideoCapture(str(source))
    try:
        size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return size if min(size) > 0 else None
    finally:
        capture.release()


def _hide_facecam(source: Path, box, margin: float = 0.01) -> str:
    """Paint the streamer's own cam out of the gameplay picture so it only shows once, in the overlay."""
    size = source_size(source)
    if size is None:
        return ""
    w, h = size
    x0, y0 = max(0, int((box.x - margin) * w)) // 2 * 2, max(0, int((box.y - margin) * h)) // 2 * 2
    cw = (min(w, math.ceil((box.x + box.w + margin) * w)) - x0) // 2 * 2
    ch = (min(h, math.ceil((box.y + box.h + margin) * h)) - y0) // 2 * 2
    if cw < 8 or ch < 8:
        return ""
    # A cam against the side of the frame is covered by the gameplay beside it, mirrored, so it reads as more game.
    if x0 == 0 and 2 * cw <= w:
        fill = f"crop={cw}:{ch}:{cw}:{y0},hflip"
    elif x0 + cw >= w - 2 and x0 >= cw:
        fill = f"crop={cw}:{ch}:{x0 - cw}:{y0},hflip"
    else:  # otherwise a smooth wash of its own colours: no streaks, and nothing of the face survives
        fill = f"crop={cw}:{ch}:{x0}:{y0},scale=4:4:flags=area,scale={cw}:{ch}:flags=bicubic"
    return f"split[hidemain][hidecam];[hidecam]{fill}[hidden];[hidemain][hidden]overlay={x0}:{y0},"


def build_ffmpeg_command(source: Path, output: Path, start: float, end: float, mode: str = "crop", captions: Path | None = None, facecam: FacecamOverlay | None = None, effects=None, *, export_fps=None, silent_audio=False, watermark=None, export_preset=None) -> list:
    if mode not in FILTERS:
        raise ValueError(f"Unknown mode {mode!r}; choose from {', '.join(FILTERS)}")
    if end <= start:
        raise ValueError("End time must be after start time")
    tone_map = hdr_to_sdr(source) if export_fps else ""
    filters = FILTERS[mode]
    split = (facecam is not None and facecam.rect.x == 0 and facecam.rect.y == 0
             and facecam.rect.w == WIDTH and mode == 'crop')
    main_h = HEIGHT - facecam.rect.h if split else HEIGHT
    if mode == 'crop':
        dynamic = effects is not None and effects.focal
        focal_x = 0.5 if facecam is not None or dynamic else detect_focal_x(source, start, end)
        focal = f"({effects.focal})" if dynamic else f"{focal_x:.6f}"
        filters = (
            f"scale={WIDTH}:{main_h}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{main_h}:x='max(0,min(iw-ow,iw*{focal}-ow/2))':"
            f"y=(ih-oh)/2,setsar=1"
        )
    if effects is not None and effects.zoom:
        # Zoom only the main picture so the facecam overlay stays put; aim the zoom at the tracked subject.
        y = (f"max(0,min(ih-ih/zoom,ih*({effects.focal_y})-ih/zoom/2))"
             if getattr(effects, "focal_y", None) else "ih/2-(ih/zoom/2)")
        filters += (f",zoompan=z='{effects.zoom}':x='iw/2-(iw/zoom/2)+({effects.shake or 0})':"
                    f"y='{y}':d=1:s={WIDTH}x{main_h}:fps={export_fps or 30},setsar=1")
    if effects is not None and effects.flash:
        filters += f",eq=brightness='{effects.flash}':eval=frame"
    if split:
        filters += f",pad={WIDTH}:{HEIGHT}:0:{facecam.rect.h}"
    if facecam is not None:
        box, rect = facecam.box, facecam.rect
        fit = (f"scale={rect.w}:{rect.h}:force_original_aspect_ratio=increase,crop={rect.w}:{rect.h},setsar=1"
               if split else
               f"scale={rect.w}:{rect.h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
               f"pad={rect.w}:{rect.h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
               f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.9:t=3")
        filters = (
            f"[0:v]{tone_map + ',' if tone_map else ''}split[mainin][camin];"
            f"[mainin]{_hide_facecam(source, box) if facecam.hide_source else ''}{filters}[main];"
            f"[camin]crop=w='max(2,trunc(iw*{box.w:.8f}/2)*2)':"
            f"h='max(2,trunc(ih*{box.h:.8f}/2)*2)':"
            f"x='iw*{box.x:.8f}':y='ih*{box.y:.8f}',"
            f"setsar=1,{fit}[cam];"
            f"[main][cam]overlay={rect.x}:{rect.y}"
        )
    elif tone_map:
        filters = f"{tone_map},{filters}"
    if captions is not None:
        path = str(captions.resolve())
        fonts = str(FONT_DIR.resolve())
        # Escape the option value, then the enclosing filtergraph.
        path = "".join("\\" + char if char in "\\': " else char for char in path)
        path = "".join("\\" + char if char in "\\'[],; " else char for char in path)
        fonts = "".join("\\" + char if char in "\\': " else char for char in fonts)
        fonts = "".join("\\" + char if char in "\\'[],; " else char for char in fonts)
        filters += f",ass=filename={path}:fontsdir={fonts}"
    watermark_input = []
    if watermark is not None:
        watermark_input = ["-i", str(watermark.path)]
        filters += f"[composed];[composed][1:v]overlay={watermark.x}:{watermark.y}:eof_action=repeat"
    if export_fps:
        width, height = EXPORT_SIZES[export_preset] if export_preset else ("iw", "ih")
        filters += f",scale={width}:{height}:out_color_matrix=bt709:out_range=tv,format=yuv420p"
    silent_input = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"] if silent_audio else []
    quality = [*professional_video_options(export_preset=export_preset), "-r", export_fps] if export_fps else ["-crf", "20"]
    audio = ["-b:a", "192k", "-ar", "48000"] if export_fps else ["-b:a", "128k"]
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), *watermark_input, *silent_input, "-t", f"{end - start:.3f}",
        *(["-filter_complex_threads", "2"] if export_preset else []), "-filter_complex", filters,
        "-c:v", "libx264", "-preset", "veryfast" if export_preset == "preview" else "medium",
        *quality, "-pix_fmt", "yuv420p", *(["-threads:v", "4"] if export_preset else []),
        "-c:a", "aac", *audio,
        "-movflags", "+faststart",
        str(output),
    ]


def hardware_export_command(command, preset):
    if Path("/dev/nvidiactl").exists():
        video = ["h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
    elif Path("/dev/dri").is_dir() and any(Path("/dev/dri").glob("renderD*")):
        video = ["h264_qsv", "-preset", "medium", "-global_quality", "19"]
    else:
        return None
    if preset == "preview":
        return None
    start, end = command.index("-c:v") + 1, command.index("-crf") + 2
    return [*command[:start], *video, *command[end:]]


def make_vertical_clip(source: Path, output: Path, start: float, end: float, mode: str = "crop", captions: Path | None = None, facecam: FacecamOverlay | None = None, effects=None, *, export_fps=None, silent_audio=False, watermark=None, export_preset=None, use_hardware=True) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it first")
    output.parent.mkdir(parents=True, exist_ok=True)
    if export_preset is not None:
        command = build_ffmpeg_command(source, output, start, end, mode, captions, facecam, effects,
                                       export_fps=export_fps or source_frame_rate(source) or "30/1",
                                       silent_audio=silent_audio, watermark=watermark, export_preset=export_preset)
        hardware = hardware_export_command(command, export_preset) if use_hardware else None
        if hardware:
            result = subprocess.run(hardware, capture_output=True)
            if result.returncode == 0:
                return output
            logger.info("Hardware encoder unavailable; retrying with libx264")
    elif watermark is not None:
        command = build_ffmpeg_command(source, output, start, end, mode, captions, facecam, effects,
                                       export_fps=export_fps, silent_audio=silent_audio, watermark=watermark)
    elif export_fps:
        command = build_ffmpeg_command(source, output, start, end, mode, captions, facecam, effects,
                                       export_fps=export_fps, silent_audio=silent_audio)
    elif effects is not None:
        command = build_ffmpeg_command(source, output, start, end, mode, captions, facecam, effects)
    else:
        command = build_ffmpeg_command(source, output, start, end, mode, captions) if facecam is None else build_ffmpeg_command(source, output, start, end, mode, captions, facecam)
    subprocess.run(command, check=True)
    return output


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())
