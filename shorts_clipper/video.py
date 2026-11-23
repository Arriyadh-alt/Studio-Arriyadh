"""Cut a time range and convert it to 9:16 vertical video with ffmpeg."""

import math
import shutil
import subprocess
from pathlib import Path
from statistics import median

import cv2

from .facecam import FacecamOverlay
from .captions import FONT_DIR

WIDTH, HEIGHT = 1080, 1920

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


def _source_size(source: Path) -> tuple[int, int] | None:
    capture = cv2.VideoCapture(str(source))
    try:
        size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return size if min(size) > 0 else None
    finally:
        capture.release()


def _hide_facecam(source: Path, box, margin: float = 0.01) -> str:
    """Paint the streamer's own cam out of the gameplay picture so it only shows once, in the overlay."""
    size = _source_size(source)
    if size is None:
        return ""
    w, h = size
    x0, y0 = max(1, int((box.x - margin) * w)), max(1, int((box.y - margin) * h))
    x1 = min(w - 1, math.ceil((box.x + box.w + margin) * w))
    y1 = min(h - 1, math.ceil((box.y + box.h + margin) * h))
    return f"delogo=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}," if x1 - x0 >= 4 and y1 - y0 >= 4 else ""


def build_ffmpeg_command(source: Path, output: Path, start: float, end: float, mode: str = "crop", captions: Path | None = None, facecam: FacecamOverlay | None = None, effects=None) -> list:
    if mode not in FILTERS:
        raise ValueError(f"Unknown mode {mode!r}; choose from {', '.join(FILTERS)}")
    if end <= start:
        raise ValueError("End time must be after start time")
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
                    f"y='{y}':d=1:s={WIDTH}x{main_h}:fps=30,setsar=1")
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
            f"[0:v]split[mainin][camin];[mainin]{_hide_facecam(source, box) if facecam.hide_source else ''}{filters}[main];"
            f"[camin]crop=w='max(2,trunc(iw*{box.w:.8f}/2)*2)':"
            f"h='max(2,trunc(ih*{box.h:.8f}/2)*2)':"
            f"x='iw*{box.x:.8f}':y='ih*{box.y:.8f}',"
            f"setsar=1,{fit}[cam];"
            f"[main][cam]overlay={rect.x}:{rect.y}"
        )
    if captions is not None:
        path = str(captions.resolve())
        fonts = str(FONT_DIR.resolve())
        # Escape the option value, then the enclosing filtergraph.
        path = "".join("\\" + char if char in "\\': " else char for char in path)
        path = "".join("\\" + char if char in "\\'[],; " else char for char in path)
        fonts = "".join("\\" + char if char in "\\': " else char for char in fonts)
        fonts = "".join("\\" + char if char in "\\'[],; " else char for char in fonts)
        filters += f",ass=filename={path}:fontsdir={fonts}"
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}",
        "-filter_complex", filters,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output),
    ]


def make_vertical_clip(source: Path, output: Path, start: float, end: float, mode: str = "crop", captions: Path | None = None, facecam: FacecamOverlay | None = None, effects=None) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it first")
    output.parent.mkdir(parents=True, exist_ok=True)
    if effects is not None:
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
