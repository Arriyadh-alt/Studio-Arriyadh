"""Cut a time range and convert it to 9:16 vertical video with ffmpeg."""

import shutil
import subprocess
from pathlib import Path

WIDTH, HEIGHT = 1080, 1920

FILTERS = {
    # Fill the frame by cropping the centre of the source.
    "crop": (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},setsar=1"
    ),
    # Keep the whole frame, centred over a blurred copy of itself.
    "blur": (
        f"split[bg][fg];"
        f"[bg]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},boxblur=20:5[bg];"
        f"[fg]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    ),
}


def build_ffmpeg_command(source: Path, output: Path, start: float, end: float, mode: str = "crop", captions: Path | None = None) -> list:
    if mode not in FILTERS:
        raise ValueError(f"Unknown mode {mode!r}; choose from {', '.join(FILTERS)}")
    if end <= start:
        raise ValueError("End time must be after start time")
    filters = FILTERS[mode]
    if captions is not None:
        path = str(captions.resolve())
        for char in ("\\", "'", ":", ",", "[", "]", ";", " "):
            path = path.replace(char, "\\" + char)
        # Escape once more for the filtergraph parser.
        path = path.replace("\\", "\\\\").replace("'", "\\'")
        filters += f",ass=filename={path}"
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}",
        "-filter_complex", filters,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output),
    ]


def make_vertical_clip(source: Path, output: Path, start: float, end: float, mode: str = "crop", captions: Path | None = None) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it first")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_ffmpeg_command(source, output, start, end, mode, captions), check=True)
    return output


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())
