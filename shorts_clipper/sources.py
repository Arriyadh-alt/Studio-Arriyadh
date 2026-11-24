"""Authorized local source files: uploaded once, then cut and analyzed like a YouTube source."""

import json
import re
import subprocess
from pathlib import Path

from .video import hdr_to_sdr

UPLOAD_SUFFIXES = (".mp4", ".mov", ".m4v", ".mkv", ".webm")
UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")


def source_track_kinds(path: Path) -> set[str]:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                             "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return {stream["codec_type"] for stream in json.loads(result.stdout).get("streams", [])}


def probe_source(path: Path) -> dict:
    """Duration and title for an uploaded file; raises ValueError for files without audio and video."""
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                             "-of", "json", str(path)], capture_output=True, text=True)
    if result.returncode:
        raise ValueError("This file is not a playable video")
    info = json.loads(result.stdout or "{}")
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    duration = float((info.get("format") or {}).get("duration") or 0)
    if "video" not in kinds or "audio" not in kinds or duration <= 0:
        raise ValueError("The file needs both a video and an audio track")
    return {"duration": duration}


def local_section(source: Path, output_dir: Path, start: float, end: float, *, professional=False) -> Path:
    """Cut `start`..`end` of a local file into an MP4, re-encoding so cuts are frame accurate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"section_{float(start)!r}-{float(end)!r}.mp4"
    tone_map = hdr_to_sdr(source) if professional else ""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}",
                    *(["-vf", tone_map] if tone_map else []),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", str(output)], check=True, capture_output=True)
    return output


def local_analysis_media(source: Path, output_dir: Path) -> tuple[Path, Path]:
    """Compact mono audio and a low-resolution copy for whole-file analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio, video = output_dir / "analysis_audio.m4a", output_dir / "analysis_video.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "aac", "-b:a", "64k", str(audio)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-an", "-vf", "scale=-2:160",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", str(video)], check=True, capture_output=True)
    return audio, video
