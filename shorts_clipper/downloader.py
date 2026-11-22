"""Download a section of a YouTube video with yt-dlp."""

from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func


def download_section(url: str, output_dir: Path, start: float, end: float, max_height: int = 1080) -> Path:
    """Download only `start`..`end` (seconds) of `url` as an MP4 and return its path.

    Raises ValueError if the range starts past the end of the video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    window = f"{float(start)!r}-{float(end)!r}"
    options = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / f"%(id)s_{window}.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get("duration")
        if duration and start >= duration:
            raise ValueError(f"Start time is past the end of the video ({duration:.0f}s long)")
        info = ydl.process_ie_result(info, download=True)
        files = [Path(d["filepath"]) for d in info.get("requested_downloads", []) if d.get("filepath")]
    if not files or not files[0].exists():
        raise FileNotFoundError("Download finished but the video file was not found")
    return files[0]
