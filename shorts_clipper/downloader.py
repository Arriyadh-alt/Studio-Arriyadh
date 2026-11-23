"""Download a section of a YouTube video with yt-dlp."""

import os
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError, download_range_func

from .youtube import YouTubeAccessError, blocking_message, cookie_options


def download_analysis_media(url: str, output_dir: Path, cookies: Path | None = None,
                            proxy: str | None = None) -> tuple[Path, Path | None]:
    """Download full-length compact audio and optional low-resolution video."""
    proxy = proxy or os.environ.get("CLIPPER_YTDLP_PROXY")
    if proxy and not proxy.startswith("http://"):
        raise YouTubeAccessError("YouTube section downloads require an http:// proxy. "
                                 "Use an http:// proxy so ffmpeg cannot bypass the proxy setting.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with cookie_options(cookies, proxy=proxy) as authentication:
        for kind, selector in (("audio", "ba[abr<=80]/ba/wa"),
                               ("video", "bv[height<=160]/wv")):
            options = {"format": selector, "outtmpl": str(output_dir / f"%(id)s_analysis_{kind}.%(ext)s"),
                       "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True}
            try:
                with yt_dlp.YoutubeDL({**options, **authentication}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    info = ydl.process_ie_result(info, download=True)
                    files = [Path(d["filepath"]) for d in info.get("requested_downloads", [])
                             if d.get("filepath")]
                if not files or not files[0].exists():
                    raise FileNotFoundError("Download finished but the media file was not found")
                paths.append(files[0])
            except (DownloadError, FileNotFoundError) as error:
                if isinstance(error, DownloadError):
                    message = blocking_message(error, cookies, proxy)
                    if message:
                        raise YouTubeAccessError(message) from None
                if kind == "audio":
                    raise
                paths.append(None)
    return paths[0], paths[1]


def download_section(url: str, output_dir: Path, start: float, end: float, max_height: int = 1080,
                     cookies: Path | None = None, proxy: str | None = None) -> Path:
    """Download only `start`..`end` (seconds) of `url` as an MP4 and return its path.

    Raises ValueError if the range starts past the end of the video.
    """
    proxy = proxy or os.environ.get("CLIPPER_YTDLP_PROXY")
    if proxy and not proxy.startswith("http://"):
        raise YouTubeAccessError("YouTube section downloads require an http:// proxy. "
                                 "Use an http:// proxy so ffmpeg cannot bypass the proxy setting.")
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
    with cookie_options(cookies, proxy=proxy) as authentication:
        try:
            with yt_dlp.YoutubeDL({**options, **authentication}) as ydl:
                info = ydl.extract_info(url, download=False)
                duration = info.get("duration")
                if duration and start >= duration:
                    raise ValueError(f"Start time is past the end of the video ({duration:.0f}s long)")
                info = ydl.process_ie_result(info, download=True)
                files = [Path(d["filepath"]) for d in info.get("requested_downloads", []) if d.get("filepath")]
        except DownloadError as error:
            message = blocking_message(error, cookies, proxy)
            if message:
                raise YouTubeAccessError(message) from None
            raise
    if not files or not files[0].exists():
        raise FileNotFoundError("Download finished but the video file was not found")
    return files[0]
