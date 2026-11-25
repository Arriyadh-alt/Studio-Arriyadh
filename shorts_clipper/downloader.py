"""Download a section of a YouTube video with yt-dlp."""

import os
import math
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError, download_range_func

from .youtube import YouTubeAccessError, blocking_message, cookie_options


class _AnalysisSizeError(YouTubeAccessError):
    pass


def download_analysis_media(url: str, output_dir: Path, cookies: Path | None = None,
                            proxy: str | None = None, should_continue=None, *,
                            prior_audio: Path | None = None, on_audio=None, on_start=None) -> tuple[Path, Path | None]:
    """Download full-length compact audio and optional low-resolution video."""
    max_bytes = int(os.environ.get("CLIPPER_ANALYSIS_MAX_BYTES", "1500000000"))
    max_hours = float(os.environ.get("CLIPPER_ANALYSIS_MAX_HOURS", "12"))
    if max_bytes <= 0 or not math.isfinite(max_hours) or max_hours <= 0:
        raise ValueError("Analysis download limits must be positive and finite")

    def check_size(progress):
        if (progress.get("downloaded_bytes") or 0) > max_bytes:
            raise _AnalysisSizeError(f"AI analysis media exceeds the {max_bytes:,}-byte per-file limit.")

    proxy = proxy or os.environ.get("CLIPPER_YTDLP_PROXY")
    if proxy and not proxy.startswith("http://"):
        raise YouTubeAccessError("YouTube section downloads require an http:// proxy. "
                                 "Use an http:// proxy so ffmpeg cannot bypass the proxy setting.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [prior_audio] if prior_audio else []
    if prior_audio and on_audio is not None:
        on_audio(prior_audio)
    with cookie_options(cookies, proxy=proxy) as authentication:
        if on_start is not None:
            on_start()
        for kind, selector in (("audio", "ba[abr<=80]/ba/wa"),
                               ("video", "bv[height<=160]/wv")):
            if kind == "audio" and prior_audio:
                continue
            if should_continue is not None:
                should_continue()
            options = {"format": selector, "outtmpl": str(output_dir / f"%(id)s_analysis_{kind}.%(ext)s"),
                       "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True,
                       "max_filesize": max_bytes, "progress_hooks": [check_size],
                       "socket_timeout": 25, "retries": 2, "fragment_retries": 2}
            try:
                with yt_dlp.YoutubeDL({**options, **authentication}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if (info.get("duration") or 0) > max_hours * 3600:
                        raise YouTubeAccessError(f"AI analysis is limited to replays of {max_hours:g} hours or less.")
                    info = ydl.process_ie_result(info, download=True)
                    files = [Path(d["filepath"]) for d in info.get("requested_downloads", [])
                             if d.get("filepath")]
                if not files or not files[0].exists():
                    raise FileNotFoundError("Download finished but the media file was not found")
                paths.append(files[0])
                if kind == "audio" and on_audio is not None:
                    on_audio(files[0])
            except (_AnalysisSizeError, DownloadError, FileNotFoundError) as error:
                if kind == "audio" and isinstance(error, DownloadError):
                    message = blocking_message(error, cookies, proxy)
                    if message:
                        raise YouTubeAccessError(message) from None
                if kind == "audio":
                    raise
                paths.append(None)
    return paths[0], paths[1]


def download_section(url: str, output_dir: Path, start: float, end: float, max_height: int = 1080,
                     cookies: Path | None = None, proxy: str | None = None, *, analysis=False, on_start=None) -> Path:
    """Download a frame-accurate range, using compact formats for analysis.

    Raises ValueError if the range starts past the end of the video.
    """
    proxy = proxy or os.environ.get("CLIPPER_YTDLP_PROXY")
    if proxy and not proxy.startswith("http://"):
        raise YouTubeAccessError("YouTube section downloads require an http:// proxy. "
                                 "Use an http:// proxy so ffmpeg cannot bypass the proxy setting.")
    output_dir.mkdir(parents=True, exist_ok=True)
    window = f"{float(start)!r}-{float(end)!r}"
    options = {
        "format": (f"bv[height<={max_height}]+ba[abr<=96]/bv[height<={max_height}]+ba/"
                   f"b[height<={max_height}]/ba[abr<=96]/ba" if analysis
                   else f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b"),
        "merge_output_format": "mkv" if analysis else "mp4",
        "outtmpl": str(output_dir / f"%(id)s_{window}.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    options.update(socket_timeout=25, retries=2, fragment_retries=2,
                   external_downloader_args={"ffmpeg_i": ["-rw_timeout", "25000000"]})
    with cookie_options(cookies, proxy=proxy) as authentication:
        if on_start is not None:
            on_start()
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
