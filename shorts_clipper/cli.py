"""Command-line entry point."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .captions import CaptionStyle, FONTS, to_ass, to_srt, transcribe
from .downloader import download_section
from .timeutils import format_timestamp, parse_timestamp
from .video import FILTERS, make_vertical_clip, video_duration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shorts-clipper",
        description="Cut a YouTube video into a 9:16 vertical clip for Shorts / TikTok.",
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("start", help="start time: SS, MM:SS or HH:MM:SS")
    parser.add_argument("end", help="end time: SS, MM:SS or HH:MM:SS")
    parser.add_argument("-o", "--output", type=Path, default=Path("short.mp4"), help="output file (default: short.mp4)")
    parser.add_argument("-m", "--mode", choices=sorted(FILTERS), default="crop",
                        help="crop: fill the frame from the centre (default); blur: fit the frame over a blurred background")
    parser.add_argument("--keep-download", type=Path, metavar="DIR",
                        help="keep the downloaded section in DIR instead of a temp folder")
    parser.add_argument("--cookies", type=Path, metavar="FILE",
                        help="Netscape cookies.txt (overrides CLIPPER_YTDLP_COOKIES; source is never rewritten)")
    parser.add_argument("--no-captions", action="store_true", help="disable local transcription and captions")
    parser.add_argument("--model", default=os.environ.get("CLIPPER_WHISPER_MODEL", "base"), help="faster-whisper model name or local model directory (default: CLIPPER_WHISPER_MODEL or base)")
    parser.add_argument("--language", default=None, help="speech language code, e.g. en (default: auto-detect)")
    parser.add_argument("--caption-preset", choices=("pop", "classic", "minimal"), default="pop")
    parser.add_argument("--caption-position", choices=("bottom", "middle", "top"), default="bottom")
    parser.add_argument("--caption-size", type=int, default=80, help="font size at 1080x1920 (clamped to 40–130)")
    parser.add_argument("--caption-font", choices=FONTS, default=None, help="font family (default: Montserrat for pop, DejaVu Sans otherwise)")
    parser.add_argument("--srt", action="store_true", help="also save subtitles next to the output (requires captions)")
    parser.add_argument("--local", action="store_true", help="treat URL as a local video path instead")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        start, end = parse_timestamp(args.start), parse_timestamp(args.end)
        if end <= start:
            raise ValueError("End time must be after start time")
        if args.srt and args.no_captions:
            raise ValueError("--srt cannot be combined with --no-captions")

        with tempfile.TemporaryDirectory() as tmp:
            if args.local:
                section = Path(tmp) / "section.mp4"
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", args.url,
                                "-t", str(end - start), "-c:v", "libx264", "-c:a", "aac", str(section)], check=True)
            else:
                print(f"Downloading {format_timestamp(start)} -> {format_timestamp(end)} of {args.url} ...", flush=True)
                section = download_section(args.url, args.keep_download or Path(tmp), start, end,
                                           cookies=args.cookies)

            subtitles = None
            if not args.no_captions:
                print(f"Transcribing locally ({args.model}) ...", flush=True)
                captions = transcribe(section, args.model, args.language, clip_end=video_duration(section))
                if not captions:
                    print("No speech detected; saving video without captions.", flush=True)
                subtitles = Path(tmp) / "captions.ass"
                style = CaptionStyle.from_dict({"preset": args.caption_preset, "position": args.caption_position,
                                                "size": args.caption_size, "font": args.caption_font})
                subtitles.write_text(to_ass(captions, style), encoding="utf-8")
                if args.srt:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.with_suffix(".srt").write_text(to_srt(captions), encoding="utf-8")

            print(f"Converting to 9:16 ({args.mode} mode) ...", flush=True)
            make_vertical_clip(section, args.output, 0, video_duration(section), args.mode, subtitles)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"Error: ffmpeg failed (exit code {error.returncode})", file=sys.stderr)
        return 1
    except Exception as error:  # e.g. yt-dlp's DownloadError for a bad URL
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved {args.output}")
    return 0
