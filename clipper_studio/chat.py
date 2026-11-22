"""YouTube live-chat replay extraction."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import yt_dlp
from yt_dlp.utils import DownloadCancelled, DownloadError


MAX_REPLAY_SECONDS = 300
MAX_REPLAY_BYTES = 128 * 1024 * 1024
REPLAY_LIMIT_MESSAGE = "This chat replay exceeded Studio’s processing limit. Try a shorter or quieter livestream replay."
REPLAY_DOWNLOAD_ERROR = "Studio couldn’t download the chat replay. Try again or choose another stream."


class ChatReplayError(ValueError):
    pass


@dataclass(frozen=True)
class Message:
    offset_seconds: float
    author: str
    text: str


def _text(value):
    if "simpleText" in value:
        return value["simpleText"]
    return "".join(run.get("text", "") or next(iter(run.get("emoji", {}).get("shortcuts", [])), "")
                   or next(iter(run.get("emoji", {}).get("searchTerms", [])), "")
                   or run.get("emoji", {}).get("emojiId", "") for run in value.get("runs", []))


def parse_chat(lines):
    messages = []

    def visit(node, offset=None):
        if isinstance(node, list):
            for child in node:
                visit(child, offset)
        elif isinstance(node, dict):
            if "videoOffsetTimeMsec" in node:
                try:
                    offset = float(node["videoOffsetTimeMsec"]) / 1000
                except (ValueError, TypeError):
                    offset = None
            for key, child in node.items():
                if key in ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer",
                           "liveChatPaidStickerRenderer") and offset is not None and offset >= 0:
                    message = child.get("message", child.get("headerSubtext", {}))
                    messages.append(Message(offset, _text(child.get("authorName", {})), _text(message)))
                else:
                    visit(child, offset)

    for line in lines:
        if line.strip():
            visit(json.loads(line))
    return sorted(messages, key=lambda message: message.offset_seconds)


def fetch_chat(url: str, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory, prefix="fetch-") as temporary:
        work = Path(temporary)
        download_start = time.monotonic()

        def check_replay_limit(progress):
            if ((progress.get("downloaded_bytes") or 0) > MAX_REPLAY_BYTES or
                    time.monotonic() - download_start > MAX_REPLAY_SECONDS):
                raise DownloadCancelled(REPLAY_LIMIT_MESSAGE)

        options = {"skip_download": True, "writesubtitles": True, "subtitleslangs": ["live_chat"],
                   "outtmpl": str(work / "replay.%(ext)s"), "noplaylist": True,
                   "quiet": True, "noprogress": True, "retries": 2, "fragment_retries": 2,
                   "socket_timeout": 20, "progress_hooks": [check_replay_limit]}
        with yt_dlp.YoutubeDL(options) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
            except DownloadError as error:
                raise ChatReplayError(_video_error(str(error))) from error
            status = info.get("live_status")
            if status == "not_live":
                raise ChatReplayError("This is a regular YouTube video. Paste a finished livestream link with chat replay.")
            if status in ("is_live", "is_upcoming", "post_live") or info.get("is_live"):
                raise ChatReplayError("This livestream replay isn’t ready yet. Try again after the stream ends and chat replay is available.")
            if not info.get("subtitles", {}).get("live_chat"):
                raise ChatReplayError("This livestream has no live-chat replay. Choose a finished stream with chat replay enabled.")
            duration = info.get("duration")
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ChatReplayError("YouTube hasn’t published the replay duration yet. Try again later.")
            download_start = time.monotonic()
            try:
                ydl.process_ie_result(info, download=True)
            except DownloadCancelled as error:
                raise ChatReplayError(REPLAY_LIMIT_MESSAGE) from error
            except DownloadError as error:
                raise ChatReplayError(REPLAY_DOWNLOAD_ERROR) from error

        replay = work / "replay.live_chat.json"
        if not replay.is_file():
            raise ChatReplayError(REPLAY_DOWNLOAD_ERROR)
        if replay.stat().st_size > MAX_REPLAY_BYTES:
            raise ChatReplayError(REPLAY_LIMIT_MESSAGE)
        try:
            with replay.open(encoding="utf-8") as stream:
                messages = parse_chat(stream)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ChatReplayError("Studio couldn’t read this chat replay. Try again or choose another stream.") from error
        if not messages:
            raise ChatReplayError("This chat replay has no timestamped messages. Choose a stream with an active chat.")
        replay.replace(directory / replay.name)
    return {key: info.get(key) for key in ("id", "title", "duration", "thumbnail")}, messages


def _video_error(message):
    message = message.casefold()
    if any(text in message for text in ("private video", "video is private", "members-only", "members only",
                                         "video is unavailable", "video unavailable", "this video is not available",
                                         "not available in your country", "confirm your age", "has been removed")):
        return "This YouTube video is private, restricted, or unavailable. Check the link and use a publicly accessible livestream replay."
    return "Studio couldn’t load this YouTube video. Check the link and try again, or choose another public replay."
