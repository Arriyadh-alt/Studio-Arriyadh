"""YouTube live-chat replay extraction."""

import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import yt_dlp
from yt_dlp.utils import DownloadCancelled, DownloadError

from shorts_clipper.youtube import blocking_message, cookie_options

from .chat_buckets import ChatBuckets, Message


METADATA_KEYS = ("id", "title", "duration", "thumbnail", "channel", "uploader")

REPLAY_LIMIT_MESSAGE = "This chat replay exceeded Studio’s processing limit. Try a shorter or quieter livestream replay."
REPLAY_DOWNLOAD_ERROR = "Studio couldn’t download the chat replay. Try again or choose another stream."
REPLAY_READ_ERROR = "Studio couldn’t read this chat replay. Try again or choose another stream."
DISK_RESERVE_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
logger = logging.getLogger(__name__)


class ChatReplayError(ValueError):
    pass


def _text(value):
    if "simpleText" in value:
        return value["simpleText"]
    return "".join(run.get("text", "") or next(iter(run.get("emoji", {}).get("shortcuts", [])), "")
                   or next(iter(run.get("emoji", {}).get("searchTerms", [])), "")
                   or run.get("emoji", {}).get("emojiId", "") for run in value.get("runs", []))


def iter_chat(lines):
    def visit(node, offset=None):
        if isinstance(node, list):
            for child in node:
                yield from visit(child, offset)
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
                    yield Message(offset, _text(child.get("authorName", {})), _text(message)), child.get("id")
                else:
                    yield from visit(child, offset)

    for line in lines:
        if line.strip():
            yield from visit(json.loads(line))


def parse_chat(lines):
    return sorted((message for message, _ in iter_chat(lines)), key=lambda message: message.offset_seconds)


def validate_metadata(info, require_chat=True, require_live=True):
    status = info.get("live_status")
    never_live = status == "not_live" or (status is None and not info.get("was_live") and not info.get("is_live"))
    if require_live and never_live:
        raise ChatReplayError("This is a regular YouTube video, with no live-chat replay. Paste a finished livestream link with chat replay.")
    if status in ("is_live", "is_upcoming", "post_live") or info.get("is_live"):
        raise ChatReplayError("This livestream replay isn’t ready yet. Try again after the stream ends and chat replay is available.")
    if require_chat and not info.get("subtitles", {}).get("live_chat"):
        raise ChatReplayError("This livestream has no live-chat replay. Choose a finished stream with chat replay enabled.")
    duration = info.get("duration")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ChatReplayError("YouTube hasn’t published the replay duration yet. Try again later.")


def fetch_metadata(url: str, require_chat=True, require_live=True):
    options = {"skip_download": True, "writesubtitles": False, "noplaylist": True,
               "quiet": True, "noprogress": True, "socket_timeout": 10,
               "retries": 1, "extractor_retries": 1}
    with cookie_options() as authentication, yt_dlp.YoutubeDL({**options, **authentication}) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except DownloadError as error:
            raise ChatReplayError(_video_error(str(error))) from error
    validate_metadata(info, require_chat=require_chat, require_live=require_live)
    return info


def _backoff(attempt):
    return min(8, 2 ** max(0, attempt - 1) + random.uniform(0, .5))


def _transient(error):
    message = str(error).casefold()
    return isinstance(error, (ConnectionError, TimeoutError)) or any(
        part in message for part in ("timed out", "timeout", "connection reset", "connection refused",
                                     "temporarily unavailable", "remote end closed", "incomplete read",
                                     "http error 500", "http error 502", "http error 503", "http error 504"))


def _ingest_attempt(info, work, writer, should_continue):
    path = work / "replay.live_chat.json"
    partial = path.with_name(path.name + ".part")
    position, downloaded = 0, 0
    start = time.monotonic()
    overlong = False
    guarded = set()

    def record_guard(guard):
        if guard not in guarded:
            guarded.add(guard)
            logger.warning("Chat replay guard=%s elapsed=%.2fs bytes=%d", guard, time.monotonic() - start, downloaded)

    def consume(replay, final=False):
        nonlocal position, overlong
        if not replay.is_file():
            return
        with replay.open("rb") as stream:
            stream.seek(position)
            while line := stream.readline(MAX_LINE_BYTES + 1):
                if not line.endswith(b"\n") and len(line) <= MAX_LINE_BYTES and not final:
                    break
                position = stream.tell()
                if overlong or len(line) > MAX_LINE_BYTES:
                    overlong = not line.endswith(b"\n")
                    writer.buckets.warning = REPLAY_READ_ERROR
                    record_guard("line_bytes")
                    continue
                try:
                    for message, message_id in iter_chat((line,)):
                        writer.add(message, message_id)
                except (UnicodeError, json.JSONDecodeError, RecursionError):
                    writer.buckets.warning = REPLAY_READ_ERROR
            if writer.limited:
                record_guard("interval_entries")

    def progress(state):
        nonlocal downloaded
        should_continue()
        downloaded = state.get("downloaded_bytes") or downloaded
        if state.get("status") == "finished" and path.is_file():
            consume(path, final=True)
        elif partial.is_file():
            consume(partial)
        if shutil.disk_usage(work).free < DISK_RESERVE_BYTES:
            record_guard("disk_reserve")
            raise DownloadCancelled("Chat replay stopped to preserve available disk space.")

    options = {"skip_download": True, "writesubtitles": True, "subtitleslangs": ["live_chat"],
               "outtmpl": str(work / "replay.%(ext)s"), "noplaylist": True,
               "quiet": True, "noprogress": True, "retries": 2, "fragment_retries": 2,
               "retry_sleep_functions": {"http": _backoff, "fragment": _backoff},
               "socket_timeout": 20, "progress_hooks": [progress]}
    try:
        progress({})
        with cookie_options(writeback=False) as authentication, yt_dlp.YoutubeDL({**options, **authentication}) as ydl:
            ydl.process_ie_result(info, download=True)
    finally:
        consume(path if path.is_file() else partial, final=path.is_file())
        writer.flush()
        if writer.limited:
            record_guard("interval_entries")
    if not path.is_file():
        raise ChatReplayError(REPLAY_DOWNLOAD_ERROR)


def fetch_chat(url: str, directory: Path, info=None, should_continue=None):
    info = info if info is not None else fetch_metadata(url)
    validate_metadata(info)
    directory.mkdir(parents=True, exist_ok=True)
    should_continue = should_continue or (lambda: None)
    with TemporaryDirectory(dir=directory, prefix="fetch-") as staging:
        buckets = ChatBuckets(Path(staging) / "buckets.sqlite", info["duration"])
        failure = None
        with buckets.writer() as writer:
            for attempt in range(1, 4):
                should_continue()
                try:
                    with TemporaryDirectory(dir=staging, prefix="attempt-") as temporary:
                        _ingest_attempt(info, Path(temporary), writer, should_continue)
                    failure = None
                    break
                except DownloadCancelled as error:
                    failure = str(error)
                    break
                except (DownloadError, ConnectionError, TimeoutError) as error:
                    failure = blocking_message(error) or REPLAY_DOWNLOAD_ERROR
                    if attempt == 3 or not _transient(error):
                        break
                    time.sleep(_backoff(attempt))
                except ChatReplayError as error:
                    failure = str(error)
                    break
        buckets.warning = failure or buckets.warning
        if not buckets:
            raise ChatReplayError(buckets.warning or "This chat replay has no timestamped messages. Choose a stream with an active chat.")
        target = directory / buckets.path.name
        buckets.path.replace(target)
        buckets.path = target
    return {key: info.get(key) for key in METADATA_KEYS}, buckets


def _video_error(message):
    message = message.casefold()
    if any(text in message for text in ("private video", "video is private", "members-only", "members only",
                                         "video is unavailable", "video unavailable", "this video is not available",
                                         "not available in your country", "confirm your age", "has been removed")):
        return "This YouTube video is private, restricted, or unavailable. Check the link and use a publicly accessible livestream replay."
    blocked = blocking_message(message)
    if blocked:
        return blocked
    return "Studio couldn’t load this YouTube video. Check the link and try again, or choose another public replay."
