"""YouTube live-chat replay extraction."""

import json
from dataclasses import dataclass
from pathlib import Path

import yt_dlp


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


def validate_metadata(info):
    if not info.get("subtitles", {}).get("live_chat"):
        raise ValueError("This video has no live-chat replay (disabled, unavailable, or not a livestream VOD).")
    if (not info.get("duration") or info.get("is_live") or
            not (info.get("was_live") or info.get("live_status") == "was_live")):
        raise ValueError("A finished livestream VOD with a known duration is required.")


def fetch_metadata(url: str):
    with yt_dlp.YoutubeDL({"skip_download": True, "writesubtitles": False, "noplaylist": True,
                          "quiet": True, "noprogress": True, "socket_timeout": 10,
                          "retries": 1, "extractor_retries": 1}) as ydl:
        info = ydl.extract_info(url, download=False)
    validate_metadata(info)
    return info


def fetch_chat(url: str, directory: Path, info=None):
    info = info if info is not None else fetch_metadata(url)
    validate_metadata(info)
    directory.mkdir(parents=True, exist_ok=True)
    options = {"skip_download": True, "writesubtitles": True, "subtitleslangs": ["live_chat"],
               "outtmpl": str(directory / "replay.%(ext)s"), "noplaylist": True,
               "quiet": True, "noprogress": True, "retries": 3, "socket_timeout": 30}
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.process_ie_result(info, download=True)
    files = list(directory.glob("*.live_chat.json"))
    if not files:
        raise ValueError("Live-chat replay download did not produce a chat file.")
    with files[0].open(encoding="utf-8") as stream:
        messages = parse_chat(stream)
    if not messages:
        raise ValueError("Live-chat replay contains no timestamped messages.")
    return {key: info.get(key) for key in ("id", "title", "duration", "thumbnail")}, messages
