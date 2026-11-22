"""Private cookie copies and actionable YouTube access errors."""

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path


class YouTubeAccessError(RuntimeError):
    pass


def _sanitize_cookies(contents):
    kept = []
    for line in contents.splitlines(keepends=True):
        if not line.strip() or (line.startswith("#") and not line.startswith("#HttpOnly_")):
            kept.append(line)
            continue
        fields = line.split("\t")
        if len(fields) == 7 and re.fullmatch(r"[0-9]+", fields[4]):
            kept.append(line)
    return "".join(kept)


def blocking_message(error, cookies=None):
    message = str(error).casefold()
    if not (re.search(r"\b429\b", message) or any(text in message for text in (
            "too many requests", "not a bot", "bot check", "bot-check", "sign in", "sign-in"))):
        return None
    configured = bool(cookies or os.environ.get("CLIPPER_YTDLP_COOKIES") or os.environ.get("YOUTUBE_COOKIES"))
    detail = ("Cookies are configured but may be rejected or expired." if configured else
              "No cookies configured.")
    return "YouTube is blocking this server — add or refresh YouTube cookies (see README). " + detail


@contextmanager
def cookie_options(cookies=None):
    source = cookies or os.environ.get("CLIPPER_YTDLP_COOKIES")
    contents = None if source else os.environ.get("YOUTUBE_COOKIES")
    if not source and not contents:
        yield {}
        return
    data_dir = None
    if contents:
        data_dir = Path(os.environ.get("CLIPPER_DATA_DIR", "~/clipper-data")).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clipper-cookies-", dir=data_dir) as temporary:
        private = Path(temporary) / "cookies.txt"
        private.touch(mode=0o600)
        try:
            if source:
                contents = Path(source).expanduser().read_text(encoding="utf-8")
            private.write_text(_sanitize_cookies(contents), encoding="utf-8")
        except OSError:
            raise YouTubeAccessError(
                "Could not prepare YouTube cookies. Use a readable Netscape cookies.txt via "
                "CLIPPER_YTDLP_COOKIES/--cookies or set YOUTUBE_COOKIES (see README).") from None
        yield {"cookiefile": str(private)}
