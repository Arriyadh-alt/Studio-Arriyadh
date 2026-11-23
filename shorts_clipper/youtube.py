"""Private cookie copies and actionable YouTube access errors."""

import fcntl
import logging
import os
import re
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from urllib.parse import unquote, urlsplit


class YouTubeAccessError(RuntimeError):
    pass


def _sanitize_cookies(contents):
    kept = []
    for line in contents.splitlines(keepends=True):
        entry = line.removeprefix("#HttpOnly_")
        if not entry.strip() or entry.startswith("#"):
            kept.append(line)
            continue
        fields = entry.split("\t")
        if len(fields) == 7 and (not fields[4] or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", fields[4])):
            kept.append(line)
    return "".join(kept)


def blocking_message(error, cookies=None, proxy=None):
    message = str(error).casefold()
    if not ("youtube is blocking this server" in message or re.search(r"\b429\b", message) or any(text in message for text in (
            "too many requests", "not a bot", "bot check", "bot-check", "sign in", "sign-in"))):
        return None
    configured = bool(cookies or os.environ.get("CLIPPER_YTDLP_COOKIES") or os.environ.get("YOUTUBE_COOKIES"))
    detail = ("Cookies are configured but may be rejected or expired." if configured else
              "No cookies configured.")
    if proxy or os.environ.get("CLIPPER_YTDLP_PROXY"):
        detail += " The configured proxy may be blocked too."
    return "YouTube is blocking this server — add or refresh YouTube cookies (see README). " + detail


def _redact_proxy(message, proxy):
    sensitive = {proxy, repr(proxy)[1:-1], unquote(proxy)}
    try:
        parsed = urlsplit(proxy)
        sensitive.update(value for value in (parsed.username, parsed.password) if value)
    except ValueError:
        pass
    sensitive.update(unquote(value) for value in tuple(sensitive))
    message = str(message)
    for value in sorted(sensitive, key=len, reverse=True):
        message = message.replace(value, "[redacted]")
    return re.sub(r"(?i)\b(?:https?|socks[45]h?)://[^\s'\"<>]+", "[redacted URL]", message)


class _ProxyLogger:
    def __init__(self, proxy):
        self.proxy = proxy
        self.logger = logging.getLogger(__name__)

    def debug(self, message):
        self.logger.debug(_redact_proxy(message, self.proxy))

    def warning(self, message):
        self.logger.warning(_redact_proxy(message, self.proxy))

    def error(self, message):
        self.logger.error(_redact_proxy(message, self.proxy))


def _proxy_error(error, proxy):
    message = _redact_proxy(error, proxy)
    try:
        return type(error)(message)
    except Exception:
        return YouTubeAccessError(message)


@contextmanager
def _cookie_lock(source):
    try:
        descriptor = os.open(str(source) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError:
        raise YouTubeAccessError("Could not lock YouTube cookies. The cookie directory must be writable.") from None
    with os.fdopen(descriptor, "a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _save_cookies(source, private):
    temporary = None
    try:
        contents = _sanitize_cookies(private.read_text(encoding="utf-8"))
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=source.parent,
                                         prefix=f".{source.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, source)
    except (OSError, UnicodeError):
        raise YouTubeAccessError("Could not save refreshed YouTube cookies. The cookie directory must be writable.") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def cookie_options(cookies=None, proxy=None):
    proxy = proxy or os.environ.get("CLIPPER_YTDLP_PROXY")
    options = {"proxy": proxy, "logger": _ProxyLogger(proxy)} if proxy else {}
    source = cookies or os.environ.get("CLIPPER_YTDLP_COOKIES")
    contents = None if source else os.environ.get("YOUTUBE_COOKIES")
    if not source and not contents:
        try:
            yield options
        except Exception as error:
            if proxy:
                raise _proxy_error(error, proxy) from None
            raise
        return
    data_dir = None
    if contents:
        data_dir = Path(os.environ.get("CLIPPER_DATA_DIR", "~/clipper-data")).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source).expanduser().resolve() if source else None
    with (_cookie_lock(source) if source else nullcontext()), tempfile.TemporaryDirectory(
            prefix="clipper-cookies-", dir=data_dir) as temporary:
        private = Path(temporary) / "cookies.txt"
        private.touch(mode=0o600)
        try:
            if source:
                contents = source.read_text(encoding="utf-8")
            private.write_text(_sanitize_cookies(contents), encoding="utf-8")
        except (OSError, UnicodeError):
            raise YouTubeAccessError(
                "Could not prepare YouTube cookies. Use a readable Netscape cookies.txt via "
                "CLIPPER_YTDLP_COOKIES/--cookies or set YOUTUBE_COOKIES (see README).") from None
        blocked = False
        try:
            try:
                yield {**options, "cookiefile": str(private)}
            except BaseException as error:
                blocked = blocking_message(error, cookies, proxy) is not None
                raise
            finally:
                if source and not blocked:
                    _save_cookies(source, private)
        except Exception as error:
            if proxy:
                raise _proxy_error(error, proxy) from None
            raise
