import logging
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from stat import S_IMODE

import pytest
from yt_dlp.cookies import YoutubeDLCookieJar
from yt_dlp.utils import DownloadError

from clipper_studio.chat import ChatReplayError, fetch_chat, fetch_metadata
from shorts_clipper.downloader import download_section
from shorts_clipper import youtube
from shorts_clipper.youtube import YouTubeAccessError, blocking_message, cookie_options


COOKIE_TEXT = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\ttest\tfake\n"
REFRESHED_COOKIES = COOKIE_TEXT.replace("\tfake\n", "\trefreshed\n")
PROXY = "http://proxy-user:proxy-password@proxy.invalid:8080"


@pytest.fixture(autouse=True)
def cookie_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("CLIPPER_YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("YOUTUBE_COOKIES", raising=False)
    monkeypatch.delenv("CLIPPER_YTDLP_PROXY", raising=False)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))


@pytest.mark.parametrize("configuration", ["none", "path", "text"])
@pytest.mark.parametrize("operation,proxy", [
    (operation, proxy) for operation in ("metadata", "chat", "section")
    for proxy in (None, PROXY, "socks5://proxy-user:proxy-password@proxy.invalid:1080")
    if operation != "section" or proxy is None or proxy.startswith("http://")])
def test_cookie_options_reach_all_download_stages(monkeypatch, tmp_path, operation, configuration, proxy):
    source = tmp_path / "source.txt"
    source.write_text(COOKIE_TEXT)
    source.chmod(0o400)
    if proxy:
        monkeypatch.setenv("CLIPPER_YTDLP_PROXY", proxy)
    if configuration == "path":
        monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(source))
    elif configuration == "text":
        monkeypatch.setenv("YOUTUBE_COOKIES", COOKIE_TEXT)
    seen = []

    class Downloader:
        def __init__(self, options):
            self.options = options
        def __enter__(self):
            if proxy:
                assert self.options["proxy"] == proxy
                assert "logger" in self.options
            else:
                assert "proxy" not in self.options and "logger" not in self.options
            if configuration == "none":
                assert "cookiefile" not in self.options
            else:
                private = Path(self.options["cookiefile"])
                seen.append(private)
                expected = source.read_text() if configuration == "path" else COOKIE_TEXT
                assert private != source and private.read_text() == expected
                assert S_IMODE(private.stat().st_mode) == 0o600
                assert S_IMODE(private.parent.stat().st_mode) == 0o700
                if configuration == "text":
                    assert tmp_path / "data" in private.parents
            return self
        def __exit__(self, *args):
            if configuration != "none":
                Path(self.options["cookiefile"]).write_text(REFRESHED_COOKIES)
        def extract_info(self, url, download):
            assert not download
            if operation != "section":
                assert not self.options["writesubtitles"]
            return {"duration": 100, "live_status": "was_live", "subtitles": {"live_chat": [{}]}}
        def process_ie_result(self, info, download):
            assert download
            if configuration != "none":
                expected = source.read_text() if configuration == "path" else COOKIE_TEXT
                assert Path(self.options["cookiefile"]).read_text() == expected
            if operation == "chat":
                assert self.options["writesubtitles"]
                output = Path(self.options["outtmpl"]).parent / "replay.live_chat.json"
                output.write_bytes((Path(__file__).parent / "fixtures/chat.jsonl").read_bytes())
                return info
            output = tmp_path / "output" / "section.mp4"
            output.write_bytes(b"video")
            return {"requested_downloads": [{"filepath": str(output)}]}

    monkeypatch.setattr("yt_dlp.YoutubeDL", Downloader)
    if operation == "metadata":
        assert fetch_metadata("https://youtu.be/test")["duration"] == 100
    elif operation == "chat":
        assert len(fetch_chat("https://youtu.be/test", tmp_path / "output")[1]) == 3
    else:
        assert download_section("https://youtu.be/test", tmp_path / "output", 0, 10).read_bytes() == b"video"
    assert source.read_text() == (REFRESHED_COOKIES if configuration == "path" else COOKIE_TEXT)
    assert S_IMODE(source.stat().st_mode) == (0o600 if configuration == "path" else 0o400)
    assert len(seen) == (0 if configuration == "none" else 2 if operation == "chat" else 1)
    assert all(not path.exists() for path in seen)


def test_explicit_cookie_path_overrides_env_and_env_path_overrides_text(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit.txt"
    explicit.write_text(COOKIE_TEXT)
    env = tmp_path / "env.txt"
    environment_cookies = COOKIE_TEXT.replace("\tfake\n", "\tenvironment-file\n")
    env.write_text(environment_cookies)
    monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(env))
    monkeypatch.setenv("YOUTUBE_COOKIES", "environment text")
    with cookie_options(explicit) as options:
        assert Path(options["cookiefile"]).read_text() == COOKIE_TEXT
    with cookie_options() as options:
        assert Path(options["cookiefile"]).read_text() == environment_cookies


@pytest.mark.parametrize("configuration", ["path", "text", "explicit"])
def test_malformed_cookie_lines_are_dropped_without_logging_secrets(
        monkeypatch, tmp_path, capfd, caplog, configuration):
    valid = (COOKIE_TEXT + "\n# Export comment\n" +
             "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\thttp_only\tvalid-value\n")
    malformed = [
        "malformed-cookie-secret\n",
        ".youtube.com\tTRUE\t/\tTRUE\t0\tmissing-field-secret\n",
        ".youtube.com\tTRUE\t/\tTRUE\t0\tname\textra-field-secret\textra\n",
        ".youtube.com\tTRUE\t/\tTRUE\tinvalid\tname\tinvalid-expiry-secret\n",
        ".youtube.com\tTRUE\t/\tTRUE\t.5\tname\tmissing-integer-expiry-secret\n",
        ".youtube.com\tTRUE\t/\tTRUE\t1.\tname\tmissing-fraction-expiry-secret\n",
        ".youtube.com\tTRUE\t/\tTRUE\t-1\tname\tnegative-expiry-secret\n",
        "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\tinvalid\tname\thttp-only-secret\n",
    ]
    contents = valid + "".join(malformed)
    source = tmp_path / "cookies.txt"
    source.write_text(contents)
    source.chmod(0o400)
    if configuration == "path":
        monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(source))
    elif configuration == "text":
        monkeypatch.setenv("YOUTUBE_COOKIES", contents)
    with cookie_options(source if configuration == "explicit" else None) as options:
        private = Path(options["cookiefile"])
        assert private.read_text() == valid
        assert S_IMODE(private.stat().st_mode) == 0o600
        jar = YoutubeDLCookieJar(str(private))
        jar.load()
        assert {cookie.name for cookie in jar} == {"test", "http_only"}
    captured = capfd.readouterr()
    assert captured.out == captured.err == ""
    assert all(line.strip() not in caplog.text for line in malformed)
    assert source.read_text() == (contents if configuration == "text" else valid)
    assert not private.exists()


@pytest.mark.parametrize("expiry", ["", "0", "1700000000", "1700000000.5"])
@pytest.mark.parametrize("prefix", ["", "#HttpOnly_"])
def test_yt_dlp_accepted_expiries_are_preserved(monkeypatch, capfd, caplog, expiry, prefix):
    contents = ("# Netscape HTTP Cookie File\n" +
                f"{prefix}.youtube.com\tTRUE\t/\tTRUE\t{expiry}\taccepted\tfake-value\n")
    monkeypatch.setenv("YOUTUBE_COOKIES", contents)
    with cookie_options() as options:
        assert Path(options["cookiefile"]).read_text() == contents
        jar = YoutubeDLCookieJar(options["cookiefile"])
        jar.load()
        cookies = list(jar)
        assert len(cookies) == 1
        assert cookies[0].name == "accepted"
        assert cookies[0].expires == (int(float(expiry)) if expiry and float(expiry) else None)
    captured = capfd.readouterr()
    assert captured.out == captured.err == caplog.text == ""


def test_missing_cookie_source_fails_clearly_without_falling_back(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(tmp_path / "missing"))
    monkeypatch.setenv("YOUTUBE_COOKIES", COOKIE_TEXT)
    with pytest.raises(YouTubeAccessError, match="Could not prepare YouTube cookies"):
        with cookie_options():
            pytest.fail("Missing configured path must not silently disable cookies")


def test_empty_cookie_variables_leave_options_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", "")
    monkeypatch.setenv("YOUTUBE_COOKIES", "")
    with cookie_options() as options:
        assert options == {}
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("error", ["Sign in to confirm you’re not a bot", "Sign in to confirm you're not a bot",
                                   "Please sign in", "HTTP Error 429: Too Many Requests", "HTTP 429", "bot-check"])
@pytest.mark.parametrize("configured", [False, True])
def test_blocking_error_classification(monkeypatch, error, configured):
    if configured:
        monkeypatch.setenv("YOUTUBE_COOKIES", COOKIE_TEXT)
    message = blocking_message(error)
    assert message.startswith("YouTube is blocking this server — add or refresh YouTube cookies (see README)")
    assert ("configured but may be rejected or expired" in message) if configured else "No cookies configured" in message
    assert blocking_message("ffmpeg exited with code 1") is None


@pytest.mark.parametrize("stage", ["metadata", "chat", "section_metadata", "section_download"])
@pytest.mark.parametrize("configuration", ["none", "path", "text"])
@pytest.mark.parametrize("proxy", [None, PROXY])
def test_blocked_downloads_surface_actionable_errors_and_clean_up(monkeypatch, tmp_path, stage, configuration, proxy):
    configured = configuration != "none"
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_TEXT)
    source.chmod(0o400)
    if proxy:
        monkeypatch.setenv("CLIPPER_YTDLP_PROXY", proxy)
    if configuration == "path":
        monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(source))
    elif configuration == "text":
        monkeypatch.setenv("YOUTUBE_COOKIES", COOKIE_TEXT)
    seen = []

    class Downloader:
        def __init__(self, options):
            self.options = options
            if configured:
                seen.append(Path(options["cookiefile"]))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            if configured:
                assert seen[-1].is_file()
                seen[-1].write_text(REFRESHED_COOKIES)
        def extract_info(self, *args, **kwargs):
            if stage in ("metadata", "section_metadata"):
                raise DownloadError("Sign in to confirm you're not a bot")
            return {"duration": 100, "live_status": "was_live", "subtitles": {"live_chat": [{}]}}
        def process_ie_result(self, *args, **kwargs):
            raise DownloadError("HTTP Error 429: Too Many Requests")

    monkeypatch.setattr("yt_dlp.YoutubeDL", Downloader)
    error_type = YouTubeAccessError if stage.startswith("section") else ChatReplayError
    with pytest.raises(error_type, match="YouTube is blocking this server") as raised:
        if stage.startswith("section"):
            download_section("https://youtu.be/test", tmp_path / "output", 0, 10)
        elif stage == "metadata":
            fetch_metadata("https://youtu.be/test")
        else:
            fetch_chat("https://youtu.be/test", tmp_path / "output", info={
                "duration": 100, "live_status": "was_live", "subtitles": {"live_chat": [{}]}})
    assert ("configured but may be rejected or expired" in str(raised.value)) if configured else "No cookies configured" in str(raised.value)
    assert ("proxy may be blocked too" in str(raised.value)) == bool(proxy)
    assert source.read_text() == COOKIE_TEXT
    assert S_IMODE(source.stat().st_mode) == 0o400
    assert all(not path.exists() for path in seen)


def test_explicit_proxy_overrides_environment(monkeypatch):
    monkeypatch.setenv("CLIPPER_YTDLP_PROXY", PROXY)
    explicit = "socks5://another-user:another-password@other.invalid:1080"
    with cookie_options(proxy=explicit) as options:
        assert options["proxy"] == explicit


@pytest.mark.parametrize("scheme", ["https", "socks4", "socks5", "socks5h", "HTTP"])
@pytest.mark.parametrize("configuration", ["env", "explicit"])
def test_section_rejects_non_http_proxy_before_any_request(monkeypatch, tmp_path, capfd, caplog, scheme, configuration):
    proxy = f"{scheme}://proxy-user:proxy-password@proxy.invalid:8080"
    if configuration == "env":
        monkeypatch.setenv("CLIPPER_YTDLP_PROXY", proxy)

    def unexpected_request(*args, **kwargs):
        pytest.fail("An unsupported section proxy must fail before constructing yt-dlp")

    monkeypatch.setattr("yt_dlp.YoutubeDL", unexpected_request)
    output = tmp_path / "output"
    with pytest.raises(YouTubeAccessError, match="Use an http:// proxy") as raised:
        download_section("https://youtu.be/test", output, 0, 10,
                         proxy=proxy if configuration == "explicit" else None)
    captured = capfd.readouterr()
    message = str(raised.value) + caplog.text + captured.out + captured.err
    assert not output.exists()
    assert all(secret not in message for secret in (proxy, "proxy-user", "proxy-password"))


@pytest.mark.parametrize("proxy", [PROXY, "socks5://proxy-user:p%40ss%3Aword@proxy.invalid:1080"])
@pytest.mark.parametrize("cookies", [False, True])
def test_proxy_is_redacted_in_logs_errors_and_tracebacks(monkeypatch, caplog, capfd, proxy, cookies):
    monkeypatch.setenv("CLIPPER_YTDLP_PROXY", proxy)
    if cookies:
        monkeypatch.setenv("YOUTUBE_COOKIES", COOKIE_TEXT)
    caplog.set_level(logging.DEBUG, logger="shorts_clipper.youtube")
    failure = f"Unable to connect through {proxy}; proxy-user p@ss:word"
    with pytest.raises(DownloadError) as raised:
        with cookie_options() as options:
            for level in ("debug", "warning", "error"):
                getattr(options["logger"], level)(f"Proxy request failed: {proxy!r}")
            options["logger"].error("Proxy authentication failed for proxy-user with p@ss:word")
            raise DownloadError(failure)
    captured = capfd.readouterr()
    output = caplog.text + captured.out + captured.err + "".join(traceback.format_exception(
        type(raised.value), raised.value, raised.value.__traceback__))
    assert "[redacted]" in output
    for secret in (proxy, "proxy-user", "proxy-password"):
        assert secret not in output
    if "p%40ss%3Aword" in proxy:
        assert "p@ss:word" not in output and "p%40ss%3Aword" not in output


@pytest.mark.parametrize("configuration", ["path", "explicit"])
def test_cookie_writeback_is_atomic_private_and_sanitized(monkeypatch, tmp_path, configuration):
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_TEXT)
    source.chmod(0o400)
    if configuration == "path":
        monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(source))
    replaced = []
    replace = os.replace

    def replace_cookie(temporary, destination):
        temporary, destination = Path(temporary), Path(destination)
        assert destination == source and temporary.parent == source.parent
        assert temporary != source and source.read_text() == COOKIE_TEXT
        assert temporary.read_text() == REFRESHED_COOKIES
        assert S_IMODE(temporary.stat().st_mode) == 0o600
        replaced.append(temporary)
        replace(temporary, destination)

    monkeypatch.setattr(youtube.os, "replace", replace_cookie)
    with source.open() as original:
        with cookie_options(source if configuration == "explicit" else None) as options:
            private = Path(options["cookiefile"])
            private.write_text(REFRESHED_COOKIES + "malformed refreshed cookie secret\n")
            assert source.read_text() == COOKIE_TEXT
        assert original.read() == COOKIE_TEXT
    assert source.read_text() == REFRESHED_COOKIES
    assert S_IMODE(source.stat().st_mode) == 0o600
    assert len(replaced) == 1 and not replaced[0].exists()
    assert not private.exists()
    assert S_IMODE(Path(str(source) + ".lock").stat().st_mode) == 0o600


@pytest.mark.parametrize("message", ["Please sign in", "Sign in to confirm you're not a bot", "HTTP Error 429"])
def test_explicit_cookie_path_is_not_written_back_on_block(tmp_path, message):
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_TEXT)
    with pytest.raises(DownloadError):
        with cookie_options(source) as options:
            private = Path(options["cookiefile"])
            private.write_text(REFRESHED_COOKIES)
            raise DownloadError(message)
    assert source.read_text() == COOKIE_TEXT
    assert not private.exists()


@pytest.mark.parametrize("contents", [REFRESHED_COOKIES, ""])
@pytest.mark.parametrize("error_type", [DownloadError, RuntimeError])
def test_cookie_refresh_is_discarded_after_any_failure(tmp_path, contents, error_type):
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_TEXT)
    with pytest.raises(error_type, match="ffmpeg exited"):
        with cookie_options(source) as options:
            private = Path(options["cookiefile"])
            private.write_text(contents)
            raise error_type("ffmpeg exited with code 1")
    assert source.read_text() == COOKIE_TEXT
    assert not private.exists()


def test_env_cookie_text_is_never_written_back(monkeypatch, tmp_path):
    monkeypatch.setenv("YOUTUBE_COOKIES", COOKIE_TEXT)

    def unexpected_replace(*args):
        pytest.fail("Environment cookie text must not be written back")

    monkeypatch.setattr(youtube.os, "replace", unexpected_replace)
    with cookie_options() as options:
        private = Path(options["cookiefile"])
        private.write_text(REFRESHED_COOKIES)
    assert os.environ["YOUTUBE_COOKIES"] == COOKIE_TEXT
    assert not private.exists()
    assert list((tmp_path / "data").iterdir()) == []


def test_failed_atomic_replace_preserves_source_and_cleans_up(monkeypatch, tmp_path):
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_TEXT)

    def failed_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(youtube.os, "replace", failed_replace)
    with pytest.raises(YouTubeAccessError, match="Could not save refreshed YouTube cookies"):
        with cookie_options(source) as options:
            private = Path(options["cookiefile"])
            private.write_text(REFRESHED_COOKIES)
    assert source.read_text() == COOKIE_TEXT
    assert not private.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cookies.txt", "cookies.txt.lock"]


def test_cookie_lock_serializes_read_call_and_writeback(monkeypatch, tmp_path):
    source = tmp_path / "cookies.txt"
    source.write_text(COOKIE_TEXT)
    waiting = threading.Event()
    entered = threading.Event()
    flock = youtube.fcntl.flock

    def observed_lock(file, operation):
        if threading.current_thread().name.startswith("cookie-refresh"):
            waiting.set()
        return flock(file, operation)

    def refresh_again():
        with cookie_options(source) as options:
            entered.set()
            private = Path(options["cookiefile"])
            assert private.read_text() == REFRESHED_COOKIES
            private.write_text(REFRESHED_COOKIES.replace("refreshed", "refreshed-again"))

    monkeypatch.setattr(youtube.fcntl, "flock", observed_lock)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cookie-refresh") as pool:
        with cookie_options(source) as options:
            pending = pool.submit(refresh_again)
            assert waiting.wait(5)
            assert not entered.is_set()
            Path(options["cookiefile"]).write_text(REFRESHED_COOKIES)
        pending.result(timeout=5)
    assert source.read_text() == REFRESHED_COOKIES.replace("refreshed", "refreshed-again")
