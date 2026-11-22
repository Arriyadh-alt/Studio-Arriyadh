from pathlib import Path
from stat import S_IMODE

import pytest
from yt_dlp.utils import DownloadError

from clipper_studio.chat import ChatReplayError, fetch_chat, fetch_metadata
from shorts_clipper.downloader import download_section
from shorts_clipper.youtube import YouTubeAccessError, blocking_message, cookie_options


COOKIE_TEXT = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\ttest\tfake\n"


@pytest.fixture(autouse=True)
def cookie_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("CLIPPER_YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("YOUTUBE_COOKIES", raising=False)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))


@pytest.mark.parametrize("operation", ["metadata", "chat", "section"])
@pytest.mark.parametrize("configuration", ["none", "path", "text"])
def test_cookie_options_reach_all_download_stages(monkeypatch, tmp_path, operation, configuration):
    source = tmp_path / "source.txt"
    source.write_text(COOKIE_TEXT)
    source.chmod(0o400)
    if configuration == "path":
        monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(source))
    elif configuration == "text":
        monkeypatch.setenv("YOUTUBE_COOKIES", COOKIE_TEXT)
    seen = []

    class Downloader:
        def __init__(self, options):
            self.options = options
        def __enter__(self):
            if configuration == "none":
                assert "cookiefile" not in self.options
            else:
                private = Path(self.options["cookiefile"])
                seen.append(private)
                assert private != source and private.read_text() == COOKIE_TEXT
                assert S_IMODE(private.stat().st_mode) == 0o600
                assert S_IMODE(private.parent.stat().st_mode) == 0o700
                if configuration == "text":
                    assert tmp_path / "data" in private.parents
            return self
        def __exit__(self, *args):
            if configuration != "none":
                Path(self.options["cookiefile"]).write_text("rewritten cookie jar")
        def extract_info(self, url, download):
            assert not download
            if operation != "section":
                assert not self.options["writesubtitles"]
            return {"duration": 100, "live_status": "was_live", "subtitles": {"live_chat": [{}]}}
        def process_ie_result(self, info, download):
            assert download
            if configuration != "none":
                assert Path(self.options["cookiefile"]).read_text() == COOKIE_TEXT
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
    assert source.read_text() == COOKIE_TEXT
    assert S_IMODE(source.stat().st_mode) == 0o400
    assert len(seen) == (0 if configuration == "none" else 2 if operation == "chat" else 1)
    assert all(not path.exists() for path in seen)


def test_explicit_cookie_path_overrides_env_and_env_path_overrides_text(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit.txt"
    explicit.write_text(COOKIE_TEXT)
    env = tmp_path / "env.txt"
    env.write_text("environment file")
    monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(env))
    monkeypatch.setenv("YOUTUBE_COOKIES", "environment text")
    with cookie_options(explicit) as options:
        assert Path(options["cookiefile"]).read_text() == COOKIE_TEXT
    with cookie_options() as options:
        assert Path(options["cookiefile"]).read_text() == "environment file"


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
@pytest.mark.parametrize("configured", [False, True])
def test_blocked_downloads_surface_actionable_errors_and_clean_up(monkeypatch, tmp_path, stage, configured):
    if configured:
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
            fetch_chat("https://youtu.be/test", tmp_path / "output")
    assert ("configured but may be rejected or expired" in str(raised.value)) if configured else "No cookies configured" in str(raised.value)
    assert all(not path.exists() for path in seen)
