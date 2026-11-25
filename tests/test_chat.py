import fcntl
import json
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest
from yt_dlp import YoutubeDL
from yt_dlp.downloader.common import FileDownloader
from yt_dlp.utils import DownloadError

from clipper_studio.chat import (CHAT_TIMEOUT_ERROR, ChatReplayError, ChatWatchdog, Message,
                                 fetch_chat, fetch_metadata, parse_chat)
from clipper_studio.spikes import detect_spikes


FIXTURE = Path(__file__).parent / "fixtures/chat.jsonl"
REPLAY_URL = "https://www.youtube.com/watch?v=LIxBN4AHMEE"
REPLAY_INFO = {
    "live_status": "was_live",
    "duration": 42403,
    "subtitles": {"live_chat": [{"ext": "json"}]},
}


def test_parse_fixture():
    with FIXTURE.open() as stream:
        assert parse_chat(stream) == [Message(12.5, "Alice", "WOW :fire:"),
                                      Message(15, "Bob", "Great stream!"), Message(20, "Carol", "Sticker")]


def test_empty_and_malformed():
    assert parse_chat(["", "{}", '{"videoOffsetTimeMsec":null}']) == []
    with pytest.raises(ValueError):
        parse_chat(["broken json"])


def test_watchdog_keeps_partial_buckets_and_reports_fragment_offset(tmp_path, monkeypatch):
    now = 0
    monkeypatch.setattr("clipper_studio.chat.time", SimpleNamespace(monotonic=lambda: now))
    progress = []
    watchdog = ChatWatchdog(600, lambda: None, progress.append)

    def replay(operation, args, *, timeout, check):
        nonlocal now
        assert operation == "replay"
        output = args[1] / "replay.live_chat.json.part"
        output.write_bytes(replay_line(1, 90, "Viewer") + b'{"videoOffsetTimeMsec":"120000","actions":[]}\n')
        now = 10
        check()
        now = 56
        check()

    monkeypatch.setattr("clipper_studio.chat.run_download", replay)
    info = dict(REPLAY_INFO, duration=600)
    _, buckets = fetch_chat(REPLAY_URL, tmp_path, info, watchdog)
    assert progress == [20]
    assert len(buckets) == 1
    assert buckets.warning == CHAT_TIMEOUT_ERROR
    assert buckets.path.is_file()


def test_watchdog_bounds_slow_but_advancing_chat_and_has_a_hard_cap(monkeypatch):
    now = 0
    monkeypatch.setattr("clipper_studio.chat.time", SimpleNamespace(monotonic=lambda: now))
    slow = ChatWatchdog(6600, lambda: None, lambda _: None)
    for now in (30, 60, 89):
        slow.advanced(now)
        assert not slow.expired()
    now = 90
    slow.advanced(now)
    assert slow.expired()

    now = 0
    capped = ChatWatchdog(43200, lambda: None, lambda _: None)
    for now in range(30, 600, 30):
        capped.advanced(now * 70)
        assert not capped.expired()
    now = 600
    capped.advanced(42000)
    assert capped.expired()


@pytest.fixture
def stub_yt_dlp(monkeypatch):
    class Downloader:
        info = REPLAY_INFO
        extraction_error = None
        download = None
        extractions = 0
        downloads = 0

        def __init__(self, options):
            assert options["skip_download"]
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, *args, **kwargs):
            type(self).extractions += 1
            assert kwargs == {"download": False}
            assert not self.options["writesubtitles"]
            if self.extraction_error:
                raise DownloadError(self.extraction_error)
            return self.info

        def process_ie_result(self, info, download):
            type(self).downloads += 1
            assert self.options["writesubtitles"]
            assert download and info == REPLAY_INFO
            output = Path(self.options["outtmpl"]).parent / "replay.live_chat.json"
            if self.download:
                self.download(output, self.options["progress_hooks"][0])
            else:
                output.write_bytes(FIXTURE.read_bytes())

    monkeypatch.setattr("clipper_studio.chat.yt_dlp.YoutubeDL", Downloader)
    return Downloader


@pytest.mark.parametrize("info,message", [
    ({"duration": 500, "live_status": "not_live"}, "regular YouTube video"),
    ({"duration": 500, "live_status": "was_live"}, "no live-chat replay"),
    ({"live_status": "is_live"}, "replay isn’t ready yet"),
    ({"live_status": "is_upcoming"}, "replay isn’t ready yet"),
    ({"live_status": "post_live"}, "replay isn’t ready yet"),
    (dict(REPLAY_INFO, duration=None), "hasn’t published the replay duration"),
    (dict(REPLAY_INFO, duration=0), "hasn’t published the replay duration"),
    (dict(REPLAY_INFO, duration=float("inf")), "hasn’t published the replay duration"),
    (dict(REPLAY_INFO, duration=float("nan")), "hasn’t published the replay duration"),
])
def test_missing_replay_status(stub_yt_dlp, tmp_path, info, message):
    stub_yt_dlp.info = info
    with pytest.raises(ChatReplayError, match=message):
        fetch_chat(REPLAY_URL, tmp_path)
    assert stub_yt_dlp.downloads == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("error,message", [
    ("ERROR: [youtube] LIxBN4AHMEE: Private video. Sign in.", "private, restricted, or unavailable"),
    ("ERROR: [youtube] LIxBN4AHMEE: This video is unavailable", "private, restricted, or unavailable"),
    ("Failed to connect", "couldn’t load this YouTube video"),
])
def test_youtube_error(stub_yt_dlp, tmp_path, error, message):
    stub_yt_dlp.extraction_error = error
    with pytest.raises(ChatReplayError, match=message):
        fetch_chat(REPLAY_URL, tmp_path)


def test_long_replay_download(stub_yt_dlp, tmp_path):
    metadata, messages = fetch_chat(REPLAY_URL, tmp_path)
    assert metadata["duration"] == 42403
    assert len(messages) == 3
    with messages.reader() as (group, count):
        assert group(2) == (Message(12.5, "Alice", "WOW :fire:"),)
        assert count(3) == 1


def test_chat_download_releases_shared_cookie_lock(stub_yt_dlp, tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    initial = "# Netscape HTTP Cookie File\n"
    cookies.write_text(initial)
    cookies.with_suffix(".txt.lock").touch(mode=0o600)
    monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(cookies))

    def download(self, output, hook):
        private = Path(self.options["cookiefile"])
        assert private != cookies and private.read_text() == initial
        assert private.stat().st_mode & 0o777 == 0o600
        with cookies.with_suffix(".txt.lock").open("r+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        private.write_text(initial + "# chat changes\n")
        output.write_bytes(FIXTURE.read_bytes())

    stub_yt_dlp.download = download
    _, messages = fetch_chat(REPLAY_URL, tmp_path / "chat", REPLAY_INFO)
    assert len(messages) == 3
    assert cookies.read_text() == initial


def test_disk_guard_retains_partial_download(stub_yt_dlp, tmp_path, monkeypatch, caplog):
    enough_space = [True]
    monkeypatch.setattr("clipper_studio.chat.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=10**10 if enough_space[0] else 0))

    def download(self, output, progress):
        output.with_suffix(".json.part").write_bytes(FIXTURE.read_bytes())
        enough_space[0] = False
        progress({"downloaded_bytes": 1234})

    stub_yt_dlp.download = download
    _, buckets = fetch_chat(REPLAY_URL, tmp_path)
    assert len(buckets) == 3
    assert "available disk space" in buckets.warning
    assert "guard=disk_reserve elapsed=" in caplog.text and "bytes=1234" in caplog.text


def replay_line(index, offset=100, author="Viewer", text="OMG"):
    renderer = {"id": str(index), "authorName": {"simpleText": author}, "message": {"simpleText": text}}
    action = {"videoOffsetTimeMsec": str(offset * 1000),
              "actions": [{"addChatItemAction": {"item": {"liveChatTextMessageRenderer": renderer}}}]}
    return (json.dumps({"replayChatItemAction": action}) + "\n").encode()


def test_streams_large_replay_and_dedupes_deliveries_not_reactions(stub_yt_dlp, tmp_path, monkeypatch):
    count = 160_000
    now = [0]
    monkeypatch.setattr("clipper_studio.chat.time.monotonic", lambda: now[0])

    def download(self, output, progress):
        partial = output.with_suffix(".json.part")
        with partial.open("wb") as file:
            for index in range(count):
                file.write(replay_line(index, 100 + index % 40, f"Viewer {index % 100}"))
                if index % 4000 == 3999:
                    file.flush()
                    now[0] += 30
                    progress({"downloaded_bytes": 200 * 1024 * 1024})
            file.write(replay_line(0, 100, "Viewer 0"))
        partial.replace(output)
        progress({"status": "finished", "downloaded_bytes": 200 * 1024 * 1024})

    stub_yt_dlp.download = download
    tracemalloc.start()
    try:
        info, buckets = fetch_chat(REPLAY_URL, tmp_path, REPLAY_INFO)
        _, activity = detect_spikes(buckets, info["duration"])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(buckets) == count and not buckets.warning
    assert sum(point["count"] for point in activity) == count
    assert peak < 12 * 1024 * 1024


@pytest.mark.parametrize("retry_options", [{}, {"frag_index": 1}], ids=["http", "fragment"])
def test_streamed_download_backs_off_through_yt_dlp_retry_callback(
        stub_yt_dlp, tmp_path, monkeypatch, retry_options):
    waits = []
    monkeypatch.setattr("clipper_studio.chat.time.sleep", waits.append)
    monkeypatch.setattr("clipper_studio.chat.random.uniform", lambda *_: .25)

    def download(self, output, progress):
        with YoutubeDL(self.options) as ydl:
            downloader = FileDownloader(ydl, self.options)
            for attempt in (1, 2):
                downloader.report_retry(TimeoutError("Network timed out"), attempt, 2, **retry_options)
        output.write_bytes(FIXTURE.read_bytes())

    stub_yt_dlp.download = download
    _, buckets = fetch_chat(REPLAY_URL, tmp_path, REPLAY_INFO)
    assert waits == [1.25, 2.25]
    assert len(buckets) == 3 and not buckets.warning
    assert stub_yt_dlp.downloads == 1


def test_retries_transient_download_and_scores_like_original(stub_yt_dlp, tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr("clipper_studio.chat.time.sleep", waits.append)
    monkeypatch.setattr("clipper_studio.chat.random.uniform", lambda *_: .25)
    data = [replay_line(i, 100 + i % 4, f"Viewer {i % 10}", ["OMG", "🔥", "what"][i % 3])
            for i in range(50)]

    def download(self, output, progress):
        partial = output.with_suffix(".json.part")
        if type(self).downloads < 3:
            partial.write_bytes(b"".join(data[:20]))
            progress({"downloaded_bytes": partial.stat().st_size})
            raise DownloadError("HTTP Error 503: Service Unavailable")
        output.write_bytes(b"".join(data))

    stub_yt_dlp.download = download
    _, buckets = fetch_chat(REPLAY_URL, tmp_path, REPLAY_INFO)
    assert waits == [1.25, 2.25]
    assert len(buckets) == len(data) and not buckets.warning
    assert detect_spikes(buckets, REPLAY_INFO["duration"]) == detect_spikes(parse_chat(data), REPLAY_INFO["duration"])


def test_preflight_metadata_reused_for_bounded_chat_download(stub_yt_dlp, tmp_path):
    info = fetch_metadata(REPLAY_URL)
    assert stub_yt_dlp.extractions == 1 and stub_yt_dlp.downloads == 0
    result, messages = fetch_chat(REPLAY_URL, tmp_path, info)
    assert stub_yt_dlp.extractions == 1 and stub_yt_dlp.downloads == 1
    assert result["duration"] == 42403 and len(messages) == 3


@pytest.mark.parametrize("contents,message", [(None, "couldn’t download"), ("", "no timestamped messages"),
                                             ("broken", "couldn’t read")])
def test_chat_download_empty_and_malformed(stub_yt_dlp, tmp_path, contents, message):
    def download(self, output, progress):
        if contents is not None:
            output.write_text(contents)
    stub_yt_dlp.download = download
    with pytest.raises(ChatReplayError, match=message):
        fetch_chat(REPLAY_URL, tmp_path)
    assert list(tmp_path.iterdir()) == []
