from pathlib import Path

import pytest
from yt_dlp.utils import DownloadError

from clipper_studio.chat import ChatReplayError, Message, fetch_chat, parse_chat


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


@pytest.fixture
def stub_yt_dlp(monkeypatch):
    class Downloader:
        info = REPLAY_INFO
        extraction_error = None
        download = None

        def __init__(self, options):
            assert options["skip_download"]
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, *args, **kwargs):
            if self.extraction_error:
                raise DownloadError(self.extraction_error)
            return self.info

        def process_ie_result(self, info, download):
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
])
def test_missing_replay_status(stub_yt_dlp, tmp_path, info, message):
    stub_yt_dlp.info = info
    with pytest.raises(ChatReplayError, match=message):
        fetch_chat(REPLAY_URL, tmp_path)


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
    assert (tmp_path / "replay.live_chat.json").read_bytes() == FIXTURE.read_bytes()


@pytest.mark.parametrize("limit", ["time", "size", "final_size"])
def test_replay_limit_cleans_up_incomplete_download(stub_yt_dlp, tmp_path, monkeypatch, limit):
    now = [0]
    monkeypatch.setattr("clipper_studio.chat.time.monotonic", lambda: now[0])
    monkeypatch.setattr("clipper_studio.chat.MAX_REPLAY_SECONDS", 10)
    monkeypatch.setattr("clipper_studio.chat.MAX_REPLAY_BYTES", 20)

    def download(self, output, progress):
        output.write_bytes(b"a" * 21)
        now[0] = 11 if limit == "time" else 1
        progress({"downloaded_bytes": 21 if limit == "size" else 1})

    stub_yt_dlp.download = download
    with pytest.raises(ChatReplayError, match="exceeded Studio’s processing limit"):
        fetch_chat(REPLAY_URL, tmp_path)
    assert list(tmp_path.iterdir()) == []
