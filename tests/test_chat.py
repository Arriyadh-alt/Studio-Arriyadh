from pathlib import Path

import pytest

from clipper_studio.chat import Message, fetch_chat, fetch_metadata, parse_chat


def test_parse_fixture():
    with (Path(__file__).parent / "fixtures/chat.jsonl").open() as stream:
        assert parse_chat(stream) == [Message(12.5, "Alice", "WOW :fire:"),
                                      Message(15, "Bob", "Great stream!"), Message(20, "Carol", "Sticker")]


def test_empty_and_malformed():
    assert parse_chat(["", "{}", '{"videoOffsetTimeMsec":null}']) == []
    with pytest.raises(ValueError):
        parse_chat(["broken json"])


def test_no_replay(monkeypatch, tmp_path):
    class Downloader:
        def __init__(self, options):
            assert options["skip_download"]
            assert not options["writesubtitles"]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            assert kwargs == {"download": False}
            return {"duration": 500}
        def process_ie_result(self, *args, **kwargs):
            pytest.fail("No-chat videos must fail before replay download")
    monkeypatch.setattr("clipper_studio.chat.yt_dlp.YoutubeDL", Downloader)
    with pytest.raises(ValueError, match="no live-chat replay"):
        fetch_chat("https://youtube.com/watch?v=test", tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fields", [{"is_live": True, "was_live": True}, {"was_live": False},
                                  {"was_live": True, "duration": None}])
def test_metadata_requires_finished_replay(monkeypatch, fields):
    class Downloader:
        def __init__(self, options):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            return {"duration": 500, "subtitles": {"live_chat": [{}]}, **fields}
    monkeypatch.setattr("clipper_studio.chat.yt_dlp.YoutubeDL", Downloader)
    with pytest.raises(ValueError, match="finished livestream VOD"):
        fetch_metadata("https://youtu.be/test")


def test_preflight_metadata_reused_for_chat(monkeypatch, tmp_path):
    info = {"id": "test", "title": "Replay", "duration": 500, "was_live": True,
            "subtitles": {"live_chat": [{}]}}
    calls = []

    class Downloader:
        def __init__(self, options):
            self.options = options
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            calls.append("metadata")
            assert not self.options["writesubtitles"]
            return info
        def process_ie_result(self, received, download):
            calls.append("download")
            assert received is info and download
            assert self.options["writesubtitles"]
            fixture = Path(__file__).parent / "fixtures/chat.jsonl"
            (tmp_path / "replay.live_chat.json").write_text(fixture.read_text())

    monkeypatch.setattr("clipper_studio.chat.yt_dlp.YoutubeDL", Downloader)
    metadata = fetch_metadata("https://youtu.be/test")
    result, messages = fetch_chat("https://youtu.be/test", tmp_path, metadata)
    assert calls == ["metadata", "download"]
    assert result["title"] == "Replay" and len(messages) == 3


@pytest.mark.parametrize("contents,error", [(None, "did not produce a chat file"),
                                          ("", "no timestamped messages"), ("broken", "Expecting value")])
def test_chat_download_empty_and_malformed(monkeypatch, tmp_path, contents, error):
    class Downloader:
        def __init__(self, options):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def process_ie_result(self, *args, **kwargs):
            if contents is not None:
                (tmp_path / "replay.live_chat.json").write_text(contents)
    monkeypatch.setattr("clipper_studio.chat.yt_dlp.YoutubeDL", Downloader)
    info = {"duration": 500, "was_live": True, "subtitles": {"live_chat": [{}]}}
    with pytest.raises(ValueError, match=error):
        fetch_chat("https://youtu.be/test", tmp_path, info)
