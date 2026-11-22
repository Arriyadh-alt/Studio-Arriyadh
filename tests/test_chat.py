from pathlib import Path

import pytest

from clipper_studio.chat import Message, fetch_chat, parse_chat


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
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, *args, **kwargs):
            return {"duration": 500}
    monkeypatch.setattr("clipper_studio.chat.yt_dlp.YoutubeDL", Downloader)
    with pytest.raises(ValueError, match="no live-chat replay"):
        fetch_chat("https://youtube.com/watch?v=test", tmp_path)
