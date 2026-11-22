from unittest.mock import patch

from shorts_clipper import cli
from shorts_clipper.captions import Caption


def test_default_captions_and_srt(tmp_path):
    output = tmp_path / "out.mp4"
    with patch.object(cli, "download_section", return_value=tmp_path / "source.mp4"), \
         patch.object(cli, "video_duration", return_value=10), \
         patch.object(cli, "transcribe", return_value=[Caption(0, 1, "Hello")]) as transcribe, \
         patch.object(cli, "make_vertical_clip") as render:
        assert cli.main(["URL", "0", "10", "--srt", "-o", str(output)]) == 0
        transcribe.assert_called_once_with(tmp_path / "source.mp4", "base", None, clip_end=10)
        assert render.call_args.args[-1].name == "captions.ass"
    assert "Hello" in output.with_suffix(".srt").read_text()


def test_no_captions_skips_transcription(tmp_path):
    with patch.object(cli, "download_section", return_value=tmp_path / "source.mp4"), \
         patch.object(cli, "video_duration", return_value=10), \
         patch.object(cli, "transcribe") as transcribe, \
         patch.object(cli, "make_vertical_clip") as render:
        assert cli.main(["URL", "0", "10", "--no-captions"]) == 0
        transcribe.assert_not_called()
        assert render.call_args.args[-1] is None


def test_empty_speech_and_transcription_error(tmp_path, capsys):
    with patch.object(cli, "download_section", return_value=tmp_path / "source.mp4"), \
         patch.object(cli, "video_duration", return_value=10), \
         patch.object(cli, "transcribe", return_value=[]) as transcribe, \
         patch.object(cli, "make_vertical_clip"):
        assert cli.main(["URL", "0", "10"]) == 0
        assert "No speech detected" in capsys.readouterr().out
        transcribe.side_effect = RuntimeError("model unavailable")
        assert cli.main(["URL", "0", "10"]) == 1
        assert "model unavailable" in capsys.readouterr().err


def test_conflicting_flags():
    assert cli.main(["URL", "0", "10", "--srt", "--no-captions"]) == 1


def test_cookies_option_is_forwarded(tmp_path):
    cookies = tmp_path / "cookies.txt"
    with patch.object(cli, "download_section", return_value=tmp_path / "source.mp4") as download, \
         patch.object(cli, "video_duration", return_value=10), \
         patch.object(cli, "make_vertical_clip"):
        assert cli.main(["URL", "0", "10", "--no-captions", "--cookies", str(cookies)]) == 0
        assert download.call_args.kwargs["cookies"] == cookies
