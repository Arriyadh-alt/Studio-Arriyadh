from pathlib import Path

import pytest

from shorts_clipper.video import build_ffmpeg_command


def test_command_cuts_range_and_sets_filter():
    cmd = build_ffmpeg_command(Path("in.mp4"), Path("out.mp4"), 10, 25, "crop")
    assert cmd[cmd.index("-ss") + 1] == "10.000"
    assert cmd[cmd.index("-t") + 1] == "15.000"
    assert "crop=1080:1920" in cmd[cmd.index("-filter_complex") + 1]
    assert cmd[-1] == "out.mp4"


def test_command_rejects_bad_range_and_mode():
    with pytest.raises(ValueError):
        build_ffmpeg_command(Path("a"), Path("b"), 5, 5)
    with pytest.raises(ValueError):
        build_ffmpeg_command(Path("a"), Path("b"), 0, 5, "stretch")


@pytest.mark.parametrize("mode", ["crop", "blur"])
def test_captions_follow_vertical_filter(mode):
    cmd = build_ffmpeg_command(Path("a"), Path("b"), 0, 5, mode, Path("captions.ass"))
    assert ",ass=filename=" in cmd[cmd.index("-filter_complex") + 1]
