import shutil
import subprocess
from pathlib import Path

import pytest

from shorts_clipper.captions import Caption, to_ass
from shorts_clipper.video import build_ffmpeg_command, make_vertical_clip


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


def test_render_captions_with_filtergraph_characters_in_path(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for the caption rendering integration test")
    directory = tmp_path / "a,b;c[d]'e:f space"
    directory.mkdir()
    source = tmp_path / "source.mp4"
    subtitles = directory / "captions.ass"
    output = tmp_path / "captioned.mp4"
    subtitles.write_text(to_ass([Caption(0, 1, "Visible caption")]), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=black:s=160x90:r=4:d=0.5", "-c:v", "libx264", str(source)], check=True)
    make_vertical_clip(source, output, 0, 0.5, captions=subtitles)
    frame = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
                           check=True, capture_output=True).stdout
    assert len(frame) == 1080 * 1920
    assert max(frame) > 100
