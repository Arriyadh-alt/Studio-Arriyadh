import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import pytest

from shorts_clipper.captions import Caption, to_ass
from shorts_clipper.video import build_ffmpeg_command, crop_left, detect_focal_x, make_vertical_clip


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


@pytest.mark.parametrize('focus, expected', [(0, 0), (.5, 420), (1, 840), (.25, 0), (.75, 840)])
def test_focal_crop_clamps_to_source(focus, expected):
    assert crop_left(1920, 1080, focus) == expected


def test_focal_missing_source_falls_back(tmp_path):
    assert detect_focal_x(tmp_path / 'missing.mp4', 0, 10) == .5


@pytest.mark.parametrize('faces, expected', [([], .5), ([(100, 20, 80, 80), (400, 10, 20, 20)], 140 / 640)])
def test_face_sampling_largest_face_and_fallback(tmp_path, faces, expected):
    source = tmp_path / 'source.mp4'
    source.touch()
    capture, detector = MagicMock(), MagicMock()
    capture.read.return_value = (True, np.zeros((360, 640, 3), dtype=np.uint8))
    detector.empty.return_value = False
    detector.detectMultiScale.return_value = faces
    profile = MagicMock()
    profile.empty.return_value = True
    with patch.object(cv2, 'VideoCapture', return_value=capture), patch.object(cv2, 'CascadeClassifier', side_effect=[detector, profile]):
        assert detect_focal_x(source, 10, 20) == expected
    assert capture.set.call_count == 5
    capture.set.assert_any_call(cv2.CAP_PROP_POS_MSEC, 11000)
    capture.release.assert_called_once()


def test_face_sampling_median_and_decode_failure(tmp_path):
    source = tmp_path / 'source.mp4'
    source.touch()
    capture, detector = MagicMock(), MagicMock()
    capture.read.side_effect = [(True, np.zeros((360, 640, 3), dtype=np.uint8))] * 3 + [(False, None)] * 2
    detector.empty.return_value = False
    detector.detectMultiScale.side_effect = [[(0, 0, 40, 40)], [(300, 0, 40, 40)], [(600, 0, 40, 40)]]
    profile = MagicMock()
    profile.empty.return_value = True
    with patch.object(cv2, 'VideoCapture', return_value=capture), patch.object(cv2, 'CascadeClassifier', side_effect=[detector, profile]):
        assert detect_focal_x(source, 0, 10) == .5
    capture.release.assert_called_once()


def test_face_detection_error_falls_back(tmp_path):
    source = tmp_path / 'source.mp4'
    source.touch()
    capture = MagicMock()
    with patch.object(cv2, 'VideoCapture', return_value=capture), patch.object(cv2, 'CascadeClassifier', side_effect=cv2.error('bad cascade')):
        assert detect_focal_x(source, 0, 10) == .5
    capture.release.assert_called_once()


def test_command_uses_focal_point_and_wide_blur():
    with patch('shorts_clipper.video.detect_focal_x', return_value=.25):
        cmd = build_ffmpeg_command(Path('a'), Path('b'), 0, 5)
    assert 'iw*0.250000-ow/2' in cmd[cmd.index('-filter_complex') + 1]
    cmd = build_ffmpeg_command(Path('a'), Path('b'), 0, 5, 'blur')
    filters = cmd[cmd.index('-filter_complex') + 1]
    assert 'gt(iw/ih,2)' in filters
    assert 'saturation=0.35' in filters


def test_profile_speaker_preferred_over_lower_audience(tmp_path):
    source = tmp_path / 'source.mp4'
    source.touch()
    capture, detector, profile = MagicMock(), MagicMock(), MagicMock()
    capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    detector.empty.return_value = profile.empty.return_value = False
    detector.detectMultiScale.return_value = [(100, 300, 80, 80)]
    profile.detectMultiScale.side_effect = [[], [(300, 50, 60, 60)]] * 5
    with patch.object(cv2, 'VideoCapture', return_value=capture), patch.object(cv2, 'CascadeClassifier', side_effect=[detector, profile]):
        assert detect_focal_x(source, 0, 10) == (640 - 300 - 30) / 640
