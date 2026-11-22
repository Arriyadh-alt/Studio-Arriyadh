from pathlib import Path
import shutil
import subprocess
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from shorts_clipper.captions import MARGIN_BOTTOM, MARGIN_RIGHT
from shorts_clipper.facecam import (SourceBox, OutputRect, detect_facecam,
                                    find_facecam_border, layout_facecam,
                                    auto_layout_facecam, save_source_still)
from shorts_clipper.video import build_ffmpeg_command, make_vertical_clip


def test_geometry():
    assert SourceBox(-.1, .8, .5, .5) == SourceBox(0, .8, .5, .2)
    for values in [(1, 0, .2, .2), (0, 0, -1, .2), (float('nan'), 0, 1, 1)]:
        with pytest.raises(ValueError):
            SourceBox(*values)
    with pytest.raises(ValueError):
        OutputRect(1000, 0, 200, 200)


@pytest.mark.parametrize('corner', ['top-left', 'top-right', 'bottom-left', 'bottom-right'])
@pytest.mark.parametrize('width', [.1, .42, .9])
def test_layout(corner, width):
    spec = layout_facecam(SourceBox(.7, .1, .2, .3), (1920, 1080), corner, width)
    r = spec.rect
    assert 270 <= r.w <= 648
    assert abs(r.w/r.h - (.2*1920)/(.3*1080)) < .01
    assert r.x+r.w <= 1080-MARGIN_RIGHT-24
    assert r.y+r.h < 1920-MARGIN_BOTTOM-240


def test_border():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (400, 30), (620, 200), (255, 255, 255), 3)
    box = find_facecam_border(frame, SourceBox(470/640, 70/360, 50/640, 50/360))
    assert box is not None
    assert abs(box.x-400/640) < .01
    assert abs(box.w-220/640) < .02
    assert find_facecam_border(np.zeros_like(frame), SourceBox(.1, .1, .1, .1)) is None


@pytest.mark.parametrize('faces,found', [
    ([[(480, 50, 45, 45)]]*8, True),
    ([[(100, 20, 200, 200)]]*8, False),
    ([[(i*70, 50, 30, 30)] for i in range(8)], False),
    ([[]]*8, False),
    ([[(480, 50, 45, 45)]]*4+[[]]*4, False),
])
def test_detection(faces, found):
    detector, profile = MagicMock(), MagicMock()
    detector.empty.return_value = False
    profile.empty.return_value = True
    detector.detectMultiScale.side_effect = faces
    with patch('shorts_clipper.facecam._sample_frames', return_value=[np.zeros((360, 640, 3), np.uint8)]*8), patch(
            'shorts_clipper.facecam.cv2.CascadeClassifier', side_effect=[detector, profile]):
        box = detect_facecam(Path('source'), 0, 10)
    assert (box is not None) == found
    if found:
        assert box.w > 45/640 and box.h > 45/360


def test_detection_failure():
    assert detect_facecam(Path('nonexistent'), 0, 10) is None
    with patch('shorts_clipper.facecam._sample_frames', side_effect=RuntimeError('decode')):
        assert detect_facecam(Path('source'), 0, 10) is None


def test_auto_layout():
    blank = np.zeros((480, 270, 3), np.uint8)
    box = SourceBox(.1, .1, .2, .2)
    with patch('shorts_clipper.facecam._sample_frames', return_value=[blank]*8):
        assert auto_layout_facecam(Path('source'), 0, 1, box) == layout_facecam(box, (270, 480))
    busy = blank.copy()
    busy[:180, :140] = np.random.default_rng(1).integers(0, 256, (180, 140, 3), dtype=np.uint8)
    with patch('shorts_clipper.facecam._sample_frames', return_value=[busy]*8):
        assert auto_layout_facecam(Path('source'), 0, 1, box).rect.x > 24
    with patch('shorts_clipper.facecam._sample_frames', return_value=[]):
        with pytest.raises(ValueError):
            auto_layout_facecam(Path('source'), 0, 1, box)


@pytest.mark.parametrize('mode', ['crop', 'blur'])
def test_filters(mode):
    spec = layout_facecam(SourceBox(.7, .1, .2, .3), (640, 360))
    with patch('shorts_clipper.video.detect_focal_x', side_effect=AssertionError('must use center')):
        command = build_ffmpeg_command(Path('source'), Path('out'), 0, 1, mode, Path('subs.ass'), spec)
    graph = command[command.index('-filter_complex')+1]
    assert '[0:v]split[mainin][camin]' in graph
    assert '[camin]crop=' in graph and 'drawbox=' in graph
    assert graph.index('[main][cam]overlay=') < graph.index(',ass=')
    if mode == 'crop':
        assert 'iw*0.500000' in graph


@pytest.mark.parametrize('mode', ['crop', 'blur'])
def test_real_overlay(tmp_path, mode):
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg required')
    source, output = tmp_path/'source.mp4', tmp_path/'out.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=blue:s=640x360:r=4:d=0.5,drawbox=x=0:y=0:w=128:h=128:color=red:t=fill',
                    '-c:v', 'libx264', str(source)], check=True)
    spec = layout_facecam(SourceBox(0, 0, .2, 128/360), (640, 360))
    make_vertical_clip(source, output, 0, .5, mode, facecam=spec)
    capture = cv2.VideoCapture(str(output))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape[:2] == (1920, 1080)
    r = spec.rect
    assert frame[r.y+r.h//2, r.x+r.w//2, 2] > 200
    still = save_source_still(source, tmp_path/'still.jpg')
    assert cv2.imread(str(still)).shape[:2] == (360, 640)


def test_still_errors(tmp_path):
    with pytest.raises(ValueError):
        save_source_still(tmp_path/'missing', tmp_path/'still.jpg')
    with pytest.raises(ValueError):
        save_source_still(tmp_path/'missing', tmp_path/'still.jpg', -1)
