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
                                    auto_layout_facecam, save_source_still,
                                    find_persistent_border, BOTTOM_SAFE_EDGE,
                                    _hud_mask, _saliency_map)
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
    if corner.startswith('bottom'):
        assert r.y+r.h == BOTTOM_SAFE_EDGE
    else:
        assert r.y == 24


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


@pytest.mark.parametrize('name', ['jacksepticeye', 'markiplier'])
def test_real_corner_face_regression(name):
    frame = cv2.imread(str(Path(__file__).parent/'fixtures'/f'facecam_{name}.jpg'))
    assert frame is not None
    with patch('shorts_clipper.facecam._sample_frames', return_value=[frame]*8):
        box = detect_facecam(Path('source'), 0, 20)
    assert box is not None
    assert box.x < .1 and box.y < .1
    assert box.x+box.w > .14 and box.y+box.h > .2
    assert .16 < box.x+box.w < .21
    assert .27 < box.y+box.h < .33


@pytest.mark.parametrize('mirror_x,mirror_y', [(False,False),(True,False),(False,True),(True,True)])
def test_persistent_border_rejects_moving_gameplay(mirror_x, mirror_y):
    frames=[]
    for index in range(8):
        frame=np.full((360,640,3),30,dtype=np.uint8)
        cv2.rectangle(frame,(190+index*35,40),(210+index*35,300),(255,255,255),-1)
        frame[:108,:120]=180
        if mirror_x:
            frame=cv2.flip(frame,1)
        if mirror_y:
            frame=cv2.flip(frame,0)
        frames.append(frame)
    face=SourceBox((640-40-48 if mirror_x else 40)/640,
                   (360-20-48 if mirror_y else 20)/360,48/640,48/360)
    box=find_persistent_border(frames,face)
    assert box is not None
    assert abs(box.w-120/640)<.01 and abs(box.h-108/360)<.01
    assert find_persistent_border([np.zeros_like(frames[0])]*8,face) is None
    assert find_persistent_border([],face) is None


@pytest.mark.parametrize('intensity', [10, 255])
@pytest.mark.parametrize('mode', ['crop', 'blur'])
def test_auto_never_uses_bottom_even_when_top_is_busier(intensity, mode):
    blank=np.zeros((480,270,3),np.uint8)
    changed=blank.copy()
    changed[:160]=intensity
    box=SourceBox(.1,.1,.2,.2)
    with patch('shorts_clipper.facecam._sample_frames',return_value=[blank,changed]*4):
        spec=auto_layout_facecam(Path('source'),0,1,box,mode)
    assert spec.rect.y == 24


def test_auto_size_stays_readable():
    frames=[np.full((480,270,3),v,np.uint8) for v in (0,255)*4]
    with patch('shorts_clipper.facecam._sample_frames',return_value=frames):
        spec=auto_layout_facecam(Path('source'),0,1,SourceBox(.1,.1,.2,.2))
    assert spec.rect.w/1080 >= .38
    assert spec.rect.w < layout_facecam(SourceBox(.1,.1,.2,.2), (270,480)).rect.w
    assert spec.rect.y==24


def test_auto_layout():
    blank = np.zeros((480, 270, 3), np.uint8)
    box = SourceBox(.1, .1, .2, .2)
    with patch('shorts_clipper.facecam._sample_frames', return_value=[blank]*8):
        assert auto_layout_facecam(Path('source'), 0, 1, box) == layout_facecam(box, (270, 480))
    busy = blank.copy()
    busy[:180, :140] = np.random.default_rng(1).integers(0, 256, (180, 140, 3), dtype=np.uint8)
    with patch('shorts_clipper.facecam._sample_frames', return_value=[busy]*8):
        assert auto_layout_facecam(Path('source'), 0, 1, box).rect.x > 24
    with patch('shorts_clipper.facecam._sample_frames', return_value=[cv2.flip(busy,1)]*8):
        assert auto_layout_facecam(Path('source'), 0, 1, box).rect.x == 24
    with patch('shorts_clipper.facecam._sample_frames', return_value=[]):
        with pytest.raises(ValueError):
            auto_layout_facecam(Path('source'), 0, 1, box)


@pytest.mark.parametrize('mode', ['crop', 'blur'])
def test_filters(mode):
    from shorts_clipper.captions import FONT_DIR
    spec = layout_facecam(SourceBox(.7, .1, .2, .3), (640, 360))
    with patch('shorts_clipper.video.detect_focal_x', side_effect=AssertionError('must use center')):
        command = build_ffmpeg_command(Path('source'), Path('out'), 0, 1, mode, Path('subs.ass'), spec)
    graph = command[command.index('-filter_complex')+1]
    assert '[0:v]split[mainin][camin]' in graph
    assert '[camin]crop=' in graph and 'drawbox=' in graph
    assert graph.index('[main][cam]overlay=') < graph.index(',ass=')
    assert f':fontsdir={FONT_DIR}' in graph
    if mode == 'crop':
        assert 'iw*0.500000' in graph


@pytest.mark.parametrize('mode', ['crop', 'blur'])
@pytest.mark.parametrize('captioned', [False, True])
def test_real_overlay(tmp_path, mode, captioned):
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg required')
    source, output = tmp_path/'source.mp4', tmp_path/'out.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=blue:s=640x360:r=4:d=0.5,drawbox=x=0:y=0:w=128:h=128:color=red:t=fill',
                    '-c:v', 'libx264', str(source)], check=True)
    spec = layout_facecam(SourceBox(0, 0, .2, 128/360), (640, 360))
    subtitles = None
    if captioned:
        from shorts_clipper.captions import Caption, CaptionStyle, to_ass
        subtitles = tmp_path/'captions.ass'
        subtitles.write_text(to_ass([Caption(0, 1, 'Visible animated caption')], CaptionStyle()), encoding='utf-8')
    make_vertical_clip(source, output, 0, .5, mode, captions=subtitles, facecam=spec)
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


@pytest.mark.parametrize('position', ['top-left', 'top-right', 'bottom-left', 'bottom-right'])
@pytest.mark.parametrize('box', [SourceBox(.1,.1,.2,.15), SourceBox(.1,.1,.05,.8),
                                 SourceBox(0,0,.209,.279)])
def test_aspect_preserved_to_nearest_output_pixel(position, box):
    r = layout_facecam(box, (1920,1080), position).rect
    aspect = box.w*1920/(box.h*1080)
    assert min(abs(r.h-r.w/aspect), abs(r.w-r.h*aspect)) <= 2
    assert r.y >= 24 and r.y+r.h <= BOTTOM_SAFE_EDGE


def test_hud_and_saliency_signals():
    blank = np.zeros((480,270),np.uint8)
    assert not np.any(_hud_mask(blank))
    assert not np.any(_saliency_map(blank))
    text = blank.copy()
    for y in range(18,110,18):
        cv2.putText(text, 'SCORE 1250', (8,y), cv2.FONT_HERSHEY_SIMPLEX, .4, 255, 1)
    assert _hud_mask(text)[:120,:110].mean() > .05
    saliency = _saliency_map(text)
    assert np.isfinite(saliency).all() and saliency[:120,:110].mean() > saliency[200:].mean()
    with patch('shorts_clipper.facecam._sample_frames', return_value=[cv2.cvtColor(text,cv2.COLOR_GRAY2BGR)]*8), patch(
            'shorts_clipper.facecam._saliency_map', return_value=np.zeros_like(blank)), patch(
            'shorts_clipper.facecam.cv2.Canny', return_value=blank):
        assert auto_layout_facecam(Path('source'),0,1,SourceBox(.1,.1,.2,.2)).rect.x > 24


def test_fallback_contains_entire_face_track_with_margin():
    detector, profile = MagicMock(), MagicMock()
    detector.empty.return_value = False
    profile.empty.return_value = True
    faces = [(480+i*3, 50+i*2, 30, 30) for i in range(8)]
    detector.detectMultiScale.side_effect = [[f] for f in faces]
    with patch('shorts_clipper.facecam._sample_frames', return_value=[np.zeros((360,640,3),np.uint8)]*8), patch(
            'shorts_clipper.facecam.cv2.CascadeClassifier', side_effect=[detector,profile]):
        box = detect_facecam(Path('source'),0,1)
    assert box is not None
    assert box.x < min(f[0] for f in faces)/640
    assert box.y < min(f[1] for f in faces)/360
    assert box.x+box.w > max(f[0]+f[2] for f in faces)/640
    assert box.y+box.h > max(f[1]+f[3] for f in faces)/360


@pytest.mark.parametrize('mode', ['crop', 'blur'])
def test_director_effects_precede_facecam_and_captions(mode):
    from shorts_clipper.director import Effects
    spec = layout_facecam(SourceBox(0,0,.2,.3), (640,360))
    command = build_ffmpeg_command(Path('source'),Path('out'),0,1,mode,Path('sub.ass'),spec,
                                   Effects(zoom='1.2',shake='4',flash='0.1',focal='0.8'))
    graph = command[command.index('-filter_complex')+1]
    assert graph.index('zoompan=') < graph.index("eq=brightness='0.1'") < graph.index('[main];')
    assert graph.index('[main][cam]overlay=') < graph.index(',ass=')
    assert 'force_original_aspect_ratio=decrease:force_divisible_by=2' in graph
    assert f'pad={spec.rect.w}:{spec.rect.h}' in graph
    assert 'zoompan' not in graph.split('[main];')[1]


@pytest.mark.parametrize('mode', ['crop', 'blur'])
def test_facecam_pixels_stay_fixed_through_director_zoom(tmp_path, mode):
    from shorts_clipper.director import Plan, Segment, effects_for
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg required')
    source = tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i',
                    'color=blue:s=640x360:r=30:d=1,drawbox=x=0:y=0:w=128:h=96:color=red:t=fill',
                    '-c:v','libx264',str(source)],check=True)
    spec = layout_facecam(SourceBox(0,0,.2,96/360),(640,360))
    plan = Plan([Segment(0,1)],zooms=[[.3,.8,.35,.1,'punch']],shakes=[[.4,.8]],flashes=[.5])
    output = make_vertical_clip(source,tmp_path/'out.mp4',0,1,mode,facecam=spec,effects=effects_for(plan))
    capture = cv2.VideoCapture(str(output))
    for second in (.05,.5,.9):
        capture.set(cv2.CAP_PROP_POS_MSEC,second*1000)
        ok, frame = capture.read()
        assert ok
        r = spec.rect
        red = frame[r.y+10:r.y+r.h-10,r.x+10:r.x+r.w-10]
        assert np.mean(red[:,:,2]) > 240
        assert np.mean(red[:,:,:2]) < 10
        assert np.mean(frame[r.y+r.h+8,r.x+r.w//2,:2]) > 20
    capture.release()
