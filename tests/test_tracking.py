import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from shorts_clipper import tracking as T
from shorts_clipper.facecam import FacecamOverlay, OutputRect, SourceBox, layout_facecam
from shorts_clipper.video import build_ffmpeg_command

CAM = SourceBox(.75, .05, .22, .3)


def make_source(path: Path, appear=5.0, seconds=12, fps=10, x=.15):
    w, h = 320, 180
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    rng = np.random.default_rng(1)
    background = cv2.GaussianBlur(rng.integers(40, 90, (h, w), dtype=np.uint8), (0, 0), 3)
    for i in range(seconds * fps):
        t = i / fps
        frame = background.copy()
        # streamer moving in the facecam all the time
        cx = int((CAM.x + CAM.w / 2) * w + 12 * np.sin(t * 3))
        cv2.circle(frame, (cx, int((CAM.y + CAM.h / 2) * h)), 12, 230, -1)
        if t >= appear:  # the event the streamer reacts to
            ox = int((x + .02 * np.sin(t * 2)) * w)
            cv2.rectangle(frame, (ox - 14, 90), (ox + 14, 140), 255, -1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    return make_source(tmp_path_factory.mktemp('tracking') / 'source.mp4')


def test_backtracks_from_reaction_to_trigger_and_keeps_it_in_frame(source):
    reaction = T.analyze_reaction(source, 12, 8.0, CAM)
    trigger = reaction.trigger
    assert 4.4 <= trigger['trigger'] <= 6.0
    assert trigger['trigger'] <= trigger['notice'] <= trigger['reaction'] == 8.0
    assert trigger['target'][0] < .35  # the event, not the facecam on the right
    layouts = T.layout_options(layout_facecam(CAM, (320, 180), 'top-right'), (320, 180))
    result = T.plan_reframe(reaction, (0, 12), layouts)
    assert result['validation']['checks']['trigger_visible']
    assert not result['needs_review']
    centre = T._camera_at(result['path'], 7.0)
    assert abs(centre - trigger['target'][0]) < result['width'] / 2


def test_static_center_crop_would_miss_the_trigger(source):
    reaction = T.analyze_reaction(source, 12, 8.0, CAM)
    width = T.window_width(1920, reaction.footage.aspect)
    centre = T.camera_path(reaction.track, width, fixed=.5)
    report = T.validate(reaction.track, centre, width, reaction.trigger, (0, 12))
    assert not report['checks']['trigger_visible']


def test_manual_target_and_fixed_crop(source):
    reaction = T.analyze_reaction(source, 12, 8.0, CAM)
    layouts = T.layout_options(None, (320, 180))
    picked = T.plan_reframe(reaction, (0, 12), layouts, target={'t': 6.0, 'x': .15, 'y': .64})
    assert abs(picked['target'][0] - .15) < .08
    fixed = T.plan_reframe(reaction, (0, 12), layouts, follow='fixed', x=.9)
    assert {round(p[1], 3) for p in fixed['path']} == {round(1 - fixed['width'] / 2, 3)}
    assert fixed['attempts'] == 1


def test_flags_manual_review_when_trigger_cannot_be_framed(source):
    reaction = T.analyze_reaction(source, 12, 8.0, CAM)
    layouts = T.layout_options(None, (320, 180))
    result = T.plan_reframe(reaction, (0, 12), layouts, follow='fixed', x=1.0)
    assert result['needs_review']


def test_layout_choice_and_split_geometry():
    track = T.Track(np.arange(8.0), np.full(8, .5), np.full(8, .5), np.full(8, .3), np.ones(8))
    trigger = {'trigger': 1, 'reaction': 6, 'confidence': .8}
    assert T.choose_layout(track, trigger, False, 16 / 9) == 'gameplay'
    assert T.choose_layout(track, trigger, True, 16 / 9) == 'split'
    track.spread[:] = .2  # wide but still fits the corner-cam crop (real Windah footage sits near here)
    assert T.choose_layout(track, trigger, True, 16 / 9) == 'gameplay'
    track.spread[:] = 0
    assert T.choose_layout(track, trigger, True, 16 / 9) == 'gameplay'
    assert T.choose_layout(track, {**trigger, 'confidence': .05}, True, 16 / 9) == 'reaction'
    layouts = T.layout_options(layout_facecam(CAM, (1920, 1080), 'top-right'), (1920, 1080))
    split, top, height = layouts['split']
    assert T.is_split(split) and top == split.rect.h and top + height == 1920
    assert layouts['reaction'][0].rect.w > layouts['gameplay'][0].rect.w


def test_keyframes_map_to_edited_timeline():
    path = [(t, .2 + .05 * t, .5) for t in np.arange(0, 10, .25)]
    segments = [SimpleNamespace(start=2, end=4, role='main'), SimpleNamespace(start=6, end=8, role='main')]
    keys = T.to_output_time(path, segments)
    assert keys[0][1] == pytest.approx(.3) and keys[-1][0] == pytest.approx(4)
    assert keys[-1][1] == pytest.approx(.2 + .05 * 8)
    expression = T.keyframe_expression(keys)
    assert expression.startswith('0.3000') and 'clip((t-' in expression
    assert T.keyframe_expression([(0, .5, .5), (1, .5, .5)]) == '0.5000'


def test_dynamic_crop_follows_with_facecam_and_split_layout(tmp_path):
    effects = SimpleNamespace(zoom=None, shake=None, flash=None, focal='0.2+(0.1)*clip((t-1)/1,0,1)', focal_y=None)
    overlay = layout_facecam(CAM, (1920, 1080), 'top-right')
    command = build_ffmpeg_command(tmp_path / 's.mp4', tmp_path / 'o.mp4', 0, 5, facecam=overlay, effects=effects)
    graph = command[command.index('-filter_complex') + 1]
    assert 'iw*(0.2+(0.1)*clip((t-1)/1,0,1))' in graph
    split = FacecamOverlay(CAM, OutputRect(0, 0, 1080, 700))
    command = build_ffmpeg_command(tmp_path / 's.mp4', tmp_path / 'o.mp4', 0, 5, facecam=split, effects=effects)
    graph = command[command.index('-filter_complex') + 1]
    assert 'crop=1080:1220' in graph and 'pad=1080:1920:0:700' in graph and 'overlay=0:0' in graph


def test_source_facecam_is_hidden_in_the_gameplay_picture(tmp_path):
    source = tmp_path / 's.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=1280x720:d=1',
                    '-f', 'lavfi', '-i', 'anullsrc', '-t', '1', '-pix_fmt', 'yuv420p', str(source)], check=True)
    corner = SourceBox(.8, .7, .2, .3)
    split = FacecamOverlay(corner, OutputRect(0, 0, 1080, 700))
    command = build_ffmpeg_command(source, tmp_path / 'o.mp4', 0, 1, facecam=split)
    graph = command[command.index('-filter_complex') + 1]
    assert '[mainin]split[hidemain][hidecam];[hidecam]crop=270:224:1010:496,' in graph
    subprocess.run(command, check=True)
    corner_overlay = layout_facecam(corner, (1280, 720), 'top-right')
    command = build_ffmpeg_command(source, tmp_path / 'o.mp4', 0, 1, facecam=corner_overlay)
    assert '[mainin]split[hidemain][hidecam];[hidecam]crop=270:224:1010:496,' in command[command.index('-filter_complex') + 1]
    shared = FacecamOverlay(corner, OutputRect(0, 0, 1080, 700), hide_source=False)
    command = build_ffmpeg_command(source, tmp_path / 'o.mp4', 0, 1, facecam=shared)
    assert 'delogo' not in command[command.index('-filter_complex') + 1]


def test_hold_does_not_drift_before_the_move():
    keys = [(t / 2, .3 if t < 10 else .3 + .05 * (t - 9), .5) for t in range(16)]
    expression = T.keyframe_expression(keys)
    value = lambda t: eval(expression.replace('clip(', '_clip(').replace('t', str(t)).replace('_cli' + str(t) + '(', '_clip('),
                           {'_clip': lambda v, lo, hi: min(max(v, lo), hi)})
    assert value(2.0) == pytest.approx(.3) and value(4.5) == pytest.approx(.3)
    assert value(7.5) == pytest.approx(.3 + .05 * 6)


def test_trigger_must_be_visible_when_it_appears():
    times = np.arange(0, 10, .25)
    track = T.Track(times, np.full(len(times), .5), np.full(len(times), .5), np.zeros(len(times)), np.ones(len(times)))
    trigger = {'trigger': 2.0, 'event': 2.0, 'notice': 4.0, 'reaction': 6.0, 'end': 7.0, 'confidence': .8,
               'target': [.05, .5], 'box': [0, .4, .1, .2]}
    path = [(t, .5, .5) for t in times]
    report = T.validate(track, path, .32, trigger, (0, 10))
    assert report['checks']['subject_in_frame'] and not report['checks']['trigger_visible']


def test_keyframes_never_ramp_over_zero_time_at_cuts():
    keys = [(0, .1, .5), (2, .3, .5), (2, .3005, .5), (2.5, .5, .5)]
    assert '/0.000' not in T.keyframe_expression(keys)


def test_target_time_uses_nearest_sample():
    footage = T.Footage(np.arange(0, 3, .25), np.zeros((12, 2, 2)))
    assert footage.index(1.26) == 5 and footage.index(1.4) == 6
