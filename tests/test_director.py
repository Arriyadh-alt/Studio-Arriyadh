import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shorts_clipper.captions import Caption, CaptionStyle, to_ass
from shorts_clipper.director import (DirectorOptions, Plan, Segment, _loop_seconds, cut_source, effects_for,
                                     framing_conflicts, framing_fix, hook_band, hook_title_ass, loudness,
                                     make_seamless_loop, measure, mix_audio, mix_command, normalize_loudness,
                                     plan_edit, plan_record, probe, quality_check, remap_captions)
from shorts_clipper.facecam import OutputRect
from shorts_clipper.video import build_ffmpeg_command, make_vertical_clip


def word(start, text):
    return SimpleNamespace(start=start, end=start + 0.4, word=text, probability=1.0)


def captions_for(times, loud_at=None):
    words = [word(t, ("WOW!" if loud_at is not None and abs(t - loud_at) < 0.01 else "hello")) for t in times]
    return [Caption(chunk[0].start, chunk[-1].end, " ".join(w.word for w in chunk), chunk)
            for chunk in (words[i:i + 4] for i in range(0, len(words), 4))]


def levels(duration, quiet=(), loud=()):
    data = np.full(int(duration / 0.1), -24.0)
    data += np.sin(np.arange(len(data))) * 2
    for a, b in quiet:
        data[int(a / .1):int(b / .1)] = -80
    for a, b in loud:
        data[int(a / .1):int(b / .1)] = -6
    return data


def test_options_ignore_bad_values():
    assert DirectorOptions.from_dict({"enabled": "yes", "music": "loud", "sfx": False}) == DirectorOptions(sfx=False)
    assert DirectorOptions.from_dict(None).enabled
    assert DirectorOptions.from_dict(None).loop
    assert not DirectorOptions.from_dict({"loop": False}).loop


def test_plan_hooks_the_peak_cuts_dead_air_and_keeps_order():
    times = [t / 2 for t in range(60) if not 12 <= t < 16]
    plan = plan_edit(levels(30, quiet=[(6, 8)], loud=[(19, 21)]), captions_for(times, loud_at=19.5), 30,
                     DirectorOptions(), peak=20, spike_score=4, hook_text="No way")
    hook, *main = plan.segments
    assert hook.role == "hook" and 1.5 <= hook.length <= 3.0 and 17 <= hook.start <= 21
    assert plan.hook["text"] == "No way"
    assert all(s.role == "main" for s in main) and len(main) == 2
    assert main[0].end <= 6.5 and main[1].start >= 7.5 and main[-1].end == 30
    assert any(z[-1] == "reaction" for z in plan.zooms) and plan.shakes
    assert {kind for _, kind in plan.sfx} >= {"whoosh", "hit"}
    assert 0 <= plan.score["overall"] <= 100 and plan.summary()["cuts"] == 1
    assert plan.map_time(25) == pytest.approx(hook.length + main[0].length + 25 - main[1].start)


def test_plan_respects_toggles_and_length_cap():
    times = [t / 2 for t in range(120)]
    plan = plan_edit(levels(60, loud=[(40, 42)]), captions_for(times), 60,
                     DirectorOptions(hook=True, edits=False, sfx=False, music="on"), peak=41)
    assert plan.duration <= 60.0001 and not plan.sfx and plan.music
    assert [s.role for s in plan.segments][1:] == ["main"]
    off = plan_edit(levels(20), [], 20, DirectorOptions(hook=False, edits=False, music="off"))
    assert off.segments == [Segment(0, 20)] and not off.zooms and not off.music


def test_music_auto_only_under_quiet_speechless_audio():
    quiet = levels(20) - 20
    assert plan_edit(quiet, [], 20, DirectorOptions(hook=False)).music
    assert not plan_edit(levels(20), captions_for([t / 2 for t in range(40)]), 20, DirectorOptions(hook=False)).music


def test_remap_captions_follow_cuts_and_duplicate_hook():
    plan = Plan([Segment(10, 12, "hook"), Segment(0, 5), Segment(8, 14)])
    shown = remap_captions(captions_for([1, 2, 9, 11]), plan)
    starts = [round(w.start, 2) for c in shown for w in c.words]
    assert starts == [1.0, 3.0, 4.0, 8.0, 10.0]
    assert all(a.end <= b.start for a, b in zip(shown, shown[1:]))


def test_effects_and_filters_are_time_based():
    plan = Plan([Segment(0, 3, "hook"), Segment(5, 10)], zooms=[[0, 3, .16, .25, "hook"]], shakes=[[4, 4.5]],
                flashes=[3], focal=[.3, .6])
    effects = effects_for(plan)
    assert "in_time" in effects.zoom and "in_time" in effects.shake and "t-3.000" in effects.flash
    with_effects = build_ffmpeg_command(Path("a"), Path("b"), 0, 8, effects=effects)
    graph = with_effects[with_effects.index("-filter_complex") + 1]
    assert "zoompan=z='min(1.35" in graph and "eq=brightness=" in graph and "iw*(0.3000+" in graph
    assert "zoompan" not in build_ffmpeg_command(Path("a"), Path("b"), 0, 8)[6]


def test_mix_command_ducks_music_and_normalizes():
    plan = Plan([Segment(0, 10)], sfx=[[1, "whoosh"], [2, "hit"]], music=True)
    command = mix_command(Path("in.mp4"), Path("out.mp4"), plan)
    graph = command[command.index("-filter_complex") + 1]
    assert "sidechaincompress" in graph and "amix=inputs=4" in graph and "loudnorm=I=-14" in graph
    assert "adelay=2000|2000" in graph


def test_hook_title_and_framing_fix():
    ass = hook_title_ass(to_ass([], CaptionStyle()), "Big {moment}", 2.5)
    assert "Style: Hook" in ass and "BIG ｛MOMENT｝" in ass and ",0:00:02.50," in ass
    style, note = framing_fix(CaptionStyle(position="bottom"), OutputRect(24, 1400, 450, 450))
    assert style.position == "middle" and "facecam" in note
    assert framing_fix(CaptionStyle(position="bottom"), OutputRect(24, 24, 450, 450))[1] is None


@pytest.mark.parametrize("style", ["static", "teaser", "story"])
def test_long_hooks_wrap_in_every_title_style(style):
    import re

    text = "What was the hardest part of learning to compete?"
    ass = hook_title_ass(None, text, 3, static=style == "static", story=style == "story")
    dialogue = next(line for line in ass.splitlines() if line.startswith("Dialogue:"))
    label = re.sub(r"\{[^}]*}", "", dialogue.split(",", 9)[9])
    assert r"\N" in label
    assert label.replace(r"\N", " ") == (text if style == "static" else text.upper())


def test_hook_band_clears_a_top_corner_facecam():
    facecam = OutputRect(414, 24, 450, 520)
    top, bottom = hook_band("Thank you", facecam, CaptionStyle())
    assert top >= 544 + 20 and not framing_conflicts(facecam, CaptionStyle(), (top, bottom))
    ass = hook_title_ass(to_ass([], CaptionStyle()), "Thank you", 2, top=top)
    assert f",8,110,110,{top},1" in ass
    assert hook_band("Thank you", OutputRect(24, 1400, 450, 450), CaptionStyle(position="middle"))[0] == 380


@pytest.mark.parametrize('height', [340, 978])
@pytest.mark.parametrize('size', [80, 130])
def test_manual_bottom_caption_band_moves_above_camera(height, size):
    from shorts_clipper.director import caption_band
    rect = OutputRect(412,1600-height,454,height)
    style, note = framing_fix(CaptionStyle(position='bottom',size=size), rect)
    assert note
    assert caption_band(style,size)[1] <= rect.y
    assert not framing_conflicts(rect,style)


def test_story_title_avoids_tall_manual_bottom_camera():
    from shorts_clipper.director import STORY_SIZE
    from shorts_clipper.facecam import SourceBox, layout_facecam
    overlay = layout_facecam(SourceBox(.1,.1,.1,.8),(1920,1080),'bottom-right')
    text = 'Windah takes one wrong turn and instantly regrets it'
    style, _ = framing_fix(CaptionStyle(size=130),overlay.rect)
    band = hook_band(text,overlay.rect,None,preferred=760,size=STORY_SIZE)
    assert not framing_conflicts(overlay.rect,style,band,hook_captions_overlap=False)
    assert overlay.rect.y+overlay.rect.h == 1600


def test_wrapped_captions_stay_above_tall_bottom_camera():
    from shorts_clipper.captions import styled_lines, caption_height
    from shorts_clipper.director import caption_band
    from shorts_clipper.facecam import SourceBox, layout_facecam
    overlay = layout_facecam(SourceBox(.1,.1,.1,.8),(1920,1080),'bottom-right')
    style = CaptionStyle(size=130)
    caption = Caption(0,1,'amazing reaction hilarious moment')
    height = caption_height([caption],style,overlay.rect.y-284)
    style, _ = framing_fix(style,overlay.rect,height)
    lines = styled_lines(caption.text,style)
    assert len(lines) >= 4
    assert caption_band(style,height=height)[1] <= overlay.rect.y
    assert not framing_conflicts(overlay.rect,style,caption_height=height)


@pytest.mark.parametrize('story', [False,True])
def test_rendered_text_leaves_tall_bottom_camera_visible(tmp_path, story):
    import shutil
    import cv2
    from shorts_clipper.captions import FONT_DIR, caption_height
    from shorts_clipper.director import STORY_SIZE
    from shorts_clipper.facecam import SourceBox, layout_facecam
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg required')
    style = CaptionStyle(size=130)
    captions = [Caption(2,3,'amazing reaction hilarious moment')]
    hook = {'style':'story','text':'Windah takes one wrong turn and instantly regrets it'} if story else None
    overlay = layout_facecam(SourceBox(.1,.1,.1,.8),(1920,1080),'bottom-right')
    limit = overlay.rect.y-284
    height = caption_height(captions,style,limit)
    style, _ = framing_fix(style,overlay.rect,height)
    ass = to_ass(captions,style,max_height=limit)
    if hook:
        band = hook_band(hook['text'],overlay.rect,None,preferred=760,size=STORY_SIZE)
        ass = hook_title_ass(ass,hook['text'],2,top=band[0],story=True,max_height=band[1]-band[0])
    subtitles = tmp_path/'text.ass'
    subtitles.write_text(ass)
    r = overlay.rect
    for time in ([1,2.5] if story else [2.5]):
        output = tmp_path/f'frame-{time}.png'
        graph = (f"drawbox=x={r.x}:y={r.y}:w={r.w}:h={r.h}:color=red:t=fill,"
                 f"ass=filename={subtitles}:fontsdir={FONT_DIR}")
        subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=blue:s=1080x1920:r=2:d=3',
                        '-vf',graph,'-ss',str(time),'-frames:v','1',str(output)],check=True)
        frame = cv2.imread(str(output))
        camera = frame[r.y+4:r.y+r.h-4,r.x+4:r.x+r.w-4]
        assert np.mean(camera[:,:,2]) > 245
        assert np.max(camera[:,:,:2]) < 25


def test_caption_height_handles_empty_minimal_and_font_fallback(monkeypatch):
    from shorts_clipper.captions import caption_height
    assert caption_height([],CaptionStyle()) == 0
    captions = [Caption(0,1,'amazing reaction hilarious moment')]
    assert caption_height(captions,CaptionStyle(size=130)) > caption_height(captions,CaptionStyle(size=80))
    assert caption_height(captions,CaptionStyle(preset='minimal')) < caption_height(captions,CaptionStyle())
    monkeypatch.setattr('shorts_clipper.captions.style_font',lambda *a: None)
    assert caption_height(captions,CaptionStyle()) > 0


def test_one_eight_line_caption_does_not_shrink_camera():
    from shorts_clipper.captions import caption_height, styled_lines
    from shorts_clipper.facecam import SourceBox, layout_facecam
    style = CaptionStyle(size=130)
    caption = Caption(15,16,'amazing reaction hilarious moment amazing reaction hilarious moment')
    assert len(styled_lines(caption.text,style)) == 8
    overlay = layout_facecam(SourceBox(0,0,.2,.25),(1920,1080),'bottom-right')
    assert overlay.rect.w >= 410
    limit = overlay.rect.y-284
    assert caption_height([caption],style,limit) <= limit
    short = Caption(0,1,'Hello')
    ass = to_ass([short,caption],style,max_height=limit)
    events = [line for line in ass.splitlines() if line.startswith('Dialogue:')]
    assert r'\fs' not in events[0]
    assert r'\fs' in events[1]
    assert style.size == 130


def test_late_caption_does_not_push_hook_over_top_camera():
    from shorts_clipper.director import hook_layout
    from shorts_clipper.facecam import SourceBox, layout_facecam
    overlay = layout_facecam(SourceBox(.1,.1,.1,.8),(1920,1080))
    style = CaptionStyle(size=130)
    captions = [Caption(15,16,'amazing reaction hilarious moment')]
    hook = {'text':'Windah takes one wrong turn and instantly regrets it','duration':3}
    band, height, overlap = hook_layout(hook,overlay.rect,style,captions,1476-overlay.rect.y-overlay.rect.h)
    assert height == 0 and not overlap
    assert band[0] >= overlay.rect.y+overlay.rect.h or band[1] <= overlay.rect.y


@pytest.mark.parametrize('story,end', [(False,3.7),(True,3)])
def test_hook_layout_uses_only_its_actual_time_window(story,end):
    from shorts_clipper.captions import caption_height
    from shorts_clipper.director import hook_layout
    hook = {'text':'Watch this','duration':3,'style':'story' if story else 'normal'}
    active = Caption(end-.2,end+.2,'Hello')
    captions = [Caption(-1,0,'Ended'),active,Caption(end,end+1,'amazing reaction hilarious moment')]
    _, height, overlap = hook_layout(hook,None,CaptionStyle(),captions)
    assert overlap and height == caption_height([active],CaptionStyle())
    assert hook_layout(hook,None,None,[])[1:] == (0,False)


def test_hook_fallback_excludes_camera_even_when_caption_covers_screen():
    rect = OutputRect(24,24,450,600)
    style = CaptionStyle(position='middle',size=130)
    band = hook_band('Watch this',rect,style,caption_height=1800)
    assert band[0] >= rect.y+rect.h or band[1] <= rect.y


def test_hook_scales_title_instead_of_camera_when_space_is_tight():
    from shorts_clipper.director import STORY_SIZE, hook_height
    rect = OutputRect(648,622,218,978)
    text = 'Windah takes one wrong turn and instantly regrets it'
    band = hook_band(text,rect,None,preferred=760,size=STORY_SIZE)
    assert band[1] <= rect.y and band[1]-band[0] < hook_height(text,STORY_SIZE)
    ass = hook_title_ass(None,text,3,top=band[0],story=True,max_height=band[1]-band[0])
    font_size = int(next(line for line in ass.splitlines() if line.startswith('Style: Hook')).split(',')[2])
    assert font_size < STORY_SIZE and hook_height(text,font_size) <= band[1]-band[0]
    with pytest.raises(ValueError,match='No title space'):
        hook_band(text,OutputRect(0,0,1080,1920))


def test_framing_check_flags_overlaps():
    facecam = OutputRect(414, 24, 450, 520)
    assert framing_conflicts(facecam, CaptionStyle(), (380, 530)) == ["Hook title covers the facecam"]
    assert framing_conflicts(OutputRect(24, 1300, 450, 450), CaptionStyle(), None) == ["Captions cover the facecam"]
    assert framing_conflicts() == []


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("director") / "source.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=12", "-f", "lavfi",
                    "-i", "aevalsrc='if(between(t,4,6),0,(0.2+0.6*between(t,8,9))*sin(2*PI*220*t))':d=12:s=48000",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_full_director_render(source, tmp_path):
    captions = captions_for([t / 2 for t in range(24) if not 8 <= t < 12], loud_at=8.5)
    plan = plan_edit(loudness(source, 12), captions, 12, DirectorOptions(music="on"), peak=8.5, hook_text="Watch this")
    assert plan.segments[0].role == "hook" and plan.removed > 1
    directed = cut_source(source, tmp_path / "cut.mp4", plan)
    plan.focal = [.5] * len(plan.segments)
    shown = remap_captions(captions, plan)
    subtitles = tmp_path / "c.ass"
    subtitles.write_text(hook_title_ass(to_ass(shown, CaptionStyle()), "Watch this", 2), encoding="utf-8")
    video = make_vertical_clip(directed, tmp_path / "clip.mp4", 0, plan.duration, "crop", subtitles,
                               effects=effects_for(plan))
    mixed = mix_audio(video, tmp_path / "mixed.mp4", plan)
    checks = {c["check"]: c for c in quality_check(mixed, plan, shown)}
    assert checks["Format"]["status"] == "pass" and "1080×1920" in checks["Format"]["detail"]
    assert checks["Audio balance"]["status"] == "pass"
    assert checks["Subtitle timing"]["status"] == "pass"
    plan.music, plan.sfx = False, [[1.0, "pop"]]
    quiet = {c["check"]: c for c in quality_check(mix_audio(video, tmp_path / "plain.mp4", plan), plan, shown)}
    assert quiet["Audio balance"]["status"] == "pass"
    json.dumps(plan_record(plan, DirectorOptions(), quality_check(video, plan, shown)))


def test_loop_transition_adapts_to_boundaries_and_captions():
    head = np.zeros((4, 2, 2, 3))
    tail = np.full_like(head, 0.8)
    assert _loop_seconds(head, tail, 10, []) == 0.30
    assert _loop_seconds(head, tail, 10, [Caption(9.5, 9.95, "Final line")]) == 0.12
    tail[-2] = 0
    assert _loop_seconds(head, tail, 10, []) == 0.12
    assert _loop_seconds(head, head, 10, []) == 0.12


@pytest.mark.parametrize("audio_end", [2.0, 2.15])
def test_loop_export_matches_first_and_last_frames_without_cutting_audio(tmp_path, audio_end):
    original, looped = tmp_path / "original.mp4", tmp_path / "looped.mp4"
    subprocess.run(["ffmpeg", "-v", "error",
                    "-f", "lavfi", "-i", "color=c=blue:s=160x284:r=30:d=1",
                    "-f", "lavfi", "-i", "color=c=red:s=160x284:r=30:d=1",
                    "-f", "lavfi", "-i", f"sine=f=407:d={audio_end}:r=48000",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                    "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-r", "30",
                    "-c:a", "aac", str(original)], check=True)

    assert make_seamless_loop(original, looped) == {"status": "applied", "seconds": 0.3}
    assert probe(looped) == probe(original)

    def colors(video):
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=2:2:flags=area",
                                       "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4, 3).mean(axis=1)

    before, after = colors(original), colors(looped)
    assert before.shape == after.shape == (60, 3)
    assert np.max(np.abs(before[:51] - after[:51])) < 5
    assert np.max(np.abs(before[0] - before[-1])) > 240
    assert np.max(np.abs(after[0] - after[-1])) < 5
    assert np.all(np.diff(after[50:, 2]) > 0)

    def samples(video):
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a", "-t", str(audio_end),
                                       "-f", "f32le", "-"])
        return np.frombuffer(raw, dtype=np.float32)

    before, after = samples(original), samples(looped)
    assert len(before) == len(after) == round(audio_end * 48_000)
    assert np.corrcoef(before[1000:-1000], after[1000:-1000])[0, 1] > 0.99
    assert np.max(np.abs(after[:100])) < 0.04
    assert np.max(np.abs(after[-100:])) < 0.04


@pytest.mark.parametrize("tail", ["quiet", "active", "spoken"])
def test_loop_audio_bridge_matches_opening_only_over_quiet_speech_free_tail(tmp_path, tail):
    original, looped = tmp_path / "original.mp4", tmp_path / "looped.mp4"
    ending = ".25" if tail == "active" else ".0002"
    envelope = f"if(lt(t,0.3),.55,if(gt(t,1.7),{ending},.07))"
    audio = f"aevalsrc='{envelope}*sin(2*PI*340*t)|{envelope}*sin(2*PI*460*t)':s=48000:d=2.013"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=160x284:r=30:d=2.02",
                    "-f", "lavfi", "-i", audio, "-c:v", "libx264", "-c:a", "aac",
                    str(original)], check=True)
    speech = [Caption(1.6, 2.01, "Final line")] if tail == "spoken" else []
    assert make_seamless_loop(original, looped, speech=speech)["status"] == "applied"

    def samples(file):
        length = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                              "stream=duration_ts", "-of", "csv=p=0", str(file)]))
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(file), "-map", "0:a:0", "-f", "f32le", "-"])
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)[:length]

    def rms(samples):
        return np.sqrt(np.mean(samples ** 2))

    before, after = samples(original), samples(looped)
    assert np.corrcoef(before[4800:88000].ravel(), after[4800:88000].ravel())[0, 1] > 0.995
    if tail == "quiet":
        assert abs(len(after) - 61 * 48000 // 30) <= 48
        assert rms(before[-4800:]) < 0.001
        assert 0.9 < rms(after[-4800:]) / rms(after[:4800]) < 1.1
        assert np.max(np.abs(after[-1] - after[0])) < 0.05
    else:
        assert after.shape == before.shape
        assert np.isclose(rms(before[-4800:]), rms(after[-4800:]), rtol=0.1, atol=0.001)


def test_api_render_runs_director_and_rerender_can_disable_it(monkeypatch, tmp_path, source):
    import shutil
    from fastapi.testclient import TestClient
    from clipper_studio.api import create_app
    from shorts_clipper.captions import Transcript
    captions = captions_for([t / 2 for t in range(24) if not 8 <= t < 12], loud_at=8.5)
    monkeypatch.setattr("clipper_studio.api.fetch_metadata", lambda url, *_: {"id": "x", "duration": 600, "title": "Stream"})
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration": 600, "title": "Stream"}, []))
    monkeypatch.setattr("clipper_studio.api.detect_spikes",
                        lambda *a, **k: ([{"start": 10, "end": 22, "peak": 18.5, "score": 3.0}], []))
    monkeypatch.setattr("clipper_studio.api.download_section",
                        lambda url, directory, *a, **kw: Path(shutil.copy(source, directory / "source.mp4")))
    monkeypatch.setattr("clipper_studio.api.transcribe", lambda p, **k: Transcript(list(captions), "en"))
    monkeypatch.setattr("clipper_studio.api.generate_thumbnail", lambda video, output, text, peak: output.write_bytes(b"jpg"))
    monkeypatch.setattr("clipper_studio.api.detect_facecam", lambda *a: None)
    loop_inputs = []

    def record_loop(video, output, visible, *, speech, **kwargs):
        loop_inputs.append((visible, speech))
        return make_seamless_loop(video, output, visible, speech=speech, **kwargs)

    monkeypatch.setattr("clipper_studio.api.make_seamless_loop", record_loop)
    app = create_app(tmp_path, render_workers=1)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url": "https://www.youtube.com/watch?v=LIxBN4AHMEE",
                                             "clips": 1, "captions": False, "director": {"music": "on"}}).json()["id"]
        app.state.pending.join()
        clip = client.get(f"/api/jobs/{jid}").json()["clips"][0]
        assert clip["status"] == "ready", clip.get("error")
        result = clip["director_result"]
        assert result["status"] == "used" and result["edits"]["hook"] and result["edits"]["music"]
        assert clip["director"]["loop"] and result["edits"]["loop"]["status"] == "applied"
        assert len(loop_inputs) == 1 and loop_inputs[0][0] == [] and loop_inputs[0][1]
        assert {q["check"] for q in result["quality"]} >= {"Format", "Audio balance", "Subtitle timing", "Framing"}
        assert clip["director"]["music"] == "on"
        assert client.post(f"/api/jobs/{jid}/clips/{clip['id']}/render",
                           json={"start": 10, "end": 22, "director": {"loop": False}}).status_code == 202
        app.state.pending.join()
        clip = client.get(f"/api/jobs/{jid}").json()["clips"][0]
        assert clip["status"] == "ready" and clip["director_result"]["status"] == "used"
        assert not clip["director"]["loop"] and clip["director_result"]["edits"]["loop"] is None
        assert client.post(f"/api/jobs/{jid}/clips/{clip['id']}/render",
                           json={"start": 10, "end": 22, "director": {"enabled": False}}).status_code == 202
        app.state.pending.join()
        clip = client.get(f"/api/jobs/{jid}").json()["clips"][0]
        assert clip["status"] == "ready" and clip["director_result"] == {"status": "disabled"}
        assert client.post(f"/api/jobs/{jid}/clips/{clip['id']}/render",
                           json={"start": 10, "end": 22, "director": {"music": "loud"}}).status_code == 422


def test_hook_snaps_to_whole_words_in_continuous_speech():
    from shorts_clipper.director import find_hook, transcript_words
    words = transcript_words([Caption(0, 30, "", [SimpleNamespace(start=t * 0.45, end=t * 0.45 + 0.44, word=" yes",
                                                                   probability=1) for t in range(66)])])
    start, end, _ = find_hook(levels(30, loud=[(14, 16)]), words, 30, 15)
    assert end - start <= 3.0
    assert not any(w.start < edge < w.end for w in words for edge in (start, end))


def test_rendered_offset_maps_peaks_onto_the_edit():
    from clipper_studio.api import rendered_offset
    clip = {"director_result": {"edits": {"timeline": [[0, 5, 2.5], [8, 14, 7.5]]}}}
    assert rendered_offset(clip, 10) == 9.5 and rendered_offset(clip, 6) is None
    assert rendered_offset({"director_result": {"status": "disabled"}}, 10) == 10


def test_normalize_loudness_brings_a_quiet_mix_to_target(source, tmp_path):
    quiet = tmp_path / "quiet.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-c:v", "copy", "-af", "volume=-14dB",
                    "-c:a", "aac", str(quiet)], check=True)
    assert measure(quiet, audio_only=True)["lufs"] < -16
    lufs = normalize_loudness(quiet)
    stats = measure(quiet)
    assert abs(lufs + 14) <= 0.5 and abs(stats["lufs"] + 14) <= 0.5 and stats["true_peak"] <= -0.5


def test_quality_check_framing_warns_on_hook_over_facecam(source):
    plan = Plan([Segment(0, 12)])
    checks = {c["check"]: c for c in quality_check(source, plan, [], layout={
        "rect": OutputRect(414, 24, 450, 520), "style": CaptionStyle(), "hook": (380, 530)})}
    assert checks["Framing"]["status"] == "warn" and "Hook title covers the facecam" in checks["Framing"]["detail"]


def test_loudness_correction_keeps_audio_in_sync(tmp_path):
    from shorts_clipper.director import gain_command
    clip = tmp_path / "impulse.mov"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=s=64x64:d=1", "-f", "lavfi", "-i",
                    "aevalsrc='if(between(t,0.5,0.501),0.3,0)':d=1:s=48000", "-c:v", "libx264", "-c:a", "pcm_s16le",
                    "-shortest", str(clip)], check=True)
    out = tmp_path / "level.mp4"
    subprocess.run(gain_command(clip, out, 2.0), check=True)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "s16le", "-ac", "1", "-ar", "48000", "-"],
                         capture_output=True, check=True).stdout
    audio = np.abs(np.frombuffer(raw, dtype=np.int16))
    assert abs(int(np.argmax(audio)) - 24000) < 48 and len(audio) >= 47000


def test_api_story_voiceover_and_channel_for_older_jobs(monkeypatch, tmp_path, source):
    import shutil
    import wave
    from fastapi.testclient import TestClient
    from clipper_studio.api import create_app
    from shorts_clipper.captions import Transcript
    captions = captions_for([t / 2 for t in range(24) if not 8 <= t < 12], loud_at=8.5)
    lookups = []

    def metadata(url, *_):
        lookups.append(url)
        return {"id": "x", "duration": 600, "title": "Stream", "channel": "Valkyrae", "uploader": "Valkyrae"}

    def voice(text, language, output, cache, voice, category, seed, delivery):
        assert cache == tmp_path.resolve() / "voices" and (voice, delivery) == ("trailer", "suspense")
        with wave.open(str(output), "wb") as file:
            file.setnchannels(1), file.setsampwidth(2), file.setframerate(22050)
            file.writeframes(b"\0\0" * 22050 * 2)
        return {"seconds": 2.0, "voice": voice, "voice_name": "Movie Trailer", "mood": delivery, "engine": "neural",
                "script": "Watch closely. " + text}

    monkeypatch.setattr("clipper_studio.api.fetch_metadata", metadata)
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration": 600, "title": "Stream"}, []))
    monkeypatch.setattr("clipper_studio.api.detect_spikes",
                        lambda *a, **k: ([{"start": 10, "end": 22, "peak": 18.5, "score": 3.0}], []))
    monkeypatch.setattr("clipper_studio.api.download_section",
                        lambda url, directory, *a, **kw: Path(shutil.copy(source, directory / "source.mp4")))
    monkeypatch.setattr("clipper_studio.api.transcribe", lambda p, **k: Transcript(list(captions), "en"))
    monkeypatch.setattr("clipper_studio.api.generate_thumbnail", lambda video, output, text, peak: output.write_bytes(b"jpg"))
    monkeypatch.setattr("clipper_studio.api.detect_facecam", lambda *a: None)
    monkeypatch.setattr("clipper_studio.api.narrate", voice)
    app = create_app(tmp_path, render_workers=1)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url": "https://www.youtube.com/watch?v=LIxBN4AHMEE", "clips": 1,
                                             "director": {"voiceover": True, "voice": "trailer",
                                                          "delivery": "suspense"}}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        clip = job["clips"][0]
        assert clip["status"] == "ready", clip.get("error")
        hook = clip["director_result"]["edits"]["hook"]
        assert hook["voiceover"] and hook["streamer"] == "Valkyrae" and "Valkyrae" in hook["text"]
        assert hook["narration"]["voice_name"] == "Movie Trailer" and hook["narration"]["mood"] == "suspense"
        assert job["metadata"]["channel"] == "Valkyrae" and len(lookups) == 2
        assert client.post(f"/api/jobs/{jid}/clips/{clip['id']}/render",
                           json={"start": 10, "end": 22, "director": {"voiceover": True, "voice": "trailer",
                                                                     "delivery": "suspense"}}).status_code == 202
        app.state.pending.join()
        assert len(lookups) == 2


def test_plan_keeps_the_reaction_trigger_from_being_cut():
    times = [t / 2 for t in range(60) if not 20 <= t < 28]
    args = (levels(30, quiet=[(10, 14)], loud=[(19, 21)]), captions_for(times, loud_at=19.5), 30,
            DirectorOptions(hook=False))
    plain = plan_edit(*args, peak=20)
    assert not any(s.start <= 10.5 and s.end >= 20 for s in plain.segments)
    kept = plan_edit(*args, peak=20, trigger=11)
    assert any(s.start <= 10.2 and s.end >= 20 for s in kept.segments)
