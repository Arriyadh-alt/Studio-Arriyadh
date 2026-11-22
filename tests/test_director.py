import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shorts_clipper.captions import Caption, CaptionStyle, to_ass
from shorts_clipper.director import (DirectorOptions, Plan, Segment, cut_source, effects_for, framing_conflicts, framing_fix,
                                     hook_band, hook_title_ass, normalize_loudness, measure, loudness, mix_audio, mix_command, plan_edit, plan_record, quality_check,
                                     remap_captions)
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


def test_hook_band_clears_a_top_corner_facecam():
    facecam = OutputRect(414, 24, 450, 520)
    top, bottom = hook_band("Thank you", facecam, CaptionStyle())
    assert top >= 544 + 20 and not framing_conflicts(facecam, CaptionStyle(), (top, bottom))
    ass = hook_title_ass(to_ass([], CaptionStyle()), "Thank you", 2, top=top)
    assert f",8,110,110,{top},1" in ass
    assert hook_band("Thank you", OutputRect(24, 1400, 450, 450), CaptionStyle(position="middle"))[0] == 380


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


def test_api_render_runs_director_and_rerender_can_disable_it(monkeypatch, tmp_path, source):
    import shutil
    from fastapi.testclient import TestClient
    from clipper_studio.api import create_app
    from shorts_clipper.captions import Transcript
    captions = captions_for([t / 2 for t in range(24) if not 8 <= t < 12], loud_at=8.5)
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration": 600, "title": "Stream"}, []))
    monkeypatch.setattr("clipper_studio.api.detect_spikes",
                        lambda *a, **k: ([{"start": 10, "end": 22, "peak": 18.5, "score": 3.0}], []))
    monkeypatch.setattr("clipper_studio.api.download_section",
                        lambda url, directory, *a: Path(shutil.copy(source, directory / "source.mp4")))
    monkeypatch.setattr("clipper_studio.api.transcribe", lambda p, **k: Transcript(list(captions), "en"))
    monkeypatch.setattr("clipper_studio.api.generate_thumbnail", lambda video, output, text, peak: output.write_bytes(b"jpg"))
    monkeypatch.setattr("clipper_studio.api.detect_facecam", lambda *a: None)
    app = create_app(tmp_path, render_workers=1)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": "https://www.youtube.com/watch?v=LIxBN4AHMEE",
                                             "clips": 1, "director": {"music": "on"}}).json()["id"]
        app.state.pending.join()
        clip = client.get(f"/api/jobs/{jid}").json()["clips"][0]
        assert clip["status"] == "ready", clip.get("error")
        result = clip["director_result"]
        assert result["status"] == "used" and result["edits"]["hook"] and result["edits"]["music"]
        assert {q["check"] for q in result["quality"]} >= {"Format", "Audio balance", "Subtitle timing", "Framing"}
        assert clip["director"]["music"] == "on"
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
