import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from clipper_studio.api import DirectorSpec, create_app, director_config
from shorts_clipper.captions import Caption
from shorts_clipper.director import DirectorOptions, loudness, probe, remap_captions
from shorts_clipper.editing import authenticity_report, plan_source_edit, record_operations
from test_api import REPLAY_URL


SPEECH = [Caption(.3, 2.0, "Let me show you."), Caption(6.3, 7.4, "Here is the plan."),
          Caption(7.8, 8.9, "Wait for it!")]


def test_adaptive_timing_keeps_speech_and_payoff():
    levels = np.full(100, -20.)
    levels[23:49] = -90.
    natural = DirectorOptions.from_dict(director_config("natural", DirectorSpec()))
    options = dict(peak=8.5, trigger=8.4, category="funny")
    plan, tone = plan_source_edit(levels, SPEECH, 10, natural, mode="natural", **options)
    assert tone == "funny" and len(plan.segments) == 2
    assert 0 < plan.removed < 2.6 and plan.map_time(8.5) is not None
    assert plan.hook["text"] == "“Wait for it!”" and plan.voiceover is None
    assert all(s.role == "main" for s in plan.segments)
    assert max(z[2] for z in plan.zooms) <= .035
    assert not (plan.sfx or plan.music or plan.shakes or plan.flashes)
    enhanced, _ = plan_source_edit(levels, SPEECH, 10, natural, mode="enhanced", **options)
    assert enhanced.zooms[0][2] > plan.zooms[0][2]
    emotional, tone = plan_source_edit(levels, SPEECH, 10, natural, mode="natural", category="emotional", peak=8.5)
    assert tone == "emotional" and not emotional.removed and not emotional.zooms
    podcast, tone = plan_source_edit(levels, SPEECH, 10, DirectorOptions(preset="podcast"), mode="natural",
                                     category="educational")
    assert tone == "podcast" and podcast.hook["text"] == "“Let me show you.”"


def test_authenticity_uses_source_and_actual_operations():
    levels = np.full(100, -20.)
    plan, style = plan_source_edit(levels, SPEECH, 10, DirectorOptions(hook_text="Guaranteed to win"),
                                   mode="natural", category="gaming", peak=8.5)
    shown = remap_captions(SPEECH, plan)
    job = {"url": REPLAY_URL, "metadata": {"channel": "The Source", "id": "video-id"}}
    clip = {"start": 30., "end": 40., "peak": 38.5, "captions": True, "title": "Wait for it!"}
    report = authenticity_report(job, clip, plan, SPEECH, shown, 10, style=style, hook="New world record")
    assert [report["checks"][k]["status"] for k in ("context", "subtitles", "title", "hook", "thumbnail")] == [
        "pass", "pass", "pass", "review", "review"]
    assert report["provenance"]["channel"] == "The Source"
    assert report["provenance"]["retained_ranges"] == [[30., 40.]]
    assert any("copyrighted" in f for f in report["flags"])
    corrected = [Caption(.3, 2., "Please let me show you.")]
    report = authenticity_report(job, clip, plan, SPEECH, corrected, 10)
    assert report["checks"]["subtitles"]["status"] == "review"
    plan.voiceover = "narration.wav"
    record_operations(report, plan, mixed=False)
    assert not report["operations"]["synthetic_audio"]
    record_operations(report, plan, mixed=True)
    assert report["operations"]["synthetic_voiceover"] and not report["operations"]["generated_footage"]
    assert "Disclose" in report["disclosure"]


def test_mode_defaults_are_independent():
    requested = DirectorSpec(sfx=True, loop=True, music="on", voiceover=True)
    natural = director_config("natural", requested)
    enhanced = director_config("enhanced", requested)
    assert (natural["sfx"], natural["loop"], natural["music"], natural["voiceover"]) == (False, False, "off", True)
    assert (enhanced["sfx"], enhanced["loop"], enhanced["music"], enhanced["voiceover"]) == (True, True, "on", True)
    for mode in ("natural", "enhanced"):
        assert director_config(mode, DirectorSpec())["voiceover"] is False
    assert director_config("full", DirectorSpec()) == DirectorSpec().model_dump()


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("mode-source") / "source.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=480x270:r=24000/1001:d=10",
        "-f", "lavfi", "-i", "aevalsrc=0.15*sin(2*PI*320*t)*not(between(t\\,2.3\\,4.9)):d=10",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path),
    ], check=True)
    return path


@pytest.mark.parametrize("mode", ["natural", "enhanced", "full"])
def test_editing_modes_render_and_persist(monkeypatch, tmp_path, source, mode):
    metadata = {"id": "LIxBN4AHMEE", "title": "The sample stream", "channel": "Source channel", "duration": 10.0}
    moment = dict(start=0., end=10., peak=8.5, score=84, title="Wait for it!", description="The reaction.",
                  reason="Reaction", category="funny", signals={"audio": .9}, transcript="Wait for it!")
    monkeypatch.setattr("clipper_studio.api.fetch_metadata", lambda *a, **k: metadata)
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a, **k: (metadata, []))
    monkeypatch.setattr("clipper_studio.api.download_analysis_media", lambda *a, **k: (source, source))
    monkeypatch.setattr("clipper_studio.api.analyze_media", lambda *a, **k: [moment])
    monkeypatch.setattr("clipper_studio.api.download_section", lambda *a, **k: source)
    monkeypatch.setattr("clipper_studio.api.transcribe", lambda *a, **k: SPEECH)
    monkeypatch.setattr("clipper_studio.api.loudness", loudness)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        request = {"url": REPLAY_URL, "review": False, "facecam": {"enabled": False},
                   "tracking": {"enabled": False}, "director": {"voiceover": False, "loop": mode == "enhanced"}}
        if mode != "full":
            request["editing_mode"] = mode
        assert client.post("/api/jobs", json={**request, "editing_mode": "other"}).status_code == 422
        jid = client.post("/api/jobs", json=request).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["request"]["editing_mode"] == mode
        clip = job["clips"][0]
        assert clip["status"] == "ready", job
        assert clip["editing_mode"] == mode
        report = clip["authenticity"]
        assert report["provenance"]["url"] == REPLAY_URL and report["provenance"]["channel"] == "Source channel"
        assert report["checks"]["subtitles"]["status"] == "pass"
        assert clip["director_result"]["status"] == "used"
        video_path = tmp_path / jid / clip["id"] / "clip.mp4"
        result = probe(video_path, export_details=True)
        v = next(s for s in result["streams"] if s["codec_type"] == "video")
        a = next(s for s in result["streams"] if s["codec_type"] == "audio")
        assert (v["width"], v["height"], v["codec_name"], v["pix_fmt"], a["codec_name"]) == (1080, 1920, "h264", "yuv420p", "aac")
        with video_path.open("rb") as mp4:
            header = mp4.read(128)
        assert b"ftyp" in header and b"moov" in header
        if mode == "full":
            assert v["avg_frame_rate"] == "30/1"
        else:
            assert report["style"] == "funny" and report["checks"]["context"]["status"] == "pass"
            assert not report["operations"]["synthetic_audio"] and not report["operations"]["generated_footage"]
            assert clip["director_result"]["edits"]["cuts"] > 0
            assert report["operations"]["loop_blend"] == (mode == "enhanced")
            assert v["r_frame_rate"] == "24000/1001" and a["sample_rate"] == "48000"
            assert (v["color_space"], v["color_transfer"], v["color_primaries"]) == ("bt709", "bt709", "bt709")
            assert 12_000_000 <= int(v["bit_rate"]) <= 20_000_000
        if mode == "natural":
            base = f"/api/jobs/{jid}/clips/{clip['id']}"
            edit = client.post(base + "/thumbnail", json={"text": "New record"})
            assert edit.status_code == 200
            assert edit.json()["authenticity"]["checks"]["thumbnail"]["status"] == "review"
            monkeypatch.setattr(app.state.pending, "put", lambda *a, **kw: None)
            assert client.post(base + "/render", json={"start": 0, "end": 10, "editing_mode": "enhanced"}).status_code == 202
            queued = client.get(f"/api/jobs/{jid}").json()["clips"][0]
            assert queued["editing_mode"] == "enhanced" and "authenticity" not in queued
            assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200
            restored = client.get(f"/api/jobs/{jid}").json()["clips"][0]
            assert restored["editing_mode"] == "natural" and restored["authenticity"] == edit.json()["authenticity"]
