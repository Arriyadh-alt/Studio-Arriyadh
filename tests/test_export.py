import io
import json
import subprocess
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from clipper_studio.api import create_app
from shorts_clipper.director import DirectorOptions, plan_edit
from shorts_clipper.duration import boundary_options
from shorts_clipper.export import check_export
from shorts_clipper.video import build_ffmpeg_command, make_vertical_clip


def test_presets_follow_completed_speech_and_do_not_add_filler():
    words = [{"start": t, "end": t + 1.2, "word": "context." if t % 6 == 0 else "context"}
             for t in range(12, 61, 2)]
    activity = np.zeros(90)
    activity[12:61] = .4
    options = boundary_options(words, activity, 90, (12., 36.), 30, 27)
    assert "story" in options
    assert 40 <= options["story"]["end"] - options["story"]["start"] <= 55
    assert options["story"]["start"] < 27 < 30 < options["story"]["end"]
    assert options["auto"]["end"] >= 61
    assert "quick" not in options and "extended" not in options
    for bounds in options.values():
        assert not any(w["start"] < bounds["end"] < w["end"] for w in words)

    activity[34:] = 0
    options = boundary_options(words[:11], activity, 90, (12., 48.), 30, 27)
    assert "story" not in options and "extended" not in options


def test_quick_preset_is_not_offered_if_it_would_drop_the_setup():
    words = [{"start": 8, "end": 11.8, "word": "Setup."},
             {"start": 12.5, "end": 13.2, "word": "Context."},
             {"start": 33, "end": 35.8, "word": "Payoff."}]
    activity = np.zeros(60)
    activity[12:36] = .4
    options = boundary_options(words, activity, 60, (12., 36.), 30, 27)
    assert "quick" not in options
    assert options["standard"]["start"] <= 8
    assert options["standard"]["end"] >= 35.8


def test_approved_source_timeline_is_not_recropped_by_the_director():
    levels = np.full(600, -38.)
    levels[188:229] = -8.
    director = DirectorOptions(hook=True, story=True, edits=True)
    planned = plan_edit(levels, [], 60, director, 19, story={}, approved=True)
    assert planned.duration == 60
    assert [(s.start, s.end) for s in planned.segments] == [(0, 60)]


@pytest.mark.parametrize("preset,size", [("preview", "360:640"), ("standard", "720:1280"), ("full_hd", "1080:1920")])
def test_export_commands_use_separate_presets(tmp_path, monkeypatch, preset, size):
    monkeypatch.setattr("shorts_clipper.video.hdr_to_sdr", lambda path: "")
    command = build_ffmpeg_command(tmp_path / "original.mp4", tmp_path / "export.mp4", 0, 60, "blur",
                                   export_fps="24000/1001", export_preset=preset)
    assert f"scale={size}:out_color_matrix=bt709" in command[command.index("-filter_complex") + 1]
    assert command[command.index("-crf") + 1] == ("23" if preset == "preview" else "19")
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-r") + 1] == "24000/1001"


def test_approved_60_seconds_matches_individual_and_batch_download(monkeypatch, tmp_path):
    source = tmp_path / "original.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=0x253040:s=720x1280:r=6:d=62",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=62",
                    "-c:v", "libx264", "-threads", "2", "-preset", "ultrafast", "-c:a", "aac", str(source)],
                   check=True)
    moment = {"start": 0., "end": 30., "peak": 18., "score": 80, "title": "Keep the ending"}
    monkeypatch.setattr("clipper_studio.api.local_analysis_media", lambda *a: (source, source))
    monkeypatch.setattr("clipper_studio.api.analyze_media", lambda *a, **kw: [moment])
    monkeypatch.setattr("clipper_studio.api.transcribe", lambda *a, **kw: [])

    def levels(source, duration):
        signal = np.full(round(duration * 10), -38.)
        signal[188:229] = -8.
        return signal

    monkeypatch.setattr("clipper_studio.api.loudness", levels)
    app = create_app(tmp_path / "data", render_workers=1)
    with TestClient(app) as client:
        upload = client.post("/api/uploads", content=source.read_bytes(), headers={"X-Filename": "original.mp4"})
        assert upload.status_code == 201, upload.text
        job = client.post("/api/jobs", json={"upload": upload.json()["id"], "captions": False,
                          "facecam": {"enabled": False}, "tracking": {"enabled": False},
                          "director": {"loop": False, "music": "off", "sfx": False}}).json()
        base = f"/api/jobs/{job['id']}"
        app.state.pending.join()
        mid = client.get(base).json()["moments"][0]["id"]
        selected = client.patch(f"{base}/moments/{mid}",
                                json={"start": 1, "end": 61, "duration_preset": "extended", "status": "selected"})
        assert selected.status_code == 200, selected.text
        queued = client.post(base + "/moments/render", json={"export_preset": "standard"})
        assert queued.status_code == 202, queued.text
        cid = queued.json()["clips"][0]
        app.state.pending.join()
        clip = client.get(base).json()["clips"][0]
        assert clip["status"] == "ready", clip.get("error")
        assert clip["director_result"]["status"] == "used"
        assert clip["export_qc"]["status"] == "passed", clip["export_qc"]
        assert clip["export_qc"]["editorial_status"] == "needs_review"
        assert client.post(base + "/rights", json={"basis": "owned", "confirmed": True}).status_code == 200
        preview = client.get(clip["file_url"])
        download = client.get(clip["download_url"])
        assert preview.status_code == download.status_code == 200
        assert preview.content == download.content
        saved = tmp_path / "download.mp4"
        saved.write_bytes(download.content)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                                    "-of", "json", str(saved)]))
        assert float(probe["format"]["duration"]) == pytest.approx(60., abs=.05)
        video, audio = probe["streams"]
        assert (video["width"], video["height"], video["avg_frame_rate"]) == (720, 1280, "6/1")
        assert (video["codec_name"], video["pix_fmt"], video["color_space"]) == ("h264", "yuv420p", "bt709")
        assert (audio["codec_name"], audio["sample_rate"]) == ("aac", "48000")
        assert client.patch(f"{base}/clips/{cid}", json={"kept": True}).status_code == 200
        bundle = client.get(base + "/download-kept")
        assert bundle.status_code == 200
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert archive.read("highlight-01.mp4") == download.content


def test_export_preserves_a_delayed_audio_onset(tmp_path):
    source, output = tmp_path / "delayed.mp4", tmp_path / "preview.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=360x640:r=24:d=4",
                    "-itsoffset", "1", "-f", "lavfi", "-i", "sine=d=2", "-c:v", "libx264",
                    "-c:a", "aac", str(source)], check=True)
    make_vertical_clip(source, output, 0, 4, export_preset="preview", use_hardware=False)
    result = check_export(output, "preview", 4, "24/1", None)
    assert next(c for c in result["checks"] if c["check"] == "Audio alignment")["status"] == "pass"
    samples = np.frombuffer(subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output),
                                                   "-vn", "-f", "f32le", "-"]), dtype="<f4")
    assert np.max(np.abs(samples[:24000])) < .001
    assert np.max(np.abs(samples[57600:72000])) > .05
