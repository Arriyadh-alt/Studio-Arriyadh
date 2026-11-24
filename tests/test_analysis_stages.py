import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from clipper_studio.api import create_app
from clipper_studio.chat import ChatReplayError
from clipper_studio.media_analysis import analyze_ranges, media_ranges
from test_api import REPLAY_URL, media, saved_job


def test_restart_during_segment_scan_resumes_from_disk(tmp_path, monkeypatch):
    duration = 3800
    ranges = media_ranges(duration)
    assert ranges[0].end > ranges[1].start
    assert media_ranges(13 * 3600)[-1].end == 13 * 3600
    blocked, release = threading.Event(), threading.Event()
    preflights, chats, downloads, audio_scans = [], [], [], []

    def metadata(*args):
        preflights.append(args)
        return {"duration": duration}

    def chat(*args):
        chats.append(args)
        return {"duration": duration}, []

    def download(url, folder, start, end, **options):
        assert options["analysis"]
        downloads.append(start)
        if start == ranges[1].start and downloads.count(start) == 1:
            blocked.set()
            assert release.wait(10)
        output = folder / "source.mkv"
        output.write_bytes(b"media")
        return output

    def scan(path, length):
        start = int(path.parent.name.split("-")[0])
        audio_scans.append(start)
        levels = np.full(round(length), -80.)
        energy = np.zeros_like(levels)
        for timestamp, strength in [(902, 1), (1850, .8)]:
            peak = timestamp - start
            if 0 <= peak < len(energy):
                energy[peak] = strength
        return levels, energy, np.zeros_like(energy)

    monkeypatch.setattr("clipper_studio.api.fetch_metadata", metadata)
    monkeypatch.setattr("clipper_studio.api.fetch_chat", chat)
    monkeypatch.setattr("clipper_studio.api.download_section", download)
    monkeypatch.setattr("clipper_studio.media_analysis._audio", scan)
    monkeypatch.setattr("clipper_studio.media_analysis._visual", lambda p, d, n: np.zeros(n))
    monkeypatch.setattr("clipper_studio.media_analysis._transcribe", lambda path, lo, hi, lang: [
        {"word": "no way", "start": lo + 5, "end": lo + 6}])

    app = create_app(tmp_path)
    try:
        with TestClient(app) as client:
            jid = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
            assert blocked.wait(5)
            state = client.get(f"/api/jobs/{jid}").json()
            assert state["stages"]["source_validation"]["status"] == "completed"
            assert state["stages"]["audio_analysis"]["status"] == "processing"
            assert state["stages"]["audio_analysis"]["progress"] > 0
            assert (tmp_path / jid / "checkpoints" / "media" / ranges[0].key / "source.mkv").exists()
    finally:
        release.set()
    app.state.pending.join()
    with TestClient(create_app(tmp_path)) as client:
        client.app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed"
        assert len(preflights) == len(chats) == 1
        assert downloads.count(ranges[0].start) == audio_scans.count(ranges[0].start) == 1
        assert audio_scans == [segment.start for segment in ranges]
        assert job["moments"][0]["peak"] == 902
        assert job["moments"][0]["start"] == 884
        assert job["moments"][0]["transcript"] == "no way"
        assert job["moments"][1]["peak"] == 1850
        assert job["moments"][1]["start"] == 1832
        assert job["moments"][1]["transcript"] == "no way"
        stages = job["stages"]
        for name in ("source_validation", "audio_analysis", "video_analysis", "transcript_generation",
                     "chat_analysis", "viral_moment_detection"):
            assert stages[name]["status"] == "completed"
            assert stages[name]["progress"] == 100 and stages[name]["artifacts"]


@pytest.mark.parametrize("fail_download", [False, True])
def test_retry_failed_video_reuses_audio_and_preserves_existing_moment(tmp_path, monkeypatch, fail_download):
    downloads, audios, videos, transcript = [], [], [], []
    monkeypatch.setattr("clipper_studio.api.fetch_metadata", lambda *a: {"duration": 240})

    def chat(*args):
        raise ChatReplayError("No replay chat")

    def download(url, folder, **kwargs):
        downloads.append(kwargs.get("prior_audio"))
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "source.mkv"
        path.touch()
        if fail_download and len(downloads) == 1:
            return path, None
        return path, path

    def audio(path, duration):
        audios.append(path)
        energy = np.zeros(240)
        energy[110] = 1
        return np.full(240, -80.), energy, np.zeros(240)

    def visual(path, duration, count):
        videos.append(path)
        if not fail_download and len(videos) == 1:
            raise RuntimeError("video decode failed")
        return np.zeros(count)

    def speak(*args):
        transcript.append(args)
        return []

    monkeypatch.setattr("clipper_studio.api.fetch_chat", chat)
    monkeypatch.setattr("clipper_studio.api.download_analysis_media", download)
    monkeypatch.setattr("shorts_clipper.moments._audio", audio)
    monkeypatch.setattr("shorts_clipper.moments._visual", visual)
    monkeypatch.setattr("shorts_clipper.moments._transcribe", speak)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
        base = f"/api/jobs/{jid}"
        app.state.pending.join()
        original = client.get(base).json()
        assert original["status"] == "completed"
        assert original["stages"]["video_analysis"]["status"] == "failed"
        assert client.post(f"{base}/stages/audio_analysis/retry").status_code == 409
        assert client.post(f"{base}/stages/video_analysis/retry").status_code == 202
        app.state.pending.join()
        job = client.get(base).json()
        assert job["stages"]["video_analysis"]["status"] == "completed"
        assert job["stages"]["video_analysis"]["attempts"] == 2
        assert job["stages"]["audio_analysis"] == original["stages"]["audio_analysis"]
        assert len(audios) == len(transcript) == 1
        assert len(videos) == (1 if fail_download else 2)
        assert downloads == ([None, audios[0]] if fail_download else [None])
        assert job["moments"] == original["moments"]


def test_resume_failed_analysis_without_revalidating_source(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("clipper_studio.api.fetch_metadata", lambda *a: {"duration": 400})

    def download(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise TimeoutError("The media connection timed out")
        return None, None

    def no_chat(*args):
        raise ChatReplayError("No chat replay")

    monkeypatch.setattr("clipper_studio.api.download_analysis_media", download)
    monkeypatch.setattr("clipper_studio.api.fetch_chat", no_chat)
    monkeypatch.setattr("clipper_studio.api.analyze_media", lambda *a, **k: [])
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
        base = f"/api/jobs/{jid}"
        app.state.pending.join()
        failed = client.get(base).json()
        assert failed["status"] == "failed"
        monkeypatch.setattr("clipper_studio.api.fetch_metadata", lambda *a: None)
        assert client.post(f"{base}/resume").status_code == 202
        app.state.pending.join()
        job = client.get(base).json()
        assert job["status"] == "completed"
        assert job["stages"]["source_validation"] == failed["stages"]["source_validation"]
        assert len(calls) == 2


def test_segmented_podcast_keeps_question_and_answer_in_one_conversation(monkeypatch):
    duration = 1000
    first, second = media_ranges(duration)
    calls = []
    monkeypatch.setattr("clipper_studio.media_analysis._audio",
                        lambda path, length: tuple(np.zeros(round(length)) for _ in range(3)))
    monkeypatch.setattr("clipper_studio.media_analysis.voice_features",
                        lambda path, length: np.zeros((round(length * 4), 25)))

    def transcribe(segment, language):
        return [{"word": word, "start": start - segment.start, "end": start - segment.start + .3}
                for word, start in [("Why?", 899), ("Because", 901)]]

    def podcast(audio, video, length, **kwargs):
        calls.append((length, kwargs["transcribe_fn"](audio, None), kwargs["features_fn"](audio, length).shape))
        return []

    monkeypatch.setattr("clipper_studio.media_analysis.transcribe_all", transcribe)
    monkeypatch.setattr("clipper_studio.media_analysis.analyze_podcast", podcast)
    assert second.start < 899 < 901 < first.end
    analyze_ranges(duration, 8, 18, None, fetch=lambda segment: segment,
                   checkpoint=lambda stage, key, work: work(), expect=lambda *a: None,
                   progress=lambda *a: None, check=lambda: None, podcast=True)
    assert len(calls) == 1
    assert calls[0][0] == duration
    assert [(w["word"], w["start"]) for w in calls[0][1]] == [("Why?", 899), ("Because", 901)]
    assert calls[0][2] == (4000, 25)


def test_retry_final_rendering_only_queues_failed_clips(tmp_path, monkeypatch, media):
    app = create_app(tmp_path)
    job = saved_job("renderjob", "running")
    ready = job["clips"][0]
    job["clips"].append(dict(ready, id="broken", start=100, end=130, status="queued"))
    app.state.store.save(job)
    ready_file = tmp_path / "renderjob" / ready["id"] / "clip.mp4"
    ready_file.parent.mkdir()
    ready_file.write_bytes(b"existing export")
    render_calls = []

    def convert(source, output, *args, **kwargs):
        render_calls.append(output)
        if len(render_calls) <= 3:
            raise RuntimeError("Render temporarily unavailable")
        output.write_bytes(b"new export")

    monkeypatch.setattr("clipper_studio.api.make_vertical_clip", convert)
    monkeypatch.setattr("clipper_studio.api.time.sleep", lambda *a: None)
    with TestClient(app) as client:
        app.state.pending.join()
        base = "/api/jobs/renderjob"
        job = client.get(base).json()
        assert job["status"] == "completed_with_errors"
        assert job["stages"]["final_rendering"]["status"] == "partially_completed"
        assert client.post(f"{base}/stages/final_rendering/retry").status_code == 202
        app.state.pending.join()
        job = client.get(base).json()
        assert job["status"] == "completed"
        stage = job["stages"]["final_rendering"]
        assert stage["status"] == "completed" and stage["attempts"] == 4
        assert stage["artifacts"] == ["first/clip.mp4", "broken/clip.mp4"]
        assert ready_file.read_bytes() == b"existing export"
        assert len(render_calls) == 4
