import io
import threading
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from clipper_studio.api import JobRequest, Store, create_app, title_slug
from clipper_studio.chat import ChatReplayError, Message
from shorts_clipper.youtube import YouTubeAccessError, blocking_message


REPLAY_URL = "https://www.youtube.com/watch?v=LIxBN4AHMEE"


def test_caption_request_defaults():
    from clipper_studio.api import RenderRequest
    for request in (JobRequest(url=REPLAY_URL), RenderRequest(start=0, end=10)):
        assert request.captions is True
        assert request.language == "auto" and request.caption_style is None
    with pytest.raises(ValidationError):
        RenderRequest(start=0, end=10, language="not-a-code")


def test_caption_fonts_served_for_preview(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        for font in ('Montserrat-Black.ttf','Anton-Regular.ttf','BebasNeue-Regular.ttf'):
            response = client.get('/caption-fonts/' + font)
            assert response.status_code == 200 and response.content[:4] == b'\x00\x01\x00\x00'
        assert client.get('/caption-fonts/missing.ttf').status_code == 404


@pytest.mark.parametrize("legacy", [False, True])
def test_caption_options_worker_and_rerender(monkeypatch, tmp_path, media, legacy):
    from shorts_clipper.captions import Caption, Transcript
    calls, styles = [], []
    def speech(source, **kwargs):
        calls.append(kwargs)
        return Transcript([Caption(0, 1, "Halo dunia!")], "id")
    def serialize(captions, style):
        styles.append(style)
        return "ASS"
    monkeypatch.setattr("clipper_studio.api.transcribe", speech)
    monkeypatch.setattr("clipper_studio.api.to_ass", serialize)
    monkeypatch.setattr("clipper_studio.api.generate_thumbnail", lambda video, output, text, peak: output.write_bytes(text.encode()))
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration":600}, []))
    monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([{"start":10,"end":40,"peak":10.5}], []))
    app = create_app(tmp_path, render_workers=1)
    if legacy:
        job = saved_job("old", "running")
        job["clips"][0].update(status="queued", captions=True, peak=10.5)
        app.state.store.save(job)
    with TestClient(app) as client:
        jid = "old" if legacy else client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":REPLAY_URL,
            "language":"id", "caption_style":{"preset":"minimal","size":100}}).json()["id"]
        app.state.pending.join()
        clip = client.get(f"/api/jobs/{jid}").json()["clips"][0]
        assert clip["status"] == "ready" and clip["caption_language"] == "id"
        assert clip["thumbnail_hook"] == "HALO DUNIA" and clip["thumbnail_version"]
        assert client.get(clip["thumbnail_url"]).content == b"HALO DUNIA"
        assert calls[-1]["language"] == (None if legacy else "id")
        assert styles[-1].preset == ("pop" if legacy else "minimal")
        if not legacy:
            assert clip["language"] == "id" and clip["caption_style"]["size"] == 100
        base = f"/api/jobs/{jid}/clips/{clip['id']}"
        response = client.post(base + "/render", json={"start":12,"end":35,"language":"en",
            "caption_style":{"preset":"pop","position":"top","size":999,"highlight_color":"#123456"}})
        assert response.status_code == 202
        app.state.pending.join()
        clip = client.get(f"/api/jobs/{jid}").json()["clips"][0]
        assert clip["status"] == "ready" and clip["language"] == "en"
        assert calls[-1]["language"] == "en"
        assert styles[-1].position == "top" and styles[-1].size == 130
        assert styles[-1].highlight_color == "#123456"
        assert client.post(base + "/render", json={"start":12,"end":35,"captions":False}).status_code == 202
        app.state.pending.join()
        assert client.get(f"/api/jobs/{jid}").json()["clips"][0]["caption_language"] is None


@pytest.mark.parametrize("phase", ["metadata", "chat", "section"])
@pytest.mark.parametrize("configured", [False, True])
def test_cookie_blocking_is_a_friendly_job_error(monkeypatch, tmp_path, phase, configured):
    monkeypatch.delenv("CLIPPER_YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("YOUTUBE_COOKIES", raising=False)
    if configured:
        monkeypatch.setenv("YOUTUBE_COOKIES", "mock cookies")
    calls = []

    def fail(*args):
        calls.append(phase)
        error = YouTubeAccessError if phase == "section" else ChatReplayError
        raise error(blocking_message("HTTP Error 429: Too Many Requests"))

    if phase == "metadata":
        monkeypatch.setattr("clipper_studio.api.fetch_metadata", fail)
    elif phase == "chat":
        monkeypatch.setattr("clipper_studio.api.fetch_chat", fail)
    else:
        monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration": 600}, []))
        monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: (
            [{"start": 10, "end": 40}, {"start": 100, "end": 130}], []))
        monkeypatch.setattr("clipper_studio.api.download_section", fail)
    app = create_app(tmp_path, render_workers=1)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url": REPLAY_URL}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed"
        assert job["error"].startswith("YouTube is blocking this server")
        if configured:
            assert "configured but may be rejected or expired" in job["error"]
        else:
            assert "No cookies configured" in job["error"]
        assert all(clip["status"] == "failed" for clip in job["clips"])
        assert calls == [phase]


@pytest.fixture(autouse=True)
def metadata(monkeypatch):
    def fetch(*args):
        return {"duration": 600, "was_live": True, "subtitles": {"live_chat": [{}]}}
    monkeypatch.setattr("clipper_studio.api.fetch_metadata", fetch)
    return fetch


@pytest.fixture
def media(monkeypatch):
    monkeypatch.setattr("clipper_studio.api.download_section", lambda url, directory, *a: directory / "source.mp4")
    monkeypatch.setattr("clipper_studio.api.video_duration", lambda p: 30)
    monkeypatch.setattr("clipper_studio.api.transcribe", lambda p: [])
    monkeypatch.setattr("clipper_studio.api.make_vertical_clip", lambda source, output, *a, **kw: output.write_bytes(b"video"))
    monkeypatch.setattr("clipper_studio.api.subprocess.run", lambda command, **k: Path(command[-1]).write_bytes(b"jpg"))


def wait_job(client, jid, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{jid}").json()
        if predicate(job):
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job did not reach expected state: {job}")


def saved_job(jid, status="completed"):
    return {"id": jid, "url": "https://youtu.be/ready", "status": status,
            "request": {"mode": "crop", "captions": False, "clips": 2, "lead": 18},
            "metadata": {"duration": 600}, "analysis_complete": True, "progress": 1,
            "clips": [{"id": "first", "status": "ready", "start": 10, "end": 40,
                       "mode": "crop", "captions": False, "kept": True}]}


def test_pipeline_and_file_endpoints(monkeypatch, tmp_path):
    stages = []
    save = Store.save
    def record_stage(store, job):
        stages.append(job['stage'])
        save(store, job)
    monkeypatch.setattr(Store, "save", record_stage)
    monkeypatch.setattr("clipper_studio.api.time.sleep", lambda seconds: None)
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: (
        {"title": "🔴 Test!", "duration": 600, "thumbnail": None},
        [Message(t, str(i), "W") for t in (100, 300, 500) for i in range(20)]))
    download_attempts = []
    def section(url, directory, start, end):
        download_attempts.append(start)
        if len(download_attempts) == 1:
            raise RuntimeError("Transient download failure")
        path = directory / "source.mp4"
        path.write_bytes(b"source")
        return path
    monkeypatch.setattr("clipper_studio.api.download_section", section)
    monkeypatch.setattr("clipper_studio.api.video_duration", lambda p: 40)
    monkeypatch.setattr("clipper_studio.api.transcribe", lambda p, **kwargs: [])
    render_attempts = []
    def convert(source, output, *args, watermark):
        assert watermark.path.is_file()
        render_attempts.append(output)
        if len(render_attempts) == 1:
            raise RuntimeError("Transient render failure")
        output.write_bytes(b"0123456789")
    monkeypatch.setattr("clipper_studio.api.make_vertical_clip", convert)
    def thumbnail(command, **kwargs):
        from pathlib import Path
        Path(command[-1]).write_bytes(b"jpg")
    monkeypatch.setattr("clipper_studio.api.subprocess.run", thumbnail)
    app = create_app(tmp_path, render_workers=1)
    with TestClient(app) as client:
        response = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":REPLAY_URL, "clips":3})
        assert response.status_code == 202
        jid = response.json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed"
        assert len(job["clips"]) == 3
        for number in range(1, 4):
            for stage in ("Downloading", "Transcribing", "Rendering"):
                assert f"Clip {number} of 3 · {stage}" in stages
        assert [c["number"] for c in job["clips"]] == [1, 2, 3]
        assert all(c["status"] == "ready" for c in job["clips"])
        clip = job["clips"][0]
        assert clip["attempt"] == 3
        assert len(download_attempts) == 5
        assert job["created_at"]
        assert client.get(f"/api/jobs/{jid}/download-kept").status_code == 409
        assert client.patch(f"/api/jobs/{jid}/clips/{clip['id']}", json={"kept":True}).json()["kept"]
        archive = client.get(f"/api/jobs/{jid}/download-kept")
        assert archive.status_code == 200
        assert 'filename="test-kept-clips.zip"' in archive.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
            assert bundle.read("highlight-01.mp4") == b"0123456789"
            assert bundle.read("highlight-01.jpg") == b"jpg"
        assert client.get(f"/api/jobs/{jid}").json()["clips"][0]["kept"]
        response = client.get(clip["file_url"], headers={"Range":"bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert response.headers["content-range"] == "bytes 2-5/10"
        assert 'filename="test-highlight-01.mp4"' in client.get(clip["download_url"]).headers["content-disposition"]
        assert client.get(clip["thumbnail_url"]).content == b"jpg"
        base = f"/api/jobs/{jid}/clips/{clip['id']}"
        assert client.post(base + "/render", json={"start":10,"end":90}).status_code == 422
        assert client.post(base + "/render", json={"start":10,"end":40,"mode":"blur","captions":False}).status_code == 202
        app.state.pending.join()
        assert client.get(f"/api/jobs/{jid}").json()["clips"][0]["start"] == 10
        def render_fail(*args, **kwargs):
            raise RuntimeError("ffmpeg failed")
        monkeypatch.setattr("clipper_studio.api.make_vertical_clip", render_fail)
        assert client.post(base + "/render", json={"start":10,"end":40}).status_code == 202
        app.state.pending.join()
        failed = client.get(f"/api/jobs/{jid}").json()
        assert failed["status"] == "completed_with_errors"
        assert failed["clips"][0]["error"] == "ffmpeg failed"
        assert failed["clips"][0]["attempt"] == 3
        assert client.get(clip["file_url"]).status_code == 409
        assert client.delete(base).status_code == 200
        assert [c["number"] for c in client.get(f"/api/jobs/{jid}").json()["clips"]] == [2, 3]
        second = job["clips"][1]
        client.patch(f"/api/jobs/{jid}/clips/{second['id']}", json={"kept":True})
        with zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{jid}/download-kept").content)) as bundle:
            assert bundle.namelist() == ["highlight-02.mp4", "highlight-02.jpg"]
        assert client.get(clip["file_url"]).status_code == 404
        assert len(client.get("/api/jobs").json()) == 1


def test_thumbnail_regeneration_preserves_video_and_auto_hook(tmp_path, monkeypatch, media):
    store = Store(tmp_path)
    job = saved_job("thumbjob")
    job["metadata"]["title"] = "Final round showdown"
    store.save(job)
    directory = tmp_path / job["id"] / "first"
    directory.mkdir()
    (directory / "clip.mp4").write_bytes(b"video")

    def make_thumbnail(video, output, text, peak):
        assert video.read_bytes() == b"video"
        output.write_bytes(text.encode())

    monkeypatch.setattr("clipper_studio.api.generate_thumbnail", make_thumbnail)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        clip_url = "/api/jobs/thumbjob/clips/first"
        thumbnail_url = clip_url + "/thumbnail"
        edited = client.post(thumbnail_url, json={"text": "  tidak   mungkin!  "})
        assert edited.status_code == 200
        assert edited.json()["thumbnail_text"] == "TIDAK MUNGKIN!"
        jpg = client.get(thumbnail_url + "?download=true")
        assert jpg.headers["content-type"] == "image/jpeg"
        assert jpg.headers["content-disposition"] == 'attachment; filename="final-round-showdown-highlight-01.jpg"'
        assert jpg.content == b"TIDAK MUNGKIN!"
        assert client.get(clip_url + "/video").content == b"video"
        assert client.post(thumbnail_url, json={"text": "x" * 65}).status_code == 422

        assert client.post(clip_url + "/render", json={"start": 10, "end": 40, "captions": False}).status_code == 202
        app.state.pending.join()
        assert client.get(thumbnail_url).content == b"TIDAK MUNGKIN!"
        reset = client.post(thumbnail_url)
        assert reset.status_code == 200
        assert reset.json()["thumbnail_text"] is None
        assert reset.json()["thumbnail_version"] != edited.json()["thumbnail_version"]
        assert client.get(thumbnail_url).content == b"FINAL ROUND SHOWDOWN"

        def unavailable(*args):
            raise ValueError("No frames")

        monkeypatch.setattr("clipper_studio.api.generate_thumbnail", unavailable)
        assert client.post(thumbnail_url).status_code == 500
        assert client.get(thumbnail_url).content == b"FINAL ROUND SHOWDOWN"


def test_errors_and_empty(monkeypatch, tmp_path, metadata):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/jobs").json() == []
        assert client.get("/").status_code == 200
        assert "Clipper Studio" in client.get("/").text
        assert client.get("/manifest.webmanifest").json()["icons"][0]["sizes"] == "192x192"
        assert client.get("/icon-512.png").headers["content-type"] == "image/png"
        assert client.get("/api/jobs/missing").status_code == 404
        assert client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":"https://example.org"}).status_code == 422
        assert client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":REPLAY_URL, "lead":10}).status_code == 422
        def fail(*args):
            raise ChatReplayError("no live-chat replay")
        monkeypatch.setattr("clipper_studio.api.fetch_metadata", fail)
        jid = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":"https://www.youtube.com/watch?v=LIxBN4AHMEE"}).json()["id"]
        app.state.pending.join()
        assert client.get(f"/api/jobs/{jid}").json()["error"] == "no live-chat replay"
        monkeypatch.setattr("clipper_studio.api.fetch_metadata", metadata)
        monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration":100}, []))
        jid = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":REPLAY_URL}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed" and job["clips"] == []


def test_restart_and_busy_conflicts(tmp_path, media):
    app = create_app(tmp_path)
    client = TestClient(app)
    jid = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":REPLAY_URL}).json()["id"]
    store = app.state.store
    job = store.get(jid)
    job.update(saved_job(jid, "running"))
    job.pop("analysis_complete")
    job["clips"][0].update(id="testclip", status="rendering")
    job["clips"].extend([{"id": "ready", "status": "ready", "kept": True},
                         {"id": "failed", "status": "failed", "error": "old error"}])
    store.save(job)
    base = f"/api/jobs/{jid}/clips/testclip"
    assert client.delete(base).status_code == 409
    assert client.post(base + "/render", json={"start":10,"end":40}).status_code == 409
    restarted = create_app(tmp_path)
    with TestClient(restarted) as client:
        restarted.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed_with_errors"
        assert [c["status"] for c in job["clips"]] == ["ready", "ready", "failed"]
        assert job["clips"][1] == {"id": "ready", "status": "ready", "kept": True,
                                   "number": 2, "queue_position": None,
                                   "facecam": {"enabled": True, "position": "auto", "size": 0.42, "box": None},
                                   "tracking": {"enabled": True, "layout": "auto", "follow": "auto", "x": None, "target": None},
                                   "director": {"enabled": False, "hook": True, "edits": True, "sfx": True,
                                                "music": "auto", "story": True, "voiceover": False,
                                                "hook_text": None, "voice": "auto", "delivery": "auto", "loop": True}}
        assert job["clips"][2]["error"] == "old error"


@pytest.mark.parametrize("url", [
    " https://youtu.be/LIxBN4AHMEE?si=abc&t=23 ",
    "https://m.youtube.com/watch?v=LIxBN4AHMEE&list=PL123&index=3",
    "https://youtube.com/live/LIxBN4AHMEE?feature=share",
    REPLAY_URL,
])
def test_normalize_replay_url(url):
    assert JobRequest(url=url).url == REPLAY_URL


@pytest.mark.parametrize("url", [
    "this is not a link", "https://www.youtube.com/@NASA/streams",
    "https://www.youtube.com/playlist?list=PL123", "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/watch?v=LIxBN4AHMEE&v=dQw4w9WgXcQ",
    "https://www.youtube.com:invalid/watch?v=LIxBN4AHMEE",
    "https://user@www.youtube.com/watch?v=LIxBN4AHMEE",
    "https://www.youtube.com.example.org/watch?v=LIxBN4AHMEE",
    "https://youtu.be/LIxBN4AHMEE/extra", "https://[youtube.com/watch?v=LIxBN4AHMEE",
])
def test_reject_non_video_urls(url):
    with pytest.raises(ValidationError, match="Paste a YouTube video link"):
        JobRequest(url=url)


def test_queued_job_survives_new_client_and_worker_continues(monkeypatch, tmp_path):
    started, release = threading.Event(), threading.Event()
    calls = []

    def fetch(url, directory, info):
        calls.append(url)
        if url == REPLAY_URL:
            started.set()
            assert release.wait(5)
            raise ChatReplayError("This livestream has no live-chat replay.")
        return {"duration": 11949, "title": "Long stream"}, [Message(11940, "user", "W")]

    monkeypatch.setattr("clipper_studio.api.fetch_chat", fetch)
    app = create_app(tmp_path, analysis_workers=1)
    with TestClient(app) as client:
        try:
            first = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":"https://youtu.be/LIxBN4AHMEE"}).json()["id"]
            assert started.wait(5)
            second = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":"https://m.youtube.com/watch?v=puMjBYns9L0"}).json()["id"]
            assert first != second
            refreshed = TestClient(app)
            assert refreshed.get(f"/api/jobs/{first}").json()["status"] == "running"
            queued = wait_job(refreshed, second, lambda job: job["metadata"] is not None and job["queue_position"] == 1)
            assert queued["status"] == "queued"
            assert queued["stage"] == "Queued (position 1)"
            recent = {job["id"]: job for job in refreshed.get("/api/jobs?summary=true").json()}
            assert recent[second]["queue_position"] == 1
            assert "clips" not in recent[second] and "activity" not in recent[second]
            assert calls == [REPLAY_URL]
        finally:
            release.set()
        app.state.pending.join()
        assert client.get(f"/api/jobs/{first}").json()["error"] == "This livestream has no live-chat replay."
        job = client.get(f"/api/jobs/{second}").json()
        assert job["status"] == "completed"
        assert job["activity"][-1]["time"] == 11945
        recent = {job["id"]: job for job in client.get("/api/jobs?summary=true").json()}
        assert recent[second]["clip_count"] == 0 and "activity" not in recent[second]
        assert calls == [REPLAY_URL, "https://www.youtube.com/watch?v=puMjBYns9L0"]


def test_delete_queued_during_autopilot(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration":600}, []))
    monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: (
        [{"start":10,"end":40}, {"start":100,"end":130}], []))
    def section(*args):
        entered.set()
        assert release.wait(5)
        return args[1] / "source.mp4"
    monkeypatch.setattr("clipper_studio.api.download_section", section)
    monkeypatch.setattr("clipper_studio.api.video_duration", lambda p: 30)
    monkeypatch.setattr("clipper_studio.api.make_vertical_clip", lambda source, output, *a, **kw: output.write_bytes(b"video"))
    monkeypatch.setattr("clipper_studio.api.subprocess.run", lambda command, **k: Path(command[-1]).write_bytes(b"jpg"))
    app = create_app(tmp_path, render_workers=1)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":REPLAY_URL, "captions":False}).json()["id"]
        try:
            assert entered.wait(5)
            job = client.get(f"/api/jobs/{jid}").json()
            assert job["stage"] == "Clip 1 of 2 · Downloading"
            first, second = job["clips"]
            base = f"/api/jobs/{jid}/clips/"
            assert client.post(base + first["id"] + "/render", json={"start":10,"end":40}).status_code == 409
            assert client.delete(base + first["id"]).status_code == 409
            assert client.delete(base + second["id"]).status_code == 200
        finally:
            release.set()
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed"
        assert [c["number"] for c in job["clips"]] == [1]


def test_rerender_ready_while_running(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    jid = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url":REPLAY_URL}).json()["id"]
    job = app.state.store.get(jid)
    job.update(status="running", progress=0.6, metadata={"duration":100},
               clips=[{"id":"ready", "status":"ready"}, {"id":"rendering", "status":"rendering"}])
    app.state.store.save(job)
    base = f"/api/jobs/{jid}/clips/"
    assert client.post(base + "ready/render", json={"start":10,"end":40}).status_code == 202
    assert client.post(base + "rendering/render", json={"start":10,"end":40}).status_code == 409
    saved = app.state.store.get(jid)
    assert saved["status"] == "running" and saved["progress"] == 0.6


def test_slug_and_completed_restart(tmp_path):
    assert title_slug({"metadata":{"title":"🔴"}}) == "clipper-studio"
    assert title_slug({"metadata":{"title":"A" * 50}}) == "a" * 40
    assert title_slug({"metadata":None}) == "clipper-studio"
    app = create_app(tmp_path)
    app.state.store.save({"id":"done", "status":"completed", "clips":[
        {"id":"first", "status":"ready"}, {"id":"second", "status":"ready"}]})
    with TestClient(app) as client:
        assert client.get("/api/jobs/done").json()["status"] == "completed"
        assert client.get("/api/jobs?summary=true").json()[0]["clip_count"] == 2
        assert [c["number"] for c in client.get("/api/jobs/done").json()["clips"]] == [1, 2]
        assert client.delete("/api/jobs/done/clips/first").status_code == 200
        assert client.get("/api/jobs/done").json()["clips"][0]["number"] == 2
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/jobs/done").json()["clips"][0]["number"] == 2


@pytest.mark.parametrize("late_error", [False, True])
def test_analysis_does_not_block_rerender_or_no_chat(monkeypatch, tmp_path, metadata, media, late_error):
    entered, release = threading.Event(), threading.Event()
    fetched = []

    def chat(url, *args):
        fetched.append(url)
        entered.set()
        assert release.wait(5)
        if late_error:
            raise ChatReplayError("Late download error after cancellation")
        return {"duration": 600}, []

    def preflight(url):
        if url.endswith("bbbbbbbbbbb"):
            raise ChatReplayError("This video has no live-chat replay (disabled, unavailable, or not a livestream VOD).")
        return metadata(url)

    monkeypatch.setattr("clipper_studio.api.fetch_chat", chat)
    monkeypatch.setattr("clipper_studio.api.fetch_metadata", preflight)
    app = create_app(tmp_path, analysis_workers=1, render_workers=1, metadata_workers=1)
    app.state.store.save(saved_job("readyjob"))
    with TestClient(app) as client:
        slow = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa"}).json()["id"]
        try:
            assert entered.wait(5)
            nochat = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url": "https://www.youtube.com/watch?v=bbbbbbbbbbb"}).json()["id"]
            failed = wait_job(client, nochat, lambda job: job["status"] == "failed")
            assert "no live-chat replay" in failed["error"]
            assert client.post("/api/jobs/readyjob/clips/first/render",
                               json={"start": 11, "end": 40, "captions": False}).status_code == 202
            ready = wait_job(client, "readyjob", lambda job: job["status"] == "completed")
            assert ready["clips"][0]["rendered_at"]
            assert not release.is_set()
            queued = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url": "https://www.youtube.com/watch?v=ccccccccccc"}).json()["id"]
            waiting = wait_job(client, queued, lambda job: job["metadata"] is not None and job["queue_position"] == 1)
            assert waiting["stage"] == "Queued (position 1)"
            assert client.post(f"/api/jobs/{queued}/cancel").status_code == 200
            assert client.post(f"/api/jobs/{slow}/cancel").status_code == 200
        finally:
            release.set()
        app.state.pending.join()
        assert fetched == ["https://www.youtube.com/watch?v=aaaaaaaaaaa"]
        assert client.get(f"/api/jobs/{slow}").json()["status"] == "cancelled"
        assert client.get(f"/api/jobs/{queued}").json()["status"] == "cancelled"
        assert client.post("/api/jobs/missing/cancel").status_code == 404
        assert client.post("/api/jobs/readyjob/cancel").status_code == 409


def test_cancel_render_and_rerender_ignores_stale_work(monkeypatch, tmp_path, media):
    entered, release = threading.Event(), threading.Event()
    downloads, conversions = [], []

    def section(url, directory, start, end):
        downloads.append(start)
        if len(downloads) == 1:
            entered.set()
            assert release.wait(5)
        return directory / "source.mp4"

    def convert(source, output, *args, **kwargs):
        conversions.append(source)
        output.write_bytes(b"new video")

    monkeypatch.setattr("clipper_studio.api.download_section", section)
    monkeypatch.setattr("clipper_studio.api.make_vertical_clip", convert)
    app = create_app(tmp_path, render_workers=1)
    job = saved_job("cancelrender", "running")
    job["clips"][0]["status"] = "queued"
    job["clips"].append(dict(job["clips"][0], id="second", start=100, end=130))
    app.state.store.save(job)
    with TestClient(app) as client:
        try:
            assert entered.wait(5)
            waiting = client.get("/api/jobs/cancelrender").json()
            assert waiting["clips"][1]["queue_position"] == 1
            assert client.post("/api/jobs/cancelrender/cancel").status_code == 200
            cancelled = client.get("/api/jobs/cancelrender").json()
            assert [c["status"] for c in cancelled["clips"]] == ["cancelled", "cancelled"]
            assert client.post("/api/jobs/cancelrender/clips/first/render",
                               json={"start": 11, "end": 40, "captions": False}).status_code == 202
            queued = client.get("/api/jobs/cancelrender").json()
            assert queued["stage"] == "Queued (position 1)"
            assert queued["clips"][0]["queue_position"] == 1
        finally:
            release.set()
        app.state.pending.join()
        ready = client.get("/api/jobs/cancelrender").json()
        assert ready["clips"][0]["status"] == "ready"
        assert ready["clips"][0]["kept"]
        assert ready["clips"][1]["status"] == "cancelled"
        assert downloads == [10, 11]
        assert len(conversions) == 1


@pytest.mark.parametrize("running", [False, True])
def test_cancel_rerender_restores_previous_ready_render(monkeypatch, tmp_path, media, running):
    entered, release = threading.Event(), threading.Event()
    downloads = []

    def section(url, directory, start, end):
        downloads.append(start)
        if not release.is_set():
            entered.set()
            assert release.wait(5)
        return directory / "source.mp4"

    monkeypatch.setattr("clipper_studio.api.download_section", section)
    app = create_app(tmp_path, render_workers=1)
    job = saved_job("restore")
    base = "/api/jobs/restore/clips/first"
    job["clips"][0].update(file_url=base + "/video", download_url=base + "/download",
                           thumbnail_url=base + "/thumbnail", rendered_at=123)
    original = dict(job["clips"][0])
    app.state.store.save(job)
    output = tmp_path / "restore" / "first"
    output.mkdir()
    (output / "clip.mp4").write_bytes(b"original video")
    (output / "thumbnail.jpg").write_bytes(b"original thumbnail")
    if not running:
        blocker = saved_job("blocker", "running")
        blocker["clips"][0].update(status="queued", start=100, end=130)
        app.state.store.save(blocker)
    with TestClient(app) as client:
        try:
            if not running:
                assert entered.wait(5)
            assert client.post(base + "/render", json={"start": 11, "end": 39,
                               "mode": "blur", "captions": True}).status_code == 202
            if running:
                assert entered.wait(5)
            queued = client.get("/api/jobs/restore").json()["clips"][0]
            assert queued["status"] == ("rendering" if running else "queued")
            assert client.patch(base, json={"kept": False}).status_code == 200
            assert client.post("/api/jobs/restore/cancel").status_code == 200
            restored = client.get("/api/jobs/restore").json()["clips"][0]
            for key, value in original.items():
                if key != "kept":
                    assert restored[key] == value
            assert not restored["kept"] and "previous_render" not in restored
            assert client.get(base + "/video").content == b"original video"
            assert client.get(base + "/download").content == b"original video"
            assert client.get(base + "/thumbnail").content == b"original thumbnail"
        finally:
            release.set()
        app.state.pending.join()
        assert downloads == ([11] if running else [100])
        assert client.get(base + "/video").content == b"original video"
        assert client.post(base + "/render", json={"start": 12, "end": 38,
                           "mode": "blur", "captions": False}).status_code == 202
        app.state.pending.join()
        replaced = client.get("/api/jobs/restore").json()["clips"][0]
        assert replaced["status"] == "ready" and replaced["start"] == 12
        assert "previous_render" not in replaced
        assert client.get(base + "/video").content == b"video"


def test_restart_analysis_empty_and_cancelled_jobs(monkeypatch, tmp_path):
    fetched = []

    def chat(url, *args):
        fetched.append(url)
        return {"duration": 600}, []

    monkeypatch.setattr("clipper_studio.api.fetch_chat", chat)
    app = create_app(tmp_path)
    interrupted = saved_job("analysis", "running")
    interrupted.update(clips=[], analysis_complete=False)
    app.state.store.save(interrupted)
    cancelled = saved_job("cancelled", "cancelled")
    cancelled["clips"][0]["status"] = "cancelled"
    app.state.store.save(cancelled)
    empty = saved_job("empty", "running")
    empty["clips"] = []
    app.state.store.save(empty)
    with TestClient(app) as client:
        app.state.pending.join()
        assert client.get("/api/jobs/analysis").json()["status"] == "completed"
        assert client.get("/api/jobs/empty").json()["status"] == "completed"
        assert client.get("/api/jobs/cancelled").json()["status"] == "cancelled"
        assert fetched == [interrupted["url"]]


def test_pool_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIPPER_METADATA_WORKERS", "2")
    monkeypatch.setenv("CLIPPER_ANALYSIS_WORKERS", "4")
    monkeypatch.setenv("CLIPPER_RENDER_WORKERS", "1")
    pending = create_app(tmp_path).state.pending
    assert (pending.metadata_workers, pending.analysis_workers, pending.render_workers) == (2, 4, 1)
    with pytest.raises(ValueError, match="between 1 and 32"):
        create_app(tmp_path, render_workers=0)


def test_parallel_renders_preserve_concurrent_review_updates(monkeypatch, tmp_path, media):
    entered = threading.Barrier(3)
    release = threading.Event()

    def section(url, directory, *args):
        entered.wait(timeout=5)
        assert release.wait(5)
        return directory / "source.mp4"

    monkeypatch.setattr("clipper_studio.api.download_section", section)
    app = create_app(tmp_path, render_workers=2)
    job = saved_job("parallel", "running")
    job["clips"][0]["status"] = "queued"
    job["clips"].append(dict(job["clips"][0], id="second", start=100, end=130))
    app.state.store.save(job)
    with TestClient(app) as client:
        try:
            entered.wait(timeout=5)
            assert client.patch("/api/jobs/parallel/clips/first", json={"kept": False}).status_code == 200
            assert [c["status"] for c in client.get("/api/jobs/parallel").json()["clips"]] == ["rendering", "rendering"]
        finally:
            release.set()
        app.state.pending.join()
        completed = client.get("/api/jobs/parallel").json()
        assert completed["status"] == "completed" and completed["progress"] == 1
        assert [c["status"] for c in completed["clips"]] == ["ready", "ready"]
        assert [c["kept"] for c in completed["clips"]] == [False, True]


def test_unexpected_analysis_failure_is_friendly_and_pool_continues(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration": 600}, []))

    def broken(*args, **kwargs):
        raise RuntimeError("internal detector detail")

    monkeypatch.setattr("clipper_studio.api.detect_spikes", broken)
    app = create_app(tmp_path, analysis_workers=1)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url": REPLAY_URL}).json()["id"]
        app.state.pending.join()
        failed = client.get(f"/api/jobs/{jid}").json()
        assert failed["status"] == "failed"
        assert failed["error"] == "Studio couldn’t process this replay. Try submitting it again."
        assert "internal detector detail" in caplog.text
        monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([], []))
        next_job = client.post("/api/jobs", json={"ai_analysis":False, "review":False, "url": REPLAY_URL}).json()["id"]
        app.state.pending.join()
        assert client.get(f"/api/jobs/{next_job}").json()["status"] == "completed"
