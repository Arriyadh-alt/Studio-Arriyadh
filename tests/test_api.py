import io
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from clipper_studio.api import JobRequest, Store, create_app, title_slug
from clipper_studio.chat import ChatReplayError, Message


REPLAY_URL = "https://www.youtube.com/watch?v=LIxBN4AHMEE"


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
        [Message(t, "user", "W") for t in (100, 300, 500) for _ in range(20)]))
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
    monkeypatch.setattr("clipper_studio.api.transcribe", lambda p: [])
    render_attempts = []
    def convert(source, output, *args):
        render_attempts.append(output)
        if len(render_attempts) == 1:
            raise RuntimeError("Transient render failure")
        output.write_bytes(b"0123456789")
    monkeypatch.setattr("clipper_studio.api.make_vertical_clip", convert)
    def thumbnail(command, **kwargs):
        from pathlib import Path
        Path(command[-1]).write_bytes(b"jpg")
    monkeypatch.setattr("clipper_studio.api.subprocess.run", thumbnail)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/jobs", json={"url":REPLAY_URL, "clips":3})
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
        def render_fail(*args):
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
            assert bundle.namelist() == ["highlight-02.mp4"]
        assert client.get(clip["file_url"]).status_code == 404
        assert len(client.get("/api/jobs").json()) == 1


def test_errors_and_empty(monkeypatch, tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/jobs").json() == []
        assert client.get("/").status_code == 200
        assert "Clipper Studio" in client.get("/").text
        assert client.get("/manifest.webmanifest").json()["icons"][0]["sizes"] == "192x192"
        assert client.get("/icon-512.png").headers["content-type"] == "image/png"
        assert client.get("/api/jobs/missing").status_code == 404
        assert client.post("/api/jobs", json={"url":"https://example.org"}).status_code == 422
        assert client.post("/api/jobs", json={"url":REPLAY_URL, "lead":10}).status_code == 422
        def fail(*args):
            raise ChatReplayError("no live-chat replay")
        monkeypatch.setattr("clipper_studio.api.fetch_chat", fail)
        jid = client.post("/api/jobs", json={"url":REPLAY_URL}).json()["id"]
        app.state.pending.join()
        assert client.get(f"/api/jobs/{jid}").json()["error"] == "no live-chat replay"
        monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration":100}, []))
        jid = client.post("/api/jobs", json={"url":REPLAY_URL}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed" and job["clips"] == []


def test_restart_and_busy_conflicts(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    jid = client.post("/api/jobs", json={"url":REPLAY_URL}).json()["id"]
    store = app.state.store
    job = store.get(jid)
    job["metadata"] = {"duration":100}
    job["clips"] = [{"id":"testclip", "status":"rendering"}]
    store.save(job)
    base = f"/api/jobs/{jid}/clips/testclip"
    assert client.delete(base).status_code == 409
    assert client.post(base + "/render", json={"start":10,"end":40}).status_code == 409
    restarted = create_app(tmp_path)
    with TestClient(restarted) as client:
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed"
        assert job["clips"][0]["status"] == "failed"


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

    def fetch(url, directory):
        calls.append(url)
        if url == REPLAY_URL:
            started.set()
            assert release.wait(5)
            raise ChatReplayError("This livestream has no live-chat replay.")
        return {"duration": 11949, "title": "Long stream"}, [Message(11940, "user", "W")]

    monkeypatch.setattr("clipper_studio.api.fetch_chat", fetch)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        try:
            first = client.post("/api/jobs", json={"url":"https://youtu.be/LIxBN4AHMEE"}).json()["id"]
            assert started.wait(5)
            second = client.post("/api/jobs", json={"url":"https://m.youtube.com/watch?v=puMjBYns9L0"}).json()["id"]
            assert first != second
            refreshed = TestClient(app)
            assert refreshed.get(f"/api/jobs/{first}").json()["status"] == "running"
            queued = refreshed.get(f"/api/jobs/{second}").json()
            assert queued["status"] == "queued"
            assert "jobs run one at a time" in queued["stage"]
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
    monkeypatch.setattr("clipper_studio.api.make_vertical_clip", lambda source, output, *a: output.write_bytes(b"video"))
    monkeypatch.setattr("clipper_studio.api.subprocess.run", lambda command, **k: Path(command[-1]).write_bytes(b"jpg"))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url":REPLAY_URL, "captions":False}).json()["id"]
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
    jid = client.post("/api/jobs", json={"url":REPLAY_URL}).json()["id"]
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
