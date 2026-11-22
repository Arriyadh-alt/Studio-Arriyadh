import io
import threading
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from clipper_studio.api import create_app, title_slug
from clipper_studio.chat import Message


def test_pipeline_and_file_endpoints(monkeypatch, tmp_path):
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
        response = client.post("/api/jobs", json={"url":"https://youtube.com/watch?v=test", "clips":3})
        assert response.status_code == 202
        jid = response.json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed"
        assert len(job["clips"]) == 3
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
        assert client.post("/api/jobs", json={"url":"https://youtu.be/test", "lead":10}).status_code == 422
        def fail(*args):
            raise ValueError("no live-chat replay")
        monkeypatch.setattr("clipper_studio.api.fetch_chat", fail)
        jid = client.post("/api/jobs", json={"url":"https://youtu.be/test"}).json()["id"]
        app.state.pending.join()
        assert client.get(f"/api/jobs/{jid}").json()["error"] == "no live-chat replay"
        monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration":100}, []))
        jid = client.post("/api/jobs", json={"url":"https://youtu.be/test"}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed" and job["clips"] == []


def test_restart_and_busy_conflicts(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    jid = client.post("/api/jobs", json={"url":"https://youtu.be/test"}).json()["id"]
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
        jid = client.post("/api/jobs", json={"url":"https://youtu.be/test", "captions":False}).json()["id"]
        try:
            assert entered.wait(5)
            job = client.get(f"/api/jobs/{jid}").json()
            assert job["stage"] == "Downloading highlight 1 of 2"
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
    jid = client.post("/api/jobs", json={"url":"https://youtu.be/test"}).json()["id"]
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
        assert [c["number"] for c in client.get("/api/jobs/done").json()["clips"]] == [1, 2]
        assert client.delete("/api/jobs/done/clips/first").status_code == 200
        assert client.get("/api/jobs/done").json()["clips"][0]["number"] == 2
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/jobs/done").json()["clips"][0]["number"] == 2
