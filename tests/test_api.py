import io
import zipfile

from fastapi.testclient import TestClient

from clipper_studio.api import Store, create_app
from clipper_studio.chat import Message


def test_pipeline_and_file_endpoints(monkeypatch, tmp_path):
    stages = []
    save = Store.save
    def record_stage(store, job):
        stages.append(job['stage'])
        save(store, job)
    monkeypatch.setattr(Store, "save", record_stage)
    monkeypatch.setattr("clipper_studio.api.time.sleep", lambda seconds: None)
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: (
        {"title": "Test", "duration": 600, "thumbnail": None},
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
        for number in range(1, 4):
            for stage in ("Downloading", "Transcribing", "Rendering"):
                assert f"Clip {number} of 3 · {stage}" in stages
        assert all(c["status"] == "ready" for c in job["clips"])
        clip = job["clips"][0]
        assert clip["attempt"] == 3
        assert len(download_attempts) == 5
        assert job["created_at"]
        assert client.get(f"/api/jobs/{jid}/download-kept").status_code == 409
        assert client.patch(f"/api/jobs/{jid}/clips/{clip['id']}", json={"kept":True}).json()["kept"]
        archive = client.get(f"/api/jobs/{jid}/download-kept")
        assert archive.status_code == 200
        with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
            assert bundle.read("highlight-1.mp4") == b"0123456789"
        assert client.get(f"/api/jobs/{jid}").json()["clips"][0]["kept"]
        response = client.get(clip["file_url"], headers={"Range":"bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert response.headers["content-range"] == "bytes 2-5/10"
        assert "attachment" in client.get(clip["download_url"]).headers["content-disposition"]
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
    job["clips"] = [{"id":"testclip", "status":"queued"}]
    store.save(job)
    base = f"/api/jobs/{jid}/clips/testclip"
    assert client.delete(base).status_code == 409
    assert client.post(base + "/render", json={"start":10,"end":40}).status_code == 409
    restarted = create_app(tmp_path)
    with TestClient(restarted) as client:
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed"
        assert job["clips"][0]["status"] == "failed"
