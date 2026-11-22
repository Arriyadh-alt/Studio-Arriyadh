from fastapi.testclient import TestClient

from clipper_studio.api import create_app
from clipper_studio.chat import Message


def test_pipeline_and_file_endpoints(monkeypatch, tmp_path):
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: (
        {"title": "Test", "duration": 600, "thumbnail": None},
        [Message(t, "user", "W") for t in (100, 300, 500) for _ in range(20)]))
    def section(url, directory, start, end):
        path = directory / "source.mp4"
        path.write_bytes(b"source")
        return path
    monkeypatch.setattr("clipper_studio.api.download_section", section)
    monkeypatch.setattr("clipper_studio.api.video_duration", lambda p: 40)
    monkeypatch.setattr("clipper_studio.api.transcribe", lambda p: [])
    def convert(source, output, *args):
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
        assert all(c["status"] == "ready" for c in job["clips"])
        clip = job["clips"][0]
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
        assert client.get(clip["file_url"]).status_code == 409
        assert client.delete(base).status_code == 200
        assert client.get(clip["file_url"]).status_code == 404
        assert len(client.get("/api/jobs").json()) == 1


def test_errors_and_empty(monkeypatch, tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/jobs").json() == []
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
