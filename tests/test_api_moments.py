import pytest
from fastapi.testclient import TestClient

from clipper_studio.api import JobRequest, Store, create_app
from clipper_studio.chat import ChatReplayError
from shorts_clipper.youtube import YouTubeAccessError
from test_api import REPLAY_URL, media, metadata, saved_job


def proposal(score=80):
    return dict(start=100, end=140, peak=118, score=score, title="Shock", description="A surprised reaction.",
                reason="Loud shout", category="shock", signals={"audio": .9}, transcript="no way")


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration": 600}, []))
    monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([proposal(4)], []))

    def download(url, directory):
        directory.mkdir()
        path = directory / "audio.wav"
        path.touch()
        return path, None

    monkeypatch.setattr("clipper_studio.api.download_analysis_media", download)
    monkeypatch.setattr("clipper_studio.api.analyze_media", lambda *a, **k: [proposal()])


def test_review_and_endpoints(monkeypatch, tmp_path, analysis):
    assert JobRequest(url=REPLAY_URL).review and JobRequest(url=REPLAY_URL).ai_analysis
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed" and job["clips"] == [] and job["analysis_complete"]
        assert job["stage"] == "1 moments ready for review"
        m = job["moments"][0]
        assert m["rank"] == 1 and m["sources"] == ["ai", "chat"]
        assert not (tmp_path / jid / "analysis").exists()
        base = f"/api/jobs/{jid}/moments"
        assert client.post(base + "/render").status_code == 409
        for data in ({"start": -1}, {"start": 140}, {"end": 700}, {"end": 200}):
            assert client.patch(base + "/" + m["id"], json=data).status_code == 422
        response = client.patch(base + "/" + m["id"], json={"status": "selected", "start": 101})
        assert response.json()["duration"] == 39
        queued = []
        monkeypatch.setattr(app.state.pending, "put", lambda *a, **k: queued.append((a, k)))
        ids = client.post(base + "/render").json()["clips"]
        assert len(ids) == 1 and queued[0][1]["priority"]
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["clips"][0]["moment_id"] == m["id"]
        assert job["moments"][0]["status"] == "rendered"
        assert client.patch(base + "/" + m["id"], json={"status": "selected"}).status_code == 409
        assert client.post(base + "/render", json={"ids": [m["id"]]}).status_code == 409
        assert client.delete(f"/api/jobs/{jid}/clips/{ids[0]}").status_code == 200
        m = client.get(f"/api/jobs/{jid}").json()["moments"][0]
        assert m["status"] == "suggested" and m["clip_id"] is None


@pytest.mark.parametrize("source", ["ai", "chat", "empty", "chatless_empty", "both_fail"])
def test_source_fallbacks(monkeypatch, tmp_path, analysis, source):
    def fail_chat(*args):
        raise ChatReplayError("Chat disabled")

    def fail_ai(*args, **kwargs):
        raise YouTubeAccessError("YouTube is blocking this server")

    if source in ("ai", "chatless_empty", "both_fail"):
        monkeypatch.setattr("clipper_studio.api.fetch_chat", fail_chat)
        monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([], []))
    if source in ("chat", "both_fail"):
        monkeypatch.setattr("clipper_studio.api.download_analysis_media", fail_ai)
    if source in ("empty", "chatless_empty"):
        monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([], []))
        monkeypatch.setattr("clipper_studio.api.analyze_media", lambda *a, **k: [])
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        if source == "both_fail":
            assert job["status"] == "failed" and "YouTube is blocking" in job["error"]
        elif source in ("empty", "chatless_empty"):
            assert job["status"] == "completed"
            assert job["stage"] == "No viral moments found"
        else:
            assert job["status"] == "completed" and job["moments"][0]["sources"] == [source]
            assert job["chat_error" if source == "ai" else "ai_error"]


def test_autorender_and_old_jobs(tmp_path, analysis, media):
    app = create_app(tmp_path)
    app.state.store.save(saved_job("old"))
    assert Store(tmp_path).get("old")["moments"] == []
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL, "review": False,
                          "captions": False, "director": {"enabled": False}}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["clips"][0]["status"] == "ready"
        assert job["clips"][0]["moment_id"] == job["moments"][0]["id"]


def test_chat_disabled_preflight_and_batch(monkeypatch, tmp_path, analysis):
    from clipper_studio.chat import validate_metadata

    info = {"duration": 600, "was_live": True, "subtitles": {}}
    calls = []

    def preflight(url, require_chat=True):
        calls.append(require_chat)
        validate_metadata(info, require_chat)
        return info

    def chat(*args):
        validate_metadata(info)

    def analyze(*args, **kwargs):
        kwargs["progress"](.5, "Speech analysis")
        return [proposal(), {**proposal(70), "start": 300, "end": 340, "peak": 318}]

    monkeypatch.setattr("clipper_studio.api.fetch_metadata", preflight)
    monkeypatch.setattr("clipper_studio.api.fetch_chat", chat)
    monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([], []))
    monkeypatch.setattr("clipper_studio.api.analyze_media", analyze)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert calls == [False] and job["status"] == "completed"
        assert job["analysis_detail"] == "Speech analysis"
        assert len(job["moments"]) == 2 and job["chat_error"]
        app.state.store.save({**job, "clips": [{**saved_job("old")["clips"][0], "number": 7}]})
        monkeypatch.setattr(app.state.pending, "put", lambda *a, **k: None)
        ids = [m["id"] for m in job["moments"]]
        response = client.post(f"/api/jobs/{jid}/moments/render", json={"ids": ids})
        assert len(response.json()["clips"]) == 2
        clips = client.get(f"/api/jobs/{jid}").json()["clips"]
        assert [c["number"] for c in clips] == [7, 8, 9]


def test_story_context_includes_moment(monkeypatch, tmp_path):
    import json
    from clipper_studio.api import story_context
    from shorts_clipper.story import llm_hooks

    clip = {**proposal(), "moment_id": "moment", "id": "clip"}
    context = story_context({"clips": [clip]}, clip, "en", tmp_path, tmp_path)
    assert context["moment"] == {k: clip[k] for k in ("title", "description", "category")}
    monkeypatch.setenv("CLIPPER_LLM_API_KEY", "test")
    requests = []

    def request(req, **kwargs):
        requests.append(json.loads(req.data))
        raise RuntimeError("offline")

    monkeypatch.setattr("shorts_clipper.story.urllib.request.urlopen", request)
    assert llm_hooks(context, "shock", "en") == []
    assert "Detected moment:" in requests[0]["messages"][0]["content"]
    assert "A surprised reaction." in requests[0]["messages"][0]["content"]
