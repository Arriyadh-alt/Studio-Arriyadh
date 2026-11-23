import subprocess

import pytest
from fastapi.testclient import TestClient

from clipper_studio.api import JobRequest, create_app
from shorts_clipper.captions import Caption, Transcript
from shorts_clipper.director import Word
from test_api import REPLAY_URL


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("upload") / "episode.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=15:d=30",
                    "-f", "lavfi", "-i", "sine=f=180:d=30", "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(path)], check=True)
    return path


def moment():
    return {"start": 2.0, "end": 26.0, "peak": 9.0, "score": 77, "category": "educational", "title": "The alarm clock rule",
            "description": "Speaker 2 on sleep", "reason": "A complete question and answer about sleep",
            "transcript": "[Speaker 1] Why? [Speaker 2] Because.", "signals": {},
            "podcast": {"dimensions": {"entertaining": 10, "educational": 80, "surprising": 40, "emotional": 5,
                                       "controversial": 0, "retention": 70},
                        "hook": {"text": "Never set two alarms.", "start": 9.0, "end": 11.0, "strength": 70},
                        "teaser": {"text": "Never set two alarms.", "start": 9.0, "end": 11.0},
                        "context": {"score": 90, "notes": ["self-contained"]}, "topic": "sleep",
                        "speakers": [{"id": 1, "label": "Speaker 2", "seconds": 18}],
                        "turns": [[2, 6, 0], [6.5, 26, 1]], "commentary": "Speaker 2 explains the alarm clock rule.",
                        "kind": "qa", "question": "Why?", "before": ""}}


def captions(path, language=None, clip_end=None):
    words = [Word(i + 0.1, i + 0.6, "clok" if i == 3 else "word") for i in range(int(clip_end or 20) - 1)]
    return Transcript([Caption(w.start, w.end, w.word, [w]) for w in words], language="en")


def test_job_request_sources():
    with pytest.raises(ValueError):
        JobRequest()
    with pytest.raises(ValueError):
        JobRequest(url=REPLAY_URL, upload="a" * 32)
    request = JobRequest(upload="a" * 32, ai_analysis=False)
    assert request.ai_analysis and request.content_type == "gaming"
    assert JobRequest(url=REPLAY_URL, content_type="podcast").podcast.version == "original"


def test_upload_podcast_review_render_and_rights(monkeypatch, tmp_path, video):
    seen = {}

    def analyze(audio, video_path, duration, **kwargs):
        seen.update(duration=duration, audio=audio.exists(), **kwargs)
        return [moment()]

    monkeypatch.setattr("clipper_studio.api.analyze_podcast", analyze)
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: pytest.fail("podcasts skip chat replay"))
    monkeypatch.setattr("clipper_studio.api.transcribe", captions)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.post("/api/uploads", content=b"x", headers={"X-Filename": "a.exe"}).status_code == 400
        assert client.post("/api/uploads", content=b"x").status_code == 400
        assert client.post("/api/uploads", content=b"not video", headers={"X-Filename": "a.mp4"}).status_code == 400
        upload = client.post("/api/uploads", content=video.read_bytes(), headers={"X-Filename": "My Episode.mp4"}).json()
        assert upload["filename"] == "My Episode.mp4" and 29 < upload["duration"] < 31
        jid = client.post("/api/jobs", json={"upload": upload["id"], "content_type": "podcast",
                                             "podcast": {"layout": "wide"}}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed", job
        assert job["metadata"]["title"] == "My Episode" and seen["audio"] and 29 < seen["duration"] < 31
        m = job["moments"][0]
        assert m["podcast"]["context"]["score"] == 90 and m["sources"] == ["ai"]
        assert client.get(f"/api/jobs/{jid}/source").status_code == 200
        edits = {"layout": "wide", "version": "commentary", "hook_text": "Two alarms ruin sleep",
                 "caption_edits": [{"find": "clok", "replace": "clock"}]}
        assert client.patch(f"/api/jobs/{jid}/moments/{m['id']}", json={"podcast": edits, "status": "selected"}).status_code == 200
        cid = client.post(f"/api/jobs/{jid}/moments/render").json()["clips"][0]
        app.state.pending.join()
        clip = client.get(f"/api/jobs/{jid}").json()["clips"][0]
        assert clip["status"] == "ready", clip.get("error")
        assert clip["podcast"]["version"] == "commentary" and clip["podcast"]["caption_edits"][0]["replace"] == "clock"
        result = clip["director_result"]
        assert result["status"] == "used" and result["options"]["preset"] == "podcast"
        assert result["edits"]["commentary"]["label"] == "AI commentary" and result["edits"]["sound_effects"] == 0
        assert result["edits"]["voice_clarity"]
        assert clip["tracking_result"]["layout"] == "wide"
        assert client.get(f"/api/jobs/{jid}/clips/{cid}/video").status_code == 200  # preview stays available
        assert client.get(f"/api/jobs/{jid}/clips/{cid}/download").status_code == 403
        assert client.post(f"/api/jobs/{jid}/rights", json={"basis": "licensed", "confirmed": False}).status_code == 422
        assert client.post(f"/api/jobs/{jid}/rights", json={"basis": "licensed", "confirmed": True}).status_code == 200
        assert client.get(f"/api/jobs/{jid}/clips/{cid}/download").status_code == 200
        text = client.get(f"/api/jobs/{jid}/clips/{cid}/description").json()["text"]
        assert "Source: My Episode" in text and "AI-generated commentary" in text and "Licensed" in text


def test_gaming_jobs_keep_chat_and_open_exports(monkeypatch, tmp_path):
    from clipper_studio.api import export_description, require_rights
    require_rights({"request": {"content_type": "gaming"}})
    with pytest.raises(Exception):
        require_rights({"request": {"content_type": "podcast"}})
    job = {"url": REPLAY_URL, "metadata": {"title": "Ep 1", "channel": "Show"},
           "rights": {"basis": "permission", "attribution": True, "include_link": True}}
    text = export_description(job, {"title": "Clip"})
    assert "by Show" in text and REPLAY_URL in text
    job["rights"]["include_link"] = False
    assert REPLAY_URL not in export_description(job, {"title": "Clip"})
