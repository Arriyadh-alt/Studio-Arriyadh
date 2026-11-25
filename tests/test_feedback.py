import json
import socket

import numpy as np
import pytest
from fastapi.testclient import TestClient

from clipper_studio.api import create_app
from clipper_studio.feedback import FeedbackStore, calibrated_weights
from shorts_clipper.captions import Caption
from shorts_clipper.director import DirectorOptions, plan_record
from shorts_clipper.editing import plan_source_edit
from test_api import saved_job


PATH = "/api/feedback"
LINK = "https://www.youtube.com/shorts/AbCdEf12345"


@pytest.fixture
def app(tmp_path):
    app = create_app(tmp_path)
    job = saved_job("feedbackjob")
    job.update(url="https://www.youtube.com/watch?v=Original123",
               metadata={"title": "Original stream", "duration": 600, "channel": "Original creator"})
    job["clips"][0].update(title="The comeback", moment_id="moment1", sources=["ai", "chat"], score=82,
                           editing_mode="enhanced", score_breakdown={"hook": 78, "story": 86},
                           score_penalties=[{"reason": "abrupt ending", "points": 3}],
                           score_explanation="The trigger and reaction are both visible.",
                           director_result={"status": "used", "edits": {"duration": 26.5, "hook": {
                               "text": "How did that work?", "style": "story", "writer": "llm"}}},
                           authenticity={"checks": {"context": {"status": "review", "detail": "Review setup."}}})
    job["clips"].append({"id": "chatclip", "start": 20, "end": 40, "status": "ready", "sources": ["chat"], "score": 91})
    app.state.store.save(job)
    return app


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def entry(**overrides):
    return {"job_id": "feedbackjob", "clip_id": "first", "publish_url": LINK,
            "confirmed_published": True, **overrides}


def test_manual_only_no_outgoing_publish(app, client, monkeypatch):
    def block_connection(*args, **kwargs):
        pytest.fail("Recording performance must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", block_connection)
    schema = client.get("/openapi.json").json()["paths"]
    assert not any("publish" in path.split("/") for path in schema)
    assert {path for path, verbs in schema.items() if "upload" in path and "post" in verbs} == {"/api/uploads"}
    before = app.state.store.get("feedbackjob")
    assert client.get(PATH).json() == []
    assert client.post(PATH, json=entry(confirmed_published=False)).status_code == 422
    assert client.post(PATH, json=entry(confirmed_published=1)).status_code == 422
    missing_confirmation = entry()
    del missing_confirmation["confirmed_published"]
    assert client.post(PATH, json=missing_confirmation).status_code == 422
    assert client.get(PATH).json() == []
    assert client.post(PATH, json=entry(approved=True)).status_code == 422
    response = client.post(PATH, json=entry(metrics={"views": 0}, publish_date="2026-11-20"))
    assert response.status_code == 201
    record = response.json()
    assert record["metrics"] == {"views": 0}
    assert record["publish_url"] == "https://www.youtube.com/watch?v=AbCdEf12345"
    assert record["source"]["channel"] == "Original creator"
    assert record["source"]["url"] == before["url"]
    assert app.state.store.get("feedbackjob") == before
    assert FeedbackStore(app.state.store.root).records() == [record]


def test_snapshot_crud_and_integrity(app, client):
    clip, chat = client.get(PATH + "/clips").json()
    assert chat["editorial_score"] is None
    assert clip["duration"] == 26.5 and clip["duration_basis"] == "render"
    assert clip["hook"] == "How did that work?"
    flags = {flag["check"]: flag["detail"] for flag in clip["integrity"]}
    assert "No rights confirmation" in flags["Copyright / permission"]
    assert "Cropping, subtitles" in flags["Reused content"]
    assert flags["Source context"] == "Review setup."
    assert "generated text hook" in flags["Synthetic-content disclosure"]
    record = client.post(PATH, json=entry(title="Actual title", hook_style="question", metrics={
        "average_percentage_viewed": 113.4, "engaged_views": 20, "viewed_percent": 62.5,
        "swiped_away_percent": 37.5, "average_view_duration": 30.6, "likes": 8,
        "comments": 1, "shares": 2, "subscribers_gained": 3,
        "retention_graph": [{"position_percent": 100, "retained_percent": 76}, {"position_percent": 0, "retained_percent": 120}],
    })).json()
    path = PATH + "/" + record["id"]
    assert record["editorial_score"] == 82
    assert record["score_breakdown"] == {"hook": 78, "story": 86}
    assert record["score_penalties"][0]["points"] == 3
    assert record["score_explanation"].startswith("The trigger")
    assert record["metrics"]["retention_graph"][0]["retained_percent"] == 120
    assert client.post(PATH, json=entry(publish_url="https://youtu.be/AbCdEf12345")).status_code == 409
    job = app.state.store.get("feedbackjob")
    job["clips"][0]["score"] = 95
    app.state.store.save(job)
    patched = client.patch(path, json={"metrics": {"likes": 12, "shares": None}}).json()
    assert patched["editorial_score"] == 82
    assert patched["metrics"]["engaged_views"] == 20
    assert patched["metrics"]["likes"] == 12 and patched["metrics"]["shares"] is None
    assert client.get(path).json() == patched
    assert client.delete(path).json() == {"deleted": record["id"]}
    assert client.get(path).status_code == 404


def test_render_score_snapshot_does_not_replace_editorial_score(app, client):
    render_score = {"overall": 62, "hook": 65, "pacing": 71, "emotion": 70, "dialogue": 80, "chat": 0}
    job = app.state.store.get("feedbackjob")
    job["clips"][0]["director_result"]["score"] = render_score
    job["clips"][1]["director_result"] = {"status": "used", "score": render_score}
    app.state.store.save(job)
    ai, chat = client.get(PATH + "/clips").json()
    assert ai["score_breakdown"]["hook"] == 78 and ai["director_score"] == render_score
    assert chat["editorial_score"] is None and chat["director_score"] == render_score
    published = client.post(PATH, json=entry(clip_id="chatclip", metrics={"average_percentage_viewed": 80})).json()
    assert published["director_score"] == render_score
    compared = client.get(PATH + "/comparison").json()
    assert compared["sample_size"] == 0 and compared["director"]["sample_size"] == 1


def test_metrics_are_cleared_only_for_a_different_video_id(client):
    original = {"views": 42, "average_percentage_viewed": 72,
                "retention_graph": [{"position_percent": 0, "retained_percent": 100}]}
    record = client.post(PATH, json=entry(metrics=original)).json()
    path = PATH + "/" + record["id"]
    assert client.patch(path, json={"publish_url": "https://youtu.be/AbCdEf12345?si=test"}).json()["metrics"] == original
    response = client.patch(path, json={"publish_url": "https://www.youtube.com/shorts/AbCdEf12346", "metrics": original})
    assert response.status_code == 200
    assert response.json()["metrics"] == {}
    assert client.get(path).json()["metrics"] == {}
    assert client.get(PATH + "/comparison").json()["sample_size"] == 0
    assert client.patch(path, json={"metrics": {"views": 9}}).json()["metrics"] == {"views": 9}


@pytest.mark.parametrize("commentary, expected", [(None, "source quote"), ("Here is what is happening.", "commentary")])
def test_snapshot_distinguishes_enhanced_podcast_hook_styles(app, client, commentary, expected):
    job = app.state.store.get("feedbackjob")
    options = DirectorOptions(preset="podcast", commentary=commentary)
    plan, _ = plan_source_edit(np.full(100, -20.), [Caption(.3, 2., "Let me show you.")], 10,
                               options, mode="enhanced", category="educational")
    assert plan.hook["style"] == "source"
    job["clips"][0]["director_result"] = plan_record(plan, options, [])
    app.state.store.save(job)
    response = client.post(PATH, json=entry())
    assert response.status_code == 201
    assert response.json()["hook_style"] == expected
    assert client.patch(PATH + "/" + response.json()["id"], json={"hook_style": expected}).status_code == 200


def test_csv_and_json_import(client):
    # A YouTube Studio per-video export has no Clipper IDs and may include a Total row.
    content = ('\ufeffContent,Video title,Video publish time, VIEWS ,Engaged views,'
               'Stayed to watch (%),Average view duration,Average percentage viewed (%),Comments,Subscribers\n'
               'Total,,,2000,,,,,,\n'
               'AbCdEf12345,Actual title,"Nov 2, 2026","1,234",800,61.2%,0:00:30,113.2,—,4\n')
    request = {"format": "csv", "content": content, "job_id": "feedbackjob", "clip_id": "first"}
    assert client.post(PATH + "/import", json=request).status_code == 422
    response = client.post(PATH + "/import", json={**request, "confirmed_published": True})
    assert response.status_code == 200
    record, = response.json()
    assert record["publish_date"] == "2026-11-02"
    assert record["metrics"] == {"views": 1234, "engaged_views": 800, "viewed_percent": 61.2,
                                 "average_view_duration": 30, "average_percentage_viewed": 113.2, "subscribers_gained": 4}
    request = {"format": "json", "confirmed_published": True, "content": json.dumps([
        {"publish_url": LINK, "metrics": {"likes": 13}},
        {"job_id": "feedbackjob", "clip_id": "chatclip", "publish_url": "https://youtu.be/AbCdEf12346",
         "metrics": {"likes": 0}},
    ])}
    response = client.post(PATH + "/import", json=request)
    assert response.status_code == 200
    assert response.json()[0]["id"] == record["id"]
    assert response.json()[0]["metrics"]["views"] == 1234
    assert response.json()[1]["metrics"] == {"likes": 0}


def test_invalid_import_is_atomic(client):
    uploads = [
        entry(metrics={"viewed_percent": 101}),
        entry(metrics={"views": True}),
        entry(metrics={"average_view_duration": "-1"}),
        entry(metrics={"retention_graph": [{"position_percent": 101, "retained_percent": 50}]}),
        entry(publish_url="https://www.youtube.com.invalid/shorts/AbCdEf12345"),
    ]
    for packet in uploads:
        assert client.post(PATH, json=packet).status_code == 422
    response = client.post(PATH + "/import", json={"format": "json", "confirmed_published": True,
        "content": json.dumps([entry(), entry(clip_id="gone", publish_url="https://youtu.be/AbCdEf12346")])})
    assert response.status_code == 404
    assert client.get(PATH).json() == []
    response = client.post(PATH + "/import", json={"format": "csv", "confirmed_published": True,
        "job_id": "feedbackjob", "clip_id": "first", "content": "Content,Views\nAbCdEf12345,1\nAbCdEf12346,2"})
    assert response.status_code == 422
    assert "link the published URL" in response.json()["detail"]


def test_calibration_uses_only_measured_matching_samples(tmp_path, client, monkeypatch):
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path))
    weights = {"hook": .2, "story": .2, "emotion": .2, "visual": .15, "payoff": .15, "retention": .1}
    data = FeedbackStore(tmp_path)
    records = [
        dict(id=str(i), metrics={"average_percentage_viewed": 40 + 10 * i, "views": 100}, editorial_score=100 - i * 10,
             score_breakdown={"hook": i * 10, "story": 100 - i * 10, "visual": 60},
             hook_style="curiosity gap", duration=44, editing_mode="enhanced")
        for i in range(5)
    ]
    data.save(records[:-1] + [{**records[-1], "metrics": {"views": 1000}}])
    assert calibrated_weights(weights) == weights
    summary = client.get(PATH + "/calibration").json()
    assert summary["durations"] == [] and summary["weight_source"] == "default"
    assert summary["default_weights"] == summary["baseline_weights"] == summary["effective_weights"] == weights
    data.save(records)
    result = calibrated_weights(weights)
    assert result.keys() == weights.keys()
    assert sum(result.values()) == pytest.approx(1)
    assert result["hook"] > weights["hook"] and result["story"] < weights["story"]
    assert result["visual"] == weights["visual"]
    assert all(abs(result[key] - value) <= .05 for key, value in weights.items())
    summary = client.get(PATH + "/calibration").json()
    assert summary["weight_source"] == "calibrated"
    assert summary["default_weights"] == weights and summary["effective_weights"] == result
    assert summary["sample_size"] == summary["component_sample_size"] == 5
    for key, label in (("durations", "40–55s"), ("hook_styles", "curiosity gap"), ("editing_modes", "enhanced")):
        assert summary[key] == [{"label": label, "sample_size": 5, "median_percentage_viewed": 60}]
    compare = client.get(PATH + "/comparison").json()
    assert compare["correlation"] == -1 and compare["components"]["hook"] == 1
    assert compare["percentile_gaps"]["0"] == -100 and compare["percentile_gaps"]["4"] == 100
    assert client.get(PATH + "/comparison?metric=views").json()["correlation"] is None


def test_calibration_summary_reports_configured_scoring_baseline(tmp_path, client):
    config = tmp_path / "viral-scoring.json"
    config.write_text(json.dumps({"weights": {"hook": 0}}))
    summary = client.get(PATH + "/calibration").json()
    assert summary["weight_source"] == "configured"
    assert summary["default_weights"]["hook"] == .2
    assert summary["effective_weights"] == summary["baseline_weights"]
    assert summary["effective_weights"]["hook"] == 0
    assert sum(summary["effective_weights"].values()) == pytest.approx(1)
    config.write_text(json.dumps({"weights": {"hook": -1}}))
    assert client.get(PATH + "/calibration").json()["weight_source"] == "default"
