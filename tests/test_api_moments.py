import threading

import pytest
from fastapi.testclient import TestClient

from clipper_studio.api import CHAT_FALLBACK_WARNING, JobRequest, Store, create_app
from clipper_studio.chat import ChatReplayError, REPLAY_LIMIT_MESSAGE
from shorts_clipper.youtube import YouTubeAccessError
from test_api import REPLAY_URL, media, metadata, saved_job


def proposal(score=80):
    return dict(start=100, end=140, peak=118, score=score, title="Shock", description="A surprised reaction.",
                reason="Loud shout", category="shock", signals={"audio": .9}, transcript="no way")


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.setattr("clipper_studio.api.fetch_chat", lambda *a: ({"duration": 600}, []))
    monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([proposal(4)], []))

    def download(url, directory, should_continue=None):
        if should_continue:
            should_continue()
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


@pytest.mark.parametrize("source", ["oversized", "disabled", "regular_video", "unexpected"])
def test_ai_off_still_falls_back_when_chat_is_unavailable(monkeypatch, tmp_path, analysis, source):
    def chat(*args):
        if source == "unexpected":
            raise RuntimeError("bad chat parser")
        if source in ("disabled", "regular_video"):
            from clipper_studio.chat import validate_metadata

            validate_metadata({"live_status": "not_live" if source == "regular_video" else "was_live",
                               "duration": 600})
        raise ChatReplayError(REPLAY_LIMIT_MESSAGE)

    def download(*args, **kwargs):
        assert kwargs["should_continue"]() is None
        job = app.state.store.jobs()[0]
        assert job["chat_warning"] == CHAT_FALLBACK_WARNING
        return None, None

    monkeypatch.setattr("clipper_studio.api.fetch_chat", chat)
    monkeypatch.setattr("clipper_studio.api.download_analysis_media", download)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL, "ai_analysis": False}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed" and not job["error"]
        assert job["chat_error"] and job["chat_warning"] == CHAT_FALLBACK_WARNING
        assert job["moments"][0]["sources"] == ["ai"]


def test_ai_starts_without_waiting_for_chat(monkeypatch, tmp_path, analysis):
    ai_started = threading.Event()

    def chat(*args):
        assert ai_started.wait(3)
        raise ChatReplayError("Chat replay unavailable")

    def download(*args, **kwargs):
        ai_started.set()
        return None, None

    monkeypatch.setattr("clipper_studio.api.fetch_chat", chat)
    monkeypatch.setattr("clipper_studio.api.download_analysis_media", download)
    app = create_app(tmp_path, analysis_workers=1)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
        app.state.pending.join()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed" and job["chat_error"] == "Chat replay unavailable"


def test_retry_chat_only_preserves_existing_reviews_and_clips(monkeypatch, tmp_path, analysis):
    entered, release = threading.Event(), threading.Event()

    def chat(url, directory, info, should_continue):
        if info is not None:
            raise ChatReplayError(REPLAY_LIMIT_MESSAGE)
        entered.set()
        assert release.wait(5)
        return {"duration": 600}, []

    def spikes(*args, **kwargs):
        return [proposal(4), {**proposal(4), "start": 300, "end": 340, "peak": 315}], []

    monkeypatch.setattr("clipper_studio.api.fetch_chat", chat)
    monkeypatch.setattr("clipper_studio.api.detect_spikes", spikes)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        key = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
        base = f"/api/jobs/{key}"
        app.state.pending.join()
        old = app.state.store.get(key)
        moment = old["moments"][0]
        moment.update(start=103, duration=37, status="rendered", clip_id="oldclip")
        old["clips"] = [{**saved_job("old")["clips"][0], "id": "oldclip", "moment_id": moment["id"],
                         "number": 1, "status": "ready"}]
        app.state.store.save(old)
        old = app.state.store.get(key)
        monkeypatch.setattr("clipper_studio.api.download_analysis_media",
                            lambda *a, **k: pytest.fail("chat retry must not download AI media"))
        try:
            assert client.post(base + "/chat/retry").status_code == 202
            assert entered.wait(5)
            job = client.get(base).json()
            assert job["chat_retrying"] and job["status"] == "completed"
            assert client.post(base + "/chat/retry").status_code == 409
        finally:
            release.set()
        app.state.pending.join()
        job = app.state.store.get(key)
        assert job["moments"][0] == moment and job["clips"] == old["clips"]
        assert len(job["moments"]) == 2 and job["moments"][1]["sources"] == ["chat"]
        assert job["chat_error"] is None and not job["chat_retrying"]
        assert client.post(base + "/chat/retry").status_code == 202
        app.state.pending.join()
        assert app.state.store.get(key)["moments"] == job["moments"]


def test_chat_retry_updates_an_untouched_suggestion(monkeypatch, tmp_path, analysis):
    monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([
        proposal(8), {**proposal(4), "start": 143, "end": 178, "peak": 145},
    ], []))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL, "ai_analysis": False}).json()["id"]
        app.state.pending.join()
        original, neighboring = app.state.store.get(jid)["moments"]
        assert original["end"] < neighboring["start"]
        monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([
            {**proposal(90), "start": 110, "end": 150, "peak": 128, "message_sample": ["WOW"]},
            {**proposal(6), "start": 105, "end": 145, "peak": 125, "message_sample": ["hi"]},
        ], []))
        assert client.post(f"/api/jobs/{jid}/chat/retry").status_code == 202
        app.state.pending.join()
        moment, = app.state.store.get(jid)["moments"]
        assert moment["id"] == original["id"]
        assert moment["rank"] == original["rank"]
        assert (moment["start"], moment["end"], moment["peak"]) == (110, 150, 128)
        assert moment["score"] > original["score"]
        assert moment["message_sample"] == ["WOW"]


@pytest.mark.parametrize("sources,status,reviewed", [
    (["chat"], "suggested", True), (["ai"], "rejected", False), (["ai"], "selected", False),
])
def test_retry_peak_does_not_replace_or_overlap_protected_moment(sources, status, reviewed):
    from clipper_studio.ranking import add_chat_moments, merge_moments

    original, protected = merge_moments([], [proposal(8), {
        **proposal(4), "start": 143, "end": 178, "peak": 145,
    }], 600, 18)
    protected.update(sources=sources, status=status, reviewed=reviewed)
    snapshot = dict(protected)
    candidate = {**proposal(90), "start": 110, "end": 150, "peak": 128}
    result = add_chat_moments([original, protected], [candidate], 600, 18)
    assert result == [snapshot]


def test_retry_chat_matches_initial_merge_and_preserves_reviewed_moments():
    from clipper_studio.ranking import add_chat_moments, merge_moments

    ai = [proposal(80)]
    chat = [proposal(90), {**proposal(70), "start": 120, "end": 160, "peak": 143}]
    first_run = merge_moments(ai, chat, 600, 18)
    assert [m["sources"] for m in first_run] == [["ai", "chat"], ["chat"]]
    reviewed, rendered = merge_moments([], [
        {**proposal(15), "start": 260, "end": 280, "peak": 271},
        {**proposal(14), "start": 360, "end": 380, "peak": 371},
    ], 600, 18)
    reviewed.update(reviewed=True, rank=3)
    rendered.update(status="rendered", clip_id="clip", rank=4)

    existing = [*first_run, reviewed, rendered]
    assert add_chat_moments(existing, chat, 600, 18) == existing


def test_retry_drops_proposals_overlapping_either_range_of_reviewed_ai_moment():
    from clipper_studio.ranking import add_chat_moments, merge_moments

    raw_ai = [proposal(80)]
    reviewed, = merge_moments(raw_ai, [proposal(90)], 600, 18)
    reviewed.update(reviewed=True, start=300, end=330, duration=30)
    fresh = [
        proposal(90),
        {**proposal(70), "start": 120, "end": 160, "peak": 143},
        {**proposal(60), "start": 320, "end": 345, "peak": 335},
    ]
    assert add_chat_moments([reviewed], fresh, 600, 18) == [reviewed]


def test_retry_rebuilds_untouched_ai_and_chat_from_current_results(monkeypatch, tmp_path, analysis):
    from clipper_studio.ranking import chat_moment, merge_moments

    stale_spike = {**proposal(9), "start": 250, "end": 280, "peak": 265}
    new_spike = {**proposal(20), "start": 101, "end": 143, "peak": 119, "message_sample": ["Wow"]}
    monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([stale_spike], []))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
        app.state.pending.join()
        original = app.state.store.get(jid)["moments"]
        assert [m["sources"] for m in original] == [["ai"], ["chat"]]
        monkeypatch.setattr("clipper_studio.api.analyze_media",
                            lambda *a, **k: pytest.fail("chat retry must not rerun AI"))
        monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([new_spike], []))
        assert client.post(f"/api/jobs/{jid}/chat/retry").status_code == 202
        app.state.pending.join()
        moment, = app.state.store.get(jid)["moments"]
        expected, = merge_moments([proposal()], [chat_moment(new_spike)], 600, 18)
        assert moment == {**expected, "id": original[0]["id"]}
        assert app.state.store.get(jid)["ai_candidates"] == [proposal()]


def test_retry_preserves_edited_ai_only_moment_and_blocks_its_original_range(monkeypatch, tmp_path, analysis):
    monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([], []))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL}).json()["id"]
        app.state.pending.join()
        original, = app.state.store.get(jid)["moments"]
        base = f"/api/jobs/{jid}"
        edited = client.patch(f"{base}/moments/{original['id']}", json={"start": 300, "end": 330})
        assert edited.status_code == 200
        monkeypatch.setattr("clipper_studio.api.detect_spikes", lambda *a, **k: ([
            proposal(90),
            {**proposal(70), "start": 120, "end": 160, "peak": 143},
        ], []))
        assert client.post(f"{base}/chat/retry").status_code == 202
        app.state.pending.join()
        assert app.state.store.get(jid)["moments"] == [edited.json()]


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

    def preflight(url, require_chat=True, require_live=True):
        calls.append(require_chat)
        validate_metadata(info, require_chat, require_live)
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


def test_status_patch_preserves_fractional_boundaries(tmp_path):
    app = create_app(tmp_path)
    job = saved_job("fractional")
    job["moments"] = [{**proposal(), "id": "moment", "status": "suggested",
                       "start": 100.1, "end": 160.10000000000002, "duration": 60.00000000000002}]
    app.state.store.save(job)
    with TestClient(app) as client:
        url = "/api/jobs/fractional/moments/moment"
        response = client.patch(url, json={"status": "selected"})
        assert response.status_code == 200
        assert response.json()["start"] == 100.1 and response.json()["end"] == 160.10000000000002
        assert client.patch(url, json={"end": 160.10000000000002}).status_code == 200
        assert client.patch(url, json={"end": 160.101}).status_code == 422
        app.state.store.save({**job, "moments": [{**job["moments"][0], "start": 100.4,
                                                  "end": 159.7, "duration": 59.3}]})
        response = client.patch(url, json={"start": 100, "end": 160})
        assert response.status_code == 200
        assert response.json()["duration"] == 60
