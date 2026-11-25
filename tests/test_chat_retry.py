from fastapi.testclient import TestClient

from clipper_studio.api import create_app
from clipper_studio.chat import DownloadCancelled, _download_replay, fetch_metadata
from test_api import media
from test_chat import REPLAY_URL, replay_line, stub_yt_dlp


def test_chat_retry_retains_more_complete_buckets(stub_yt_dlp, tmp_path, monkeypatch, media):
    monkeypatch.setattr("clipper_studio.api.fetch_metadata", fetch_metadata)

    def download(operation, args, kwargs=None, *, timeout, check):
        if operation == "metadata":
            return fetch_metadata(*args, **kwargs)
        assert operation == "replay"
        return _download_replay(*args, lambda _: check())

    monkeypatch.setattr("clipper_studio.chat.run_download", download)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        jid = client.post("/api/jobs", json={"url": REPLAY_URL, "ai_analysis": False}).json()["id"]
        app.state.pending.join()
        original = app.state.store.get(jid)
        assert original["status"] == "completed" and original["message_count"] == 3
        buckets = tmp_path / jid / "chat/buckets.sqlite"
        previous_buckets = buckets.read_bytes()

        def partial(self, output, progress):
            output.with_suffix(".json.part").write_bytes(replay_line(1, 12, "Viewer"))
            raise DownloadCancelled("Disk guard tripped")

        stub_yt_dlp.download = partial
        assert client.post(f"/api/jobs/{jid}/chat/retry").status_code == 202
        app.state.pending.join()
        job = app.state.store.get(jid)
        assert job["message_count"] == original["message_count"]
        assert job["activity"] == original["activity"]
        assert job["moments"] == original["moments"]
        assert job["chat_error"] == original["chat_error"]
        assert buckets.read_bytes() == previous_buckets

        def complete(self, output, progress):
            output.write_bytes(b"".join(replay_line(i, 100 + i, "Viewer") for i in range(5)))

        stub_yt_dlp.download = complete
        assert client.post(f"/api/jobs/{jid}/chat/retry").status_code == 202
        app.state.pending.join()
        job = app.state.store.get(jid)
        assert job["message_count"] == 5
        assert buckets.read_bytes() != previous_buckets
