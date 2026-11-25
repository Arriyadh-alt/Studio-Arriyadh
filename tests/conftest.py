import pytest


@pytest.fixture
def stub_export_validation(monkeypatch):
    monkeypatch.setattr("clipper_studio.api.source_frame_rate", lambda path: "30/1")
    monkeypatch.setattr("clipper_studio.api.has_audio", lambda path: True)
    monkeypatch.setattr("clipper_studio.api.source_size", lambda path: None)
    monkeypatch.setattr("clipper_studio.api.check_export", lambda *args, **kwargs: {"status": "passed"})


@pytest.fixture(autouse=True)
def no_unstubbed_api_media_download(monkeypatch):
    monkeypatch.setattr("clipper_studio.api.download_analysis_media", lambda *a, **k: (None, None))


@pytest.fixture(autouse=True)
def no_director_audio(monkeypatch, request):
    """API tests stub ffmpeg, so AI Director falls back to unedited renders there."""
    if request.module.__name__.endswith(("test_director", "test_podcast_api")):
        return

    def unavailable(*args):
        raise RuntimeError("no audio")
    monkeypatch.setattr("clipper_studio.api.loudness", unavailable)
