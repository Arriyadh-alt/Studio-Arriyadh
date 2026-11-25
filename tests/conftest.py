import pytest


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
