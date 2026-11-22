import pytest


@pytest.fixture(autouse=True)
def no_director_audio(monkeypatch, request):
    """API tests stub ffmpeg, so AI Director falls back to unedited renders there."""
    if request.module.__name__.endswith("test_director"):
        return

    def unavailable(*args):
        raise RuntimeError("no audio")
    monkeypatch.setattr("clipper_studio.api.loudness", unavailable)
