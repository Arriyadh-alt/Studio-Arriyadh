import subprocess

import pytest

from shorts_clipper.moments import analyze_media


@pytest.fixture
def media(tmp_path):
    audio, video = tmp_path / "audio.wav", tmp_path / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "aevalsrc=0.002*sin(2*PI*220*t)+if(between(t\\,178\\,186)\\,0.7*(0.6+0.4*sin(2*PI*5*t))*sin(2*PI*440*t)\\,0):s=16000:d=360",
                    str(audio)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=black:s=64x36:r=1:d=360,drawbox=color=white:t=fill:enable='between(t,177,185)'",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    return audio, video


@pytest.mark.parametrize("with_video,with_words", [(True, True), (False, True), (False, False)])
def test_planted_reaction(media, with_video, with_words):
    audio, video = media
    calls, updates = [], []

    def transcribe(path, start, end, language):
        calls.append((path, start, end, language))
        return [{"start": 179, "end": 180, "word": "no way!"},
                {"start": 181, "end": 182, "word": "wkwk"}] if with_words else []

    results = analyze_media(audio, video if with_video else None, 360, clips=2,
                            transcribe_fn=transcribe, progress=lambda *args: updates.append(args))
    top = results[0]
    assert abs(top["peak"] - 180) <= 6
    assert 14 <= top["peak"] - top["start"] <= 22
    assert 0 <= top["start"] < top["peak"] < top["end"] <= 360
    assert 20 <= top["end"] - top["start"] <= 60
    assert 40 <= top["score"] <= 100
    assert len(top["title"]) <= 60
    assert (top["signals"]["visual"] is None) == (not with_video)
    assert all(0 <= v <= 1 for v in top["signals"].values() if v is not None)
    assert bool(top["transcript"]) == with_words
    assert calls and updates[-1][0] == 1
    assert all(a["end"] <= b["start"] or b["end"] <= a["start"]
               for i, a in enumerate(results) for b in results[i + 1:])


def test_flat_audio(tmp_path):
    path = tmp_path / "flat.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=220:sample_rate=16000:duration=180", str(path)], check=True)
    results = analyze_media(path, None, 180, transcribe_fn=lambda *args: [])
    assert not results or max(m["score"] for m in results) < 30


def test_errors_and_empty(tmp_path):
    with pytest.raises(ValueError):
        analyze_media(tmp_path / "missing", None, 0)
    with pytest.raises(RuntimeError):
        analyze_media(tmp_path / "missing", None, 180)


def test_optional_failures_and_refinement(media, monkeypatch):
    audio, _ = media
    monkeypatch.setenv("CLIPPER_LLM_API_KEY", "test")

    def fail(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr("shorts_clipper.moments.urllib.request.urlopen", fail)
    result = analyze_media(audio, audio.parent / "missing", 360, transcribe_fn=fail)
    assert result and result[0]["transcript"] == ""
    assert result[0]["signals"]["visual"] is None
