import json
import wave

import numpy as np
import pytest

from shorts_clipper import voices
from shorts_clipper.director import DirectorOptions
from shorts_clipper.voices import MOODS, PERSONAS, choose_voice, mood_for, narrate, script


def tone(seconds, rate=voices.RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(t * 2 * np.pi * 220) * 0.3).astype("<f4")


def test_scene_picks_the_delivery_and_a_matching_voice():
    assert mood_for("karma") == "humor" and mood_for("scare") == "suspense" and mood_for("wholesome") == "warm"
    assert mood_for(None) == "excited" and mood_for("karma", "surprise") == "surprise"
    assert choose_voice("auto", "suspense") == "trailer" and choose_voice("bestie", "suspense") == "bestie"
    assert choose_voice("nope", "humor") == "bestie"
    assert set(voices.AUTO_VOICE.values()) <= set(PERSONAS)


def test_script_reacts_before_the_line_with_the_mood_and_persona():
    lines = script('Why Did Valkyrae Say "Thank You"?', "en", "surprise", "hype", "a")
    assert lines[0].lead and lines[0].text in voices.LEAD_INS["en"]["surprise"]
    assert lines[1].text == 'Why Did Valkyrae Say "Thank You"?'
    rate, pitch, _, pause = MOODS["surprise"]
    assert lines[1].rate == rate + PERSONAS["hype"].rate and lines[1].pitch == pitch + PERSONAS["hype"].pitch
    assert lines[0].pause == pause and lines[0].rate > lines[1].rate
    assert script("Windah Kena Karma", "id", "humor", "bestie")[0].text in voices.LEAD_INS["id"]["humor"]
    assert script("Same line", "en", "excited", "hype", "a") == script("Same line", "en", "excited", "hype", "a")


def test_suspense_holds_back_the_payoff():
    lines = script('Why Did Valkyrae Say "Thank You"?', "en", "suspense", "trailer")
    assert [l.text for l in lines[1:]] == ["Why Did Valkyrae Say...", '"Thank You"?']
    assert lines[1].pause > 0 and lines[2].pitch < lines[1].pitch
    assert script("Valkyrae Was Not Ready", "en", "suspense", "trailer")[-1].text.endswith(".")


def test_narrate_writes_a_polished_wav(monkeypatch, tmp_path):
    seen = {}

    def neural(lines, voice, work):
        seen["voice"] = voice
        return [np.concatenate([np.zeros(2400, "<f4"), tone(0.6), np.zeros(2400, "<f4")]), tone(1.4)]

    monkeypatch.setattr(voices, "_neural", neural)
    out = tmp_path / "hook.wav"
    info = narrate("Momen Ketika Windah Terkena Karma!", "id", out, tmp_path, "auto", "karma", "s")
    assert seen["voice"] == "id-ID-GadisNeural" and info["engine"] == "neural" and info["mood"] == "humor"
    assert info["voice"] == "bestie" and info["script"].endswith("Momen Ketika Windah Terkena Karma!")
    with wave.open(str(out)) as wav:
        assert wav.getframerate() == 48000 and wav.getnchannels() == 1
        assert abs(wav.getnframes() / 48000 - info["seconds"]) < 0.05
    assert 1.9 < info["seconds"] < 2.6  # leading silence trimmed, the lead-in pause kept


def test_narrate_drops_the_lead_in_when_the_hook_runs_long(monkeypatch, tmp_path):
    monkeypatch.setattr(voices, "_neural", lambda lines, voice, work: [tone(1.0), tone(3.2)])
    info = narrate("A Very Long Hook Line", "en", tmp_path / "a.wav", tmp_path, "hype", "win")
    assert info["script"] == "A Very Long Hook Line!" and info["seconds"] <= voices.MAX_SECONDS + 0.05


def test_narrate_falls_back_to_the_offline_voice(monkeypatch, tmp_path):
    def broken(*a):
        raise OSError("no network")
    offline = []
    monkeypatch.setattr(voices, "_neural", broken)
    monkeypatch.setattr(voices, "_offline", lambda lines, language, cache, work: offline.append(lines) or
                        [tone(0.5) for _ in lines])
    info = narrate("Nobody Was Ready", "en", tmp_path / "b.wav", tmp_path, "trailer", "scare")
    assert info["engine"] == "offline" and offline and info["voice_name"] == "Movie Trailer"
    classic = narrate("Nobody Was Ready", "en", tmp_path / "c.wav", tmp_path, "classic", "shock")
    assert classic["engine"] == "offline" and len(offline) == 2


def test_director_options_accept_only_known_voices():
    options = DirectorOptions.from_dict({"voice": "bestie", "delivery": "humor"})
    assert (options.voice, options.delivery) == ("bestie", "humor")
    options = DirectorOptions.from_dict({"voice": "robot", "delivery": "angry"})
    assert (options.voice, options.delivery) == ("auto", "auto")


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from clipper_studio.api import create_app
    calls = []

    def fake(text, language, output, cache, voice, category, seed, delivery):
        calls.append((text, language, voice, category, delivery))
        if text == "fail":
            raise OSError("offline")
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1), wav.setsampwidth(2), wav.setframerate(48000)
            wav.writeframes(b"\0\0" * 4800)
        return {"seconds": 0.1, "voice": "hype", "voice_name": "Hype Streamer", "mood": "excited", "engine": "neural",
                "script": text}

    monkeypatch.setattr("clipper_studio.api.narrate", fake)
    with TestClient(create_app(tmp_path, render_workers=1)) as test_client:
        test_client.calls = calls
        yield test_client


def test_voice_catalog_and_preview(client):
    catalog = client.get("/api/voices").json()
    assert [v["id"] for v in catalog["voices"]] == list(PERSONAS) and catalog["deliveries"] == list(MOODS)
    body = {"voice": "hype", "text": "Can Ninja Do It?", "language": "en", "category": "win"}
    first = client.post("/api/voices/preview", json=body)
    assert first.status_code == 200 and first.headers["content-type"] == "audio/wav"
    assert first.content[:4] == b"RIFF" and first.headers["x-voice-name"] == "Hype Streamer"
    assert client.post("/api/voices/preview", json=body).content == first.content and len(client.calls) == 1
    assert client.post("/api/voices/preview", json={"language": "id"}).status_code == 200
    assert client.calls[-1][0] == voices.SAMPLES["id"] and client.calls[-1][2:] == ("auto", None, "auto")


def test_voice_preview_rejects_unknown_voices_and_reports_failures(client):
    assert client.post("/api/voices/preview", json={"voice": "robot"}).status_code == 422
    assert client.post("/api/voices/preview", json={"delivery": "angry"}).status_code == 422
    assert client.post("/api/voices/preview", json={"text": "fail"}).status_code == 503
    assert client.post("/api/jobs", json={"url": "https://www.youtube.com/watch?v=LIxBN4AHMEE",
                                          "director": {"voice": "robot"}}).status_code == 422


def test_hook_records_how_it_was_narrated():
    from shorts_clipper.director import _story_intro
    got = {}

    def speak(text, language, **kwargs):
        got.update(kwargs)
        return "v.wav", {"seconds": 3.1, "voice": "chill", "voice_name": "Chill Commentator", "mood": "warm",
                         "engine": "neural", "script": "Oh, wow. " + text}

    intro = _story_intro({"language": "en", "name": "Ninja", "voice": speak},
                         DirectorOptions(voiceover=True, voice="chill"), [], {"event": 5.0}, 20)
    assert got["voice"] == "chill" and got["delivery"] == "auto" and "category" in got
    assert intro["narration"]["voice_name"] == "Chill Commentator" and intro["duration"] >= 3.5
    assert json.dumps(intro)

    def broken(text, language, **kwargs):
        raise OSError
    intro = _story_intro({"language": "en", "voice": broken}, DirectorOptions(voiceover=True), [], {"event": 5.0}, 20)
    assert not intro["voiceover"] and "narration" not in intro


def test_voice_preview_cache_stays_bounded(client, monkeypatch):
    monkeypatch.setattr("clipper_studio.api.PREVIEW_CACHE", 3)
    for i in range(6):
        assert client.post("/api/voices/preview", json={"text": f"Line {i}"}).status_code == 200
    folder = client.app.state.store.root / "voice-previews"
    assert len(list(folder.glob("*.wav"))) == 3 and len(list(folder.glob("*.json"))) == 3
    assert client.post("/api/voices/preview", json={"text": "Line 5"}).status_code == 200 and len(client.calls) == 6
