import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from shorts_clipper.editorial import backtrack_reaction, classify_scene
from shorts_clipper.moments import analyze_media, analyze_signals
from shorts_clipper.scoring import DEFAULT_WEIGHTS, editorial_score, score_tier, scoring_config


def word(start, text):
    return dict(start=start, end=start + 1, word=text)


def test_map_and_inventory_are_not_justified_by_loud_audio():
    levels = np.full(50, -10.)
    visuals = np.zeros((50, 4))
    map_words = [word(12, "Buka peta untuk cek lokasi."), word(21, "Anjir!")]
    scene = classify_scene(levels, visuals, map_words, 0, 44, 21)
    assert scene["label"] == "map_navigation"
    assert scene["low_value"] and scene["static"] and not scene["justified"]
    visuals[20] = [.7, 20, .4, .3]
    scene = classify_scene(levels, visuals, map_words, 0, 44, 21, laughter=.7)
    assert scene["visual_event"] and not scene["justified"]
    victory = [word(15, "We won with the last item in inventory.")]
    scene = classify_scene(levels, visuals, victory, 0, 44, 21)
    assert scene["label"] == "inventory" and scene["justified"]


def test_reaction_backtracks_past_25_seconds_on_specific_evidence():
    levels = np.full(150, -34.)
    excitement = np.zeros(150)
    excitement[91:101] = .65
    words = [word(67, "Look at the gate opening."), word(94, "Oh my god!")]
    evidence = backtrack_reaction(levels, excitement, None, words, 100)
    assert evidence["verified"] and evidence["trigger_time"] == 67
    assert evidence["subject"] == "Look at the gate opening."
    assert evidence["notice_time"] == 91 and evidence["peak_time"] == 100
    evidence = backtrack_reaction(levels, excitement, None, words[1:], 100)
    assert not evidence["verified"] and evidence["trigger_time"] is None


def test_map_opening_with_loud_reaction_is_hidden(monkeypatch):
    monkeypatch.delenv("CLIPPER_LLM_API_KEY", raising=False)
    levels = np.full(250, -22.)
    excitement, laughter = np.zeros(250), np.zeros(250)
    excitement[120] = laughter[120] = 1.
    visuals = np.tile([0., 3., 0., .2], (250, 1))
    visuals[118] = [.8, 30., .3, .2]
    transcript = [word(113, "Buka peta untuk cek lokasi."), word(120, "Astaga!")]
    m, = analyze_signals((levels, excitement, laughter), visuals, 250, 1, 18,
                         lambda *_: transcript, lambda *_: None)
    assert m["scene"]["label"] == "map_navigation"
    assert m["score_tier"] == "hidden" and "low_value_scene" in m["score_penalties"]


def test_hooks_are_grounded_deduplicated_and_start_covers_trigger(monkeypatch):
    monkeypatch.delenv("CLIPPER_LLM_API_KEY", raising=False)
    levels = np.full(380, -29.)
    excitement, laughter = np.zeros(380), np.zeros(380)
    excitement[150] = excitement[240] = .9

    def transcript(lo, _, peak):
        assert lo <= peak - 31
        return [word(peak - 31, "Lihat ada musuh di depan."),
                word(peak, "Astaga!"),
                word(peak + 6, "Ternyata jalan keluar ada di belakang.")]

    moments = analyze_signals((levels, excitement, laughter), None, 380, 2, 18, transcript, lambda *_: None)
    assert len(moments) == 2
    for m in moments:
        assert m["start"] < m["trigger"] < m["peak"] < m["reaction"]["conclusion_time"] <= m["end"]
        assert not m["needs_manual_review"]
        assert len(m["hooks"]) == 3
        assert all("ternyata" not in hook.lower() for hook in m["hooks"])
    assert len({hook for m in moments for hook in m["hooks"]}) == 6
    assert any("musuh" in hook for hook in moments[0]["hooks"])


def test_retry_can_reuse_transcript_checkpoints_from_shorter_windows(tmp_path, monkeypatch):
    levels = np.full(300, -30.)
    excitement = np.zeros(300)
    excitement[150] = .9
    monkeypatch.setattr("shorts_clipper.moments._audio", lambda *_: (levels, excitement, np.zeros(300)))
    keys = []

    def cached(stage, key, work):
        if stage == "transcript_generation":
            keys.append(key)
            return [word(135, "Look at the gate opening.")] if key == "speech-127-192" else []
        return work()

    moment, = analyze_media(tmp_path / "audio.mp4", None, 300, checkpoint=cached)
    assert keys == ["speech-127-192"]
    assert moment["transcript"] == "Look at the gate opening." and moment["trigger"] == 135


@pytest.mark.parametrize(("score", "tier"), [(54, "hidden"), (55, "needs_improvement"),
                                             (70, "review"), (85, "high_priority"), (100, "high_priority")])
def test_score_thresholds(score, tier):
    assert score_tier(score) == tier


def test_load_overrides_apply_calibration_and_explain_penalty(tmp_path, monkeypatch):
    (tmp_path / "viral-scoring.json").write_text(json.dumps({
        "weights": dict.fromkeys(DEFAULT_WEIGHTS, 0) | {"hook": 1},
        "penalties": {"misleading_hook": 6},
    }))
    passed = []

    def calibrate(weights):
        passed.append(weights)
        return weights | {"hook": .5, "story": .5}

    monkeypatch.setitem(sys.modules, "clipper_studio.feedback", SimpleNamespace(calibrated_weights=calibrate))
    config = scoring_config(tmp_path)
    score = editorial_score({"hook": 100, "story": 50}, {"misleading_hook"}, {"hook": "Example"}, config)
    assert passed[0]["hook"] == 1
    assert score["score"] == 69
    assert score["score_penalties"] == {"misleading_hook": 6}
    assert score["score_tier"] == "needs_improvement"
    assert score["score_explanation"] == {"hook": "Example"}
