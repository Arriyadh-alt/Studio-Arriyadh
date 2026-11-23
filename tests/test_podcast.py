import subprocess
import wave

import numpy as np
import pytest

from shorts_clipper import podcast as P

SCRIPT = [
    (0, "Welcome back to the show, today we talk about sleep and training."),
    (1, "Thanks for having me, I am glad to be here."),
    (0, "So what is the biggest mistake people make with sleep?"),
    (1, "The biggest mistake is the alarm clock. Research shows that waking at the same time every day matters "
        "more than going to bed early. Because your body clock anchors to light in the morning, the key is "
        "consistency. Most people never learn this lesson until they burn out. Here's why it works: light resets "
        "the clock."),
    (0, "That is surprising, I honestly never expected that."),
    (1, "And it gets better with one simple habit."),
    (0, "Let's talk about something different, money and investing."),
    (1, "I disagree with the popular advice that you should buy a house first. That's not true for most people."),
    (0, "That is a hot take, I don't think everyone would agree."),
    (1, "The data shows renting and investing the difference wins seventy percent of the time."),
    (0, "Wow, haha, that is wild. I can't believe that."),
    (1, "It is the truth, and nobody wants to hear it."),
]


def build(path, gap=0.5, rate=16000):
    words, audio, t = [], [], 0.0
    for speaker, text in SCRIPT:
        for token in text.split():
            length = 0.32
            words.append({"start": round(t, 3), "end": round(t + length, 3), "word": " " + token})
            n = int(length * rate)
            x = np.arange(n) / rate
            f = 115 if speaker == 0 else 235
            tone = (np.sin(2 * np.pi * f * x) + (0.6 if speaker == 0 else 0.1) * np.sin(4 * np.pi * f * x)
                    + (0.05 if speaker == 0 else 0.5) * np.sin(10 * np.pi * f * x))
            audio.append(0.3 * tone * np.hanning(n))
            audio.append(np.zeros(int(0.06 * rate)))
            t += length + 0.06
        audio.append(np.zeros(int(gap * rate)))
        t += gap
    samples = (np.concatenate(audio) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return words, len(samples) / rate


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    path = tmp_path_factory.mktemp("podcast") / "episode.wav"
    words, duration = build(path)
    return path, words, duration


def test_speakers_are_separated(episode):
    path, words, duration = episode
    groups = P.utterances(words)
    labels = P.diarize(groups, P.voice_features(path, duration))
    items = P.sentences(words, groups, labels)
    assert len(set(labels)) == 2
    first = {s["text"].split()[0]: s["speaker"] for s in items}
    assert first["Welcome"] != first["Thanks"]
    assert first["So"] == first["Welcome"]


def test_candidates_are_complete_and_ranked(episode, monkeypatch):
    path, words, duration = episode
    monkeypatch.delenv("CLIPPER_LLM_API_KEY", raising=False)
    moments = P.analyze_podcast(path, None, duration, clips=3, transcribe_fn=lambda *a: words)
    assert moments and len(moments) <= 3
    best = moments[0]
    for key in ("title", "start", "end", "score", "category", "reason", "transcript"):
        assert best[key] not in (None, "")
    info = best["podcast"]
    assert set(info["dimensions"]) == set(P.FILTERS)
    assert 0 <= info["context"]["score"] <= 100 and info["hook"]["strength"] >= 0
    assert P.MIN_CLIP - 1 <= best["end"] - best["start"] <= P.MAX_CLIP + 1
    starts = {round(w["start"], 2) for w in words}
    assert any(abs(best["start"] + 0.15 - s) < 0.02 for s in starts)  # starts on a sentence
    assert info["commentary"] and P.grounded(info["commentary"], info["before"] + " " + best["transcript"])


def test_answer_without_its_question_loses_context():
    items = [{"start": 0, "end": 3, "text": "Why does sleep matter?", "speaker": 0, "words": 4, "terminal": True,
              "question": True, "topic": 0},
             {"start": 3.5, "end": 9, "text": "Because the body clock anchors to morning light.", "speaker": 1,
              "words": 8, "terminal": True, "question": False, "topic": 0}]
    topics = [{"first": 0}]
    with_question = P.context_check(items, topics, 0, 1)[0]
    without = P.context_check(items, topics, 1, 1)
    assert with_question > without[0]
    assert any("question" in note for note in without[1])


def test_ungrounded_commentary_is_rejected():
    assert P.grounded("The guest explains why sleep timing matters", "sleep timing matters a lot")
    assert not P.grounded("The guest was arrested in Paris", "sleep timing matters a lot")


from shorts_clipper import speakers as S

SEATS = [{"x": 0.25, "y": 0.4, "w": 0.1, "h": 0.18, "presence": 1.0},
         {"x": 0.75, "y": 0.4, "w": 0.1, "h": 0.18, "presence": 1.0}]


def test_seats_group_detections_left_to_right():
    faces = [[(0.2, 0.3, 0.1, 0.18), (0.7, 0.3, 0.1, 0.18)] for _ in range(20)] + [[(0.45, 0.1, 0.05, 0.05)]]
    seats = S.find_seats(faces)
    assert [round(s["x"], 2) for s in seats] == [0.25, 0.75]


def test_voices_match_faces_by_mouth_motion():
    times = np.arange(0, 20, 0.2)
    turns = [[0, 5, 0], [5, 10, 1], [10, 15, 0], [15, 20, 1]]
    voices = S.activity(turns, times, 2)
    motion = np.array([voices[1] * 4, voices[0] * 4])  # speaker 0 sits on the right
    mapping, confidence = S.match_speakers(turns, times, motion, SEATS)
    assert mapping == {0: 1, 1: 0} and confidence > 0.5


def test_switching_holds_shots_and_ignores_interjections():
    turns = [[0, 6, 0], [6, 6.4, 1], [6.5, 12, 0], [12, 20, 1], [20, 21, 0], [21.1, 30, 1]]
    shots, _ = S.plan_shots(turns, 30, {0: 0, 1: 1})
    assert [s[2] for s in shots] == [0, 1]
    assert all(b - a >= S.MIN_SHOT for a, b, _ in shots)
    assert abs(shots[1][0] - 12) < 0.3
    path = S.shot_path(shots, SEATS, 0.32)
    assert path[0][1] < 0.5 < path[-1][1]


def test_listener_reaction_gets_one_cutaway():
    times = np.arange(0, 20, 0.2)
    motion = np.zeros((2, len(times)))
    motion[1, (times >= 8) & (times <= 9)] = 5
    shots, cutaways = S.plan_shots([[0, 20, 0]], 20, {0: 0, 1: 1}, motion, times)
    assert cutaways == 1 and [s[2] for s in shots] == [0, 1, 0]


@pytest.mark.parametrize("layout", ["speaker", "split", "switch", "wide", "auto"])
def test_layouts(layout):
    times = np.arange(0, 20, 0.2)
    frames = np.zeros((len(times), 180, 320), dtype=np.uint8)
    faces = [[(0.2, 0.3, 0.1, 0.18), (0.7, 0.3, 0.1, 0.18)] for _ in times]
    sample = {"times": times, "frames": frames, "faces": faces, "aspect": 16 / 9}
    plan = S.frame_conversation(None, 20, [[0, 10, 0], [10, 20, 1]], layout, sample=sample)
    expected = {"auto": "switch"}.get(layout, layout)
    assert plan["layout"] == expected
    if layout == "wide":
        assert plan["mode"] == "blur" and plan["path"] is None
    else:
        assert plan["path"] and all(0 <= x <= 1 for _, x, _ in plan["path"])
    if layout == "split":
        assert plan["overlay"].rect.h == S.SPLIT_BAND and plan["overlay"].rect.y == 0


from shorts_clipper import director as D
from shorts_clipper.captions import Caption


def _captions(times):
    return [Caption(a, b, w, [D.Word(a, b, w)]) for a, b, w in times]


def test_podcast_director_trims_long_pauses_only_and_keeps_voices():
    words = [(i * 0.5 + (3.0 if i >= 20 else 0), i * 0.5 + 0.4 + (3.0 if i >= 20 else 0), "never" if i == 30 else "word")
             for i in range(40)]
    levels = np.full(int(26 / D.STEP), -20.0)
    levels[int(10.1 / D.STEP):int(13 / D.STEP)] = -60
    plan = D.plan_podcast_edit(levels, _captions(words), 26, D.DirectorOptions(preset="podcast"),
                               hook={"start": 15.0, "end": 17.0, "text": "Never do this"})
    assert plan.segments[0].role == "hook" and plan.hook["text"] == "Never do this"
    assert 1.8 <= plan.removed < 3.0  # the 3 s pause shrinks to a natural breath
    assert not plan.sfx and not plan.shakes and not plan.flashes and not plan.voiceover
    assert plan.voice_clarity and all(z[2] <= 0.06 for z in plan.zooms)
    assert D.VOICE_CLARITY in " ".join(D.mix_command(D.Path("in.mp4"), D.Path("out.mp4"), plan))


def test_commentary_version_is_labelled_and_separate():
    words = [(i * 0.5, i * 0.5 + 0.4, "word") for i in range(40)]
    options = D.DirectorOptions(preset="podcast", commentary="Speaker 2 explains the alarm clock rule.")
    plan = D.plan_podcast_edit(np.full(200, -20.0), _captions(words), 20, options,
                               voice=lambda text: ("/tmp/vo.wav", {"seconds": 2.5}))
    assert plan.segments[0].role == "intro" and plan.voiceover == "/tmp/vo.wav"
    assert plan.hook["style"] == "commentary" and plan.hook["text"].startswith(D.COMMENTARY_LABEL)
    assert plan.commentary["label"] == D.COMMENTARY_LABEL and plan.summary()["commentary"]["voiced"]
    assert D.DirectorOptions.from_dict({"preset": "podcast", "commentary": "x"}).preset == "podcast"
