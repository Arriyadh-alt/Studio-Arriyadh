import pytest

from clipper_studio.chat import Message
from clipper_studio.spikes import detect_spikes, reaction


def stream():
    return [Message(t, str(i), "hello") for t in range(0, 600, 5) for i in range(2)]


def test_spikes_onset_peak_and_non_overlap():
    messages = stream()
    for onset in (100, 300, 500):
        for delta, count in ((0, 8), (5, 20), (10, 8)):
            messages.extend(Message(onset + delta, str(i), "KEKW!!!") for i in range(count))
    spikes, activity = detect_spikes(messages, 600, clips=3)
    assert [s["onset"] for s in spikes] == [100, 300, 500]
    assert [s["peak"] for s in spikes] == [105, 305, 505]
    assert [s["start"] for s in spikes] == [82, 282, 482]
    assert len(activity) == 120
    assert all(30 <= s["end"] - s["start"] <= 60 for s in spikes)
    assert all(a["end"] <= b["start"] for a, b in zip(spikes, spikes[1:]))


def test_empty_flat_and_boundaries():
    assert detect_spikes([], 100)[0] == []
    assert detect_spikes(stream(), 600)[0] == []
    for time in (0, 99):
        spikes, _ = detect_spikes([Message(time, str(i), "W") for i in range(20)], 100)
        assert len(spikes) == 1
        assert 0 <= spikes[0]["start"] < spikes[0]["end"] <= 100


def test_nms_top_n_and_lead():
    messages = stream() + [Message(t, str(i), "wow") for t in (100, 125, 300) for i in range(20)]
    spikes, _ = detect_spikes(messages, 600, clips=2, lead=20)
    assert len(spikes) == 2
    assert spikes[0]["start"] == spikes[0]["onset"] - 20
    assert spikes[1]["start"] >= spikes[0]["end"]


@pytest.mark.parametrize("kwargs", [{"lead":14}, {"lead":21}, {"clips":0}, {"bin_seconds":0}])
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        detect_spikes([], 100, **kwargs)


def test_reaction_intensity():
    assert all(reaction(t) for t in ("🔥🔥", "OMG", "nooooo", "LOL", "W", "KEKW", "wkwk", "!!!", ":fire:"))
    assert not reaction("ordinary message")
    _, normal = detect_spikes([Message(50, str(i), f"hello {i}") for i in range(10)], 100)
    _, spam = detect_spikes([Message(50, str(i), "same") for i in range(10)], 100)
    assert spam[10]["score"] > normal[10]["score"]


def test_sustained_rise_keeps_peak_inside_window():
    messages = stream() + [Message(t, str(i), "W") for t in range(100, 200, 5) for i in range(t)]
    spikes, _ = detect_spikes(messages, 600)
    assert all(s["start"] <= s["peak"] < s["end"] and s["end"] - s["start"] <= 60 for s in spikes)
