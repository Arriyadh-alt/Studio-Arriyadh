import pytest

from clipper_studio.chat import Message
from clipper_studio.spikes import detect_spikes, reaction, reaction_category


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
    for time in (0, 79):
        spikes, _ = detect_spikes([Message(time, str(i), "W") for i in range(20)], 100)
        assert len(spikes) == 1
        assert 0 <= spikes[0]["start"] < spikes[0]["end"] <= 100
    assert detect_spikes([Message(99, str(i), "W") for i in range(20)], 100)[0] == []


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
    _, normal = detect_spikes([Message(50, str(i), "ordinary message") for i in range(10)], 100)
    _, hype = detect_spikes([Message(50, str(i), "OMG") for i in range(10)], 100)
    assert hype[10]["score"] > normal[10]["score"]
    assert hype[10]["intensity"] == 10


def test_sustained_rise_keeps_peak_inside_window():
    messages = stream() + [Message(t, str(i), "W") for t in range(100, 200, 5) for i in range(t)]
    spikes, _ = detect_spikes(messages, 600)
    assert all(s["start"] <= s["peak"] < s["end"] and s["end"] - s["start"] <= 60 for s in spikes)


def test_start_and_outro_greeting_floods_not_picked():
    messages = [Message(t, str(i), text) for t, text in ((0, "hi"), (5, "hello"),
                (595, "bye bye krew")) for i in range(1000)]
    assert detect_spikes(messages, 600)[0] == []


def test_midstream_greeting_flood_not_picked():
    messages = stream() + [Message(300, str(i), "HEY everyone!!!") for i in range(200)]
    assert detect_spikes(messages, 600)[0] == []


def test_nonreacting_boundary_flood_and_flat_reactions_not_picked():
    messages = [Message(t, str(i), "ordinary message") for t in (0, 590) for i in range(100)]
    assert detect_spikes(messages, 600)[0] == []
    flat = [Message(t, str(i), "LOL") for t in range(0, 600, 5) for i in range(10)]
    assert detect_spikes(flat, 600)[0] == []


def test_single_author_spam_not_picked():
    messages = stream() + [Message(t, "spammer", text) for t in range(200, 240)
                           for text in ("P", "E", "D", "WOW OMG!!!", "copypasta" * 50)]
    assert detect_spikes(messages, 600)[0] == []


def test_big_volume_spike_detected():
    messages = [Message(t, str(i), "ordinary message") for t in range(0, 600, 5)
                for i in range(100)]
    messages += [Message(t, str(i), "WHAT!!!") for t in (295, 300, 305) for i in range(300)]
    picks, _ = detect_spikes(messages, 600)
    assert len(picks) == 1
    assert 295 <= picks[0]['peak'] <= 305
    assert picks[0]['onset'] == 295
    assert picks[0]['start'] == 277


def test_samples_are_reaction_first_and_deduplicated():
    messages = [Message(200, str(i), text) for i, text in enumerate(
        ['hello', 'ordinary message', 'OMG', 'OMG', 'LOL', 'WHAT!!!'])]
    picks, _ = detect_spikes(messages, 600)
    assert picks[0]['message_sample'][:3] == ['OMG', 'LOL', 'WHAT!!!']
    assert picks[0]['message_sample'].count('OMG') == 1


def test_reaction_categories_ignore_unrelated_emoji_and_caps():
    assert reaction_category(':face_with_tears_of_joy:') == 'laughter'
    assert reaction_category(':rolling_on_the_floor_laughing:') == 'laughter'
    assert reaction_category(':skull:') == 'shock'
    assert reaction_category('WHAT???') == 'shock'
    assert reaction_category('LETSGO 🔥 GG') == 'hype'
    assert reaction_category('W') == 'weak_hype'
    assert reaction_category('L') == 'weak_hype'
    assert reaction_category('WWW!!!') == 'weak_hype'
    assert all(not reaction(t) for t in (':potato:', ':new_moon:', 'GOT IT', 'I AM HERE', 'P', 'E', 'D'))


def test_reaction_spike_with_capped_total_volume():
    messages = []
    for t in range(0, 600, 5):
        laughing = 35 if t in (295, 300, 305) else 5
        messages += [Message(t, str(i), 'LOL' if i < laughing else 'ordinary message') for i in range(100)]
    picks, series = detect_spikes(messages, 600)
    assert len(picks) == 1
    assert 295 <= picks[0]['peak'] <= 305
    assert picks[0]['onset'] == 295
    assert all(point['count'] == 100 for point in series)


def test_laughter_outscores_weak_hype():
    messages = [Message(t, str(i), text) for t, text in ((200, 'W'), (400, 'LOL')) for i in range(30)]
    _, series = detect_spikes(messages, 600)
    assert series[80]['score'] > series[40]['score']


def test_farewell_and_giveaway_context_suppresses_reaction_flood():
    for context in ('bye everyone', 'I GOT IT giveaway'):
        messages = [Message(300, str(i), 'NOOOO') for i in range(30)]
        messages += [Message(310, f'context{i}', context) for i in range(30)]
        assert detect_spikes(messages, 600)[0] == []


def test_sparse_reactions_can_clear_threshold_without_volume_flood():
    messages = [Message(t, str(i), 'LOL') for t in (200, 400) for i in range(5)]
    picks, _ = detect_spikes(messages, 600)
    assert [p['peak'] for p in picks] == [200, 400]
