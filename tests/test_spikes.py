import pytest

from clipper_studio.chat import Message
from clipper_studio.spikes import detect_spikes, normalized_message, reaction, reaction_category


def stream():
    return [Message(t, str(i), "hello") for t in range(0, 600, 5) for i in range(2)]


def test_spikes_onset_peak_and_non_overlap():
    messages = stream()
    for onset in (100, 300, 500):
        for delta, count in ((0, 8), (5, 20), (10, 8)):
            messages.extend(Message(onset + delta, str(i), "KEKW!!!") for i in range(count))
    spikes, activity = detect_spikes(messages, 600, clips=3)
    assert [s["onset"] for s in spikes] == [82, 282, 482]
    assert [s["peak"] for s in spikes] == [105, 305, 505]
    assert [s["start"] for s in spikes] == [64, 264, 464]
    assert len(activity) == 120
    assert all(30 <= s["end"] - s["start"] <= 60 for s in spikes)
    assert all(a["end"] <= b["start"] for a, b in zip(spikes, spikes[1:]))


def test_empty_flat_and_boundaries():
    assert detect_spikes([], 100)[0] == []
    assert detect_spikes(stream(), 600)[0] == []
    for time in (0, 79, 99):
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
    messages = [Message(t, str(i), f"ordinary message {i}") for t in (0, 590) for i in range(100)]
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
    assert picks[0]['onset'] == 277
    assert picks[0]['start'] == 259


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
        messages += [Message(t, str(i), 'LOL' if i < laughing else f'ordinary message {i}') for i in range(100)]
    picks, series = detect_spikes(messages, 600)
    assert len(picks) == 1
    assert 295 <= picks[0]['peak'] <= 305
    assert picks[0]['onset'] == 277
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


@pytest.mark.parametrize('text', ['what', 'WHAT', 'WHAT is happening', 'what??', 'wait what!', 'wtf'])
def test_emphatic_what_is_shock(text):
    assert reaction_category(text) == 'shock'


@pytest.mark.parametrize('text', ['what is your favorite holiday', 'what is your favorite holiday?',
                                  'I wonder what happened', 'somewhat'])
def test_ordinary_what_is_not_shock(text):
    assert reaction_category(text) is None


def test_sparse_stream_detects_several_independent_reaction_pairs():
    messages = [Message(t, f'viewer{t}', 'ordinary message') for t in range(0, 1800, 10)]
    for t in (200, 600, 1000, 1400):
        messages += [Message(t, 'one', 'LOL'), Message(t + 5, 'two', 'OMG')]
    picks, _ = detect_spikes(messages, 1800)
    assert len(picks) == 4
    assert all(any(t <= pick['peak'] <= t + 5 for t in (200, 600, 1000, 1400)) for pick in picks)


def test_sparse_stream_rejects_single_author_reaction_and_copypasta_spam():
    messages = [Message(t, f'viewer{t}', 'ordinary message') for t in range(0, 1800, 10)]
    messages += [Message(t, 'spammer', 'LOL') for t in range(200, 240)]
    messages += [Message(600, 'spammer', text) for _ in range(20) for text in ('P', 'E', 'D', 'copypasta')]
    assert detect_spikes(messages, 1800)[0] == []


@pytest.mark.parametrize('texts', [
    ['The moon is made of cheese!', 'THE MOOOON IS MADE OF CHEESE', 'the moon is made of cheese.'],
    ['WWWWW', 'w', 'W!!!'],
    [':potato:', ':POTATO:', ':potato:!'],
    ['🥔🥔🥔', '🥔', '🥔!'],
])
def test_many_author_normalized_waves_are_reactions(texts):
    messages = [Message(300 + i % 3 * 5, str(i), texts[i % 3]) for i in range(30)]
    picks, series = detect_spikes(messages, 600)
    assert len(picks) == 1
    assert 300 <= picks[0]['peak'] <= 310
    assert series[60]['intensity'] == 10
    assert all(normalized_message(text) == normalized_message(texts[0]) for text in texts)


def test_distinct_chatter_surge_without_reaction_words_is_picked():
    messages = [Message(t, str(i), f'ordinary message {i}')
                for t in range(0, 600, 5) for i in range(5)]
    messages += [Message(t, str(i), f'unique sentence {i}')
                 for t in (295, 300, 305) for i in range(5, 60)]
    picks, series = detect_spikes(messages, 600)
    assert len(picks) == 1
    assert 295 <= picks[0]['peak'] <= 305
    assert picks[0]['onset'] == 277
    assert series[60]['intensity'] == 0
    assert set(series[60]) == {'time', 'count', 'baseline', 'intensity', 'score'}


def test_single_author_wave_does_not_count_as_many_reactors():
    messages = [Message(t, 'one', ':potato:') for t in range(295, 310) for _ in range(20)]
    picks, series = detect_spikes(messages, 600)
    assert picks == []
    assert max(point['intensity'] for point in series) == 0


def test_mass_w_wave_has_same_weight_as_copypasta_wave():
    def wave_score(text):
        _, series = detect_spikes([Message(300, str(i), text) for i in range(10)], 600)
        return series[60]['score']
    assert wave_score('WWWWW') == wave_score('the moon is made of cheese')


def test_wave_requires_three_distinct_authors():
    messages = [Message(300, str(i), ':potato:') for i in range(2) for _ in range(20)]
    picks, series = detect_spikes(messages, 600)
    assert picks == []
    assert series[60]['intensity'] == 0


@pytest.mark.parametrize('delay', [0, .5, 12, 25])
def test_chat_delay_estimates_moment_without_changing_scores(delay):
    messages = [Message(300, str(i), 'LOL') for i in range(20)]
    picks, series = detect_spikes(messages, 600, chat_delay=delay)
    reference, reference_series = detect_spikes(messages, 600, chat_delay=0)
    assert series == reference_series
    assert picks[0]['onset'] == 300 - delay
    assert picks[0]['start'] == 300 - delay - 18
    assert picks[0]['peak'] == reference[0]['peak'] == 300
    assert set(picks[0]) == {'onset', 'peak', 'score', 'start', 'end', 'message_sample'}


@pytest.mark.parametrize('delay', [-1, float('nan'), float('inf'), -float('inf')])
def test_invalid_chat_delay(delay):
    with pytest.raises(ValueError, match='Chat delay'):
        detect_spikes([], 600, chat_delay=delay)


@pytest.mark.parametrize('peak,moment', [(285, 270), (290, 278), (295, 283)])
def test_sparse_rise_crosses_only_one_empty_bin(peak, moment):
    messages = [Message(270, 'one', 'LOL'), Message(275, 'two', 'OMG')]
    messages += [Message(peak, str(i), 'LOL') for i in range(10)]
    picks, _ = detect_spikes(messages, 600, clips=1, chat_delay=12)
    assert picks[0]['peak'] == peak
    assert picks[0]['onset'] == moment
    assert picks[0]['start'] == moment - 18
    assert picks[0]['onset'] - picks[0]['start'] == 18


def test_sparse_onset_does_not_cross_long_empty_gap():
    messages = [Message(250, 'one', 'LOL'), Message(255, 'two', 'OMG')]
    messages += [Message(290, str(i), 'LOL') for i in range(10)]
    picks, _ = detect_spikes(messages, 600, clips=1, chat_delay=0)
    assert picks[0]['onset'] == 290


def test_sparse_onset_links_independent_authors_across_short_gap():
    messages = [Message(270, 'one', 'LOL')]
    messages += [Message(280, str(i), 'LOL') for i in range(10)]
    picks, _ = detect_spikes(messages, 600, clips=1, chat_delay=0)
    assert picks[0]['peak'] == 280
    assert picks[0]['onset'] == 270


def test_contiguous_rise_is_capped_at_thirty_seconds():
    messages = [Message(t, 'one', 'LOL') for t in range(250, 290, 5)]
    messages += [Message(290, str(i), 'LOL') for i in range(10)]
    picks, _ = detect_spikes(messages, 600, clips=1, chat_delay=0)
    assert picks[0]['onset'] == 260


@pytest.mark.parametrize('delay,moment', [(0, 280), (12, 280), (23, 277), (100, 270)])
def test_moment_uses_earlier_of_peak_delay_and_rise_without_double_subtraction(delay, moment):
    messages = [Message(t, str(i), 'LOL') for t in range(280, 300, 5) for i in range(2)]
    messages += [Message(300, str(i), 'LOL') for i in range(20)]
    picks, _ = detect_spikes(messages, 600, clips=1, chat_delay=delay)
    assert picks[0]['peak'] == 300
    assert picks[0]['onset'] == moment
    assert picks[0]['start'] == moment - 18
    assert picks[0]['end'] == min(picks[0]['start'] + 60, max(picks[0]['start'] + 30, 320))


def test_sparse_rise_does_not_cross_two_separate_empty_bins():
    messages = [Message(270, 'one', 'LOL'), Message(280, 'two', 'OMG')]
    messages += [Message(290, str(i), 'LOL') for i in range(20)]
    picks, _ = detect_spikes(messages, 600, clips=1, chat_delay=0)
    assert picks[0]['onset'] == 280


def test_sparse_rise_does_not_treat_nonreacting_chat_as_an_empty_bin():
    messages = [Message(t, str(i), 'LOL' if t == 285 or (t == 275 and i == 0)
                        else f'ordinary message {i}') for t in range(0, 600, 5) for i in range(2)]
    picks, _ = detect_spikes(messages, 600, clips=1, chat_delay=0)
    assert picks[0]['peak'] == picks[0]['onset'] == 285


def test_busy_rise_does_not_cross_an_empty_bin():
    messages = [Message(t, str(i), 'LOL' if t == 275 and i < 3 else f'ordinary message {i}')
                for t in range(0, 600, 5) if t not in (280, 285) for i in range(10)]
    messages += [Message(285, str(i), 'LOL') for i in range(20)]
    picks, _ = detect_spikes(messages, 600, clips=1, chat_delay=0)
    assert picks[0]['peak'] == picks[0]['onset'] == 285


@pytest.mark.parametrize('lead', [15, 18, 20])
@pytest.mark.parametrize('tail', [1, 20])
def test_long_delay_is_bounded_to_keep_peak_inside_nonoverlapping_clips(lead, tail):
    messages = [Message(t, str(i), 'LOL') for t in (200, 400) for i in range(20)]
    picks, _ = detect_spikes(messages, 600, lead=lead, tail=tail, chat_delay=100)
    assert len(picks) == 2
    for p in picks:
        assert p['onset'] == p['peak'] - 30
        assert p['start'] == max(0, p['onset'] - lead)
        assert p['start'] <= p['onset'] <= p['peak'] < p['end'] <= 600
        assert 30 <= p['end'] - p['start'] <= 60
    assert picks[0]['end'] <= picks[1]['start']


@pytest.mark.parametrize('delay,count', [(0, 2), (100, 0)])
def test_zero_tail_rejects_windows_ending_exactly_at_the_peak(delay, count):
    messages = [Message(t, str(i), 'LOL') for t in (200, 400) for i in range(20)]
    picks, _ = detect_spikes(messages, 600, tail=0, chat_delay=delay)
    assert len(picks) == count
    for p in picks:
        assert p['peak'] < p['end']
        assert p['end'] == min(600, p['start'] + 60, max(p['start'] + 30, p['peak']))


def test_moment_delay_clamps_at_start_and_empty_chat_stays_empty():
    assert detect_spikes([], 600, chat_delay=20)[0] == []
    picks, _ = detect_spikes([Message(5, str(i), 'LOL') for i in range(20)], 100)
    assert picks[0]['onset'] == picks[0]['start'] == 0


@pytest.mark.parametrize('bin_seconds', [2, 7, 10, 40])
def test_moment_invariants_with_nondefault_bins(bin_seconds):
    messages = [Message(t, str(i), 'LOL') for t in (200, 400) for i in range(20)]
    picks, _ = detect_spikes(messages, 600, bin_seconds=bin_seconds)
    assert len(picks) == 2
    for p in picks:
        assert p['start'] == max(0, p['onset'] - 18)
        assert p['start'] <= p['onset'] <= p['peak'] < p['end'] <= 600
        assert 30 <= p['end'] - p['start'] <= 60


@pytest.mark.parametrize('text', ['gg everyone', 'ggs all', 'see ya', 'thanks for the stream',
                                  'thank you for another great livestream'])
def test_sparse_late_outro_context_suppresses_copypasta_wave(text):
    messages = [Message(550, str(i), 'bop') for i in range(6)]
    messages += [Message(555, f'outro{i}', text) for i in range(2)]
    assert detect_spikes(messages, 600)[0] == []


def test_outro_context_requires_independent_authors():
    messages = [Message(550, str(i), 'bop') for i in range(6)]
    messages += [Message(555, 'one', 'thanks for the stream') for _ in range(100)]
    assert detect_spikes(messages, 600)[0]


@pytest.mark.parametrize('time,text', [(300, 'gg everyone'), (550, 'GG'), (550, 'WOOO LETSGOO')])
def test_midstream_group_gg_and_late_gameplay_hype_remain_pickable(time, text):
    messages = [Message(time, str(i), text) for i in range(20)]
    assert detect_spikes(messages, 600)[0]
