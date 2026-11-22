from types import SimpleNamespace
from unittest.mock import patch

import pytest

from shorts_clipper.captions import (
    LINE_WIDTH, Caption, chunk_words, join_tokens, text_width, timestamp, to_ass, to_srt, wrap_caption,
)


def word(text, start, end):
    return SimpleNamespace(word=text, start=start, end=end)


def test_chunks_follow_word_times_and_punctuation():
    words = [word(t, i / 2, (i + 1) / 2) for i, t in enumerate([" One", "two", "three", "four", "five.", "Six"])]
    assert chunk_words(words) == [Caption(0, 2.5, "One two three four five."), Caption(2.5, 3, "Six")]


def test_chunks_split_on_pause_and_duration():
    assert len(chunk_words([word("Hi", 0, 1), word("there", 2, 3)])) == 2
    assert len(chunk_words([word("Hi", 0, 2), word("there", 2, 3)])) == 2


def test_empty_and_invalid_words():
    assert chunk_words([]) == []
    assert chunk_words([word(" ", 0, 1), word("bad", 1, 1)]) == []
    with pytest.raises(ValueError):
        chunk_words([], 0)


def test_srt_and_timestamp_rounding():
    assert to_srt([Caption(1.25, 2.5, "Hello!")]) == "1\n00:00:01,250 --> 00:00:02,500\nHello!\n\n"
    assert timestamp(59.9999) == "00:01:00,000"
    assert timestamp(59.9999, True) == "0:01:00.00"
    assert to_srt([]) == ""


def test_ass_style_and_literal_text():
    result = to_ass([Caption(0.5, 1, "{hello}\\N\nworld")])
    assert "PlayResY: 1920" in result
    assert "DejaVu Sans,76,&H00FFFFFF" in result
    assert "Dialogue: 0,0:00:00.50,0:00:01.00,Default,,0,0,0,,｛hello｝＼N world" in result
    assert "Dialogue:" not in to_ass([])


def test_minimum_duration_uses_silence_without_overlap():
    captions = chunk_words([word('Hello.', 0, .2), word('Next.', .5, .6), word('Last.', 2, 2.2)], clip_end=3)
    assert captions == [Caption(0, .5, 'Hello.'), Caption(.5, 1.3, 'Next.'), Caption(2, 2.8, 'Last.')]
    assert all(a.end <= b.start for a, b in zip(captions, captions[1:]))


@pytest.mark.parametrize('clip_end, expected', [(None, 10), (10, 10), (10.3, 10.3), (11, 10.6)])
def test_final_caption_never_extends_past_clip(clip_end, expected):
    captions = chunk_words([word('Last.', 9.8, 10)], clip_end=clip_end)
    assert captions[0].end == pytest.approx(expected)
    assert '00:00:10,600' not in to_srt(chunk_words([word('Last.', 9.8, 10)]))


def test_clip_cap_trims_or_omits_words_beyond_end():
    assert chunk_words([word('Last.', 9.8, 10.2), word('Extra.', 11, 12)], clip_end=10) == [Caption(9.8, 10, 'Last.')]
    assert chunk_words([], clip_end=10) == []


def test_orphan_question_and_college_merge():
    words = [word(t, i * .2, (i + 1) * .2) for i, t in enumerate(["So", "why'd", "I", "drop", "out?"])]
    assert chunk_words(words) == [Caption(0, 1, "So why'd I drop out?")]
    words = [word(t, i * .3, (i + 1) * .3) for i, t in enumerate(['I', 'never', 'graduated', 'from', 'college.'])]
    assert chunk_words(words) == [Caption(0, 1.5, 'I never graduated from college.')]


def test_orphan_does_not_merge_across_pause_or_punctuation():
    assert len(chunk_words([word('Hello.', 0, 1), word('Next', 1, 2)])) == 2
    assert len(chunk_words([word('Hello', 0, 1), word('Next', 2, 3)])) == 2


def test_quoted_orphan_preserves_space_after_number():
    words = [word(t, i * .2, (i + 1) * .2)
             for i, t in enumerate([' Say', ' the', ' number', ' 5', " 'yes'"])]
    assert chunk_words(words) == [Caption(0, 1, "Say the number 5 'yes'")]


@pytest.mark.parametrize('stem, suffix', [('Wi', '-Fi'), ('don', "'t")])
def test_pause_splits_continuation_tokens(stem, suffix):
    captions = chunk_words([word(' ' + stem, 0, .2), word(suffix, 2, 2.2)])
    assert captions == [Caption(0, .8, stem), Caption(2, 2.2, suffix)]


@pytest.mark.parametrize('stem, suffix, expected', [('Wi', '-Fi', 'Wi-Fi'), ('don', "'t", "don't")])
def test_continuations_stay_in_group_at_word_limit(stem, suffix, expected):
    words = [word(t, i * .2, (i + 1) * .2)
             for i, t in enumerate([' Use', ' the', ' new', ' ' + stem, suffix, ' next.'])]
    captions = chunk_words(words)
    assert captions[0].text == 'Use the new ' + expected + ' next.'
    words[-1].start, words[-1].end = 3, 3.2
    captions = chunk_words(words)
    assert captions[0].text == 'Use the new ' + expected
    assert captions[0].end == 1
    assert captions[1].text == 'next.'


@pytest.mark.parametrize('stem, suffix, expected', [('Wi', '-Fi', 'Wi-Fi'), ('don', "'t", "don't")])
def test_orphan_merge_rejoins_original_continuation_tokens(stem, suffix, expected):
    words = [word(t, i * .2, (i + 1) * .2)
             for i, t in enumerate([' Use', ' the', ' new', ' word', ' ' + stem, suffix])]
    assert chunk_words(words) == [Caption(0, pytest.approx(1.2), 'Use the new word ' + expected)]


@pytest.mark.parametrize('tokens, expected', [
    ([' as', ' a', ' drop', '-in'], 'as a drop-in'),
    (['drop', '-', 'in', ','], 'drop-in,'),
    (['That', "'s", ' great', '!'], "That's great!"),
    (['24', '-hour'], '24-hour'),
    ([' 24', '-hour'], '24-hour'),
    ([' 24', ' -hour'], '24 -hour'),
    ([' route', '-5'], 'route-5'),
])
def test_token_joining(tokens, expected):
    assert join_tokens(tokens) == expected


@pytest.mark.parametrize('tokens, expected', [
    (['-5', ' -3', ' -1'], '-5 -3 -1'),
    (['-5', '-3', '-1'], '-5 -3 -1'),
    (['in', " '99"], "in '99"),
    (['in', "'99"], "in '99"),
    (['drop', ' -in'], 'drop -in'),
    (['don', "'t"], "don't"),
    (['don', " 't"], "don 't"),
    (['don', '’t'], 'don’t'),
    (['value', '-5'], 'value -5'),
    ([], ''),
    ([' ', ''], ''),
])
def test_token_joining_preserves_meaningful_boundaries(tokens, expected):
    assert join_tokens(tokens) == expected


@pytest.mark.parametrize('text', ['of the finest universities', 'unwed graduate student.', 'W' * 90, '界' * 40])
def test_wrap_respects_safe_width(text):
    lines = wrap_caption(text)
    assert all(text_width(line) <= LINE_WIDTH for line in lines)
    assert ''.join(''.join(lines).split()) == ''.join(text.split())


def test_ass_asymmetric_safe_margins_and_explicit_wrap():
    result = to_ass([Caption(0, 1, 'of the finest universities')])
    assert ',90,190,420,1' in result
    assert r'\N' in result
    assert 'WrapStyle: 2' in result


def test_font_fallback_and_empty_wrap():
    with patch('shorts_clipper.captions.caption_font', return_value=None):
        assert all(text_width(line) <= LINE_WIDTH for line in wrap_caption('W' * 90))
    assert wrap_caption('') == []


def test_oversized_orphan_is_not_merged():
    captions = chunk_words([word('Hi', 0, .1), word('W' * 90, .1, .2)], max_words=1)
    assert len(captions) == 2
