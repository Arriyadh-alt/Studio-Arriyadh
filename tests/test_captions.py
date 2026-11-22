from types import SimpleNamespace

import pytest

from shorts_clipper.captions import Caption, chunk_words, timestamp, to_ass, to_srt


def word(text, start, end):
    return SimpleNamespace(word=text, start=start, end=end)


def test_chunks_follow_word_times_and_punctuation():
    words = [word(t, i / 2, (i + 1) / 2) for i, t in enumerate([" One", "two", "three", "four", "five.", "Six"])]
    assert chunk_words(words) == [Caption(0, 2, "One two three four"), Caption(2, 2.5, "five."), Caption(2.5, 3, "Six")]


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
