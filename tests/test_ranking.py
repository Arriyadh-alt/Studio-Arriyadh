import pytest

from clipper_studio.ranking import chat_moment, merge_moments


def moment(start=100, end=140, peak=118, score=70):
    return dict(start=start, end=end, peak=peak, score=score, title="Hype", reason="Shouting", category="hype")


@pytest.mark.parametrize("chat", [moment(120, 165, 141, 80), moment(130, 165, 140, 80)])
def test_agreement(chat):
    result = merge_moments([moment()], [chat], 600, 18)
    assert len(result) == 1
    m = result[0]
    assert m["sources"] == ["ai", "chat"] and m["score"] == 90
    assert m["alternatives"]["ai"] == {"start": 100, "end": 140}
    assert m["alternatives"]["chat"]["end"] == chat["end"]
    assert m["start"] == 100 and m["end"] == 160
    assert m["start"] <= m["peak"] <= m["end"] and m["duration"] <= 60
    assert "chat" in m["reason"]


def test_separate_ranked_and_empty():
    result = merge_moments([moment(score=50)], [moment(300, 340, 320, 90)], 600, 18)
    assert [m["rank"] for m in result] == [1, 2]
    assert result[0]["chat_score"] == 90 and result[0]["ai_score"] is None
    assert merge_moments([], [], 600, 18) == []
    assert chat_moment(moment(score=8))["score"] > chat_moment(moment(score=2))["score"]
