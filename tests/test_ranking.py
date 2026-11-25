from itertools import combinations
from random import Random

import pytest

from clipper_studio.ranking import add_chat_moments, chat_moment, merge_moments


def moment(start=100, end=140, peak=118, score=70):
    return dict(start=start, end=end, peak=peak, score=score, title="Hype", reason="Shouting", category="hype")


@pytest.mark.parametrize("chat", [moment(120, 165, 141, 80), moment(130, 165, 140, 80)])
def test_agreement(chat):
    result = merge_moments([moment()], [chat], 600, 18)
    assert len(result) == 1
    m = result[0]
    assert m["sources"] == ["ai", "chat"] and m["score"] == m["ai_score"] == 70
    assert m["alternatives"]["ai"] == {"start": 100, "end": 140}
    assert m["alternatives"]["chat"]["end"] == chat["end"]
    assert m["start"] == 100 and m["end"] == 160
    assert m["start"] <= m["peak"] <= m["end"] and m["duration"] <= 60
    assert "chat" in m["reason"]


def test_editorial_trigger_and_range_survive_chat_and_break_ties():
    ai = [moment(100, 140, 118), {
        **moment(200, 239, 221), "trigger": 205, "reaction": {"trigger_time": 205, "verified": True},
        "score_breakdown": {"hook": 65}, "score_tier": "review",
    }]
    merged, plain = merge_moments(ai, [moment(200, 254, 226, 90)], 600, 18)
    assert merged["score"] == plain["score"] == 70
    assert merged["sources"] == ["ai", "chat"] and merged["rank"] == 1
    assert merged["end"] == 239 and merged["trigger"] == merged["reaction"]["trigger_time"] == 205
    assert merged["score_breakdown"] == {"hook": 65}


def test_separate_ranked_and_empty():
    result = merge_moments([moment(score=50)], [moment(300, 340, 320, 90)], 600, 18)
    assert [m["rank"] for m in result] == [1, 2]
    assert result[0]["chat_score"] == 90 and result[0]["ai_score"] is None
    assert merge_moments([], [], 600, 18) == []
    assert chat_moment(moment(score=8))["score"] > chat_moment(moment(score=2))["score"]


def test_retry_matches_second_spike_when_first_extension_crosses_touched_moment():
    ai = [moment()]
    original, touched = merge_moments(ai, [moment(150, 180, 167, 60)], 600, 18)
    touched["reviewed"] = True
    chat = [moment(110, 158, 128, 90), moment(108, 138, 125, 75)]
    merged, kept = add_chat_moments([original, touched], chat, 600, 18, ai=ai)
    assert merged["sources"] == ["ai", "chat"]
    assert merged["chat_score"] == 75
    assert merged["id"] == original["id"]
    assert merged["end"] == 140 and kept == touched


def test_retry_does_not_match_a_spike_attributed_to_an_edited_moment():
    ai = [moment(100, 130, 118, 65), moment(162, 210, 164, 75)]
    spike = moment(140, 195, 143, 90)
    touched, = merge_moments(ai[1:], [spike], 600, 18)
    touched.update(reviewed=True, start=300, end=340, duration=40)
    untouched, = merge_moments(ai[:1], [], 600, 18)
    result = add_chat_moments([touched, untouched], [spike], 600, 18, ai=ai)
    assert result == [touched, {**untouched, "rank": 2}]


def test_retry_preserves_both_ai_ranges_when_chat_extension_would_overlap():
    ai = [moment(), moment(141, 180, 160, 55)]
    chat = [moment(110, 153, 128, 90)]
    existing = merge_moments(ai, [], 600, 18)
    assert merge_moments(ai, chat, 600, 18)[0]["end"] == 153
    assert add_chat_moments(existing, chat, 600, 18, ai=ai) == existing


def test_retry_untouched_ranges_never_overlap_each_other_or_touched_ranges():
    rng = Random(314159)

    def candidates(count):
        for _ in range(count):
            start = rng.randrange(20, 570)
            end = min(600, start + rng.randrange(20, 61))
            yield moment(start, end, rng.randrange(start, end), rng.randrange(1, 101))

    def overlaps(left, right):
        return max(left["start"], right["start"]) < min(left["end"], right["end"])

    def ai_key(m):
        source = m.get("alternatives", {}).get("ai", m)
        return (source["start"], source["end"], m["peak"])

    for _ in range(100):
        ai = list(candidates(6))
        existing = merge_moments(ai, candidates(10), 600, 18)
        protected = {}
        protected_ranges = []
        claimed = []
        for original in rng.sample(existing, 3):
            original["reviewed"] = True
            if "ai" in original["sources"]:
                original_range = original.get("alternatives", {}).get("ai")
                if original_range is None:
                    original_range = {k: original[k] for k in ("start", "end")}
                    original["original_ai_range"] = original_range
                protected_ranges.append({**original_range, "peak": original["peak"]})
            if original["sources"] == ["chat"]:
                original["original_chat_range"] = {k: original[k] for k in ("start", "end")}
            chat_range = original.get("alternatives", {}).get("chat") or original.get("original_chat_range")
            if chat_range:
                claimed.append((chat_range["start"], chat_range["end"]))
            moved = next(candidates(1))
            original.update(start=moved["start"], end=moved["end"], duration=moved["end"] - moved["start"])
            protected_ranges.append(original)
            protected[original["id"]] = original
        result = add_chat_moments(existing, candidates(10), 600, 18, ai=ai)
        untouched = [m for m in result if m["id"] not in protected]
        assert {m["id"]: m for m in result if m["id"] in protected} == {
            m["id"]: {**protected[m["id"]], "rank": m["rank"]} for m in result if m["id"] in protected
        }
        assert protected.keys() <= {m["id"] for m in result}
        assert not any(overlaps(a, b) for a, b in combinations(untouched, 2))
        assert not any(overlaps(a, b) for a in untouched for b in protected_ranges)
        represented_ai = [m for m in untouched if "ai" in m["sources"]]
        for proposal in ai:
            if any(overlaps(proposal, m) or abs(proposal["peak"] - m["peak"]) <= 25 for m in protected_ranges):
                continue
            if any(ai_key(proposal) != ai_key(m) and (
                    overlaps(proposal, m) or overlaps(proposal, m.get("alternatives", {}).get("ai", m)))
                   for m in represented_ai):
                continue
            assert ai_key(proposal) in {ai_key(m) for m in represented_ai}
        for m in untouched:
            chat_range = m.get("alternatives", {}).get("chat") or (m if m["sources"] == ["chat"] else None)
            if chat_range:
                signature = (chat_range["start"], chat_range["end"])
                assert signature not in claimed
                claimed.append(signature)
