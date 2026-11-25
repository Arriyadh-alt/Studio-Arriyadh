"""Normalize and rank independent chat and media suggestions."""

import uuid

from shorts_clipper.scoring import SCORE_FIELDS

from .spikes import reaction_category


def chat_moment(spike):
    samples = spike.get("message_sample", [])
    phrase = samples[0] if samples else ""
    category = {"laughter": "funny", "hype": "hype", "shock": "shock"}.get(reaction_category(phrase), "reaction")
    strength = max(0, float(spike.get("score", 0)))
    return {**spike, "score": round(100 * strength / (strength + 4)), "category": category,
            "title": (f'{category.title()}: "{phrase}"' if phrase else "Chat reaction highlight")[:60],
            "description": f"Viewers react to a {category} moment on stream.",
            "reason": f"Chat spiked with strength {strength:.1f} above its local baseline" +
                      (f"; viewers saying '{phrase[:80]}'" if phrase else "")}


def _matches(a, c):
    shorter = min(a["duration"], c["duration"])
    overlap = max(0, min(a["end"], c["end"]) - max(a["start"], c["start"]))
    return (shorter > 0 and overlap >= .4 * shorter) or abs(a["peak"] - c["peak"]) <= 25


def _overlaps(a, b):
    return max(a["start"], b["start"]) < min(a["end"], b["end"])


def _untouched(moment):
    return (moment["status"] == "suggested" and not moment.get("reviewed")
            and not moment.get("edits") and not moment.get("clip_id"))


def _chat_signature(moment):
    chat_range = moment.get("alternatives", {}).get("chat")
    if chat_range is None and moment["sources"] == ["chat"]:
        chat_range = moment.get("original_chat_range", moment)
    if chat_range is not None:
        return chat_range["start"], chat_range["end"]
    return None


def refresh_suggestions(existing, ai, chat, duration, lead, *, clips=()):
    result = list(existing)
    remaining = {i for i, moment in enumerate(existing) if _untouched(moment)}
    protected = [moment for i, moment in enumerate(existing) if i not in remaining]
    rank = max((moment["rank"] for moment in existing), default=0)
    for candidate in _retry_candidates(protected, ai, chat, duration, lead, clips=clips):
        nearby = {i for i, moment in enumerate(result) if abs(moment["peak"] - candidate["peak"]) <= 25}
        if nearby - remaining:
            continue
        if nearby:
            match = min(nearby, key=lambda i: (abs(result[i]["peak"] - candidate["peak"]), i))
            previous = result[match]
            if "ai" in previous["sources"] and "ai" not in candidate["sources"]:
                continue
            remaining.remove(match)
            result[match] = dict(candidate, id=previous["id"], rank=previous["rank"])
        else:
            rank += 1
            result.append(dict(candidate, rank=rank))
    return result


def _original_ai(moment):
    if "ai" not in moment["sources"]:
        return None
    original = moment.get("alternatives", {}).get("ai") or moment.get("original_ai_range", {})
    start, end = original.get("start", moment["start"]), original.get("end", moment["end"])
    return {**moment, "start": start, "end": end, "duration": end - start}


def _retry_candidates(protected, ai, chat, duration, lead, *, clips=()):
    protected_ranges = [*protected, *(original for original in map(_original_ai, protected) if original)]

    def conflicts(candidate):
        return (any(_matches(m, candidate) or _overlaps(m, candidate) for m in protected_ranges)
                or any(_overlaps(clip, candidate) for clip in clips))

    claimed = {signature for moment in protected if (signature := _chat_signature(moment)) is not None}
    return rank_moments(_merge_candidates(ai, chat, duration, lead, conflicts=conflicts, claimed=claimed, disjoint=True))


def add_chat_moments(existing, chat, duration, lead, *, ai=None):
    protected, remaining, legacy_ai = [], [], []
    for moment in existing:
        original = _original_ai(moment)
        if _untouched(moment):
            remaining.append(moment)
        else:
            protected.append(moment)
        if ai is None and original is not None:
            original["score"] = moment.get("ai_score")
            if original["score"] is None:
                original["score"] = moment["score"]
            if "chat" in moment["sources"]:
                original["reason"] = moment["reason"].partition("; corroborated by chat: ")[0]
                original["message_sample"] = []
            legacy_ai.append(original)

    fresh = _retry_candidates(protected, legacy_ai if ai is None else ai, chat, duration, lead)
    unmatched = []
    for candidate in fresh:
        match = next((m for m in remaining if (m["start"], m["end"]) == (candidate["start"], candidate["end"])), None)
        if match is not None:
            candidate["id"] = match["id"]
            remaining.remove(match)
        else:
            unmatched.append(candidate)
    for candidate in unmatched:
        match = min((m for m in remaining if _matches(m, candidate)),
                    key=lambda m: (("ai" in m["sources"]) != ("ai" in candidate["sources"]),
                                   abs(m["peak"] - candidate["peak"])), default=None)
        if match is not None:
            candidate["id"] = match["id"]
            remaining.remove(match)
    previous_order = {moment["id"]: i for i, moment in enumerate(existing)}
    result = [*fresh, *(dict(moment) for moment in protected)]
    result.sort(key=lambda m: previous_order.get(m["id"], len(existing)))
    return rank_moments(result)


def merge_moments(ai, chat, duration, lead):
    """Match overlap or nearby peaks, retaining source proposals and AI timing."""
    return rank_moments(_merge_candidates(ai, chat, duration, lead))


def _merge_candidates(ai, chat, duration, lead, *, conflicts=lambda moment: False, claimed=None, disjoint=False):
    def normalize(item, source):
        start = max(0., min(float(item["start"]), duration))
        end = max(start, min(float(item["end"]), duration, start + 60))
        score = max(0, min(100, round(item.get("score", 0))))
        return {"id": uuid.uuid4().hex, "rank": 0, "sources": [source],
                "start": start, "end": end, "peak": float(item.get("peak", start + lead)),
                "duration": end - start, "score": score,
                "ai_score": score if source == "ai" else None,
                "chat_score": score if source == "chat" else None,
                **{k: item.get(k, "") for k in ("title", "description", "reason", "transcript")},
                "category": item.get("category", "reaction"), "signals": item.get("signals", {}),
                "message_sample": item.get("message_sample", []), "status": "suggested", "clip_id": None,
                **({"podcast": item["podcast"]} if item.get("podcast") else {}),
                **{key: item[key] for key in SCORE_FIELDS if key in item}}

    result = [normalize(a, "ai") for a in ai]
    ai_moments = [a for a in result if not conflicts(a)]
    if disjoint:
        result = []
        for a in sorted(ai_moments, key=lambda m: -m["score"]):
            if not any(_overlaps(a, placed) for placed in result):
                result.append(a)
        ai_moments = result
    standalone = []
    for original in chat:
        c = normalize(original, "chat")
        signature = (c["start"], c["end"])
        if claimed is not None and signature in claimed:
            continue
        matches = [a for a in ai_moments if a["sources"] == ["ai"] and _matches(a, c)]
        for a in sorted(matches, key=lambda a: abs(a["peak"] - c["peak"])):
            combined = _corroborate(a, c, duration, lead)
            if not conflicts(combined) and not (
                    disjoint and any(other is not a and _overlaps(combined, other) for other in ai_moments)):
                a.update(combined)
                if claimed is not None:
                    claimed.add(signature)
                break
        else:
            if (not conflicts(c) and not any(_overlaps(a, c) for a in matches)
                    and not any(_matches(m, c) for m in standalone)):
                standalone.append(c)
                if claimed is not None:
                    claimed.add(signature)
    if not disjoint:
        return ai_moments + standalone
    placed = list(ai_moments)
    for candidate in sorted(standalone, key=lambda m: -m["score"]):
        if not any(_overlaps(candidate, other) for other in placed):
            placed.append(candidate)
    return placed


def _corroborate(a, c, duration, lead):
    alternatives = {"ai": {k: a[k] for k in ("start", "end")},
                    "chat": {k: c[k] for k in ("start", "end")}}
    # Chat peak is delayed engagement; the AI proposal anchors the true event.
    start, end = a["start"], a["end"]
    if "score_breakdown" not in a:
        start = max(0., min(start, a["peak"] - lead))
        end = min(duration, max(end, c["end"]), start + 60)
    if end < a["peak"]:
        end = min(duration, a["peak"] + 1)
        start = max(0., end - 60)
    return dict(a, alternatives=alternatives, sources=["ai", "chat"], start=start, end=end, duration=end - start,
                chat_score=c["score"],
                reason=a["reason"] + "; corroborated by chat: " + c["reason"], message_sample=c["message_sample"])


def rank_moments(moments):
    moments.sort(key=lambda m: (-m["score"], -(m.get("chat_score") or 0) if "ai" in m["sources"] else 0))
    for rank, moment in enumerate(moments, 1):
        moment["rank"] = rank
    return moments
