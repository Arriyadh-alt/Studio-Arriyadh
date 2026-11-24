"""Normalize and rank independent chat and media suggestions."""

import uuid

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


def _original_ai(moment):
    if "ai" not in moment["sources"]:
        return None
    original = moment.get("alternatives", {}).get("ai") or moment.get("original_ai_range", {})
    start, end = original.get("start", moment["start"]), original.get("end", moment["end"])
    return {**moment, "start": start, "end": end, "duration": end - start}


def add_chat_moments(existing, chat, duration, lead, *, ai=None):
    protected, remaining, legacy_ai = [], [], []
    for moment in existing:
        original = _original_ai(moment)
        untouched = (moment["status"] == "suggested" and not moment.get("reviewed")
                     and not moment.get("edits") and not moment.get("clip_id"))
        if untouched:
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

    protected_ranges = [*protected, *(original for original in map(_original_ai, protected) if original)]
    fresh = [candidate for candidate in merge_moments(legacy_ai if ai is None else ai, chat, duration, lead)
             if not any(_matches(m, candidate) or max(m["start"], candidate["start"]) < min(m["end"], candidate["end"])
                        for m in protected_ranges)]
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
    return sorted([*protected, *fresh], key=lambda m: m["rank"])


def merge_moments(ai, chat, duration, lead):
    """Match overlap or nearby peaks, retaining source proposals and AI timing."""
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
                **({"podcast": item["podcast"]} if item.get("podcast") else {})}

    result = [normalize(a, "ai") for a in ai]
    for original in chat:
        c = normalize(original, "chat")
        matches = []
        for a in result:
            if a["sources"] != ["ai"]:
                continue
            if _matches(a, c):
                matches.append(a)
        if not matches:
            if not any(m["sources"] == ["chat"] and _matches(m, c) for m in result):
                result.append(c)
            continue
        a = min(matches, key=lambda a: abs(a["peak"] - c["peak"]))
        a["alternatives"] = {"ai": {k: a[k] for k in ("start", "end")},
                             "chat": {k: c[k] for k in ("start", "end")}}
        # Chat peak is delayed engagement; the AI proposal anchors the true event.
        start = max(0., min(a["start"], a["peak"] - lead))
        end = min(duration, max(a["end"], c["end"]), start + 60)
        if end < a["peak"]:
            end = min(duration, a["peak"] + 1)
            start = max(0., end - 60)
        a.update(sources=["ai", "chat"], start=start, end=end, duration=end - start,
                 chat_score=c["score"], score=min(100, max(a["score"], c["score"]) + 10),
                 reason=a["reason"] + "; corroborated by chat: " + c["reason"],
                 message_sample=c["message_sample"])
    result.sort(key=lambda m: -m["score"])
    for rank, moment in enumerate(result, 1):
        moment["rank"] = rank
    return result
