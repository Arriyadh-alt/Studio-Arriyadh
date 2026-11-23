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
                "message_sample": item.get("message_sample", []), "status": "suggested", "clip_id": None}

    result = [normalize(a, "ai") for a in ai]
    for original in chat:
        c = normalize(original, "chat")
        matches = []
        for a in result:
            if a["sources"] != ["ai"]:
                continue
            shorter = min(a["duration"], c["duration"])
            overlap = max(0, min(a["end"], c["end"]) - max(a["start"], c["start"]))
            if (shorter > 0 and overlap >= .4 * shorter) or abs(a["peak"] - c["peak"]) <= 25:
                matches.append(a)
        if not matches:
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
