"""Boundary proposals from source speech, action and reaction timing."""

import math
import re
from typing import Literal

DurationPreset = Literal["auto", "quick", "standard", "story", "extended"]
PRESET_RANGES = {"auto": (0, 60), "quick": (15, 25), "standard": (25, 40),
                 "story": (40, 55), "extended": (55, 60)}
MAX_DURATION = PRESET_RANGES["auto"][1]
SENTENCE_END = re.compile(r"""[.!?…]["'’”)]*$""")


def boundary_options(words, activity, duration, window, peak, trigger=None, *, observed=None):
    """Only offer presets supported by the evidence surrounding the same event."""
    start, end = window
    lo, hi = max(0, start - 6), min(duration, start + MAX_DURATION)
    if observed is not None:
        lo, hi = max(lo, observed[0]), min(hi, observed[1])
    activity_end = min(hi, len(activity))
    event = min(peak, trigger) if trigger is not None else peak
    protected_end = min(duration, peak + 3.5)
    starts, ends, evidence = [], [], []
    if lo == 0:
        starts.append(0.)
    if hi == duration:
        ends.append(duration)
    for i, word in enumerate(words):
        prev = words[i - 1] if i else None
        following = words[i + 1] if i + 1 < len(words) else None
        first = int(word["start"])
        if prev is None and first >= lo + 1 and activity[first - 1] < .22:
            starts.append(word["start"])
        elif prev and (SENTENCE_END.search(prev["word"].strip()) or word["start"] - prev["end"] >= .65):
            starts.append(max(prev["end"], word["start"] - .2))
        if SENTENCE_END.search(word["word"].strip()) or (following and following["start"] - word["end"] >= .65):
            ends.append(min(word["end"] + .35, following["start"] if following else duration))
        evidence.append((word["start"], word["end"]))
    for i in range(math.ceil(lo), math.floor(activity_end)):
        if activity[i] >= .22:
            evidence.append((float(i), float(i + 1)))
        elif not any(a - .35 < i < b + .35 for a, b in evidence):
            prev = next((w for w in reversed(words) if w["end"] <= i), None)
            following = next((w for w in words if w["start"] >= i), None)
            if not prev or not following or following["start"] - prev["end"] >= .65:
                starts.append(float(i))
                ends.append(float(i))
    starts = [a for a in starts if lo <= a <= event - .25]
    ends = [b for b in ends if protected_end <= b <= hi]

    # Follow activity through the reaction and payoff; don't lengthen a story with distant,
    # unrelated speech after the event has already settled.
    outcome = protected_end
    for a, b in sorted(evidence):
        if a > outcome + 2.5:
            break
        if a >= event - 2:
            outcome = max(outcome, b)
    outcome = min(hi, outcome)
    anchor = max(lo, min(start, event - .5))
    evidence_start = min((a for a, b in evidence if b >= anchor and a <= event), default=anchor)
    earliest = min(anchor, evidence_start)
    options = {}
    for preset, (minimum, maximum) in PRESET_RANGES.items():
        pairs = [(a, b) for a in starts for b in ends
                 if minimum <= b - a <= maximum and b >= outcome - .35
                 and a >= earliest - 1.5 and b <= max(outcome + 2.0, end + .35)]
        if pairs:
            a, b = min(pairs, key=lambda pair: (abs(pair[0] - anchor) + abs(pair[1] - outcome),
                                               pair[1] - pair[0]))
            options[preset] = {"start": a, "end": b}
    return options
