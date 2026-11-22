"""Deterministic rolling-baseline activity analysis."""

import math
import re
from collections import Counter
from statistics import median


def reaction(text):
    text = text.strip()
    return bool(text and (not any(c.isalnum() for c in text) or
                         (text.isupper() and len(text) > 1) or
                         re.search(r"(.)\1{2,}|\b(lol|w|kekw|wkwk)\b|!!!|:[\w-]+:", text, re.I)))


def detect_spikes(messages, duration, clips=8, lead=18, bin_seconds=5, baseline_seconds=180,
                  min_gap=60, tail=20):
    if not math.isfinite(duration) or duration <= 0 or not 15 <= lead <= 20 or not 1 <= clips <= 50:
        raise ValueError("Positive duration, 1-50 clips and lead within 15-20 seconds are required")
    if bin_seconds <= 0 or baseline_seconds < bin_seconds or min_gap < 0 or tail < 0:
        raise ValueError("Invalid detector configuration")
    bins = [[] for _ in range(math.ceil(duration / bin_seconds))]
    for message in messages:
        if 0 <= message.offset_seconds < duration:
            bins[int(message.offset_seconds // bin_seconds)].append(message)
    series = []
    window = max(1, int(baseline_seconds / bin_seconds))
    for i, group in enumerate(bins):
        counts = Counter(m.text.strip().casefold() for m in group)
        intensity = sum(reaction(m.text) or counts[m.text.strip().casefold()] >= 3 for m in group)
        weighted = len(group) + 0.25 * intensity
        history = [point["count"] for point in series[max(0, i - window):i]]
        baseline = median(history) if history else 0
        ratio = weighted / max(1, baseline)
        z = (weighted - baseline) / math.sqrt(max(1, baseline))
        score = max(0, (ratio - 1) + z * 0.3)
        series.append({"time": i * bin_seconds, "count": len(group), "baseline": baseline,
                       "intensity": intensity, "score": round(score, 3)})
    candidates = []
    i = 0
    while i < len(series):
        point = series[i]
        if point["count"] < 4 or point["score"] < 2:
            i += 1
            continue
        onset = i
        # Walk to the first elevated bin, not the loudest bin of a gradual rise.
        threshold = max(1, point["baseline"] * 1.3)
        while onset > max(0, i - 6) and series[onset - 1]["count"] > threshold:
            onset -= 1
        end_rise = i + 1
        while (end_rise < len(series) and series[end_rise]["score"] >= 1
               and series[end_rise]["time"] <= series[onset]["time"] + 60 - lead - bin_seconds):
            end_rise += 1
        peak = max(range(i, end_rise), key=lambda n: series[n]["score"])
        start = max(0, series[onset]["time"] - lead)
        end = min(duration, start + 60, max(start + 30, series[peak]["time"] + tail))
        sample = [m.text for m in bins[peak]][:5]
        candidates.append({"onset": series[onset]["time"], "peak": series[peak]["time"],
                           "score": series[peak]["score"], "start": start, "end": end,
                           "message_sample": sample})
        i = end_rise
    chosen = []
    for candidate in sorted(candidates, key=lambda c: c["score"], reverse=True):
        if all(abs(candidate["peak"] - other["peak"]) >= min_gap and
               (candidate["end"] <= other["start"] or candidate["start"] >= other["end"])
               for other in chosen):
            chosen.append(candidate)
            if len(chosen) == clips:
                break
    return sorted(chosen, key=lambda c: c["start"]), series
