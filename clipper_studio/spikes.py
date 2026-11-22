"""Offline author-capped engagement analysis with a robust local baseline."""

import math
import re
from statistics import median


def greeting(text):
    text = re.sub(r":[\w-]+:", " ", text.casefold())
    text = re.sub(r"[^\w\s]", " ", text)
    return bool(re.match(r"^(?:hi+|h[ae]+llo+|hey+|bye+|goodbye|gn|first|good night|good morning|good evening|see you)\b", text.strip()))


def reaction(text):
    text = text.strip()
    return bool(text and not greeting(text) and (
        not any(c.isalnum() for c in text) or
        (text.isupper() and len(text) > 1) or
        re.search(r"(.)\1{2,}|\b(lol|lmao|kekw|wkwk|omg|what|wow|w|l)\b|[?!]{2,}|clip (?:it|that)|:[\w-]+:|[😂💀]", text, re.I)))


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
    reactors, signal = [], []
    for group in bins:
        weights = {}
        reacting = set()
        for message in group:
            author = message.author.strip().casefold()
            weight = 0 if greeting(message.text) else 2 if reaction(message.text) else 1
            weights[author] = max(weights.get(author, 0), weight)
            if weight == 2:
                reacting.add(author)
        reactors.append(reacting)
        signal.append(sum(weights.values()))
    radius = max(1, round(5 / bin_seconds))
    smooth = []
    for i in range(len(bins)):
        neighbors = range(max(0, i - radius), min(len(bins), i + radius + 1))
        weights = [(j, radius + 1 - abs(j - i)) for j in neighbors]
        smooth.append(sum(signal[j] * w for j, w in weights) / sum(w for _, w in weights))
    global_median = median(smooth)
    global_mad = median(abs(value - global_median) for value in smooth)
    window = max(1, math.ceil(baseline_seconds / bin_seconds))
    exclusion = max(radius, math.ceil(30 / bin_seconds))
    series = []
    for i, value in enumerate(smooth):
        history = [smooth[j] for j in range(max(0, i - window), min(len(bins), i + window + 1))
                   if abs(j - i) > exclusion]
        if len(history) >= max(8, window // 2):
            baseline = median(history)
            mad = median(abs(v - baseline) for v in history)
        else:
            baseline, mad = global_median, global_mad
        scale = max(1, 1.4826 * mad, math.sqrt(baseline), baseline * .15)
        score = max(0, (value - baseline) / scale)
        series.append({"time": i * bin_seconds, "count": len(bins[i]), "baseline": baseline,
                       "intensity": len(reactors[i]), "score": round(score, 3)})
    positive = sorted(point['score'] for point in series if point['score'] > 0)
    threshold = max(3, positive[int((len(positive) - 1) * .9)] * .35) if positive else 3
    candidates = []
    for i, point in enumerate(series):
        if point['score'] < threshold:
            continue
        nearby = range(max(0, i - radius), min(len(bins), i + radius + 1))
        if any(smooth[j] > smooth[i] or (smooth[j] == smooth[i] and j < i) for j in nearby):
            continue
        reacting = set().union(*(reactors[j] for j in nearby))
        peak_authors = {m.author.strip().casefold() for m in bins[i]}
        if len(peak_authors) < 3 or len(reacting) < max(3, math.ceil(point['baseline'] * .05)):
            continue
        all_authors = {m.author.strip().casefold() for j in nearby for m in bins[j]}
        if (point['time'] < 60 or point['time'] >= duration - 60) and len(reacting) < .7 * max(1, len(all_authors)):
            continue
        onset = i
        while onset > 0 and (i - onset + 1) * bin_seconds <= 30:
            previous = onset - 1
            if signal[previous] <= series[previous]['baseline'] + max(1, math.sqrt(series[previous]['baseline'])):
                break
            onset = previous
        start = max(0, series[onset]['time'] - lead)
        end = min(duration, start + 60, max(start + 30, point['time'] + tail))
        if end - start < min(30, duration) or point['time'] >= end:
            continue
        samples = []
        for message in sorted(bins[i], key=lambda m: (greeting(m.text), not reaction(m.text))):
            if message.text.strip().casefold() not in {s.strip().casefold() for s in samples}:
                samples.append(message.text)
            if len(samples) == 5:
                break
        candidates.append({"onset": series[onset]['time'], "peak": point['time'],
                           "score": point['score'], "start": start, "end": end,
                           "message_sample": samples})
    chosen = []
    for candidate in sorted(candidates, key=lambda c: c['score'], reverse=True):
        if all(abs(candidate['peak'] - other['peak']) >= min_gap and
               (candidate['end'] <= other['start'] or candidate['start'] >= other['end']) for other in chosen):
            chosen.append(candidate)
            if len(chosen) == clips:
                break
    return sorted(chosen, key=lambda c: c['start']), series
