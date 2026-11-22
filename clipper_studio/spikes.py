"""Offline reaction-led highlight analysis with robust local baselines."""

import math
import re
from statistics import median


def greeting(text):
    text = re.sub(r":[\w-]+:", " ", text.casefold())
    text = re.sub(r"[^\w\s]", " ", text)
    return bool(re.match(r"^(?:hi+|h[ae]+llo+|hey+|bye+|goodbye|gn|first|good night|good morning|good evening|see you)\b", text.strip()))


def reaction_category(text):
    if not text.strip() or greeting(text):
        return None
    if re.search(r"\b(?:lo+l|lmao+|lmfao+|(?:ha){2,}|kek\w*|wkwk\w*|xd+)\b|[😂🤣]|:(?:face_with_tears_of_joy|rolling_on_the_floor_laughing|laughing|joy):", text, re.I):
        return "laughter"
    if re.fullmatch(r"\s*[wl]+[!\s]*", text, re.I):
        return "weak_hype"
    if re.search(r"\b(?:omg|wtf|what|wow|no{3,})\b|\?{2,}|!{2,}|[💀☠😱😮]|:(?:skull|skull_and_crossbones|face_screaming_in_fear|astonished_face|face_with_open_mouth):", text, re.I):
        return "shock"
    if re.search(r"\b(?:let[’']?s\s*go+|letsgo+|gg+|yay+|yes{2,})\b|clip (?:it|that)|[🔥🎉]|:(?:fire|party_popper):", text, re.I):
        return "hype"
    return None


def reaction(text):
    return reaction_category(text) is not None


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
    reactors, signal, totals = [], [], []
    category_weights = {None: 0, "weak_hype": .25, "hype": .75, "laughter": 1, "shock": 1}
    for group in bins:
        weights = {}
        reacting = set()
        for message in group:
            author = message.author.strip().casefold()
            weight = category_weights[reaction_category(message.text)]
            weights[author] = max(weights.get(author, 0), weight)
            if weight:
                reacting.add(author)
        reactors.append(reacting)
        signal.append(sum(weights.values()))
        totals.append(len(weights))
    radius = max(1, round(5 / bin_seconds))

    def smooth_values(values):
        smoothed = []
        for i in range(len(bins)):
            neighbors = range(max(0, i - radius), min(len(bins), i + radius + 1))
            weights = [(j, radius + 1 - abs(j - i)) for j in neighbors]
            smoothed.append(sum(values[j] * w for j, w in weights) / sum(w for _, w in weights))
        return smoothed

    smooth = smooth_values(signal)
    volume = smooth_values(totals)
    shares = [count / max(1, total) for count, total in zip(smooth, volume)]
    window = max(1, math.ceil(baseline_seconds / bin_seconds))
    exclusion = max(radius, math.ceil(30 / bin_seconds))

    def robust_series(values, share=False):
        global_median = median(values)
        global_mad = median(abs(value - global_median) for value in values)
        points = []
        for i, value in enumerate(values):
            history = [values[j] for j in range(max(0, i - window), min(len(bins), i + window + 1))
                       if abs(j - i) > exclusion]
            if len(history) >= max(8, window // 2):
                baseline = median(history)
                mad = median(abs(v - baseline) for v in history)
            else:
                baseline, mad = global_median, global_mad
            floor = max(.05, math.sqrt(baseline * (1 - baseline) / max(1, volume[i]))) if share else max(.75, math.sqrt(baseline), baseline * .15)
            scale = max(1.4826 * mad, floor)
            points.append((baseline, max(0, (value - baseline) / scale)))
        return points

    reaction_stats = robust_series(smooth)
    share_stats = robust_series(shares, share=True)
    volume_stats = robust_series(volume)
    series = []
    for i, (baseline, z) in enumerate(reaction_stats):
        score = .7 * z + .3 * min(z * 2, share_stats[i][1]) + .1 * min(3, volume_stats[i][1])
        series.append({"time": i * bin_seconds, "count": len(bins[i]), "baseline": baseline,
                       "intensity": len(reactors[i]), "score": round(score, 3)})
    positive = sorted(point['score'] for point in series if point['score'] > 0)
    threshold = max(2.5, positive[int((len(positive) - 1) * .9)] * .35) if positive else 2.5
    candidates = []
    for i, point in enumerate(series):
        if point['score'] < threshold:
            continue
        nearby = range(max(0, i - radius), min(len(bins), i + radius + 1))
        if any(series[j]['score'] > point['score'] or (series[j]['score'] == point['score'] and j < i) for j in nearby):
            continue
        reacting = set().union(*(reactors[j] for j in nearby))
        peak_authors = {m.author.strip().casefold() for m in bins[i]}
        if len(peak_authors) < 3 or len(reacting) < max(3, math.ceil(point['baseline'] * .05)):
            continue
        all_authors = {m.author.strip().casefold() for j in nearby for m in bins[j]}
        if (point['time'] < 60 or point['time'] >= duration - 60) and len(reacting) < .7 * max(1, len(all_authors)):
            continue
        context = [m for j in range(max(0, i - exclusion), min(len(bins), i + exclusion + 1)) for m in bins[j]]
        context_authors = {m.author.strip().casefold() for m in context}
        farewells = {m.author.strip().casefold() for m in context if re.search(r"\b(?:bye+|goodbye|see you|gn|good night)\b", m.text, re.I)}
        giveaways = {m.author.strip().casefold() for m in context if re.search(r"\b(?:got it|giveaway|free nitro|discord nitro|redeem)\b", m.text, re.I)}
        if len(farewells) >= max(3, .1 * len(context_authors)) or len(giveaways) >= max(3, .2 * len(context_authors)):
            continue
        onset = i
        while onset > 0 and (i - onset + 1) * bin_seconds <= 30:
            previous = onset - 1
            if signal[previous] <= series[previous]['baseline']:
                break
            onset = previous
        start = max(0, series[onset]['time'] - lead)
        end = min(duration, start + 60, max(start + 30, point['time'] + tail))
        if end - start < min(30, duration) or point['time'] >= end:
            continue
        samples = []
        for message in sorted(bins[i], key=lambda m: -category_weights[reaction_category(m.text)]):
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
