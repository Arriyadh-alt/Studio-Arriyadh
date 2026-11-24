"""Offline reaction-led highlight analysis with robust local baselines."""

import math
import re
import unicodedata
from functools import lru_cache
from statistics import median

from .chat_buckets import BIN_SECONDS, ChatBuckets


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
    emphatic_what = (re.fullmatch(r"\s*what[?!\.\s]*", text, re.I) or
                     re.search(r"\bWHAT\b", text) or re.search(r"\bwhat\s*[?!]", text, re.I))
    if emphatic_what or re.search(r"\b(?:omg|wtf|wow|no{3,})\b|\?{2,}|!{2,}|[💀☠😱😮]|:(?:skull|skull_and_crossbones|face_screaming_in_fear|astonished_face|face_with_open_mouth):", text, re.I):
        return "shock"
    if re.search(r"\b(?:let[’']?s\s*go+|letsgo+|gg+|yay+|yes{2,})\b|clip (?:it|that)|[🔥🎉]|:(?:fire|party_popper):", text, re.I):
        return "hype"
    return None


def reaction(text):
    return reaction_category(text) is not None


def normalized_message(text):
    text = ''.join(char for char in text.casefold() if not unicodedata.category(char).startswith('P'))
    return ' '.join(re.sub(r'(.)\1+', r'\1', text).split())


def detect_spikes(messages, duration, clips=8, lead=18, bin_seconds=5, baseline_seconds=180,
                  min_gap=60, tail=20, *, chat_delay=23):
    """Estimate moments from peak lag or rise onset, reaching back at most 30 seconds."""
    if not math.isfinite(duration) or duration <= 0 or not 15 <= lead <= 20 or not 1 <= clips <= 50:
        raise ValueError("Positive duration, 1-50 clips and lead within 15-20 seconds are required")
    if bin_seconds <= 0 or baseline_seconds < bin_seconds or min_gap < 0 or tail < 0:
        raise ValueError("Invalid detector configuration")
    if not math.isfinite(chat_delay) or chat_delay < 0:
        raise ValueError("Chat delay must be finite and nonnegative")
    if isinstance(messages, ChatBuckets):
        if bin_seconds != BIN_SECONDS or duration != messages.duration:
            raise ValueError("Replay buckets must match the analysis time base")
        with messages.reader() as (group, count):
            return _score(group, count, duration, clips, lead, bin_seconds, baseline_seconds, min_gap, tail, chat_delay)
    bins = [[] for _ in range(math.ceil(duration / bin_seconds))]
    for message in messages:
        if 0 <= message.offset_seconds < duration:
            bins[int(message.offset_seconds // bin_seconds)].append(message)
    return _score(bins.__getitem__, lambda i: len(bins[i]), duration, clips, lead, bin_seconds,
                  baseline_seconds, min_gap, tail, chat_delay)


def _score(read_group, count, duration, clips, lead, bin_seconds, baseline_seconds, min_gap, tail, chat_delay):
    length = math.ceil(duration / bin_seconds)
    radius = max(1, round(5 / bin_seconds))
    group = lru_cache(maxsize=3)(read_group)

    @lru_cache(maxsize=3)
    def normalized(i):
        return tuple(normalized_message(m.text) if not greeting(m.text) else '' for m in group(i))

    @lru_cache(maxsize=3)
    def authors_by_message(i):
        authors = {}
        for message, text in zip(group(i), normalized(i)):
            if text:
                authors.setdefault(text, set()).add(message.author.strip().casefold())
        return authors

    category_weights = {None: 0, "weak_hype": .25, "hype": .75, "laughter": 1, "shock": 1}
    @lru_cache(maxsize=3)
    def reactions(i):
        neighbors = range(max(0, i - radius), min(length, i + radius + 1))
        shared_authors = {}
        for j in neighbors:
            for text, authors in authors_by_message(j).items():
                shared_authors.setdefault(text, set()).update(authors)
        waves = {text for text, authors in shared_authors.items() if len(authors) >= 3}
        weights = {}
        reacting = set()
        active = set()
        group_weights = []
        for message, text in zip(group(i), normalized(i)):
            author = message.author.strip().casefold()
            weight = category_weights[reaction_category(message.text)]
            if text in waves:
                weight = max(weight, .75)
            group_weights.append(weight)
            if message.text.strip() and not greeting(message.text):
                active.add(author)
            weights[author] = max(weights.get(author, 0), weight)
            if weight:
                reacting.add(author)
        return reacting, active, group_weights, sum(weights.values())

    signal, totals, intensities = [], [], []
    for i in range(length):
        reacting, active, _, strength = reactions(i)
        signal.append(strength)
        totals.append(len(active))
        intensities.append(len(reacting))

    def smooth_values(values):
        smoothed = []
        for i in range(length):
            neighbors = range(max(0, i - radius), min(length, i + radius + 1))
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
            history = [values[j] for j in range(max(0, i - window), min(length, i + window + 1))
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
        reaction_score = .7 * z + .3 * min(z * 2, share_stats[i][1])
        score = reaction_score + .25 * volume_stats[i][1]
        series.append({"time": i * bin_seconds, "count": count(i), "baseline": baseline,
                       "intensity": intensities[i], "score": round(score, 3)})
    positive = sorted(point['score'] for point in series if point['score'] > 0)
    typical_volume = median(volume)
    threshold_floor = min(2.5, max(1.25, math.sqrt(typical_volume)))
    threshold = max(threshold_floor, positive[int((len(positive) - 1) * .9)] * .35) if positive else threshold_floor
    minimum_reactors = 2 if typical_volume < 3 else 3
    minimum_peak_authors = 1 if minimum_reactors == 2 else 3
    candidates = []
    for i, point in enumerate(series):
        if point['score'] < threshold:
            continue
        nearby = range(max(0, i - radius), min(length, i + radius + 1))
        if any(series[j]['score'] > point['score'] or (series[j]['score'] == point['score'] and j < i) for j in nearby):
            continue
        reacting = set().union(*(reactions(j)[0] for j in nearby))
        all_authors = set().union(*(reactions(j)[1] for j in nearby))
        peak_authors = {m.author.strip().casefold() for m in group(i)}
        volume_led = (volume_stats[i][1] >= threshold and
                      volume[i] - volume_stats[i][0] >= minimum_reactors)
        participants = all_authors if volume_led else reacting
        if len(peak_authors) < minimum_peak_authors or len(participants) < max(minimum_reactors, math.ceil(point['baseline'] * .05)):
            continue
        if (point['time'] < 60 or point['time'] >= duration - 60) and len(reacting) < .7 * max(1, len(all_authors)):
            continue
        context_authors, farewells, giveaways = set(), set(), set()
        for j in range(max(0, i - exclusion), min(length, i + exclusion + 1)):
            for m in group(j):
                author = m.author.strip().casefold()
                context_authors.add(author)
                if re.search(r"\b(?:bye+|goodbye|see you|gn|good night)\b", m.text, re.I):
                    farewells.add(author)
                if re.search(r"\b(?:got it|giveaway|free nitro|discord nitro|redeem)\b", m.text, re.I):
                    giveaways.add(author)
        if len(farewells) >= max(3, .1 * len(context_authors)) or len(giveaways) >= max(3, .2 * len(context_authors)):
            continue
        onset = i
        empty_bins = 0
        for previous in range(i - 1, max(-1, i - math.floor(30 / bin_seconds) - 1), -1):
            above_reaction_baseline = signal[previous] > series[previous]['baseline']
            above_volume_baseline = volume_led and totals[previous] > volume_stats[previous][0]
            if above_reaction_baseline or above_volume_baseline:
                onset = previous
            elif minimum_reactors == 2 and not count(previous) and empty_bins == 0:
                empty_bins += 1
            else:
                break
        moment = max(0, point['time'] - 30, min(point['time'] - chat_delay, series[onset]['time']))
        start = max(0, moment - lead)
        end = min(duration, start + 60, max(start + 30, point['time'] + tail))
        if end - start < min(30, duration) or point['time'] >= end:
            continue
        if point['time'] >= duration - min(300, .15 * duration):
            reaction_context = (m for j in range(max(0, i - radius), min(length, math.ceil(end / bin_seconds)))
                                for m in group(j) if start <= m.offset_seconds < end)
            leading_context = (m for j in range(int(start // bin_seconds), i + 1)
                               for m in group(j) if start <= m.offset_seconds < end)
            outro_dominated = False
            for neighborhood in (reaction_context, leading_context):
                neighborhood_authors, outros = set(), set()
                for m in neighborhood:
                    author = m.author.strip().casefold()
                    neighborhood_authors.add(author)
                    if re.search(r"\b(?:by+e+|goodbye|see (?:you|ya)|gn|good night|ggs? (?:everyone|everybody|all)|"
                                 r"(?:thanks?|thank you) (?:so much )?for (?:the |this |another )?"
                                 r"(?:(?:great|awesome|fun|dual) )?(?:stream|live(?:stream)?))\b", m.text, re.I):
                        outros.add(author)
                outro_dominated |= len(outros) >= 2 and len(outros) > .25 * len(neighborhood_authors)
            if outro_dominated:
                continue
        samples = []
        for message, _ in sorted(zip(group(i), reactions(i)[2]), key=lambda item: -item[1]):
            if message.text.strip().casefold() not in {s.strip().casefold() for s in samples}:
                samples.append(message.text)
            if len(samples) == 5:
                break
        candidates.append({"onset": moment, "peak": point['time'],
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
