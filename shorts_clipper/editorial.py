"""Classify and explain candidates using bounded, source-relative evidence."""

import math
import re

import numpy as np


BACKTRACK_LIMIT = 45
LOW_VALUE = {
    "loading_screen": r"\b(?:loading screen|still loading|waiting to load|layar loading|lagi loading|memuat)\b",
    "static_menu": r"\b(?:settings menu|pause menu|main menu|menu pengaturan|menu utama)\b",
    "inventory": r"\b(?:inventory|item slots?|inventaris|slot barang|buka tas)\b",
    "map_navigation": r"\b(?:open(?:ing)? (?:the )?map|map (?:open|navigation|marker)|waypoint|buka peta|lihat map|titik tujuan)\b",
    "afk": r"\b(?:afk|be right back|brb|sebentar ya|tinggal dulu)\b",
    "repetitive_gameplay": r"\b(?:same level again|grinding again|ulang level|farming lagi)\b",
}
VALUABLE = {
    "victory": r"\b(?:victory|we won|i won|menang|juara)\b",
    "failure": r"\b(?:we lost|i failed|game over|kalah|gagal)\b",
    "discovery": r"\b(?:i found|we found|just discovered|ketemu|baru sadar|ternyata)\b",
    "unexpected_event": r"\b(?:out of nowhere|didn.t expect|suddenly|tiba-tiba|nggak nyangka|tidak menyangka)\b",
    "education": r"\b(?:the trick is|here.s how|because you|caranya|tipsnya|soalnya)\b",
    "storytelling": r"\b(?:the first time|then i|that.s when|waktu itu|sejak itu|terus aku)\b",
}
TRIGGER_CUE = re.compile(
    r"\b(?:look at|watch the|here comes|just hit|just broke|look out|i found|we found|"
    r"there.s (?:a|the)|lihat (?:itu|ada|si|musuh)|awas|ada musuh|baru saja|ketemu|tiba-tiba)\b", re.I)
REACTION_CUE = re.compile(
    r"\b(?:no way|oh my god|what|look|haha\w*|wkwk\w*|lol|astaga|anjir|gila|aduh|ngakak)\b", re.I)
BANNED_HOOKS = (
    "wait until you see this", "you won't believe what happened", "this is insane",
    "tunggu sampai kamu lihat", "kamu tidak akan percaya", "ini gila", 'shock: "what"',
)


def visual_signals(visual):
    if visual is None:
        return None, None, None, None
    visual = np.asarray(visual)
    if visual.ndim == 1:
        return visual, None, None, None
    return visual[:, 0], visual[:, 1], visual[:, 2], visual[:, 3] if visual.shape[1] > 3 else None


def _text(words):
    return " ".join(w["word"].strip() for w in words if w["word"].strip())


def phrases(words):
    group = []
    for word in words:
        if group and (word["start"] - group[-1]["end"] >= .65 or
                      group[-1]["word"].rstrip().endswith((".", "!", "?"))):
            yield group[0]["start"], group[-1]["end"], _text(group)
            group = []
        group.append(word)
    if group:
        yield group[0]["start"], group[-1]["end"], _text(group)


def _usable(text):
    tokens = re.findall(r"\w+", text.casefold())
    return len(tokens) >= 4 and len(set(tokens)) >= 4 and not any(p in text.casefold() for p in BANNED_HOOKS)


def classify_scene(levels, visual, words, start, end, peak, laughter=0.):
    """Keep positive corroboration separate from the scene label."""
    lo, hi = int(start), math.ceil(end)
    relative, motion, _, edges = visual_signals(visual)
    valid = motion[lo:hi] if motion is not None else np.array([])
    valid = valid[valid >= 0]
    static = len(valid) >= max(4, (hi - lo) / 2) and float(np.mean(valid < 1.1)) >= .8
    quiet = float(np.mean(levels[lo:hi] < -47)) if hi > lo else 0.
    transcript = _text(words)
    tokens = re.findall(r"\w+", transcript.lower())
    repetitive = len(tokens) >= 12 and len(set(zip(tokens, tokens[1:]))) / (len(tokens) - 1) < .38
    low = next((k for k, pattern in LOW_VALUE.items() if re.search(pattern, transcript, re.I)), None)
    valuable = next((k for k, pattern in VALUABLE.items() if re.search(pattern, transcript, re.I)), None)
    lines = [text for _, _, text in phrases(words) if _usable(text)]
    contextual_lines = [line for line in lines if not low or not re.search(LOW_VALUE[low], line, re.I)]
    visual_event = relative is not None and np.max(relative[max(lo, peak - 4):min(hi, peak + 5)], initial=0) >= .38
    moving_edges = edges[lo:hi][motion[lo:hi] >= 2] if edges is not None else []
    poor_framing = len(moving_edges) >= 4 and float(np.mean(moving_edges > .72)) >= .75
    compelling = bool((valuable and lines) or (visual_event and (contextual_lines or (not low and laughter >= .35)))
                      or (len(contextual_lines) >= 2 and
                          re.search(r"\?|\b(?:because|soalnya|karena)\b", transcript, re.I)))
    if not low:
        low = ("repetitive_gameplay" if repetitive else "long_silence" if quiet >= .72 and not lines else
               "static_footage" if static else "contextless_talk" if len(tokens) >= 4 and not lines else None)
    label = low or valuable or ("funny_gameplay" if laughter >= .35 and visual_event else
                                "interesting_conversation" if compelling and lines else
                                "emotional_reaction" if lines or visual_event else "unclassified")
    return dict(label=label, low_value=bool(low), justified=bool(low and compelling),
                static=bool(static), repetitive=bool(repetitive or low == "repetitive_gameplay"),
                silence_fraction=round(quiet, 2), visual_event=bool(visual_event), meaningful_lines=len(lines),
                poor_framing=bool(poor_framing))


def backtrack_reaction(levels, excitement, visual, words, peak):
    relative, motion, cuts, _ = visual_signals(visual)
    lo = max(1, peak - BACKTRACK_LIMIT)
    by_second = {int(start): (start, text) for start, _, text in phrases(words)
                 if TRIGGER_CUE.search(text) and _usable(text)}
    onset_floor = max(.10, float(excitement[peak]) * .32)
    notice = peak
    while notice > lo and (
        excitement[notice - 1] >= onset_floor or
        (notice > lo + 1 and excitement[notice - 2] >= onset_floor
         and levels[notice - 1] > -42)
    ):
        notice -= 1
    spoken = next((w["start"] for w in words if notice - 1 <= w["start"] <= peak and
                   REACTION_CUE.search(w["word"])), None)
    notice = min(float(notice), spoken) if spoken is not None else float(notice)
    best = None
    quiet_run = 0
    for t in range(min(peak - 1, math.floor(notice)), lo - 1, -1):
        text_cue = by_second.get(t)
        visual_cue = relative is not None and relative[t] >= .35 and relative[t] >= relative[t - 1] + .08
        reliable_cut = cuts is not None and cuts[t] >= .14
        actual_motion = motion is not None and motion[t] >= 3
        independent_audio = (t < notice - 1 and levels[t] > -40 and
                             levels[t] - np.median(levels[max(0, t - 4):t]) >= 12 and
                             np.min(excitement[t + 1:math.floor(notice)], initial=1) < onset_floor)
        evidence = []
        if text_cue:
            evidence.append("transcript cue")
        if visual_cue:
            evidence.append("scene change" if reliable_cut else "visual change")
        if independent_audio:
            evidence.append("separate audio onset")
        verified = bool(text_cue or (visual_cue and (reliable_cut or actual_motion or motion is None)
                                    and (t >= notice - 8 or independent_audio)))
        if evidence:
            strength = (4 * bool(text_cue) + 2 * bool(visual_cue) + bool(reliable_cut)
                        + bool(independent_audio) - .025 * (notice - t))
            if best is None or (verified, strength) > (best[0], best[1]):
                best = verified, strength, text_cue[0] if text_cue else float(t), evidence, text_cue
        quiet_run = quiet_run + 1 if levels[t] < -47 and not evidence else 0
        if quiet_run >= 6 and best is not None and best[0]:
            break
    return dict(trigger_time=best[2] if best else None, verified=bool(best and best[0]),
                evidence=best[3] if best else [], subject=best[4][1] if best and best[4] else None,
                notice_time=notice, peak_time=float(peak), conclusion_time=None, conclusion=None)


def finish_reaction(reaction, excitement, words, end):
    peak = reaction["peak_time"]
    post = [(a, b, text) for a, b, text in phrases(words) if peak <= a and b <= end]
    meaningful = next(((b, text) for _, b, text in post if _usable(text)), None)
    if meaningful:
        reaction.update(conclusion_time=meaningful[0], conclusion=meaningful[1])
    else:
        floor = max(.12, excitement[int(peak)] * .27)
        for t in range(int(peak) + 1, min(math.floor(end), len(excitement)) - 1):
            if max(excitement[t:t + 2]) < floor:
                reaction.update(conclusion_time=float(t), conclusion="Reaction settles")
                break


def _clock(second):
    return f"{int(second) // 60}:{int(second) % 60:02d}"


def hook_options(words, reaction, start):
    # Quote setup, not the conclusion. In the absence of speech, describe only measured evidence.
    trigger = reaction["trigger_time"]
    peak = reaction["peak_time"]
    setup = [(at, text) for at, end, text in phrases(words) if end <= peak and _usable(text)]
    quotes = {}
    for at, text in setup:
        quotes.setdefault(" ".join(text.split()[:9]).rstrip(".,!? ")[:46], at)
        if len(quotes) == 2:
            break
    transcript = _text(words)
    indonesian = bool(re.search(r"\b(?:aku|kamu|kita|ada|ini|itu|yang|buka|peta|nggak|gue|lu)\b", transcript, re.I))
    event = trigger if trigger is not None and reaction["verified"] else peak
    anchor = _clock(event)
    placement = _clock(start)
    notice = reaction["notice_time"]
    if indonesian:
        options = {f'Simak: “{quote}”': at for quote, at in quotes.items()}
        if reaction["verified"] and "scene change" in reaction["evidence"]:
            options[f"Perhatikan perubahan layar di {anchor}"] = event
        options.update({f"Reaksi mulai di {_clock(notice)}": notice,
                        f"Dari {placement} ke reaksi di {_clock(peak)}": peak,
                        f"Dengarkan perubahan suara di {_clock(peak)}": peak})
    else:
        options = {f'Before the reaction: “{quote}”': at for quote, at in quotes.items()}
        if reaction["verified"] and any("change" in cue for cue in reaction["evidence"]):
            options[f"Watch the screen change at {anchor}"] = event
        options.update({f"The reaction starts at {_clock(notice)}": notice,
                        f"From {placement} to the reaction at {_clock(peak)}": peak,
                        f"Listen for the change in sound at {_clock(peak)}": peak})
    return options


def component_scores(scene, reaction, words, excitement, start, end, laughter, hooks):
    duration = end - start
    peak = reaction["peak_time"]
    reaction_strength = float(excitement[int(peak)])
    body = _text(words)
    distinct = len(set(re.findall(r"\w+", body.lower())))
    verified = reaction["verified"]
    irrelevant = scene["low_value"] and not scene["justified"]
    context = min(1., distinct / 18) if not irrelevant else 0
    relevant_trigger = verified and not irrelevant
    outcome = reaction["conclusion_time"] is not None and not irrelevant
    quiet = scene["silence_fraction"]
    meaningful = scene["meaningful_lines"] > 0 and not irrelevant
    hook_has_quote = meaningful and any("“" in hook for hook in hooks)
    last = words[-1] if words else None
    incomplete = bool(last and last["end"] > end - .4 and not last["word"].rstrip().endswith((".", "?", "!")))
    abrupt = incomplete or (not outcome and duration >= 59)
    flags = {
        "missing_trigger": not verified,
        "low_value_scene": irrelevant,
        "static_footage": scene["static"] and not scene["justified"],
        "repetitive": scene["repetitive"] and not scene["justified"],
        "incomplete_conversation": incomplete,
        "abrupt_ending": abrupt,
        "poor_framing": scene["poor_framing"],
    }
    scores = dict(
        hook=43 + 30 * hook_has_quote + 18 * relevant_trigger + 9 * meaningful,
        story=45 + 22 * relevant_trigger + 25 * context + 8 * outcome,
        emotion=38 + 37 * reaction_strength + 15 * min(1., 2 * laughter) + 10 * meaningful,
        visual=24 if scene["static"] else 45 + 30 * (scene["visual_event"] and not irrelevant) + 22 * relevant_trigger,
        payoff=32 + 26 * outcome + 25 * relevant_trigger + 17 * meaningful,
        retention=48 + 19 * (1 - quiet) + 15 * hook_has_quote + 10 * outcome + 8 * (duration <= 35),
    )
    explanations = dict(
        hook=("The quote or event only describes a low-value scene." if irrelevant else
              "A verbatim setup quote anchors the opening." if hook_has_quote else "No specific spoken setup was available."),
        story=f"{distinct} distinct transcript words; " + ("trigger located." if verified else "trigger unverified."),
        emotion=f"Reaction peak at {_clock(peak)}; normalized audio intensity {reaction_strength:.2f}.",
        visual=("Motion concentrated at frame edges; check framing." if scene["poor_framing"] else
                "Frames mostly static." if scene["static"] else "Visual change near reaction." if scene["visual_event"]
                else "No strong visual evidence near reaction."),
        payoff=("No meaningful payoff for this scene." if irrelevant else
                f"Conclusion: {reaction['conclusion']}." if outcome else "No clear conclusion detected."),
        retention=f"{round(100 * quiet)}% quiet audio over {duration:.0f}s; estimate, not measured retention.",
    )
    return scores, {k for k, v in flags.items() if v}, explanations
