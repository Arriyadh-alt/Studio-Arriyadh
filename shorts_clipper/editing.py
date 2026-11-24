"""Source-first editing profiles and authenticity checks."""

import logging
import re
from dataclasses import dataclass

from .captions import join_tokens
from .director import (MAX_OUTPUT, STEP, Plan, Segment, SENTENCE_END, find_silences, remap_captions, score_clip,
                       transcript_words)

MODE_DEFAULTS = {
    "full": {},
    "natural": {"sfx": False, "music": "off", "story": True, "voiceover": False, "loop": False},
    "enhanced": {"sfx": False, "music": "off", "story": True, "voiceover": False, "loop": False},
}

EMOTIONAL = re.compile(r"\b(?:passed away|lost my|miss (?:him|her|them)|grief|heartbroken|meninggal|kehilangan)\b", re.I)


@dataclass(frozen=True)
class Timing:
    pause: float | None
    breathing_room: float
    zoom: float
    attack: float
    hold: float


TIMING = {
    "emotional": Timing(None, 0, 0, 0, 0),
    "funny": Timing(2.0, .38, .035, .35, 1.8),
    "podcast": Timing(2.4, .45, .025, .55, 2.3),
    "gaming": Timing(2.7, .5, .025, .5, 1.5),
}


def edit_style(category, words, podcast):
    if category in ("emotional", "heartfelt", "sad") or EMOTIONAL.search(" ".join(w.word.strip() for w in words)):
        return "emotional"
    if category in ("funny", "comedy", "humor"):
        return "funny"
    return "podcast" if podcast else "gaming"


def source_quote(words, event=None):
    sentences, sentence = [], []
    for word in words:
        if sentence and word.start - sentence[-1].end > 1.2:
            sentence = []
        sentence.append(word)
        if SENTENCE_END.search(word.word.strip()):
            text = join_tokens(w.word for w in sentence)
            if 3 <= len(sentence) and len(text) <= 62 and min(w.probability for w in sentence) >= .55:
                sentences.append((text, sentence[0].start, word.end))
            sentence = []
    if not sentences:
        return None
    return min(sentences, key=lambda line: abs((line[1] + line[2]) / 2 - event)) if event is not None else sentences[0]


def plan_source_edit(levels, captions, duration, options, *, mode, category, peak=None, trigger=None,
                     spike_score=None, title_voice=None, quote_time=None):
    """Keep the chosen highlight in source order, with cuts only in long, very quiet gaps."""
    words = transcript_words(captions)
    tone = edit_style(category, words, options.preset == "podcast")
    timing = TIMING[tone]
    event = trigger if trigger is not None else peak
    protected = []
    if event is not None:
        protected.append((event - 2, (peak if peak is not None and peak > event else event) + 2))
    if peak is not None and tone == "funny":
        protected.append((peak - 3.5, peak + 2.5))
    if options.preset != "podcast":
        protected.append((0, min(duration, 2.0)))
    quote = source_quote(words, quote_time if quote_time is not None else peak)
    if quote:
        protected.append((quote[1], quote[2]))

    segments, cursor, removed = [], 0.0, 0.0
    if options.edits and timing.pause and words:
        minimum = timing.pause - (.45 if mode == "enhanced" else 0)
        for a, b in find_silences(levels, words, 0, duration, protected, minimum=minimum):
            if max(levels[round(a / STEP):round(b / STEP)], default=0) > -45:
                continue
            a, b = a + timing.breathing_room, b - timing.breathing_room
            if b - a >= .35 and a - cursor >= .8 and duration - b >= .8:
                segments.append(Segment(cursor, a))
                cursor = b
                removed += b - a
    segments.append(Segment(cursor, duration))
    plan = Plan(segments, removed=removed)
    if options.hook:
        text = options.hook_text or (f'“{quote[0]}”' if options.story and quote else None)
        if options.commentary:
            text = f"AI commentary: {options.commentary}"
            plan.commentary = {"text": options.commentary, "label": "AI commentary", "voiced": False}
        if text:
            plan.hook = {"style": "source", "text": text, "duration": min(duration, 3.0)}
            voice_limit = min(3., MAX_OUTPUT - plan.duration - .45)
            if title_voice and options.voiceover and voice_limit >= .5:
                try:
                    voice, detail = title_voice(options.commentary or text, max_seconds=voice_limit)
                    intro_duration = max(plan.hook["duration"], detail["seconds"] + .4)
                    if plan.duration + intro_duration > MAX_OUTPUT:
                        intro_duration = detail["seconds"] + .4
                    plan.segments.insert(0, Segment(0., intro_duration, "intro"))
                    plan.voiceover = str(voice)
                    plan.hook.update(voiceover=True, narration=detail, duration=intro_duration)
                    if plan.commentary:
                        plan.commentary["voiced"] = True
                except Exception:
                    logging.getLogger(__name__).warning("Hook voice-over unavailable", exc_info=True)
    if options.edits and timing.zoom:
        focus = peak if tone == "funny" else (quote[1] if tone == "podcast" and quote else event)
        at = plan.map_time(focus) if focus is not None else None
        if at is not None and plan.duration >= 2:
            plan.zooms.append([max(0., at - timing.attack), min(plan.duration, at + timing.hold),
                               timing.zoom * (1.65 if mode == "enhanced" else 1), timing.attack, tone])
            if mode == "enhanced" and options.sfx and options.preset != "podcast":
                plan.sfx.append([at, "pop"])
    plan.music = mode == "enhanced" and (options.music == "on" or
                 (options.music == "auto" and not words and tone != "emotional"))
    plan.voice_clarity = mode == "enhanced" and options.preset == "podcast"
    plan.score, plan.suggestions = score_clip(levels, words, duration, peak, .65 if plan.hook else .4,
                                              spike_score, removed)
    plan.suggestions = [s for s in plan.suggestions if not s.startswith(("Opening is calm", "Pacing is slow", "Little speech"))]
    return plan, tone


def _tokens(text):
    return re.findall(r"\w+", (text or "").casefold())


def authenticity_report(job, clip, plan, transcript, shown, duration, *, style=None, hook=None, tracking=None):
    """Check edit structure against source speech; never imply legal or semantic clearance."""
    segments = plan.segments if plan else [Segment(0, duration)]
    main = [s for s in segments if s.role == "main"]
    words = transcript_words(transcript)
    ordered = bool(main) and all(a.end <= b.start + .001 for a, b in zip(main, main[1:]))
    full_span = (ordered and all(s.role == "main" for s in segments) and
                 sum(s.length for s in main) >= duration - .04)
    missing = sum(not any(s.start <= w.start + .025 and s.end >= w.end - .025 for s in main) for w in words)
    peak = clip.get("peak")
    event = (tracking or {}).get("trigger") or {}
    anchors = [t for t in (peak - clip["start"] if peak is not None else None, event.get("trigger")) if t is not None]
    anchor_removed = any(0 <= t < duration and not any(s.start <= t < s.end for s in main) for t in anchors)
    context_ok = ordered and all(s.role == "main" for s in segments) and not (missing or anchor_removed)
    checks = {
        "context": {"status": "pass" if context_ok and (full_span or words) else "review",
                    "detail": ("Source excerpt stays in order; review its broader context before publishing."
                               if full_span else "Source order, detected speech and event markers are retained; review the cuts."
                               if context_ok and words else "Check that the edit keeps the setup, speech and payoff.")},
    }
    expected = remap_captions(transcript, plan) if plan else transcript
    actual_text = _tokens(" ".join(c.text for c in shown))
    expected_text = _tokens(" ".join(c.text for c in expected))
    uncertain = any(w.probability < .55 for w in transcript_words(expected))
    matches = bool(actual_text) and actual_text == expected_text
    checks["subtitles"] = {
        "status": "skip" if not clip["captions"] else "pass" if matches and not uncertain else "review",
        "detail": ("Subtitles are off." if not clip["captions"] else
                   "Subtitles match the source transcript; listen for transcription errors." if matches and not uncertain else
                   "Listen for subtitle errors, low-confidence speech or manual changes."),
    }
    source_text = " " + " ".join(_tokens(" ".join(c.text for c in transcript))) + " "
    for label, text in (("title", clip.get("title")), ("hook", (plan.hook or {}).get("text") if plan else None),
                        ("thumbnail", hook)):
        tokens = _tokens(text)
        supported = tokens and " " + " ".join(tokens) + " " in source_text
        checks[label] = ({"status": "skip", "detail": "No text to check."} if not text else
                         {"status": "pass", "detail": "Text matches a phrase in the source transcript."} if supported else
                         {"status": "review", "detail": "Check wording against the source context."})
    metadata, rights = job.get("metadata") or {}, job.get("rights") or {}
    return {
        "style": style, "checks": checks, "operations": {},
        "provenance": {"url": job.get("url"), "source_id": metadata.get("id") or job.get("upload"),
                       "title": metadata.get("title"), "channel": metadata.get("channel") or metadata.get("uploader"),
                       "filename": job.get("filename"), "start": clip["start"], "end": round(clip["start"] + duration, 3),
                       "retained_ranges": [[round(clip["start"] + s.start, 3), round(clip["start"] + s.end, 3)] for s in main]},
        "flags": [("No rights confirmation was recorded at render time; verify ownership, license or permission."
                   if not rights.get("confirmed") else
                   "Rights are self-declared; verify they cover third-party footage, audio and the intended platform."),
                  "Review the source for reused content and copyrighted material. This check cannot determine ownership or platform eligibility."],
    }


def record_operations(report, plan, *, mixed):
    narration = bool(plan and mixed and plan.voiceover)
    effects_audio = bool(plan and mixed and (plan.music or plan.sfx))
    commentary = bool(plan and plan.commentary)
    held_frame = bool(plan and any(s.role == "intro" for s in plan.segments))
    report["operations"] = {"synthetic_audio": narration or effects_audio, "synthetic_voiceover": narration,
                            "generated_footage": False, "generated_commentary": commentary, "source_freeze_frame": held_frame,
                            "loop_blend": bool(plan and (plan.loop or {}).get("status") == "applied")}
    report["disclosure"] = (
        "Disclose the added AI voiceover or commentary, and check the destination platform’s rules." if narration or commentary else
        "Synthetic music or effects were added. Check the platform’s disclosure rules for those additions." if effects_audio else
        "No synthetic speech or generated footage detected in the applied edits. Review platform requirements before uploading.")
