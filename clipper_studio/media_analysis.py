"""Scan overlapping source ranges, then refine globally selected candidates."""

import math
from dataclasses import dataclass

import numpy as np

from shorts_clipper.moments import _audio, _transcribe, _visual, analyze_signals
from shorts_clipper.podcast import BANDS, BLOCK, analyze_podcast, transcribe_all, voice_features

SEGMENTED_MIN_SECONDS = 3600
CORE_SECONDS = 15 * 60
OVERLAP_SECONDS = 95


@dataclass(frozen=True)
class Range:
    first: int
    last: int
    start: int
    end: float

    @property
    def length(self):
        return self.end - self.start

    @property
    def key(self):
        return f"{self.start}-{self.end:g}"


def media_ranges(duration):
    return [Range(first, min(first + CORE_SECONDS, math.ceil(duration)),
                  max(0, first - OVERLAP_SECONDS), min(duration, first + CORE_SECONDS + OVERLAP_SECONDS))
            for first in range(0, math.ceil(duration), CORE_SECONDS)]


def _stitch(target, local, segment, block=1):
    lo, hi = round(segment.first / block), round(segment.last / block)
    offset = round((segment.first - segment.start) / block)
    count = min(hi - lo, len(local) - offset)
    if count > 0:
        target[lo:lo + count] = local[offset:offset + count]


def analyze_ranges(duration, clips, lead, language, *, fetch, checkpoint, expect, progress, check,
                   podcast=False, speakers=None):
    ranges = media_ranges(duration)
    count = len(ranges)
    expect("audio_analysis", count * (2 if podcast else 1))
    expect("video_analysis", count if not podcast else 0)
    if podcast:
        expect("transcript_generation", count)
    seconds = math.ceil(duration)
    # Loudness defaults to quiet in gaps; zero dB would shift inferred boundaries.
    signals = [np.full(seconds, -120.), np.zeros(seconds), np.zeros(seconds)]
    video = np.zeros(seconds)
    valid_video = False
    sources, words = {}, []
    features = np.zeros((math.ceil(duration / BLOCK), BANDS + 1)) if podcast else None
    last_error, speech_error = None, None
    transcript_ok = False
    for index, segment in enumerate(ranges):
        check()
        progress(.5 * index / count, f"Scanning source segment {index + 1} of {count}")

        try:
            source = fetch(segment)
            check()
        except Exception as error:
            check()
            last_error = error
            continue

        def audio_scan():
            result = _audio(source, segment.length)
            if not len(result[0]):
                raise ValueError("No usable audio in this segment")
            return [array.tolist() for array in result]

        try:
            result = checkpoint("audio_analysis", f"scan-{segment.key}", audio_scan)
        except Exception as error:
            check()
            last_error = error
            if not podcast:
                continue
            result = ()
        sources[segment.first] = source
        for target, local in zip(signals, result):
            _stitch(target, local, segment)
        if podcast:
            try:
                local_features = checkpoint("audio_analysis", f"voices-{segment.key}",
                                            lambda: voice_features(source, segment.length).tolist())
                _stitch(features, local_features, segment, BLOCK)
            except Exception:
                check()
            try:
                speech = checkpoint("transcript_generation", f"conversation-{segment.key}",
                                    lambda: transcribe_all(source, language))
                transcript_ok = True
                for word in speech:
                    start, end = segment.start + word["start"], segment.start + word["end"]
                    if segment.first <= (start + end) / 2 < segment.last:
                        words.append(dict(word=word["word"], start=start, end=end))
            except Exception as error:
                check()
                speech_error = error
        else:
            try:
                def visual_scan():
                    frames = _visual(source, segment.length, len(result[0]))
                    return frames.tolist()

                local_video = checkpoint("video_analysis", f"scan-{segment.key}", visual_scan)
                _stitch(video, local_video, segment)
                valid_video = True
            except Exception:
                check()
    if not sources:
        raise last_error or ValueError("No playable analysis segments were found")
    if podcast:
        if not transcript_ok and speech_error:
            raise speech_error
        progress(.55, "Joining conversation across segments")
        words.sort(key=lambda item: (item["start"], item["end"]))
        return analyze_podcast(None, None, duration, clips=clips, language=language, speakers=speakers,
                               progress=lambda fraction, text: progress(.55 + .45 * fraction, text),
                               transcribe_fn=lambda *_: words, features_fn=lambda *_: features,
                               audio_fn=lambda *_: signals)

    def transcript(lo, hi):
        peak = int(lo + lead + 5) if lo else int(max(0, hi - 42))
        segment = ranges[min(len(ranges) - 1, peak // CORE_SECONDS)]

        def transcribe_candidate():
            source = sources[segment.first]
            words = _transcribe(source, lo - segment.start, hi - segment.start, language)
            return [{**word, "start": word["start"] + segment.start, "end": word["end"] + segment.start}
                    for word in words]

        return checkpoint("transcript_generation", f"speech-{lo:g}-{hi:g}", transcribe_candidate)

    return analyze_signals(signals, video if valid_video else None, duration, clips, lead,
                           transcript, progress, expect)
