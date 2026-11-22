"""Local speech transcription and short, speech-timed captions."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

from PIL import ImageFont


FONT_SIZE = 76
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_BOTTOM = 90, 190, 420
LINE_WIDTH = 1080 - MARGIN_LEFT - MARGIN_RIGHT - 16
MIN_DURATION = 0.8


@lru_cache(maxsize=1)
def caption_font():
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', FONT_SIZE)
    except OSError:
        return None


def text_width(text: str) -> float:
    font = caption_font()
    if font is not None:
        return font.getlength(text)
    return sum(FONT_SIZE * (1.1 if unicodedata.east_asian_width(c) in ('W', 'F') else 0.95)
               for c in text)


def join_tokens(tokens) -> str:
    tokens = list(tokens)
    has_whitespace = any(token and token[0].isspace() for token in tokens)
    text = ''
    hyphen_continuation = False
    for token in tokens:
        stripped = token.strip()
        if not stripped:
            continue
        leading_space = token[0].isspace()
        continuation = (not leading_space and stripped[0] in "-’'"
                        and (has_whitespace or not re.match(r"[-’']\d", stripped))
                        and text)
        punctuation = stripped[0] in ',.;:!?%'
        separator = '' if not text or continuation or punctuation or (hyphen_continuation and not leading_space) else ' '
        text += separator + stripped
        hyphen_continuation = bool(continuation and stripped == '-')
    return text


def wrap_caption(text: str) -> list[str]:
    lines, line = [], ''
    for token in text.split():
        candidate = f'{line} {token}'.strip()
        if text_width(candidate) <= LINE_WIDTH:
            line = candidate
            continue
        if line:
            lines.append(line)
            line = ''
        for char in token:
            if line and text_width(line + char) > LINE_WIDTH:
                lines.append(line)
                line = ''
            line += char
    if line:
        lines.append(line)
    return lines


@dataclass
class Caption:
    start: float
    end: float
    text: str


def chunk_words(words, max_words: int = 4, clip_end: float | None = None) -> list[Caption]:
    if max_words < 1:
        raise ValueError("max_words must be positive")
    captions, group = [], []
    for word in words:
        if not word.word.strip() or word.end <= word.start:
            continue
        if group and (word.start - group[-1].end > 0.5 or word.end - group[0].start > 2.5):
            captions.append(Caption(group[0].start, group[-1].end, join_tokens(w.word for w in group)))
            group = []
        group.append(word)
        if len(group) >= max_words or word.word.rstrip().endswith((".", "!", "?", ",", ";", ":")):
            captions.append(Caption(group[0].start, group[-1].end, join_tokens(w.word for w in group)))
            group = []
    if group:
        captions.append(Caption(group[0].start, group[-1].end, join_tokens(w.word for w in group)))
    if clip_end is None and captions:
        clip_end = captions[-1].end
    if clip_end is not None:
        captions = [Caption(c.start, min(c.end, clip_end), c.text) for c in captions if c.start < clip_end]
    balanced = []
    for caption in captions:
        previous = balanced[-1] if balanced else None
        combined = f'{previous.text} {caption.text}' if previous else ''
        if (previous and len(caption.text.split()) == 1
                and not previous.text.endswith(('.', '!', '?', ',', ';', ':'))
                and caption.start - previous.end <= 0.5
                and caption.end - previous.start <= 2.5
                and len(wrap_caption(combined)) <= 2):
            previous.text = combined
            previous.end = caption.end
        else:
            balanced.append(caption)
    for i, caption in enumerate(balanced):
        next_start = balanced[i + 1].start if i + 1 < len(balanced) else clip_end
        caption.end = min(max(caption.end, caption.start + MIN_DURATION), next_start)
    return balanced


def transcribe(source: Path, model: str = "base", language: str | None = None, clip_end: float | None = None) -> list[Caption]:
    from faster_whisper import WhisperModel

    recognizer = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _ = recognizer.transcribe(str(source), language=language, word_timestamps=True)
    return chunk_words((word for segment in segments for word in (segment.words or [])), clip_end=clip_end)


def timestamp(seconds: float, ass: bool = False) -> str:
    scale = 100 if ass else 1000
    ticks = max(0, round(seconds * scale))
    seconds, fraction = divmod(ticks, scale)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if ass:
        return f"{hours}:{minutes:02}:{seconds:02}.{fraction:02}"
    return f"{hours:02}:{minutes:02}:{seconds:02},{fraction:03}"


def to_srt(captions: list[Caption]) -> str:
    return "".join(f"{i}\n{timestamp(c.start)} --> {timestamp(c.end)}\n{c.text}\n\n"
                   for i, c in enumerate(captions, 1))


def to_ass(captions: list[Caption]) -> str:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,1,2,{MARGIN_LEFT},{MARGIN_RIGHT},{MARGIN_BOTTOM},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for caption in captions:
        # Prevent transcript text from becoming ASS override tags or line escapes.
        text = caption.text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ").replace("\r", " ")
        text = r'\N'.join(wrap_caption(text))
        lines.append(f"Dialogue: 0,{timestamp(caption.start, True)},{timestamp(caption.end, True)},Default,,0,0,0,,{text}\n")
    return header + "".join(lines)
