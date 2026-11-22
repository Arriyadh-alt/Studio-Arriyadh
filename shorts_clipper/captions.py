"""Local speech transcription and short, speech-timed captions."""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
import os
import math
import subprocess
import unicodedata

from PIL import ImageFont


FONT_SIZE = 76
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_BOTTOM = 90, 190, 420
LINE_WIDTH = 1080 - MARGIN_LEFT - MARGIN_RIGHT - 16
MIN_DURATION = 0.8
FONTS = ("DejaVu Sans", "Liberation Sans", "JetBrains Mono")


@dataclass
class CaptionStyle:
    preset: str = "pop"
    font: str = "DejaVu Sans"
    size: int = 80
    position: str = "bottom"
    text_color: str = "#FFFFFF"
    highlight_color: str = "#FFE34D"
    emphasis_color: str = "#67F5B5"
    uppercase: bool = False
    emphasis: bool = True

    @classmethod
    def from_dict(cls, value=None):
        result = cls()
        if not isinstance(value, dict):
            return result
        choices = {"preset": ("pop", "classic", "minimal"), "font": FONTS,
                   "position": ("bottom", "middle", "top")}
        for key, options in choices.items():
            if value.get(key) in options:
                setattr(result, key, value[key])
        for key in ("text_color", "highlight_color", "emphasis_color"):
            color = value.get(key)
            if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                setattr(result, key, color.upper())
        size = value.get("size")
        if isinstance(size, (int, float)) and not isinstance(size, bool) and (isinstance(size, int) or math.isfinite(size)):
            result.size = max(40, min(130, round(size)))
        for key in ("uppercase", "emphasis"):
            if isinstance(value.get(key), bool):
                setattr(result, key, value[key])
        return result


def ass_color(color: str) -> str:
    if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        color = "#FFFFFF"
    return f"&H{color[5:7]}{color[3:5]}{color[1:3]}&".upper()


STOPWORDS = set("a an the and or but of to in on at for from with is are was were be been i you he she it we they this that these those my your our their just very really so have has do does not can will as if then than yang dan di ke dari itu ini aja saja sih dong nih kok udah sudah gak nggak tidak juga aku saya kamu kita kami dia mereka nya adalah dengan untuk pada dalam karena jadi ada atau pun lah ya oh eh deh tuh the".split())


def emphasized_words(tokens):
    ranked = []
    for i, token in enumerate(tokens):
        word = token.strip(".,!?;:\"'()")
        if not word:
            continue
        score = (4 if re.search(r"[\d$€£%]", token) else 0) + (3 if re.search(r"[!?]", token) else 0)
        score += 2 if len(word) > 1 and word.isupper() else 0
        score += 1 if len(word) >= 6 and word.casefold() not in STOPWORDS else 0
        if score:
            ranked.append((score, i))
    return {i for _, i in sorted(ranked, key=lambda item: (-item[0], item[1]))[:2]}


class Transcript(list):
    def __init__(self, captions=(), language=None):
        super().__init__(captions)
        self.language = language


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
    words: list = field(default_factory=list, compare=False, repr=False)


def chunk_words(words, max_words: int = 4, clip_end: float | None = None) -> list[Caption]:
    if max_words < 1:
        raise ValueError("max_words must be positive")
    groups, group = [], []
    for word in words:
        if not word.word.strip() or word.end <= word.start:
            continue
        continuation = (not word.word[0].isspace()
                        and (word.word[0] in "-’'" or (group and group[-1].word == '-')))
        if group and (word.start - group[-1].end > 0.5 or (not continuation and (
                len(group) >= max_words
                or group[-1].word.rstrip().endswith((".", "!", "?", ",", ";", ":"))
                or word.end - group[0].start > 2.5))):
            groups.append(group)
            group = []
        group.append(word)
    if group:
        groups.append(group)
    if clip_end is None and groups:
        clip_end = groups[-1][-1].end
    balanced = []
    for group in groups:
        caption = Caption(group[0].start, group[-1].end, join_tokens(w.word for w in group), list(group))
        previous = balanced[-1] if balanced else None
        combined = f'{previous.text} {caption.text}' if previous else ''
        if (previous and len(caption.text.split()) == 1
                and not previous.text.endswith(('.', '!', '?', ',', ';', ':'))
                and caption.start - previous.end <= 0.5
                and caption.end - previous.start <= 2.5
                and len(wrap_caption(combined)) <= 2):
            previous.text = combined
            previous.end = caption.end
            previous.words.extend(caption.words)
        else:
            balanced.append(caption)
    balanced = [Caption(c.start, min(c.end, clip_end), c.text, c.words) for c in balanced if c.start < clip_end]
    for i, caption in enumerate(balanced):
        next_start = balanced[i + 1].start if i + 1 < len(balanced) else clip_end
        caption.end = min(max(caption.end, caption.start + MIN_DURATION), next_start)
    return balanced


def transcribe(source: Path, model: str | None = None, language: str | None = None, clip_end: float | None = None) -> Transcript:
    from faster_whisper import WhisperModel

    recognizer = WhisperModel(model or os.environ.get("CLIPPER_WHISPER_MODEL", "base"), device="cpu", compute_type="int8")
    segments, info = recognizer.transcribe(str(source), language=None if language == "auto" else language, word_timestamps=True)
    return Transcript(chunk_words((word for segment in segments for word in (segment.words or [])), clip_end=clip_end),
                      getattr(info, "language", None))


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


def _classic_ass(captions: list[Caption]) -> str:
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


def literal_text(text):
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ").replace("\r", " ")


@lru_cache(maxsize=24)
def style_font(font, size):
    try:
        path = subprocess.run(["fc-match", "-f", "%{file}", font + ":style=Bold"],
                              check=True, capture_output=True, text=True).stdout
        return ImageFont.truetype(path, size)
    except (OSError, subprocess.SubprocessError):
        return None


def styled_lines(text, style):
    font = style_font(style.font, style.size)
    width = lambda text: font.getlength(text) if font else len(text) * style.size
    lines, line = [], ""
    # Reserve room for the animated word's overshoot without changing line breaks.
    limit = LINE_WIDTH / (1.15 if style.preset == "pop" else 1)
    for token in text.split():
        candidate = f"{line} {token}".strip()
        if width(candidate) <= limit:
            line = candidate
            continue
        if line:
            lines.append(line)
            line = ""
        for char in token:
            if line and width(line + char) > limit:
                lines.append(line)
                line = ""
            line += char
    if line:
        lines.append(line)
    return lines


def to_ass(captions: list[Caption], style: CaptionStyle | dict | None = None) -> str:
    # Preserve the historical serializer for callers supplying only static chunks.
    if style is None and not any(c.words for c in captions):
        return _classic_ass(captions)
    style = style if isinstance(style, CaptionStyle) else CaptionStyle.from_dict(style)
    if style == CaptionStyle(preset="classic"):
        return _classic_ass(captions)
    size = 76 if style.preset == "classic" and style.size == 80 else style.size
    if style.preset == "minimal":
        size = round(size * .8)
    effective = CaptionStyle(**{**style.__dict__, "size": size})
    alignment = {"bottom": 2, "middle": 5, "top": 8}[style.position]
    margin = {"bottom": MARGIN_BOTTOM, "middle": 0, "top": 260}[style.position]
    header = _classic_ass([]).replace("DejaVu Sans,76,", f"{style.font},{size},")
    header = header.replace("&H00FFFFFF,&H00FFFFFF", f"&H00{ass_color(style.text_color)[2:-1]},&H00{ass_color(style.text_color)[2:-1]}")
    header = header.replace(",2,90,190,420,1", f",{alignment},90,190,{margin},1")
    if style.preset == "minimal":
        header = header.replace(",-1,0,0,0,100,100", ",0,0,0,0,100,100").replace(",1,5,1,", ",1,2,0,")
    events = []
    for caption in captions:
        text = literal_text(caption.text)
        display = text.upper() if style.uppercase else text
        lines = styled_lines(display, effective)
        if style.preset != "pop" or not caption.words:
            rendered = r"\N".join(lines)
            if style.preset == "minimal":
                rendered = r"{\fad(120,80)}" + rendered
            windows = [(caption.start, caption.end, rendered)]
        else:
            words = [w for w in caption.words if w.start < caption.end and w.end > caption.start]
            # Prefix lengths preserve contractions and punctuation exactly as displayed.
            spans = []
            for i, word in enumerate(words):
                end = len(literal_text(join_tokens(w.word for w in words[:i + 1])))
                start = end - len(literal_text(word.word.strip()))
                if style.uppercase:
                    start, end = len(text[:start].upper()), len(text[:end].upper())
                spans.append((start, end))
            important = emphasized_words([w.word.strip() for w in words]) if style.emphasis else set()
            # Match wrapped characters back to source offsets, including split long words.
            offsets, cursor = [], 0
            for line in lines:
                row = []
                for char in line:
                    while cursor < len(display) and display[cursor] != char:
                        cursor += 1
                    row.append(cursor)
                    cursor += 1
                offsets.append(row)
            windows = []
            for active, word in enumerate(words):
                start = caption.start if active == 0 else max(caption.start, word.start)
                end = min(caption.end, max(start, words[active + 1].start)) if active + 1 < len(words) else caption.end
                if end <= start:
                    continue
                rendered_lines = []
                for line, row in zip(lines, offsets):
                    rendered, previous = "", None
                    for char, offset in zip(line, row):
                        index = next((i for i, (a, b) in enumerate(spans) if a <= offset < b), -1)
                        if index != previous:
                            color = style.highlight_color if index == active else style.emphasis_color if index in important else style.text_color
                            target_scale = 105 if index in important and index != active else 100
                            scale = 115 if index == active else 110 if active == 0 else target_scale
                            rendered += rf"{{\c{ass_color(color)}\b1\fscx{scale}\fscy{scale}"
                            if index == active or active == 0:
                                rendered += rf"\t(0,100,\fscx{target_scale}\fscy{target_scale})"
                            rendered += "}"
                            previous = index
                        rendered += char
                    rendered_lines.append(rendered)
                entrance = r"{\fad(35,0)}" if active == 0 else ""
                windows.append((start, end, entrance + r"\N".join(rendered_lines)))
        for start, end, rendered in windows:
            if round(end * 100) > round(start * 100):
                events.append(f"Dialogue: 0,{timestamp(start, True)},{timestamp(end, True)},Default,,0,0,0,,{rendered}\n")
    return header + "".join(events)
