"""Local speech transcription and short, speech-timed captions."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Caption:
    start: float
    end: float
    text: str


def chunk_words(words, max_words: int = 4) -> list[Caption]:
    if max_words < 1:
        raise ValueError("max_words must be positive")
    captions, group = [], []
    for word in words:
        if not word.word.strip() or word.end <= word.start:
            continue
        if group and (word.start - group[-1].end > 0.5 or word.end - group[0].start > 2.5):
            captions.append(Caption(group[0].start, group[-1].end, " ".join(w.word.strip() for w in group)))
            group = []
        group.append(word)
        if len(group) >= max_words or word.word.rstrip().endswith((".", "!", "?", ",", ";", ":")):
            captions.append(Caption(group[0].start, group[-1].end, " ".join(w.word.strip() for w in group)))
            group = []
    if group:
        captions.append(Caption(group[0].start, group[-1].end, " ".join(w.word.strip() for w in group)))
    return captions


def transcribe(source: Path, model: str = "base", language: str | None = None) -> list[Caption]:
    from faster_whisper import WhisperModel

    recognizer = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _ = recognizer.transcribe(str(source), language=language, word_timestamps=True)
    return chunk_words(word for segment in segments for word in (segment.words or []))


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
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,76,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,1,2,70,70,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for caption in captions:
        # Prevent transcript text from becoming ASS override tags or line escapes.
        text = caption.text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ").replace("\r", " ")
        lines.append(f"Dialogue: 0,{timestamp(caption.start, True)},{timestamp(caption.end, True)},Default,,0,0,0,,{text}\n")
    return header + "".join(lines)
