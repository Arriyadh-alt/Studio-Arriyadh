"""AI voice-over for the storytelling hook: a cast of creator voices, each delivering the hook in the
mood of the scene (excited, surprised, suspenseful, funny or warm) instead of reading it flat.

Neural voices come from Microsoft's Edge read-aloud service through edge-tts. When that is
unreachable the offline Piper voice reads the same script with the mood's pace and pitch, so a
render never fails for want of a voice.
"""

import asyncio
import hashlib
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RATE = 24000
MAX_SECONDS = 3.6
NEURAL_TIMEOUT = 25


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    neural: dict
    rate: int = 0
    pitch: int = 0


PERSONAS = {
    "hype": Persona("Hype Streamer", "Fast and loud, like a streamer losing it live",
                    {"en": "en-US-BrianNeural", "id": "id-ID-ArdiNeural"}, rate=8, pitch=3),
    "bestie": Persona("Bubbly Bestie", "Bright, playful and quick to laugh",
                      {"en": "en-US-AvaNeural", "id": "id-ID-GadisNeural"}, rate=6, pitch=5),
    "trailer": Persona("Movie Trailer", "Deep, slow and dramatic",
                       {"en": "en-US-ChristopherNeural", "id": "id-ID-ArdiNeural"}, rate=-8, pitch=-8),
    "chill": Persona("Chill Commentator", "Relaxed and warm, like telling a friend",
                     {"en": "en-US-GuyNeural", "id": "id-ID-ArdiNeural"}, rate=-2),
    "anchor": Persona("Breaking News", "Crisp news-anchor delivery",
                      {"en": "en-US-AriaNeural", "id": "id-ID-GadisNeural"}, rate=2),
    "classic": Persona("Classic (offline)", "The original offline voice", {}),
}
VOICE_CHOICES = ("auto", *PERSONAS)

# rate %, pitch Hz, volume %, pause after the lead-in (s)
MOODS = {
    "excited": (12, 5, 10, 0.12),
    "surprise": (6, 8, 8, 0.2),
    "suspense": (-8, -5, 0, 0.35),
    "humor": (8, 6, 6, 0.18),
    "warm": (-3, 2, 0, 0.22),
}
DELIVERY_CHOICES = ("auto", *MOODS)
CATEGORY_MOOD = {"karma": "humor", "fail": "humor", "funny": "humor", "scare": "suspense", "rage": "excited",
                 "win": "excited", "shock": "surprise", "wholesome": "warm"}
AUTO_VOICE = {"excited": "hype", "surprise": "hype", "humor": "bestie", "suspense": "trailer", "warm": "chill"}

# What a creator says before the line, so the narration reacts instead of reading a title.
LEAD_INS = {
    "en": {"excited": ["Let's go!", "Okay, okay!", "Yo, chat!"],
           "surprise": ["Wait, what?", "No way.", "Hold on."],
           "suspense": ["Okay, listen.", "Watch closely.", "Wait for it."],
           "humor": ["Bro.", "Oh no.", "Okay, this is funny."],
           "warm": ["Real talk.", "Oh, wow.", "Okay, this one's sweet."]},
    "id": {"excited": ["Gaskeun!", "Oke, oke!", "Woy, chat!"],
           "surprise": ["Hah, serius?", "Gak mungkin.", "Bentar, bentar."],
           "suspense": ["Guys, dengerin.", "Perhatiin baik-baik.", "Tunggu ya."],
           "humor": ["Waduh.", "Astaga.", "Ngakak, guys."],
           "warm": ["Jujur ya.", "Aduh, manis banget.", "Ini bikin terharu."]},
}
SAMPLES = {"en": "Wait until you see what happens next!", "id": "Tunggu sampai kalian lihat apa yang terjadi!"}


def mood_for(category: str | None, delivery: str | None = "auto") -> str:
    return delivery if delivery in MOODS else CATEGORY_MOOD.get(category or "", "excited")


def choose_voice(voice: str | None, mood: str) -> str:
    return voice if voice in PERSONAS else AUTO_VOICE[mood]


@dataclass
class Line:
    text: str
    rate: int
    pitch: int
    volume: int
    pause: float = 0.0
    lead: bool = False


def script(text: str, language: str, mood: str, voice: str, seed: str = "") -> list[Line]:
    """The hook as a creator would say it: a short reaction, then the line with the mood's pace and pitch."""
    persona = PERSONAS[voice]
    rate, pitch, volume, pause = MOODS[mood]
    rate, pitch = rate + persona.rate, pitch + persona.pitch
    leads = LEAD_INS.get(language, LEAD_INS["en"])[mood]
    pick = int(hashlib.sha1(f"{seed}|{text}".encode()).hexdigest(), 16)
    line = " ".join(text.replace("…", "...").split()).rstrip(" .")
    if not line.endswith(("?", "!")):
        line += "." if mood in ("suspense", "warm") else "!"
    lines = [Line(leads[pick % len(leads)], rate + 6, pitch + 4, volume + 5, pause, lead=True)]
    words = line.split()
    if mood == "suspense" and len(words) >= 4:
        quote = next((i for i, w in enumerate(words) if w.startswith('"') and i >= 2), None)
        cut = quote if quote is not None else max(2, round(len(words) * 0.6))
        lines.append(Line(" ".join(words[:cut]) + "...", rate, pitch, volume, 0.3))
        lines.append(Line(" ".join(words[cut:]), rate - 4, pitch - 3, volume))
    else:
        lines.append(Line(line, rate, pitch, volume))
    return lines


def _signed(value: int, unit: str) -> str:
    return f"{value:+d}{unit}"


def _ffmpeg(args: list, data: bytes | None = None) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", *args], input=data, capture_output=True, check=True).stdout


def _decode(path: Path, extra: str | None = None) -> np.ndarray:
    raw = _ffmpeg(["-i", str(path), *(["-af", extra] if extra else []), "-f", "f32le", "-ac", "1",
                   "-ar", str(RATE), "-"])
    return np.frombuffer(raw, dtype="<f4")


def _trim(audio: np.ndarray, threshold: float = 0.006) -> np.ndarray:
    frame = RATE // 100
    if len(audio) < frame:
        return audio
    rms = np.sqrt(np.mean(audio[:len(audio) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    loud = np.flatnonzero(rms > threshold)
    if not len(loud):
        return audio[:0]
    pad = 4 * frame
    return audio[max(0, loud[0] * frame - pad):min(len(audio), (loud[-1] + 1) * frame + pad)]


def _neural(lines: list[Line], voice: str, work: Path) -> list[np.ndarray]:
    import edge_tts

    async def speak(i, line):
        path = work / f"line{i}.mp3"
        for attempt in range(2):
            try:
                await edge_tts.Communicate(line.text, voice, rate=_signed(line.rate, "%"), pitch=_signed(line.pitch, "Hz"),
                                           volume=_signed(line.volume, "%")).save(str(path))
                return path
            except edge_tts.exceptions.NoAudioReceived:
                if attempt:
                    raise

    async def run():
        return await asyncio.wait_for(asyncio.gather(*(speak(i, line) for i, line in enumerate(lines))),
                                      NEURAL_TIMEOUT)

    return [_decode(path) for path in asyncio.run(run())]


def _offline(lines: list[Line], language: str, cache: Path, work: Path) -> list[np.ndarray]:
    from piper import PiperVoice, SynthesisConfig
    from .story import voice_model
    piper = PiperVoice.load(str(voice_model(language, cache)))
    clips = []
    for i, line in enumerate(lines):
        config = SynthesisConfig(length_scale=0.92 / (1 + line.rate / 100), noise_scale=0.8, noise_w_scale=0.9)
        chunks = list(piper.synthesize(line.text, syn_config=config))
        rate = chunks[0].sample_rate
        path = work / f"line{i}.f32"
        path.write_bytes(np.concatenate([c.audio_float_array for c in chunks]).astype("<f4").tobytes())
        shift = 2 ** (line.pitch / 60)  # about a semitone per 5 Hz on a speaking voice
        clips.append(_decode_raw(path, rate, f"asetrate={rate * shift:.0f},aresample={RATE},atempo={1 / shift:.4f}"))
    return clips


def _decode_raw(path: Path, rate: int, extra: str) -> np.ndarray:
    raw = _ffmpeg(["-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", str(path), "-af", extra,
                   "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"])
    return np.frombuffer(raw, dtype="<f4")


def narrate(text: str, language: str, output: Path, cache: Path, voice: str = "auto",
            category: str | None = None, seed: str = "", delivery: str = "auto",
            max_seconds: float = MAX_SECONDS, *, allow_truncate=True) -> dict:
    """Speak the hook to a WAV at `output`; returns its length and how it was voiced."""
    language = language if language in LEAD_INS else "en"
    mood = mood_for(category, delivery)
    voice = choose_voice(voice, mood)
    persona = PERSONAS[voice]
    lines = script(text, language, mood, voice, seed)
    engine = "neural" if persona.neural else "offline"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        clips = None
        if persona.neural:
            try:
                clips = _neural(lines, persona.neural.get(language, persona.neural["en"]), work)
            except Exception:
                logger.warning("Neural voice unavailable; using the offline voice", exc_info=True)
                engine = "offline"
        if clips is None:
            clips = _offline(lines, language, cache, work)
        clips = [_trim(c) for c in clips]
        keep = list(zip(clips, lines))
        if sum(len(c) / RATE + l.pause for c, l in keep) > max_seconds and keep[0][1].lead:
            keep = keep[1:]
        audio = np.concatenate([np.concatenate([c, np.zeros(int(l.pause * RATE), dtype="<f4")])
                                for c, l in keep] or [np.zeros(RATE // 4, dtype="<f4")])
        tempo = min(1.35, max(1.0, len(audio) / RATE / max_seconds))
        if not allow_truncate and len(audio) / RATE / tempo > max_seconds + .001:
            raise ValueError("Narration does not fit without truncation")
        raw = work / "voice.f32"
        raw.write_bytes(audio.astype("<f4").tobytes())
        seconds = min(max_seconds, len(audio) / RATE / tempo)  # the hook card never outlasts this
        chain = (f"atempo={tempo:.4f}," if tempo > 1 else "") + (
            f"atrim=duration={seconds:.3f},afade=t=out:st={max(0.0, seconds - 0.15):.3f}:d=0.15,") + (
            "highpass=f=80,acompressor=threshold=-20dB:ratio=3:attack=5:release=80:makeup=2,"
            "equalizer=f=3200:t=q:w=1.2:g=3,volume=-1dB")
        output.parent.mkdir(parents=True, exist_ok=True)
        _ffmpeg(["-y", "-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", str(raw), "-af", chain,
                 "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(output)])
    return {"seconds": round(seconds, 2), "voice": voice, "voice_name": persona.name, "mood": mood,
            "engine": engine, "script": " ".join(l.text for _, l in keep)}
