"""AI Director: plan a hook, cuts, zooms, sound and a quality check for a highlight, offline.

Every decision comes from local signals (speech timing, loudness, the chat peak),
so it runs on CPU next to Whisper and ffmpeg without any external service.
"""

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .captions import STOPWORDS, Caption, chunk_words, literal_text, _classic_ass

STEP = 0.1
FPS = 30
MAX_OUTPUT = 60.0
TARGET_LUFS = -14.0
MUSIC_MODES = ("auto", "on", "off")
EXCITED = re.compile(r"(!|\?{2,}|\b(?:wow|what|no+|yes+|omg|lol|lmao|haha+|wkwk+|anjir|anjay|gila|astaga|bro|holy|insane|crazy)\b)", re.I)


@dataclass
class DirectorOptions:
    enabled: bool = True
    hook: bool = True
    edits: bool = True
    sfx: bool = True
    music: str = "auto"

    @classmethod
    def from_dict(cls, value=None):
        if not isinstance(value, dict):
            return cls()
        result = cls()
        for key in ("enabled", "hook", "edits", "sfx"):
            if isinstance(value.get(key), bool):
                setattr(result, key, value[key])
        if value.get("music") in MUSIC_MODES:
            result.music = value["music"]
        return result


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    role: str = "main"

    @property
    def length(self):
        return self.end - self.start


@dataclass
class Plan:
    segments: list
    hook: dict | None = None
    zooms: list = field(default_factory=list)
    shakes: list = field(default_factory=list)
    flashes: list = field(default_factory=list)
    sfx: list = field(default_factory=list)
    music: bool = False
    focal: list = field(default_factory=list)
    score: dict = field(default_factory=dict)
    suggestions: list = field(default_factory=list)
    trimmed_start: float = 0.0
    removed: float = 0.0

    @property
    def duration(self):
        return sum(s.length for s in self.segments)

    def offsets(self):
        offset, result = 0.0, []
        for segment in self.segments:
            result.append(offset)
            offset += segment.length
        return result

    def map_time(self, t, include_hook=False):
        for offset, segment in zip(self.offsets(), self.segments):
            if (include_hook or segment.role != "hook") and segment.start <= t < segment.end:
                return offset + t - segment.start
        return None

    def summary(self):
        return {"hook": self.hook, "cuts": max(0, sum(s.role == "main" for s in self.segments) - 1),
                "removed_seconds": round(self.removed + self.trimmed_start, 2),
                "zooms": len(self.zooms), "transitions": len(self.flashes),
                "sound_effects": len(self.sfx), "music": self.music,
                "reframes": len({round(f, 2) for f in self.focal}) if self.focal else 0,
                "duration": round(self.duration, 2)}


# ---------------------------------------------------------------- analysis

def loudness(source: Path, duration: float) -> np.ndarray:
    """RMS level in dBFS per STEP seconds; silence where audio is missing."""
    bins = max(1, math.ceil(duration / STEP))
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-t", f"{duration:.3f}", "-vn", "-ac", "1",
                              "-ar", "16000", "-f", "s16le", "-"], check=True, capture_output=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return np.full(bins, -90.0)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    size = int(16000 * STEP)
    levels = np.full(bins, -90.0)
    for i in range(min(bins, len(samples) // size + 1)):
        chunk = samples[i * size:(i + 1) * size]
        if len(chunk):
            levels[i] = 20 * math.log10(max(1e-5, float(np.sqrt(np.mean(chunk ** 2)))))
    return levels


def transcript_words(captions) -> list:
    words = []
    for caption in captions or ():
        if caption.words:
            words.extend(Word(float(w.start), float(w.end), w.word, getattr(w, "probability", 1.0)) for w in caption.words)
        else:
            tokens = caption.text.split() or [""]
            span = (caption.end - caption.start) / len(tokens)
            words.extend(Word(caption.start + i * span, caption.start + (i + 1) * span, (" " if i else "") + token)
                         for i, token in enumerate(tokens))
    return sorted((w for w in words if w.end > w.start and w.word.strip()), key=lambda w: w.start)


def _zscores(levels):
    audible = levels[levels > -70]
    if len(audible) < 5:
        return np.zeros_like(levels)
    spread = max(3.0, float(np.std(audible)))
    return (levels - float(np.median(audible))) / spread


def _excitement(word):
    text = word.word.strip()
    score = 1.0 if EXCITED.search(text) else 0.0
    bare = text.strip(".,!?;:\"'()")
    if len(bare) > 1 and bare.isupper():
        score += 0.5
    if len(bare) >= 6 and bare.casefold() not in STOPWORDS:
        score += 0.2
    return score


def speech_mask(words, duration, pad=0.12):
    mask = np.zeros(max(1, math.ceil(duration / STEP)), dtype=bool)
    for w in words:
        mask[max(0, int((w.start - pad) / STEP)):min(len(mask), math.ceil((w.end + pad) / STEP))] = True
    return mask


def find_hook(levels, words, duration, peak):
    """Best 1.5-3 s teaser window; returns (start, end, strength 0-1) or None."""
    z = _zscores(levels)
    if duration < 9:
        return None
    best = None
    for length in (2.0, 2.6, 3.0):
        for start in np.arange(3.0, duration - length, 0.2):
            a, b = int(start / STEP), int((start + length) / STEP)
            window = z[a:b]
            if not len(window):
                continue
            inside = [w for w in words if start <= (w.start + w.end) / 2 < start + length]
            score = 0.6 * float(np.mean(window)) + 0.5 * float(np.max(window))
            score += 0.6 * sum(_excitement(w) for w in inside) + 0.15 * min(len(inside), 8)
            if peak is not None:
                score += 1.6 * math.exp(-((start + length / 2 - peak) ** 2) / (2 * 4.0 ** 2))
            if best is None or score > best[2]:
                best = (float(start), float(start + length), score)
    if best is None:
        return None
    start, end, score = best
    # Snap to word boundaries so the teaser never starts or ends mid-word.
    starting = [w.start for w in words if start - 0.5 <= w.start <= start + 0.2]
    if starting:
        start = max(0.0, min(starting) - 0.05)
    ending = [w.end for w in words if end - 0.3 <= w.end <= end + 0.6]
    if ending:
        end = min(duration, max(ending) + 0.08)
    end = min(end, start + 3.0)
    strength = max(0.0, min(1.0, score / 4.0))
    return start, end, strength


def find_silences(levels, words, start, end, protect=(), minimum=0.8):
    """Dead-air gaps (no speech, quiet) to jump-cut, each kept with breathing room."""
    audible = levels[levels > -70]
    quiet = (float(np.median(audible)) - 7) if len(audible) else -60
    mask = speech_mask(words, len(levels) * STEP)
    minimum = minimum if words else max(minimum, 1.2)
    gaps, run = [], None
    for i in range(int(start / STEP), min(len(levels), int(end / STEP))):
        silent = not mask[i] and levels[i] < quiet
        if silent and run is None:
            run = i
        elif not silent and run is not None:
            gaps.append((run * STEP, i * STEP))
            run = None
    if run is not None:
        gaps.append((run * STEP, end))
    cuts = []
    for a, b in gaps:
        if b - a < minimum or any(a < p1 and b > p0 for p0, p1 in protect):
            continue
        cuts.append((a + 0.2, b - 0.2))
    return cuts


def score_clip(levels, words, duration, peak, hook_strength, spike_score, cut_seconds):
    z = _zscores(levels)
    speaking = speech_mask(words, duration).mean() if words else 0.0
    rate = len(words) / max(1.0, duration)
    pacing = 100 * max(0.0, 1 - abs(rate - 2.8) / 2.8) * (1 - 0.5 * max(0.0, (1 - speaking) - 0.4))
    peaks = float(np.mean(z > 1.2)) if len(z) else 0.0
    excited = sum(_excitement(w) > 0.9 for w in words)
    emotion = min(100.0, 220 * peaks + 12 * excited)
    dialogue = min(100.0, 140 * speaking)
    chat = min(100.0, 22 * max(0.0, float(spike_score or 0)))
    hook = 100 * hook_strength
    weights = {"hook": .3, "pacing": .2, "emotion": .25, "dialogue": .1, "chat": .15}
    parts = {"hook": hook, "pacing": pacing, "emotion": emotion, "dialogue": dialogue, "chat": chat}
    overall = sum(parts[k] * w for k, w in weights.items())
    tips = []
    if hook < 45:
        tips.append("Opening is calm; start the clip closer to the reaction for a stronger hook.")
    if pacing < 45:
        tips.append("Pacing is slow; trim the lead-in or shorten the clip.")
    if dialogue < 30:
        tips.append("Little speech detected; captions and music carry this clip.")
    if emotion < 30:
        tips.append("Few loud reactions; consider a livelier moment.")
    if duration > 45:
        tips.append("Clips under 45 s tend to hold viewers better on Shorts and Reels.")
    if cut_seconds > 0.5:
        tips.append(f"Removed {cut_seconds:.1f} s of dead air.")
    return {"overall": int(round(overall)), **{k: int(round(v)) for k, v in parts.items()}}, tips


# ---------------------------------------------------------------- planning

def plan_edit(levels, captions, duration, options: DirectorOptions, peak=None, spike_score=None, hook_text=None) -> Plan:
    words = transcript_words(captions)
    z = _zscores(levels)
    hook = find_hook(levels, words, duration, peak) if options.hook else None
    strength = hook[2] if hook else 0.0
    if hook and strength < 0.35:
        hook = None
    main_start = 0.0
    if options.edits:
        active = [i * STEP for i in range(min(len(z), int(3 / STEP))) if z[i] > -0.5]
        first_word = words[0].start if words else None
        candidates = [t for t in (first_word, active[0] if active else None) if t is not None]
        if candidates:
            main_start = max(0.0, min(2.5, min(candidates) - 0.3))
    protect = [(peak - 1.5, peak + 3.0)] if peak is not None else []
    cuts = find_silences(levels, words, main_start, duration, protect) if options.edits else []
    pieces, cursor = [], main_start
    for a, b in cuts:
        if a - cursor >= 0.6:
            pieces.append(Segment(cursor, a))
            cursor = b
    if duration - cursor > 0.3:
        pieces.append(Segment(cursor, duration))
    elif pieces:
        pieces[-1].end = duration
    segments = ([Segment(hook[0], hook[1], "hook")] if hook else []) + pieces
    over = sum(s.length for s in segments) - MAX_OUTPUT
    while over > 0 and segments:
        last = segments[-1]
        if last.length - over >= 0.6:
            last.end -= over
            over = 0
        else:
            over -= last.length
            segments.pop()
    plan = Plan(segments, trimmed_start=main_start, removed=sum(b - a for a, b in cuts))
    offsets = plan.offsets()
    if hook:
        plan.hook = {"source_start": round(hook[0], 2), "source_end": round(hook[1], 2),
                     "duration": round(hook[1] - hook[0], 2), "text": hook_text}
        plan.zooms.append([0.0, hook[1] - hook[0], 0.16, 0.25, "hook"])
        boundary = hook[1] - hook[0]
        plan.flashes.append(boundary)
        plan.sfx.append([max(0.0, boundary - 0.25), "whoosh"])
    if options.edits:
        mains = [(o, s) for o, s in zip(offsets, segments) if s.role == "main"]
        for index, (offset, segment) in enumerate(mains[1:], 1):
            if index % 2:
                plan.zooms.append([offset, offset + segment.length, 0.07, 0.0, "cut"])
        reaction = plan.map_time(peak) if peak is not None else None
        if reaction is not None:
            plan.zooms.append([max(0.0, reaction - 0.3), min(plan.duration, reaction + 2.2), 0.2, 0.35, "reaction"])
            plan.shakes.append([max(0.0, reaction - 0.3), min(plan.duration, reaction + 0.25)])
            plan.sfx.append([max(0.0, reaction - 0.3), "hit"])
        moments = []
        for w in words:
            t = plan.map_time(w.start)
            if t is not None and _excitement(w) >= 1:
                moments.append((_excitement(w) + float(z[min(len(z) - 1, int(w.start / STEP))]), t))
        for i in range(len(z)):
            t = plan.map_time(i * STEP)
            if t is not None and z[i] > 2.0 and (i == 0 or z[i - 1] <= 2.0):
                moments.append((float(z[i]), t))
        taken = [(a, b) for a, b, *_ in plan.zooms if _[-1] != "cut"]
        for _, t in sorted(moments, reverse=True):
            if len([zm for zm in plan.zooms if zm[-1] == "punch"]) >= 4:
                break
            if all(t < a - 2.0 or t > b + 2.0 for a, b in taken) and t + 0.9 <= plan.duration:
                plan.zooms.append([t - 0.05, t + 0.9, 0.11, 0.1, "punch"])
                taken.append((t - 0.05, t + 0.9))
                if options.sfx:
                    plan.sfx.append([max(0.0, t - 0.05), "pop"])
    if not options.sfx:
        plan.sfx = []
    speaking = speech_mask(words, duration).mean() if words else 0.0
    audible = levels[levels > -70]
    quiet_mix = (float(np.median(audible)) if len(audible) else -90) < -28
    plan.music = bool(options.music == "on" or (options.music == "auto" and speaking < 0.35 and quiet_mix))
    plan.score, plan.suggestions = score_clip(levels, words, duration, peak, strength if hook else
                                              min(1.0, max(0.0, float(np.mean(z[:30])) + 0.5) / 2),
                                              spike_score, plan.removed + main_start)
    return plan


# ---------------------------------------------------------------- captions

def remap_captions(captions, plan: Plan) -> list:
    """Move transcript words onto the edited timeline, duplicating hook words."""
    words = transcript_words(captions)
    moved = []
    for offset, segment in zip(plan.offsets(), plan.segments):
        for w in words:
            middle = (w.start + w.end) / 2
            if segment.start <= middle < segment.end:
                start = offset + max(0.0, w.start - segment.start)
                end = offset + min(segment.length, w.end - segment.start)
                if end > start:
                    moved.append(Word(start, end, w.word, w.probability))
    return chunk_words(moved, clip_end=plan.duration) if moved else []


def hook_title_ass(ass_text: str | None, text: str, end: float, caption_position="bottom") -> str:
    """Add an animated hook banner over the opening seconds of an ASS script."""
    ass_text = ass_text or _classic_ass([])
    alignment, margin = (5, 0) if caption_position == "top" else (8, 380)
    style = (f"Style: Hook,Montserrat Black,88,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00101317,-1,0,0,0,100,100,0,0,"
             f"3,14,0,{alignment},110,110,{margin},1\n")
    ass_text = ass_text.replace("\n\n[Events]", "\n" + style + "\n[Events]", 1)
    label = literal_text(text.upper())
    banner = (r"{\fad(80,160)\fscx70\fscy70\t(0,160,\fscx108\fscy108)\t(160,260,\fscx100\fscy100)\1c&H4DE3FF&}" + label)
    from .captions import timestamp
    return ass_text + f"Dialogue: 1,{timestamp(0, True)},{timestamp(max(0.8, end), True)},Hook,,0,0,0,,{banner}\n"


# ---------------------------------------------------------------- rendering

def _has_audio(source: Path) -> bool:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                             "-of", "csv=p=0", str(source)], capture_output=True, text=True)
    return bool(result.stdout.strip())


def cut_command(source: Path, output: Path, plan: Plan) -> list:
    audio = _has_audio(source)
    parts, labels = [], ""
    for i, segment in enumerate(plan.segments):
        parts.append(f"[0:v]trim=start={segment.start:.3f}:end={segment.end:.3f},setpts=PTS-STARTPTS,fps={FPS}[v{i}]")
        if audio:
            parts.append(f"[0:a]atrim=start={segment.start:.3f}:end={segment.end:.3f},asetpts=PTS-STARTPTS,"
                         f"aresample=48000,afade=t=in:d=0.02,afade=t=out:st={max(0, segment.length - 0.03):.3f}:d=0.03[a{i}]")
        else:
            parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={segment.length:.3f}[a{i}]")
        labels += f"[v{i}][a{i}]"
    parts.append(f"{labels}concat=n={len(plan.segments)}:v=1:a=1[v][a]")
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-filter_complex", ";".join(parts),
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(output)]


def cut_source(source: Path, output: Path, plan: Plan) -> Path:
    subprocess.run(cut_command(source, output, plan), check=True)
    return output


def _window(a, b, ease, var="t"):
    if ease <= 0:
        return f"between({var},{a:.3f},{b:.3f})"
    return f"clip(({var}-{a:.3f})/{ease:.3f},0,1)*clip(({b:.3f}-{var})/{ease:.3f},0,1)"


@dataclass
class Effects:
    zoom: str | None = None
    shake: str | None = None
    flash: str | None = None
    focal: str | None = None


def effects_for(plan: Plan) -> Effects:
    # zoompan exposes the frame time as `in_time`; other filters use `t`.
    zoom = "+".join(f"{amount:.3f}*{_window(a, b, ease, 'in_time')}" for a, b, amount, ease, _ in plan.zooms)
    shake = "+".join(f"9*sin(in_time*55)*{_window(a, b, 0, 'in_time')}" for a, b in plan.shakes)
    flash = "+".join(f"0.55*clip(1-abs(t-{t:.3f})/0.12,0,1)" for t in plan.flashes)
    focal = None
    if plan.focal and len({round(f, 3) for f in plan.focal}) > 1:
        focal = f"{plan.focal[0]:.4f}"
        for offset, previous, value in zip(plan.offsets()[1:], plan.focal, plan.focal[1:]):
            focal += f"+({value:.4f}-{previous:.4f})*clip((t-{offset:.3f})/0.25+0.5,0,1)"
    return Effects(zoom=f"min(1.35,1+{zoom})" if zoom else None, shake=shake or None, flash=flash or None, focal=focal)


SFX = {
    # Band-passed noise swept by a fade gives an airy transition whoosh.
    "whoosh": "anoisesrc=d=0.5:c=pink:a=0.9,highpass=f=500,lowpass=f=5000,afade=t=in:d=0.3,afade=t=out:st=0.3:d=0.2,volume=0.55",
    "hit": "aevalsrc='0.9*sin(2*PI*(45+110*exp(-t*18))*t)*exp(-t*7)':d=0.7:s=48000,volume=0.9",
    "pop": "aevalsrc='0.6*sin(2*PI*(900-500*t)*t)*exp(-t*30)':d=0.18:s=48000,volume=0.6",
}
CHORDS = ((220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (261.63, 329.63, 392.0), (196.0, 246.94, 293.66))


def music_source(duration: float) -> str:
    def chord(notes):
        return "+".join(f"sin(2*PI*{f}*t)+0.4*sin(4*PI*{f}*t)" for f in notes)
    body = chord(CHORDS[3])
    for i in (2, 1, 0):
        body = f"if(lt(mod(t,8),{2 * (i + 1)}),{chord(CHORDS[i])},{body})"
    return (f"aevalsrc='0.05*({body})*(0.55+0.45*exp(-mod(t,0.5)*5))':d={duration:.3f}:s=48000,"
            f"lowpass=f=1800,afade=t=in:d=1.2,afade=t=out:st={max(0, duration - 1.5):.3f}:d=1.5,volume=0.45")


def mix_command(video: Path, output: Path, plan: Plan) -> list:
    duration = plan.duration
    parts = [f"[0:a]aresample=48000,aformat=channel_layouts=stereo,apad=whole_dur={duration:.3f}[voice0]"]
    beds = ["[voicemix]"]
    if plan.music:
        parts[0] += ";[voice0]asplit[voicemix][key]"
        parts.append(f"{music_source(duration)},aformat=channel_layouts=stereo[bed]")
        parts.append("[bed][key]sidechaincompress=threshold=0.03:ratio=10:attack=15:release=350[music]")
        beds.append("[music]")
    else:
        parts[0] = parts[0].replace("[voice0]", "[voicemix]")
    for i, (t, kind) in enumerate(plan.sfx):
        delay = max(0, round(t * 1000))
        parts.append(f"{SFX[kind]},aformat=sample_rates=48000:channel_layouts=stereo,adelay={delay}|{delay},"
                     f"apad=whole_dur={duration:.3f}[fx{i}]")
        beds.append(f"[fx{i}]")
    mix = "".join(beds) + (f"amix=inputs={len(beds)}:duration=first:dropout_transition=0,volume={len(beds)}"
                           if len(beds) > 1 else "anull")
    parts.append(f"{mix},loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11,aformat=sample_rates=48000:channel_layouts=stereo,atrim=duration={duration:.3f}[out]")
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-filter_complex", ";".join(parts),
            "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-movflags", "+faststart", str(output)]


def mix_audio(video: Path, output: Path, plan: Plan) -> Path:
    subprocess.run(mix_command(video, output, plan), check=True)
    return output


# ---------------------------------------------------------------- quality check

def caption_band(style, size=80):
    position = getattr(style, "position", "bottom")
    height = size * 2.6
    if position == "top":
        return 260, 260 + height
    if position == "middle":
        return 960 - height / 2, 960 + height / 2
    return 1920 - 420 - height, 1920 - 420


def framing_fix(style, rect):
    """Move captions off the facecam; returns (style, note or None)."""
    if rect is None or style is None:
        return style, None
    top, bottom = caption_band(style, getattr(style, "size", 80))
    if rect.y < bottom and rect.y + rect.h > top:
        new = "middle" if style.position != "middle" else ("bottom" if rect.y < 960 else "top")
        style.position = new
        return style, f"Moved captions to {new} so they don't cover the facecam"
    return style, None


def probe(video: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate:format=duration",
                             "-of", "json", str(video)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def measure(video: Path) -> dict:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video),
                             "-vf", "blackdetect=d=0.4:pix_th=0.08", "-af", "ebur128=peak=true",
                             "-f", "null", "-"], capture_output=True, text=True)
    log = result.stderr
    summary = log[log.rfind("Summary:"):] if "Summary:" in log else ""
    lufs = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary)
    black = sum(float(m) for m in re.findall(r"black_duration:([\d.]+)", log))
    return {"lufs": float(lufs.group(1)) if lufs else None,
            "true_peak": float(peak.group(1)) if peak and peak.group(1) != "-inf" else None, "black_seconds": black}


def quality_check(video: Path, plan: Plan, captions, fixes=()) -> list:
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    try:
        info = probe(video)
        streams = info.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        num, _, den = (video_stream.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0
        duration = float(info.get("format", {}).get("duration", 0))
        size_ok = (video_stream.get("width"), video_stream.get("height")) == (1080, 1920)
        ok = size_ok and video_stream.get("codec_name") == "h264" and audio_stream is not None and duration <= MAX_OUTPUT + 0.2
        add("Format", "pass" if ok else "warn",
            f"{video_stream.get('width')}×{video_stream.get('height')} · {fps:.0f} fps · {duration:.1f} s · "
            f"{video_stream.get('codec_name')}/{audio_stream.get('codec_name') if audio_stream else 'no audio'}")
    except (subprocess.CalledProcessError, ValueError, OSError) as error:
        add("Format", "warn", f"Could not inspect the video: {error}")
    stats = measure(video)
    if stats["lufs"] is None:
        add("Audio balance", "warn", "Could not measure loudness")
    else:
        within = abs(stats["lufs"] - TARGET_LUFS) <= 2 and (stats["true_peak"] is None or stats["true_peak"] <= -0.5)
        add("Audio balance", "pass" if within else "warn",
            f"{stats['lufs']:.1f} LUFS (target {TARGET_LUFS:.0f})" +
            (f" · peak {stats['true_peak']:.1f} dBFS" if stats["true_peak"] is not None else "") +
            (" · music ducked under voice" if plan.music else ""))
    if captions:
        overlaps = sum(a.end > b.start + 0.01 for a, b in zip(captions, captions[1:]))
        short = sum(c.end - c.start < 0.3 for c in captions)
        late = sum(c.end > plan.duration + 0.05 for c in captions)
        status = "pass" if not (overlaps or short or late) else "warn"
        add("Subtitle timing", status, f"{len(captions)} captions synced to the edit" +
            (f" · {overlaps} overlapping" if overlaps else "") + (f" · {short} flash by" if short else "") +
            (f" · {late} past the end" if late else ""))
    else:
        add("Subtitle timing", "skip", "No captions in this clip")
    framing = [f for f in fixes if f]
    add("Framing", "pass", "; ".join(framing) if framing else "Captions, hook and facecam stay clear of each other")
    microcuts = sum(s.length < 0.6 for s in plan.segments)
    black = stats["black_seconds"]
    add("Visual consistency", "pass" if not microcuts and black < 0.5 else "warn",
        f"{len(plan.segments)} shots, max zoom {1 + max([z[2] for z in plan.zooms] or [0]):.2f}×" +
        (f" · {black:.1f} s black frames" if black >= 0.5 else "") + (f" · {microcuts} very short shots" if microcuts else ""))
    return checks


def plan_record(plan: Plan, options: DirectorOptions, checks) -> dict:
    return {"status": "used", "options": asdict(options), "score": plan.score, "edits": plan.summary(),
            "suggestions": plan.suggestions, "quality": checks}
