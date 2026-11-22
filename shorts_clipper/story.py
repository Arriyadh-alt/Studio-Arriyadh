"""AI storytelling hook: a short line that sets up the moment, and the story arc around it.

The hook writer runs offline from the transcript, chat and stream title. When
CLIPPER_LLM_API_KEY is set it asks an OpenAI-compatible model instead and falls
back to the offline writer on any error.
"""

import hashlib
import json
import logging
import math
import os
import re
import urllib.request
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TEXT_LIMIT = 64
STEP = 0.1

CUES = {
    "karma": r"karma|kena batunya|kualat|senjata makan tuan|backfired?|serves? (?:you|him|me) right|deserved? (?:it|that)",
    "fail": r"fail(?:ed)?|oops+|died|dead|mati|kalah|lost|jatuh|nabrak|ketabrak|crash(?:ed)?|blunder|throw(?:ing|n)?|missed|meleset|ga+k? kena|nooo+",
    "scare": r"scream\w*|jump ?scare|scared|takut|kaget|setan|hantu|ghost|monster|a{3,}h*",
    "win": r"won|menang|clutch|ace|victory|let'?s go+|ayo+|mantap|berhasil|finally|akhirnya|yes{2,}",
    "funny": r"lo+l|lmao+|(?:ha){2,}|wkwk\w*|kekw|ngakak|lucu|funny",
    "rage": r"anjir|anjing|bangsat|kesel|tolol|goblok|wtf|damn|kampret|sialan|rage|angry|marah",
    "shock": r"what+|wow|omg|astaga|ya ampun|gila|no way|insane|crazy|serius|beneran",
    "wholesome": r"thank\w*|terima ?kasih|makasih|love|sayang|donat\w*|subscri\w*|gift(?:ed)?|support",
}
PATTERNS = {kind: re.compile(rf"\b(?:{cue})\b", re.I) for kind, cue in CUES.items()}
CHAT_CATEGORY = {"laughter": "funny", "shock": "shock", "hype": "win"}

TEMPLATES = {
    "en": {
        "karma": ["The Moment {who} Gets Instant Karma{in_game}!", "{who} Didn't See This Karma Coming{in_game}",
                  "Karma Came Fast For {who}{in_game}"],
        "fail": ["{who} Had One Job{in_game}...", "Watch {who}'s Plan Fall Apart{in_game}",
                 "This Is Where It Went Wrong For {who}"],
        "scare": ["{who} Was NOT Ready For This{in_game}", "Wait For What Scares {who}...",
                  "The Moment {who} Panicked{in_game}"],
        "win": ["Can {who} Pull Off The Impossible{in_game}?", "Nobody Expected {who} To Do This{in_game}",
                "Watch {who} Turn It Around{in_game}"],
        "funny": ["The Moment {who} Couldn't Stop Laughing", "This Broke {who}'s Chat{in_game}",
                  "{who} Did NOT Expect This{in_game}"],
        "rage": ["The Moment {who} Completely Lost It{in_game}", "What Made {who} This Angry?",
                 "{who}'s Patience Ran Out Right Here"],
        "shock": ["{who} Can't Believe What Just Happened{in_game}", "Wait Until You See What {who} Saw",
                  "Nobody Was Ready For This Moment{in_game}"],
        "wholesome": ["The Moment {who} Got Emotional On Stream", "{who} Did Not Expect This From Chat",
                      "This Made {who}'s Whole Stream"],
    },
    "id": {
        "karma": ["Momen Ketika {who} Terkena Karma{in_game}!", "{who} Langsung Kena Batunya{in_game}",
                  "Karma Datang Cepat Buat {who}"],
        "fail": ["Rencana {who} Berantakan Total{in_game}", "Detik-Detik {who} Bikin Blunder{in_game}",
                 "Di Sinilah Semuanya Kacau Buat {who}"],
        "scare": ["{who} Sama Sekali Nggak Siap{in_game}", "Tunggu Sampai {who} Lihat Ini...",
                  "Momen {who} Panik Total{in_game}"],
        "win": ["Bisakah {who} Melakukan Hal Mustahil{in_game}?", "Nggak Ada Yang Nyangka {who} Bisa Begini",
                "Detik-Detik {who} Membalikkan Keadaan{in_game}"],
        "funny": ["{who} Nggak Bisa Berhenti Ketawa", "Momen Yang Bikin Chat {who} Pecah",
                  "{who} Nggak Nyangka Bakal Begini{in_game}"],
        "rage": ["Momen Ketika {who} Kehilangan Kesabaran{in_game}", "Apa Yang Bikin {who} Semarah Ini?",
                 "Kesabaran {who} Habis Di Sini"],
        "shock": ["{who} Nggak Percaya Apa Yang Terjadi{in_game}", "Tunggu Sampai Kamu Lihat Ini...",
                  "Nggak Ada Yang Siap Sama Momen Ini{in_game}"],
        "wholesome": ["Momen Ketika {who} Terharu Di Live", "{who} Nggak Nyangka Dapat Ini Dari Chat",
                      "Momen Paling Hangat Di Live {who}"],
    },
}
QUOTE_TEMPLATES = {
    "en": ['Why Did {who} Say "{quote}"?', 'It All Started With "{quote}"{in_game}...'],
    "id": ['Kenapa {who} Bilang "{quote}"?', 'Semua Berawal Dari "{quote}"{in_game}...'],
}
FALLBACK_WHO = {"en": "This Streamer", "id": "Streamer Ini"}
IN_GAME = {"en": " in {game}", "id": " di {game}"}

GAMES = [
    ("GTA VI", r"gta\s*(?:vi|6)\b|grand theft auto\s*(?:vi|6)"), ("GTA V", r"gta\s*(?:v|5)\b|grand theft auto\s*(?:v|5)"),
    ("GTA", r"\bgta\b"), ("Minecraft", r"minecraft"), ("Valorant", r"valorant"), ("Fortnite", r"fortnite"),
    ("Roblox", r"roblox"), ("PUBG", r"pubg"), ("Mobile Legends", r"mobile legends?|\bmlbb\b"),
    ("Free Fire", r"free ?fire"), ("Genshin Impact", r"genshin"), ("Elden Ring", r"elden ring"),
    ("Resident Evil", r"resident evil"), ("Phasmophobia", r"phasmophobia"), ("Lethal Company", r"lethal company"),
    ("Apex Legends", r"apex legends?"), ("Warzone", r"warzone"), ("Call of Duty", r"call of duty|\bcod\b"),
    ("League of Legends", r"league of legends"), ("Dota 2", r"dota"), ("CS2", r"\bcs2\b|counter[- ]strike"),
    ("Among Us", r"among us"), ("Sons of the Forest", r"sons of the forest"), ("The Forest", r"the forest"),
    ("Dead by Daylight", r"dead by daylight|\bdbd\b"), ("Outlast", r"outlast"), ("Silent Hill", r"silent hill"),
    ("Red Dead Redemption 2", r"red dead|\brdr ?2\b"), ("Cyberpunk 2077", r"cyberpunk"), ("Terraria", r"terraria"),
    ("Stardew Valley", r"stardew"), ("Palworld", r"palworld"), ("R.E.P.O.", r"r\.?e\.?p\.?o\b"),
    ("Hogwarts Legacy", r"hogwarts"), ("Euro Truck Simulator", r"euro truck"), ("Pokémon", r"pok[eé]mon"),
    ("Five Nights at Freddy's", r"five nights|\bfnaf\b"), ("Dark Souls", r"dark souls"), ("Sekiro", r"sekiro"),
]
ID_HINTS = re.compile(r"\b(?:yang|nggak|gak|banget|aku|kita|kalian|main|bareng|sampai|tamat|seru|ini|itu)\b", re.I)


def streamer_name(metadata) -> str | None:
    """Channel name cleaned up for a title, or None when YouTube gave no channel."""
    metadata = metadata or {}
    raw = metadata.get("channel") or metadata.get("uploader")
    if not isinstance(raw, str):
        return None
    name = re.sub(r"\b(?:official|channel|ch\.?|tv|live|gaming)\b", " ", raw, flags=re.I)
    name = re.sub(r"[|•\-–—_@#]+", " ", name)
    name = " ".join(name.split())
    if not name or not re.search(r"[^\W\d_]", name):
        return None
    if len(name) > 22:
        name = " ".join(name.split()[:2])[:22].strip()
    return name if name != name.lower() else name.title()


def game_name(title) -> str | None:
    title = title or ""
    for name, pattern in GAMES:
        if re.search(pattern, title, re.I):
            return name
    return None


def hook_language(caption_language=None, title=None) -> str:
    if caption_language in ("id", "ms", "jw", "su"):
        return "id"
    if caption_language:
        return "en"
    return "id" if len(ID_HINTS.findall(title or "")) >= 2 else "en"


def classify_moment(words, messages, event, window=6.0) -> str:
    """Event type from what the streamer says around the moment and what chat reacts with."""
    scores = dict.fromkeys(CUES, 0.0)
    near = " ".join(w.word for w in words if abs(w.start - event) <= 3)
    around = " ".join(w.word for w in words if 3 < abs(w.start - event) <= window)
    for kind, pattern in PATTERNS.items():
        scores[kind] += len(pattern.findall(near)) + 0.6 * len(pattern.findall(around))
    from clipper_studio.spikes import reaction_category
    for message in messages or ():
        for kind, pattern in PATTERNS.items():
            if pattern.search(message):
                scores[kind] += 0.7
        category = CHAT_CATEGORY.get(reaction_category(message) or "")
        if category:
            scores[category] += 0.5
    scores["karma"] *= 2.0  # an explicit karma call is rare and specific
    if scores["fail"] and (scores["funny"] or scores["karma"]):
        scores["karma"] += 0.5 * scores["fail"]
    scores["shock"] *= 0.8  # generic surprise only wins without a clearer cue
    kind, best = max(scores.items(), key=lambda item: item[1])
    return kind if best >= 0.9 else "shock"


def _fit(text: str) -> str | None:
    text = " ".join(text.split())
    return text if 8 <= len(text) <= TEXT_LIMIT else None


def key_phrase(words, category, event) -> str | None:
    """The streamer's own words that mark the moment ("thank you", "no way"), to anchor the hook in this clip."""
    pattern = PATTERNS.get(category)
    near = [w for w in words if event - 3 <= w.start <= event + 4]
    for i, w in enumerate(near):
        if pattern and pattern.search(w.word):
            picked = [w.word]
            if i + 1 < len(near) and near[i + 1].start - w.end < 0.4 and not re.search(r"[.!?,]$", w.word):
                picked.append(near[i + 1].word)
            phrase = re.sub(r"[^\w' ]", "", " ".join(picked)).strip()
            if 2 <= len(phrase) <= 16:
                return phrase.title().replace("'S", "'s").replace("'T", "'t")
    return None


def template_hooks(category, language, name=None, game=None, seed="", used=(), quote=None) -> list:
    """Up to three hooks for the event, rotated by seed so clips of one stream differ. A hook built on the
    streamer's own words at the moment comes first, since it is specific to this clip."""
    language = language if language in TEMPLATES else "en"
    templates = TEMPLATES[language].get(category) or TEMPLATES[language]["shock"]
    who = name or FALLBACK_WHO[language]

    def fill(template):
        for in_game in ((IN_GAME[language].format(game=game), "") if game else ("",)):
            text = _fit(template.format(who=who, in_game=in_game, quote=quote))
            if text:
                return text
        return None

    options = [text for text in map(fill, templates) if text]
    shift = int(hashlib.sha1(str(seed).encode()).hexdigest(), 16) % len(options) if options else 0
    options = options[shift:] + options[:shift]
    if quote and category != "karma":  # an explicit karma template already names the event
        quoted = [text for text in map(fill, QUOTE_TEMPLATES[language]) if text]
        options = (quoted[:1] if name else quoted[1:2] or quoted[:1]) + options
    used = {u.casefold() for u in used if u}
    return sorted(options, key=lambda o: o.casefold() in used)[:3]


def llm_hooks(context: dict, category: str, language: str) -> list:
    """Ask an OpenAI-compatible model for hooks; [] when no key is configured or the call fails."""
    key = os.environ.get("CLIPPER_LLM_API_KEY")
    if not key:
        return []
    base = os.environ.get("CLIPPER_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("CLIPPER_LLM_MODEL", "gpt-4o-mini")
    tongue = "natural, entertaining Indonesian (bahasa gaul is fine)" if language == "id" else "natural English"
    prompt = (
        f"Write 3 different hooks for a vertical short of a livestream moment, in {tongue}. Each hook is shown as "
        f"bold text for the first 3 seconds, at most 60 characters, in Title Case. Build curiosity about what is "
        f"about to happen without revealing the outcome, and never claim anything the transcript or chat does not "
        f"support. Use the streamer's name when given.\n"
        f"Streamer: {context.get('name') or 'unknown'}\nGame: {context.get('game') or 'unknown'}\n"
        f"Stream title: {context.get('title') or ''}\nLikely event type: {category}\n"
        f"Transcript around the moment (seconds from the event):\n{context.get('transcript', '')}\n"
        f"Top chat reactions: {json.dumps(list(context.get('messages') or [])[:5], ensure_ascii=False)}\n"
        f"Avoid these hooks already used on this stream: {json.dumps(list(context.get('used') or []), ensure_ascii=False)}\n"
        'Reply with JSON only: {"hooks": ["...", "...", "..."]}')
    body = json.dumps({"model": model, "temperature": 0.8, "response_format": {"type": "json_object"},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(base + "/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            reply = json.loads(response.read())
        hooks = json.loads(reply["choices"][0]["message"]["content"]).get("hooks", [])
        return [text for text in (_fit(str(h).strip().strip('"')) for h in hooks) if text][:3]
    except Exception:
        logger.warning("LLM hook writer failed; using the offline writer", exc_info=True)
        return []


def write_hooks(context: dict, words, event: float, seed="") -> dict:
    """Pick the event type and write hook options for it."""
    context = context or {}
    language = context.get("language") or "en"
    category = classify_moment(words, context.get("messages"), event)
    transcript = " ".join(f"[{w.start - event:+.0f}s] {w.word}" if i % 6 == 0 else w.word
                          for i, w in enumerate(w for w in words if event - 12 <= w.start <= event + 10))
    options = llm_hooks({**context, "transcript": transcript}, category, language)
    writer = "llm" if options else "local"
    if not options:
        options = template_hooks(category, language, context.get("name"), context.get("game"), seed,
                                 context.get("used", ()), key_phrase(words, category, event))
    return {"options": options, "category": category, "language": language, "writer": writer,
            "streamer": context.get("name"), "game": context.get("game")}


# ---------------------------------------------------------------- arc


def _z(levels):
    levels = np.asarray(levels, dtype=float)
    audible = levels[levels > -70]
    if not len(audible):
        return np.zeros(len(levels))
    middle = float(np.median(audible))
    spread = float(np.median(np.abs(audible - middle))) * 1.4826 or 3.0
    return np.clip((levels - middle) / spread, -4, 6)


def story_arc(levels, words, duration, peak=None) -> dict:
    """Build-up start, main event and natural ending, all in source seconds.

    The event is the loudest reaction onset near the chat peak; the build-up starts at a sentence
    5-14 s earlier, and the clip ends once the reaction has died down, never mid-word.
    """
    z = _z(levels)
    smooth = np.convolve(z, np.ones(5) / 5, mode="same") if len(z) >= 5 else z
    if peak is None:
        lo, hi = int(min(8.0, duration / 3) / STEP), len(smooth)
    else:
        lo, hi = int(max(0.0, peak - 5) / STEP), int(min(duration, peak + 2) / STEP)
    if hi <= lo:
        lo, hi = 0, len(smooth)
    top = lo + int(np.argmax(smooth[lo:hi])) if hi > lo else 0
    onset = top
    while onset > max(lo, top - 30) and smooth[onset - 1] > 0.8:
        onset -= 1
    event = min(duration, max(0.0, onset * STEP))
    if peak is not None and smooth[top] < 1.0:
        event = min(duration, max(0.0, peak))

    starts = [w.start for i, w in enumerate(words) if i == 0 or w.start - words[i - 1].end >= 0.45]
    window = [s for s in starts if event - 14 <= s <= event - 5]
    wanted = event - (10 if len([w for w in words if event - 12 <= w.start < event]) > 18 else 8)
    if window:
        start = min(window, key=lambda s: abs(s - wanted)) - 0.25
    else:
        start = wanted
        inside = next((i for i, w in enumerate(words) if w.start < start < w.end), None)
        if inside is not None:
            before = words[inside - 1].end if inside else 0.0
            start = max(before, words[inside].start - 0.25)
    start = max(0.0, start)

    mask = np.zeros(len(z), dtype=bool)
    for w in words:
        mask[max(0, int((w.start - 0.1) / STEP)):min(len(mask), math.ceil((w.end + 0.1) / STEP))] = True
    end = None
    i = int(min(duration, event + 3.5) / STEP)
    limit = int(min(duration, event + 15) / STEP)
    while i + 9 <= limit:
        if not mask[i:i + 9].any() and float(np.mean(z[i:i + 9])) < 0.3:
            end = i * STEP + 0.35
            break
        i += 1
    if end is None:
        end = limit * STEP
        inside = next((i for i, w in enumerate(words) if w.start < end < w.end), None)
        if inside is not None:
            after = words[inside + 1].start if inside + 1 < len(words) else duration
            end = min(after, words[inside].end + 0.25)
    return {"start": round(start, 2), "event": round(event, 2), "end": round(min(duration, end), 2)}


def intro_length(text: str, voice: float | None = None) -> float:
    length = max(2.5, min(3.3, 1.5 + 0.04 * len(text)))
    if voice:
        length = max(length, min(4.0, voice + 0.4))
    return round(length, 2)


# ---------------------------------------------------------------- voice-over

VOICES = {"en": "en/en_US/ryan/medium/en_US-ryan-medium", "id": "id/id_ID/news_tts/medium/id_ID-news_tts-medium"}
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{}{}"


def voice_model(language: str, cache: Path) -> Path:
    path = VOICES.get(language, VOICES["en"])
    model = cache / (Path(path).name + ".onnx")
    if not model.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        for suffix in (".onnx.json", ".onnx"):
            target = cache / (Path(path).name + suffix)
            partial = target.with_name(target.name + ".part")
            urllib.request.urlretrieve(VOICE_URL.format(path, suffix), partial)
            partial.replace(target)
    return model


def synthesize_voice(text: str, language: str, output: Path, cache: Path) -> float:
    """Speak the hook with an offline Piper voice; returns its length in seconds."""
    from piper import PiperVoice, SynthesisConfig
    voice = PiperVoice.load(str(voice_model(language, cache)))
    with wave.open(str(output), "wb") as wav:
        voice.synthesize_wav(text.rstrip(".!?…") + "!", wav, syn_config=SynthesisConfig(length_scale=0.85))
    with wave.open(str(output), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())
