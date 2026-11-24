"""Local, single-process API. Run with uvicorn clipper_studio.api:app."""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import parse_qs, unquote, urlsplit

import cv2

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from cv2 import error as ImageProcessingError
from PIL import Image
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.background import BackgroundTask

from shorts_clipper.branding import BrandingSettings, LOGO_LIMIT, normalize_logo, prepare_watermark, read_logo
from shorts_clipper.captions import FONT_DIR, CaptionStyle, apply_caption_edits, caption_height, to_ass, transcribe
from shorts_clipper.downloader import download_analysis_media, download_section
from shorts_clipper.moments import analyze_media
from shorts_clipper.podcast import analyze_podcast
from shorts_clipper.sources import UPLOAD_ID, UPLOAD_SUFFIXES, local_analysis_media, local_section, probe_source
from shorts_clipper.speakers import frame_conversation
from shorts_clipper.thumbnails import TEXT_LIMIT, choose_hook, generate_thumbnail
from shorts_clipper.director import (STEP, DirectorOptions, plan_podcast_edit, Effects, caption_band, cut_source, effects_for, framing_fix, has_audio, hook_layout,
                                     hook_title_ass, loudness, make_seamless_loop, mix_audio, normalize_loudness,
                                     plan_edit, plan_record, quality_check, remap_captions)
from shorts_clipper.editing import MODE_DEFAULTS, authenticity_report, plan_source_edit, record_operations
from shorts_clipper.story import game_name, hook_language, streamer_name
from shorts_clipper.voices import DELIVERY_CHOICES, MOODS, PERSONAS, SAMPLES, VOICE_CHOICES, narrate
from shorts_clipper.video import detect_focal_x, make_vertical_clip, source_frame_rate, video_duration
from shorts_clipper.facecam import (SourceBox, detect_facecam, auto_layout_facecam,
                                    layout_facecam, save_source_still)
from shorts_clipper.tracking import analyze_reaction, keyframe_expression, layout_options, plan_reframe, to_output_time
from shorts_clipper.youtube import YouTubeAccessError

from .chat import METADATA_KEYS, ChatReplayError, fetch_chat, fetch_metadata
from .scheduler import Scheduler
from .spikes import detect_spikes
from .ranking import add_chat_moments, chat_moment, merge_moments


logger = logging.getLogger(__name__)
YOUTUBE_URL_ERROR = "Paste a YouTube video link for a finished livestream (watch, live, or youtu.be)."
ACTIVE_STATUSES = ("queued", "running")
CHAT_FALLBACK_WARNING = "Chat analysis is temporarily unavailable. AI Video Analysis will continue automatically."


class WorkCancelled(Exception):
    pass


class FacecamBox(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    x: float = Field(ge=0, lt=1, allow_inf_nan=False)
    y: float = Field(ge=0, lt=1, allow_inf_nan=False)
    w: float = Field(gt=0, le=1, allow_inf_nan=False)
    h: float = Field(gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def within_frame(self):
        if self.x + self.w > 1 or self.y + self.h > 1:
            raise ValueError("Camera area must fit inside the source frame")
        return self


class FacecamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = True
    position: Literal["auto", "top-left", "top-right", "bottom-left", "bottom-right"] = "auto"
    size: float = Field(default=0.42, ge=0.25, le=0.6, allow_inf_nan=False)
    box: FacecamBox | None = None


def _one_of(choices, name):
    def check(value):
        if value not in choices:
            raise ValueError(f"{name} must be one of {', '.join(choices)}")
        return value
    return AfterValidator(check)


VoiceChoice = Annotated[str, _one_of(VOICE_CHOICES, "voice")]
DeliveryChoice = Annotated[str, _one_of(DELIVERY_CHOICES, "delivery")]


class TrackingTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    t: float = Field(ge=0, allow_inf_nan=False)
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)


class TrackingSpec(BaseModel):
    """Reaction-aware reframing: automatic by default, with manual layout, crop and target overrides."""
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = True
    layout: Literal["auto", "gameplay", "reaction", "split"] = "auto"
    follow: Literal["auto", "fixed"] = "auto"
    x: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    target: TrackingTarget | None = None


class DirectorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = True
    hook: bool = True
    edits: bool = True
    sfx: bool = True
    music: Literal["auto", "on", "off"] = "auto"
    story: bool = True
    voiceover: bool = False
    hook_text: str | None = Field(default=None, max_length=64)
    voice: VoiceChoice = "auto"
    delivery: DeliveryChoice = "auto"
    loop: bool = True


EditingMode = Literal["natural", "enhanced", "full"]


def director_config(mode: EditingMode, requested: DirectorSpec) -> dict:
    config = {**DirectorSpec().model_dump(), **MODE_DEFAULTS[mode], **requested.model_dump(exclude_unset=True)}
    if mode == "natural":
        config.update(sfx=False, music="off", loop=False)
    return config


class VoicePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voice: VoiceChoice = "auto"
    delivery: DeliveryChoice = "auto"
    text: str | None = Field(default=None, max_length=64)
    language: Literal["en", "id"] = "en"
    category: str | None = Field(default=None, max_length=24)


PREVIEW_CACHE = 64
LEGACY_DIRECTOR = {**DirectorSpec().model_dump(), "enabled": False}


class CaptionEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    find: str = Field(min_length=1, max_length=40)
    replace: str = Field(default="", max_length=40)


class PodcastSpec(BaseModel):
    """Podcast / interview choices: framing layout, Version A (original) or B (AI commentary), and review edits."""
    model_config = ConfigDict(extra="forbid")
    layout: Literal["auto", "speaker", "split", "switch", "wide"] = "auto"
    version: Literal["original", "commentary"] = "original"
    speakers: int | None = Field(default=None, ge=1, le=3)
    hook_text: str | None = Field(default=None, max_length=64)
    commentary: str | None = Field(default=None, max_length=160)
    caption_edits: list[CaptionEdit] = Field(default_factory=list, max_length=50)


class RightsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    basis: Literal["owned", "licensed", "permission", "remix"]
    confirmed: bool
    attribution: bool = True
    include_link: bool = True
    note: str | None = Field(default=None, max_length=200)

    @field_validator("confirmed")
    @classmethod
    def must_confirm(cls, value):
        if not value:
            raise ValueError("Confirm that you are allowed to republish this source")
        return value


RIGHTS_BASIS = {"owned": "I own this content", "licensed": "Licensed for reuse",
                "permission": "Used with the creator's permission", "remix": "Permitted remix / commentary use"}


class JobRequest(BaseModel):
    url: str | None = None
    upload: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    content_type: Literal["gaming", "podcast"] = "gaming"
    podcast: PodcastSpec = Field(default_factory=PodcastSpec)
    ai_analysis: bool = True
    review: bool = True
    clips: int = Field(default=8, ge=1, le=50)
    lead: float = Field(default=18, ge=15, le=20)
    mode: Literal["crop", "blur"] = "crop"
    captions: bool = True
    facecam: FacecamSpec = Field(default_factory=FacecamSpec)
    tracking: TrackingSpec = Field(default_factory=TrackingSpec)
    language: str | None = Field(default="auto", pattern=r"^(auto|[a-z]{2,3})$")
    caption_style: dict | None = None
    editing_mode: EditingMode = "full"
    director: DirectorSpec = Field(default_factory=DirectorSpec)

    @field_validator("url")
    @classmethod
    def youtube_url(cls, value):
        if value is None:
            return None
        value = value.strip()
        try:
            parsed = urlsplit(value)
            if (parsed.scheme != "https" or parsed.username is not None
                    or parsed.port not in (None, 443) or any(char.isspace() for char in value)):
                raise ValueError
        except ValueError:
            raise ValueError(YOUTUBE_URL_ERROR) from None

        path = parsed.path.removesuffix("/")
        ids = []
        if parsed.hostname == "youtu.be":
            ids = [path.removeprefix("/")]
        elif parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
            if path == "/watch":
                ids = parse_qs(parsed.query, keep_blank_values=True).get("v", [])
            elif path.startswith("/live/"):
                ids = [path.removeprefix("/live/")]
        if len(ids) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]{11}", ids[0]):
            raise ValueError(YOUTUBE_URL_ERROR)
        return f"https://www.youtube.com/watch?v={ids[0]}"

    @model_validator(mode="after")
    def one_source(self):
        if (self.url is None) == (self.upload is None):
            raise ValueError("Paste a YouTube link or upload an authorized video file.")
        if self.upload or self.content_type == "podcast":
            self.ai_analysis = True  # no live chat to rank: conversation and media analysis do the work
        return self


class RenderRequest(BaseModel):
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)
    mode: Literal["crop", "blur"] = "crop"
    captions: bool = True
    facecam: FacecamSpec = Field(default_factory=FacecamSpec)
    tracking: TrackingSpec = Field(default_factory=TrackingSpec)
    language: str | None = Field(default="auto", pattern=r"^(auto|[a-z]{2,3})$")
    caption_style: dict | None = None
    editing_mode: EditingMode = "full"
    director: DirectorSpec = Field(default_factory=DirectorSpec)
    podcast: PodcastSpec | None = None


class KeepRequest(BaseModel):
    kept: bool


class MomentPatch(BaseModel):
    status: Literal["suggested", "selected", "rejected"] | None = None
    start: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    podcast: PodcastSpec | None = None


class MomentRender(BaseModel):
    ids: list[str] | None = None


class ThumbnailRequest(BaseModel):
    text: str | None = Field(default=None, max_length=TEXT_LIMIT)


def thumbnail_peak(clip, duration):
    peak = clip.get("peak")
    if peak is not None and 0 <= peak - clip["start"] < duration:
        return peak - clip["start"]
    return None


def rendered_offset(clip, offset):
    """Map a source-relative offset onto an AI Director edit's timeline."""
    timeline = ((clip.get("director_result") or {}).get("edits") or {}).get("timeline")
    if offset is None or not timeline:
        return offset
    for start, end, at in timeline:
        if start <= offset < end:
            return at + offset - start
    return None


def story_context(job, clip, caption_language, work, voices):
    """What the storytelling hook writer knows about the stream and this clip."""
    metadata = job.get("metadata") or {}
    used = [hook["text"] for other in job.get("clips", []) if other.get("id") != clip.get("id")
            and (hook := ((other.get("director_result") or {}).get("edits") or {}).get("hook")) and hook.get("text")]
    requested = clip.get("language") if clip.get("language") not in (None, "auto") else None

    def voice(text, language, voice="auto", category=None, delivery="auto", **limits):
        path = work / "hook-voice.wav"
        return path, narrate(text, language, path, voices, voice, category, clip.get("id", ""), delivery, **limits)

    return {"name": streamer_name(metadata), "game": game_name(metadata.get("title")),
            "title": metadata.get("title"), "language": hook_language(caption_language or requested, metadata.get("title")),
            "messages": clip.get("message_sample", ()), "used": used, "seed": clip.get("id", ""), "voice": voice,
            "moment": {k: clip.get(k) for k in ("title", "description", "category")} if clip.get("moment_id") else None}


def thumbnail_hook(job, clip, peak, captions=()):
    return choose_hook(captions, peak, clip.get("message_sample", ()), (job.get("metadata") or {}).get("title"))


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        repo = Path(__file__).resolve().parents[1]
        if self.root == repo or repo in self.root.parents:
            raise ValueError("CLIPPER_DATA_DIR must be outside the repository")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def get(self, jid):
        if not jid.isalnum():
            raise HTTPException(404, "Job not found")
        with self.lock:
            try:
                job = json.loads((self.root / jid / "job.json").read_text())
                job.setdefault("moments", [])
                job.setdefault("request", {}).setdefault("facecam", FacecamSpec().model_dump())
                job["request"].setdefault("director", dict(LEGACY_DIRECTOR))
                job["request"].setdefault("content_type", "gaming")
                for clip in job.get("clips", []):
                    clip.setdefault("facecam", FacecamSpec().model_dump())
                    clip.setdefault("tracking", TrackingSpec().model_dump())
                    clip.setdefault("director", dict(LEGACY_DIRECTOR))
                return job
            except FileNotFoundError:
                raise HTTPException(404, "Job not found") from None

    def save(self, job):
        with self.lock:
            directory = self.root / job["id"]
            directory.mkdir(exist_ok=True)
            temporary = directory / "job.json.tmp"
            temporary.write_text(json.dumps(job, allow_nan=False))
            temporary.replace(directory / "job.json")

    def jobs(self):
        with self.lock:
            return [self.get(path.parent.name) for path in sorted(self.root.glob("*/job.json"), reverse=True)]

    def save_branding(self, filename, data):
        with self.lock:
            temporary = self.root / f"{filename}.tmp"
            temporary.write_bytes(data)
            temporary.replace(self.root / filename)


COMMENTARY_SECONDS = 10.0


def is_podcast(job):
    return (job.get("request") or {}).get("content_type") == "podcast"


def require_rights(job):
    if is_podcast(job) and not (job.get("rights") or {}).get("confirmed"):
        raise HTTPException(403, "Confirm you have the rights to republish this source before exporting.")


def export_description(job, clip):
    """Upload description with source attribution and, when chosen, the original link."""
    metadata, rights = job.get("metadata") or {}, job.get("rights") or {}
    lines = [clip.get("title") or "Highlight"]
    if ((clip.get("director_result") or {}).get("edits") or {}).get("commentary"):
        lines.append("Includes AI-generated commentary, labelled in the video; the speakers' words are unedited.")
    if rights.get("attribution", True):
        who = metadata.get("channel") or metadata.get("uploader")
        lines.append("Source: " + (metadata.get("title") or job.get("filename") or "original recording")
                     + (f" by {who}" if who else ""))
    if rights.get("include_link", True) and job.get("url"):
        lines.append(f"Original: {job['url']}")
    if rights.get("basis"):
        lines.append(f"Rights: {RIGHTS_BASIS[rights['basis']]}")
    return "\n".join(lines)


def find_clip(job, cid):
    for clip in job["clips"]:
        if clip["id"] == cid:
            return clip
    raise HTTPException(404, "Clip not found")


def title_slug(job):
    title = unicodedata.normalize("NFKD", (job.get("metadata") or {}).get("title", ""))
    return re.sub(r"[^a-z0-9]+", "-", title.encode("ascii", "ignore").decode().lower()).strip("-")[:40].rstrip("-") or "clipper-studio"


def clip_number(job, clip):
    return clip.get("number") or next(i for i, c in enumerate(job["clips"], 1) if c["id"] == clip["id"])


def create_app(data_dir=None, analysis_workers=None, render_workers=None, metadata_workers=None):
    runtime_dir = str(Path(sys.executable).parent)
    if (Path(runtime_dir) / "deno").is_file() and runtime_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = runtime_dir + os.pathsep + os.environ.get("PATH", "")
    store = Store(data_dir or os.environ.get("CLIPPER_DATA_DIR", "~/clipper-data"))
    metadata_cache = {}

    def active_job(jid, generation):
        job = store.get(jid)
        if pending.stopping or job["status"] == "cancelled" or job.get("generation", 0) != generation:
            raise WorkCancelled
        return job

    def upload_path(job):
        if not job.get("upload"):
            return None
        folder = store.root / "uploads" / job["upload"]
        found = [p for p in folder.glob("source*") if p.suffix in UPLOAD_SUFFIXES] if folder.is_dir() else []
        if not found:
            raise FileNotFoundError("The uploaded source file is no longer available")
        return found[0]

    def stream_metadata(jid, generation, job):
        """Jobs made before channel names were kept get them fetched once, for the storytelling hook."""
        metadata = job.get("metadata") or {}
        if not job.get("url") or "channel" in metadata or "uploader" in metadata:
            return job
        try:
            info = fetch_metadata(job["url"], False, False)
        except Exception:
            logger.debug("Channel lookup for the storytelling hook failed", exc_info=True)
            return job
        with store.lock:
            job = active_job(jid, generation)
            job["metadata"] = {**(job.get("metadata") or {}), **{key: info.get(key) for key in ("channel", "uploader")}}
            store.save(job)
        return job

    def update(jid, generation, **fields):
        with store.lock:
            job = active_job(jid, generation)
            job.update(fields)
            store.save(job)

    def render_attempt(jid, cid, generation):
        with store.lock:
            job = active_job(jid, generation)
            clip = find_clip(job, cid)
            clip.update(status="rendering", error=None)
            spec = dict(clip)
            clip_label = f"Clip {clip_number(job, clip)} of {len(job['clips'])}"
            store.save(job)
        try:
            output_dir = store.root / jid / cid
            output_dir.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
                work = Path(temporary)
                with store.lock:
                    branding, logo = BrandingSettings.read(store.root), read_logo(store.root)
                update(jid, generation, stage=f"{clip_label} · Downloading")
                upload = upload_path(job)
                editing_mode = spec.get("editing_mode", "full")
                professional = editing_mode != "full"
                source = (local_section(upload, work, spec["start"], spec["end"], professional=professional) if upload else
                          download_section(job["url"], work, spec["start"], spec["end"]))
                duration = min(video_duration(source), spec["end"] - spec["start"])
                subtitles, captions = None, []
                caption_language = None
                fps = (source_frame_rate(source) or "30/1") if professional else None
                director = DirectorOptions.from_dict(spec.get("director") or LEGACY_DIRECTOR)
                podcast = is_podcast(job)
                choice = PodcastSpec.model_validate(spec.get("podcast") or {}) if podcast else None
                analysis = spec.get("podcast_analysis") or {}
                if podcast:
                    director.preset, director.sfx = "podcast", False
                    if not professional:
                        director.story = False
                    director.music = "on" if director.music == "on" else "off"
                    director.hook_text = choice.hook_text or director.hook_text
                    director.commentary = ((choice.commentary or analysis.get("commentary") or "").strip() or None
                                           if choice.version == "commentary" and editing_mode != "natural" else None)
                caption_config = spec.get("caption_style")
                if editing_mode == "natural":
                    caption_config = {"preset": "classic", "emphasis": False, **(caption_config or {})}
                style = CaptionStyle.from_dict(caption_config)
                transcript = None
                if spec["captions"]:
                    update(jid, generation, stage=f"{clip_label} · Transcribing")
                    captions = transcribe(source, language=spec.get("language"), clip_end=duration)
                    caption_language = getattr(captions, "language", None)
                    if choice and choice.caption_edits:
                        transcript = deepcopy(captions)
                        apply_caption_edits(captions, [e.model_dump() for e in choice.caption_edits])
                elif director.enabled:
                    update(jid, generation, stage=f"{clip_label} · Transcribing")
                    try:
                        captions = transcribe(source, language=spec.get("language"), clip_end=duration)
                    except Exception:
                        captions = []
                        logger.debug("Transcription for AI Director unavailable", exc_info=True)
                transcript = captions if transcript is None else transcript
                plan, directed, director_result = None, None, {"status": "disabled"}
                adaptive_style = None
                peak = thumbnail_peak(spec, duration)
                hook = "" if professional else thumbnail_hook(job, spec, peak, captions)
                tracking = TrackingSpec.model_validate(spec.get("tracking") or {})
                facecam_options = (FacecamSpec(enabled=False) if podcast else
                                   FacecamSpec.model_validate(spec.get("facecam", {})))
                conversation = None
                camera_box, reaction, levels = None, None, None
                tracking_result = {"status": "disabled"}
                tracking_still = work / "tracking.jpg"
                if facecam_options.box:
                    camera_box = SourceBox(**facecam_options.box.model_dump())
                elif facecam_options.enabled:
                    try:
                        camera_box = detect_facecam(source, 0, duration)
                    except Exception:
                        logger.debug("Facecam detection failed", exc_info=True)
                if podcast and tracking.enabled:
                    update(jid, generation, stage=f"{clip_label} · Finding the active speaker")
                    turns = [[a - spec["start"], b - spec["start"], who] for a, b, who in analysis.get("turns", [])
                             if b > spec["start"] and a < spec["end"]]
                    try:
                        conversation = frame_conversation(source, duration, turns, choice.layout)
                        tracking_result = {"status": "unavailable"} if conversation is None else tracking_result
                    except Exception:
                        conversation, tracking_result = None, {"status": "unavailable"}
                        logger.warning("Speaker framing unavailable for %s/%s", jid, cid, exc_info=True)
                    if conversation and conversation["mode"] == "blur":
                        spec["mode"] = "blur"
                elif tracking.enabled:
                    update(jid, generation, stage=f"{clip_label} · Finding what the streamer reacts to")
                    try:
                        levels = loudness(source, duration)
                        reaction = analyze_reaction(source, duration, peak, camera_box,
                                                    ([i * STEP for i in range(len(levels))], list(levels)))
                        tracking_result = {"status": "unavailable"} if reaction is None else tracking_result
                    except Exception:
                        reaction, tracking_result = None, {"status": "unavailable"}
                        logger.warning("Reaction tracking unavailable for %s/%s", jid, cid, exc_info=True)
                trigger = reaction.trigger if reaction is not None else None
                if director.enabled:
                    update(jid, generation, stage=f"{clip_label} · AI Director planning edits")
                    try:
                        if professional:
                            context = story_context(job, spec, caption_language or getattr(captions, "language", None),
                                                    work, store.root / "voices")
                            voice_opt_in = director.voiceover or bool(director.commentary)
                            director.voiceover = voice_opt_in
                            teaser = analysis.get("teaser") or {}
                            plan, adaptive_style = plan_source_edit(
                                levels if levels is not None else loudness(source, duration), transcript, duration,
                                director, mode=editing_mode, category=spec.get("category"), peak=peak,
                                trigger=trigger["trigger"] if trigger else None, spike_score=spec.get("score"),
                                quote_time=teaser["start"] - spec["start"] if "start" in teaser else None,
                                title_voice=lambda text, **limits: context["voice"](
                                    text, context["language"], voice=director.voice, delivery=director.delivery, **limits))
                            if len(plan.segments) > 1:
                                directed = cut_source(source, work / "directed.mp4", plan, fps=fps, professional=True)
                        elif podcast:
                            context = story_context(job, spec, caption_language, work, store.root / "voices")
                            teaser = analysis.get("teaser") or {}
                            hook_span = ({"start": teaser["start"] - spec["start"], "end": teaser["end"] - spec["start"],
                                          "text": spec.get("thumbnail_text") or spec.get("title")} if teaser else None)
                            plan = plan_podcast_edit(levels if levels is not None else loudness(source, duration), captions,
                                                     duration, director, hook_span,
                                                     voice=lambda text: context["voice"](text, context["language"],
                                                                                         voice=director.voice,
                                                                                         delivery=director.delivery,
                                                                                         max_seconds=COMMENTARY_SECONDS))
                            directed = cut_source(source, work / "directed.mp4", plan)
                        else:
                            plan = plan_edit(levels if levels is not None else loudness(source, duration), captions,
                                             duration, director, peak,
                                             spec.get("score"), spec.get("thumbnail_text") or hook,
                                             story_context(stream_metadata(jid, generation, job), spec,
                                                           caption_language or getattr(captions, "language", None), work,
                                                           store.root / "voices"),
                                             trigger=trigger["trigger"] if trigger else None)
                            directed = cut_source(source, work / "directed.mp4", plan)
                    except Exception:
                        plan, directed, adaptive_style = None, None, None
                        director_result = {"status": "unavailable"}
                        logger.warning("AI Director unavailable for %s/%s; rendering unedited", jid, cid, exc_info=True)
                render_source = directed or source
                render_duration = plan.duration if plan else duration
                if professional and plan and plan.hook and not plan.commentary:
                    hook = plan.hook["text"]
                edited_captions = remap_captions(captions, plan) if plan and (spec["captions"] or director.loop) else captions
                shown = edited_captions if spec["captions"] else []
                text_height = None
                caption_limit = None
                video = work / "clip.mp4"
                overlay = None
                result = {"status": "disabled", "box": None}
                still = work / "source.jpg"
                try:
                    save_source_still(source, still, duration / 2)
                except Exception:
                    logger.debug("Could not save source still", exc_info=True)
                try:
                    options = facecam_options
                    if options.enabled:
                        box = camera_box
                        result = {"status": "not_found", "box": None}
                        if box is not None:
                            if options.position == "auto":
                                overlay = auto_layout_facecam(render_source, 0, render_duration, box, spec["mode"], options.size)
                            else:
                                capture = cv2.VideoCapture(str(render_source))
                                try:
                                    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                                finally:
                                    capture.release()
                                overlay = layout_facecam(box, size, options.position, options.size)
                            rect = overlay.rect
                            caption_limit = max(1, rect.y-260-24 if rect.y > 24 else 1500-rect.y-rect.h-24)
                            text_height = caption_height(shown, style, caption_limit) if spec["captions"] else None
                            corner = ("top" if rect.y == 24 else "bottom") + ("-left" if rect.x == 24 else "-right")
                            result = {"status": "used", "box": vars(box), "position": corner,
                                      "size": rect.w / 1080, "origin": "manual" if options.box else "detected"}
                except Exception:
                    overlay = None
                    result = {"status": "unavailable", "box": None}
                    logger.debug("Facecam unavailable; rendering without overlay", exc_info=True)
                camera_keys = None
                if conversation is not None:
                    if conversation["overlay"] is not None:
                        overlay = conversation["overlay"]
                        rect = overlay.rect
                        caption_limit = max(1, 1500 - rect.y - rect.h - 24)
                        text_height = caption_height(shown, style, caption_limit) if spec["captions"] else None
                        result = {"status": "used", "box": vars(overlay.box), "position": "split",
                                  "size": 1.0, "origin": "speaker"}
                    camera_keys = conversation["path"] or None
                    tracking_result = {"status": "needs_review" if conversation["needs_review"] else "used",
                                       "mode": "podcast", "layout": conversation["layout"],
                                       "auto_layout": conversation["auto_layout"], "seats": conversation["seats"],
                                       "speaker_seats": conversation.get("speaker_seats"),
                                       "shots": conversation["shots"], "switches": conversation["switches"],
                                       "cutaways": conversation["cutaways"],
                                       "needs_review": conversation["needs_review"],
                                       "path": [[round(t, 2), round(x, 4)] for t, x, _ in (camera_keys or [])]}
                elif reaction is not None and spec["mode"] == "crop":
                    try:
                        size = reaction.footage.frames.shape[2], reaction.footage.frames.shape[1]
                        span = ((min(s.start for s in plan.segments), max(s.end for s in plan.segments))
                                if plan and plan.segments else (0.0, duration))
                        reframe = plan_reframe(reaction, span, layout_options(overlay, size), tracking.layout,
                                               tracking.follow, tracking.x,
                                               tracking.target.model_dump() if tracking.target else None)
                        if reframe["overlay"] is not overlay:
                            overlay = reframe["overlay"]
                            rect = overlay.rect
                            caption_limit = max(1, rect.y-260-24 if rect.y > 24 else 1500-rect.y-rect.h-24)
                            text_height = caption_height(shown, style, caption_limit) if spec["captions"] else None
                            result.update(size=round(rect.w / 1080, 3),
                                          position="split" if reframe["layout"] == "split" else result.get("position"))
                        camera_keys = reframe["path"]
                        tracking_result = {
                            "status": "needs_review" if reframe["needs_review"] else "used",
                            "layout": reframe["layout"], "auto_layout": reframe["auto_layout"],
                            "follow": tracking.follow, "attempts": reframe["attempts"],
                            "trigger": reframe["trigger"], "target": reframe["target"],
                            "validation": reframe["validation"], "needs_review": reframe["needs_review"],
                            "window": round(reframe["width"], 4),
                            "path": [[round(t, 2), round(x, 4)] for t, x, _ in camera_keys[::2]],
                            "manual_target": tracking.target.model_dump() if tracking.target else None}
                    except Exception:
                        camera_keys, tracking_result = None, {"status": "unavailable"}
                        logger.warning("Dynamic reframing unavailable for %s/%s", jid, cid, exc_info=True)
                elif reaction is not None:
                    tracking_result = {"status": "not_cropped", "trigger": trigger, "needs_review": False}
                if reaction is not None:
                    tracking_result["still_time"] = trigger["event"] if trigger else round(duration / 2, 2)
                    try:
                        save_source_still(source, tracking_still, tracking_result["still_time"])
                    except Exception:
                        logger.debug("Could not save tracking still", exc_info=True)
                fixes = []
                if plan and plan.hook and spec["captions"] and text_height is None:
                    text_height = caption_height(shown, style)
                if spec["captions"]:
                    style, note = framing_fix(style, overlay.rect if overlay else None, text_height)
                    if note:
                        fixes.append(note)
                if plan:
                    if camera_keys:
                        plan.camera = to_output_time(camera_keys, plan.segments)
                    elif spec["mode"] == "crop" and overlay is None:
                        plan.focal = [detect_focal_x(render_source, offset, offset + segment.length)
                                      for offset, segment in zip(plan.offsets(), plan.segments)]
                ass = None
                if spec["captions"]:
                    caption_options = {"max_height": caption_limit} if caption_limit is not None else {}
                    ass = to_ass(shown, style, **caption_options)
                layout = {"rect": overlay.rect if overlay else None, "style": style if spec["captions"] else None,
                          "caption_height": text_height}
                if plan and plan.hook and plan.hook.get("text"):
                    band, hook_height, overlap = hook_layout(plan.hook, layout["rect"], layout["style"], shown, caption_limit)
                    layout.update(hook=band,hook_caption_height=hook_height,hook_captions_overlap=overlap)
                    story = plan.hook.get("style") in ("story", "commentary")
                    ass = hook_title_ass(ass, plan.hook["text"], plan.hook["duration"] + (0 if story or professional else .7),
                                         top=band[0],story=story,highlight=plan.hook.get("streamer"),max_height=band[1]-band[0],
                                         static=professional)
                if ass is not None:
                    subtitles = work / "captions.ass"
                    subtitles.write_text(ass, encoding="utf-8")
                effects = ({"effects": effects_for(plan)} if plan else
                           {"effects": Effects(focal=keyframe_expression(camera_keys))} if camera_keys else {})
                if professional:
                    effects.update(export_fps=fps, silent_audio=not has_audio(render_source))
                update(jid, generation, stage=f"{clip_label} · Checking authenticity")
                authenticity = authenticity_report(job, spec, plan, transcript, shown, duration, style=adaptive_style,
                                                   hook=spec.get("thumbnail_text") or hook, tracking=tracking_result)

                def render_vertical(camera=None):
                    bands = (caption_band(style, style.size, text_height) if shown else None, layout.get("hook"))
                    watermark = prepare_watermark(work / "branding.png", branding, logo, render_source, 0,
                                                  render_duration, spec["mode"], facecam=camera, bands=bands)
                    options = {"facecam": camera} if camera is not None else {}
                    make_vertical_clip(render_source, video, 0, render_duration, spec["mode"], subtitles,
                                       watermark=watermark, **options, **effects)

                update(jid, generation, stage=f"{clip_label} · Rendering")
                if overlay is None:
                    render_vertical()
                else:
                    try:
                        render_vertical(overlay)
                    except Exception:
                        result = {"status": "unavailable", "box": None}
                        render_vertical()
                mixed_audio = False
                if plan and (not professional or plan.voiceover or plan.music or plan.sfx or plan.voice_clarity):
                    update(jid, generation, stage=f"{clip_label} · AI Director mixing audio")
                    mixed = work / "mixed.mp4"
                    try:
                        mix_audio(video, mixed, plan)
                        mixed.replace(video)
                        mixed_audio = True
                    except Exception:
                        fixes.append("Sound effects and music skipped (audio mix failed)")
                        logger.warning("AI Director audio mix failed for %s/%s", jid, cid, exc_info=True)
                    try:
                        normalize_loudness(video)
                    except Exception:
                        logger.warning("AI Director loudness correction failed for %s/%s", jid, cid, exc_info=True)
                if plan:
                    if director.loop:
                        update(jid, generation, stage=f"{clip_label} · AI Director smoothing the loop")
                        looped = work / "looped.mp4"
                        try:
                            plan.loop = make_seamless_loop(video, looped, shown, speech=edited_captions,
                                                          **({"professional": True} if professional else {}))
                            if plan.loop["status"] == "applied":
                                looped.replace(video)
                        except (OSError, ValueError, KeyError, StopIteration, subprocess.CalledProcessError):
                            plan.loop = {"status": "skipped", "reason": "Could not render loop"}
                            logger.warning("AI Director loop failed for %s/%s", jid, cid, exc_info=True)
                    update(jid, generation, stage=f"{clip_label} · AI Director quality check")
                    director_result = plan_record(plan, director, quality_check(
                        video, plan, shown, fixes, layout, tracking_result, export_fps=fps,
                        preserve_audio=professional and not (plan.voiceover or plan.music or plan.sfx or plan.voice_clarity)))
                    peak = plan.map_time(peak) if peak is not None else None
                record_operations(authenticity, plan, mixed=mixed_audio)
                update(jid, generation, stage=f"{clip_label} · Creating thumbnail")
                thumbnail = work / "thumbnail.jpg"
                try:
                    generate_thumbnail(video, thumbnail, spec.get("thumbnail_text") or hook, peak)
                except (ImageProcessingError, OSError, ValueError):
                    logger.warning("Could not style thumbnail for %s/%s; using the first frame", jid, cid)
                    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-frames:v", "1",
                                    "-vf", "scale=270:480", str(thumbnail)], check=True)
                with store.lock:
                    job = active_job(jid, generation)
                    clip = find_clip(job, cid)
                    video.replace(output_dir / "clip.mp4")
                    thumbnail.replace(output_dir / "thumbnail.jpg")
                    base = f"/api/jobs/{jid}/clips/{cid}"
                    if still.is_file():
                        still.replace(output_dir / "source.jpg")
                        clip["source_still_url"] = base + "/source-still"
                    else:
                        clip.pop("source_still_url", None)
                    clip["facecam_result"] = result
                    if tracking_still.is_file():
                        tracking_still.replace(output_dir / "tracking.jpg")
                        clip["tracking_still_url"] = base + "/tracking-still"
                    else:
                        clip.pop("tracking_still_url", None)
                    clip["tracking_result"] = tracking_result
                    clip["needs_review"] = bool(tracking_result.get("needs_review"))
                    clip["director_result"] = director_result
                    clip["authenticity"] = authenticity
                    revision = str(time.time_ns())
                    clip.update(status="ready", rendered_at=int(revision), file_url=base + "/video",
                                download_url=base + "/download", thumbnail_url=base + "/thumbnail",
                                thumbnail_version=revision, thumbnail_hook=hook, caption_language=caption_language)
                    clip.pop("previous_render", None)
                    store.save(job)
        except Exception:
            raise

    def render(jid, cid, generation):
        for attempt in range(1, 4):
            with store.lock:
                job = active_job(jid, generation)
                clip = next((c for c in job["clips"] if c["id"] == cid), None)
                if clip is None:
                    return
                if clip["status"] not in ("queued", "rendering"):
                    return
                clip.update(attempt=attempt, status="rendering", error=None)
                job.update(status="running")
                store.save(job)
            try:
                render_attempt(jid, cid, generation)
                return
            except (WorkCancelled, YouTubeAccessError):
                raise
            except Exception as error:
                with store.lock:
                    job = active_job(jid, generation)
                    clip = next((c for c in job["clips"] if c["id"] == cid), None)
                    if clip is None:
                        return
                    clip.update(status="failed" if attempt == 3 else "rendering",
                                              error=str(error))
                    job["stage"] = "Clip failed" if attempt == 3 else f"Retrying clip in {attempt * 2}s (attempt {attempt + 1}/3)"
                    store.save(job)
                if attempt < 3:
                    time.sleep(attempt * 2)

    def finish(jid, generation):
        with store.lock:
            job = active_job(jid, generation)
            if not job.get("analysis_complete"):
                return
            total = len(job["clips"])
            done = sum(c["status"] in ("ready", "failed", "cancelled") for c in job["clips"])
            job["progress"] = 0.15 + 0.85 * done / total if total else 1
            if done < total:
                store.save(job)
                return
            failed = any(c["status"] == "failed" for c in job["clips"])
            job.update(status="completed_with_errors" if failed else "completed", progress=1,
                       stage="Finished" if job["clips"] else (f"{len(job['moments'])} moments ready for review"
                             if job["moments"] else "No viral moments found"),
                       elapsed_seconds=round(time.time() - job.get("processing_started", time.time()), 2))
            store.save(job)

    def fail_job(jid, generation, error):
        if isinstance(error, (ChatReplayError, YouTubeAccessError)):
            message = str(error)
        else:
            logger.exception("Job %s failed unexpectedly", jid)
            message = "Studio couldn’t process this replay. Try submitting it again."
        try:
            with store.lock:
                job = active_job(jid, generation)
                job.update(status="failed", stage="Failed", error=message, generation=generation + 1,
                           chat_retrying=False)
                for clip in job["clips"]:
                    if clip["status"] in ("queued", "rendering"):
                        clip.update(status="failed", error=message)
                store.save(job)
                pending.cancel(jid)
                metadata_cache.pop((jid, generation), None)
        except WorkCancelled:
            pass
        except Exception:
            logger.exception("Could not save failure for job %s", jid)

    def preflight(jid, generation):
        try:
            update(jid, generation, status="running", stage="Checking stream metadata", processing_started=time.time())
            with store.lock:
                job = active_job(jid, generation)
            if job.get("upload"):
                path = upload_path(job)
                info = {"id": job["upload"], "title": Path(job.get("filename") or "Uploaded video").stem,
                        "duration": probe_source(path)["duration"], "thumbnail": None, "channel": None,
                        "uploader": None}
            else:
                info = fetch_metadata(job["url"], False, False)
            with store.lock:
                job = active_job(jid, generation)
                metadata_cache[jid, generation] = info
                job.update(status="queued", stage="Queued for analysis",
                           metadata={key: info.get(key) for key in METADATA_KEYS})
                store.save(job)
                pending.put((jid, None, generation))
        except WorkCancelled:
            pass
        except Exception as error:
            fail_job(jid, generation, error)

    def moment_clip(job, moment, number):
        clip = {k: moment[k] for k in ("start", "end", "peak", "title", "description", "category",
                                       "sources", "score", "message_sample")}
        clip.update({k: job["request"].get(k, default) for k, default in
                     (("mode", "crop"), ("captions", True), ("facecam", FacecamSpec().model_dump()),
                      ("tracking", TrackingSpec().model_dump()),
                      ("language", "auto"), ("caption_style", None), ("director", dict(LEGACY_DIRECTOR)),
                      ("editing_mode", "full"))})
        if is_podcast(job):
            choice = {**(job["request"].get("podcast") or {}), **(moment.get("edits") or {})}
            clip.update(podcast=PodcastSpec.model_validate(choice).model_dump(),
                        podcast_analysis=moment.get("podcast") or {})
        clip.update(id=uuid.uuid4().hex, moment_id=moment["id"], number=number, status="queued", error=None,
                    file_url=None, download_url=None, thumbnail_url=None)
        moment.update(status="rendered", clip_id=clip["id"])
        return clip

    def check_job(jid, generation):
        with store.lock:
            active_job(jid, generation)

    def analyze_chat(jid, generation, job, info, retry_dir=None):
        started = time.monotonic()
        try:
            directory = retry_dir if retry_dir is not None else store.root / jid / "chat"
            metadata, messages = fetch_chat(job["url"], directory, info,
                                            lambda: check_job(jid, generation))
            count, lead = job["request"]["clips"], job["request"]["lead"]
            spikes, activity = detect_spikes(messages, metadata["duration"], clips=count, lead=lead)
            moments = [chat_moment(s) for s in sorted(spikes, key=lambda s: -s.get("score", 0))[:count]]
            error = getattr(messages, "warning", None)
            fields = dict(metadata=metadata, activity=activity, message_count=len(messages))
        except WorkCancelled:
            raise
        except Exception as failure:
            error = str(failure) if isinstance(failure, ChatReplayError) else "Chat analysis was unavailable."
            if not isinstance(failure, ChatReplayError):
                logger.warning("Chat analysis failed for job %s", jid, exc_info=True)
            moments, fields = [], {}
        fields.update(chat_error=error, chat_warning=CHAT_FALLBACK_WARNING if error else None,
                      chat_seconds=round(time.monotonic() - started, 2))
        if retry_dir is None:
            update(jid, generation, **fields)
        return moments, fields

    def analyze_ai(jid, generation, job, duration, podcast, upload):
        count, lead = job["request"]["clips"], job["request"]["lead"]
        directory = store.root / jid / "analysis"
        try:
            update(jid, generation, stage="Preparing the source for AI analysis" if upload
                   else "Downloading audio for AI analysis", progress=.15)
            audio, video = (local_analysis_media(upload, directory) if upload else
                            download_analysis_media(job["url"], directory,
                                                    should_continue=lambda: check_job(jid, generation)))
            stage = "AI understanding the conversation" if podcast else "AI analyzing video, audio & speech"
            update(jid, generation, stage=stage)
            language = job["request"].get("language")
            language = None if language == "auto" else language

            def progress(fraction, text):
                update(jid, generation, stage=stage,
                       progress=.2 + .65 * max(0, min(1, fraction)), analysis_detail=text)

            if podcast:
                return analyze_podcast(audio, video, duration, clips=count, language=language, progress=progress,
                                       speakers=(job["request"].get("podcast") or {}).get("speakers"))[:count], None
            return analyze_media(audio, video, duration, clips=count, lead=lead,
                                 language=language, progress=progress)[:count], None
        except WorkCancelled:
            raise
        except Exception as error:
            update(jid, generation, ai_error=str(error) if isinstance(error, YouTubeAccessError)
                   else "AI analysis was unavailable. Try this replay again.")
            return [], error
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def retry_chat(jid, generation, job):
        directory = store.root / jid / "chat"
        directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory, prefix="retry-") as temporary:
            candidate = Path(temporary)
            chat, fields = analyze_chat(jid, generation, job, None, retry_dir=candidate)
            with store.lock:
                job = active_job(jid, generation)
                if job.get("message_count"):
                    old_count, new_count = job["message_count"], fields.get("message_count", 0)
                    previous_bins = {point["time"] for point in job["activity"] if point["count"]}
                    candidate_bins = {point["time"] for point in fields.get("activity", []) if point["count"]}
                    if (new_count < old_count or (fields["chat_error"] and
                            (not job.get("chat_error") or not previous_bins <= candidate_bins))):
                        logger.info("Retaining prior chat replay for %s: retry covered fewer messages or intervals", jid)
                        job["chat_retrying"] = False
                        store.save(job)
                        return
                if "activity" in fields:
                    duration = job["metadata"]["duration"]
                    job["moments"] = add_chat_moments(job["moments"], chat, duration, job["request"]["lead"],
                                                      ai=job.get("ai_candidates"))
                    buckets = candidate / "buckets.sqlite"
                    if buckets.is_file():
                        buckets.replace(directory / buckets.name)
                    if (chat or not fields["chat_error"]) and job["status"] == "failed" and not job.get("analysis_complete"):
                        job.update(analysis_complete=True, status="completed", progress=1, error=None,
                                   stage=f"{len(job['moments'])} moments ready for review" if job["moments"]
                                   else "No viral moments found")
                job.update(fields, chat_retrying=False)
                store.save(job)

    def analyze(jid, generation):
        try:
            with store.lock:
                job = active_job(jid, generation)
                retrying = job.get("chat_retrying", False)
                info = None if retrying else metadata_cache.pop((jid, generation))
            if retrying:
                try:
                    retry_chat(jid, generation, job)
                except WorkCancelled:
                    raise
                except Exception:
                    logger.warning("Could not save chat retry for job %s", jid, exc_info=True)
                    update(jid, generation, chat_retrying=False, chat_error="Could not save the chat retry.")
                return
            podcast, upload = is_podcast(job), upload_path(job)
            count, lead = job["request"]["clips"], job["request"]["lead"]
            duration = info["duration"]
            chat, ai, fields, ai_error = [], [], {}, None
            update(jid, generation, status="running", stage="Fetching chat replay" if not (podcast or upload)
                   else "Preparing AI analysis")
            if podcast or upload:
                ai, ai_error = analyze_ai(jid, generation, job, duration, podcast, upload)
            elif job["request"].get("ai_analysis", False):
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="chat-replay") as worker:
                    chat_task = worker.submit(analyze_chat, jid, generation, job, info)
                    ai, ai_error = analyze_ai(jid, generation, job, duration, podcast, upload)
                    chat, fields = chat_task.result()
            else:
                chat, fields = analyze_chat(jid, generation, job, info)
                if fields["chat_error"]:
                    ai, ai_error = analyze_ai(jid, generation, job, duration, podcast, upload)
            if ai_error and (podcast or upload or (fields.get("chat_error") and not chat)):
                raise ai_error
            duration = fields.get("metadata", {}).get("duration", duration)
            moments = merge_moments(ai, chat, duration, lead)
            with store.lock:
                job = active_job(jid, generation)
                clips = [] if job["request"].get("review", False) else [
                    moment_clip(job, m, i) for i, m in enumerate(moments[:count], 1)]
                job.update(clips=clips, moments=moments, ai_candidates=ai, analysis_complete=True,
                           stage="Queued for rendering")
                store.save(job)
                for clip in clips:
                    pending.put((jid, clip["id"], generation))
            finish(jid, generation)
        except WorkCancelled:
            pass
        except Exception as error:
            fail_job(jid, generation, error)

    def render_job(jid, cid, generation):
        try:
            render(jid, cid, generation)
            finish(jid, generation)
        except WorkCancelled:
            pass
        except Exception as error:
            fail_job(jid, generation, error)

    pending = Scheduler(analyze, render_job,
                        analysis_workers=analysis_workers if analysis_workers is not None else int(os.environ.get("CLIPPER_ANALYSIS_WORKERS", "3")),
                        render_workers=render_workers if render_workers is not None else int(os.environ.get("CLIPPER_RENDER_WORKERS", "2")),
                        preflight=preflight,
                        metadata_workers=metadata_workers if metadata_workers is not None else int(os.environ.get("CLIPPER_METADATA_WORKERS", "3")))

    def queued_job(job):
        job.pop("ai_candidates", None)
        analyses, renders = pending.positions()
        for clip in job["clips"]:
            clip["queue_position"] = renders.get((job["id"], clip["id"])) if clip["status"] == "queued" else None
        position = None if job.get("chat_retrying") else analyses.get(job["id"])
        waiting = [clip["queue_position"] for clip in job["clips"] if clip["queue_position"] is not None]
        if position is None and waiting and not any(c["status"] == "rendering" for c in job["clips"]):
            position = min(waiting)
        job["queue_position"] = position
        if position is not None and job["status"] in ACTIVE_STATUSES:
            job.update(status="queued", stage=f"Queued (position {position})")
        return job

    @asynccontextmanager
    async def lifespan(app):
        with store.lock:
            for job in store.jobs():
                if any("number" not in clip for clip in job["clips"]):
                    for number, clip in enumerate(job["clips"], 1):
                        clip.setdefault("number", number)
                    store.save(job)
                if job.get("chat_retrying"):
                    pending.put((job["id"], None, job.get("generation", 0)))
                if job["status"] in ACTIVE_STATUSES:
                    generation = job.get("generation", 0)
                    job.update(status="queued", stage="Resuming after restart", error=None,
                               analysis_complete=job.get("analysis_complete", bool(job["clips"])))
                    if job["analysis_complete"]:
                        for clip in job["clips"]:
                            if clip["status"] in ("queued", "rendering"):
                                clip.update(status="queued", error=None)
                                pending.put((job["id"], clip["id"], generation), priority=clip.get("render_priority", False))
                    else:
                        pending.put((job["id"], None, generation), preflight=True)
                    store.save(job)
                    if job["analysis_complete"]:
                        finish(job["id"], generation)
        pending.start()
        yield
        pending.close()

    app = FastAPI(title="Clipper Studio", lifespan=lifespan)
    app.state.store = store
    app.state.pending = pending

    @app.get("/api/branding")
    def get_branding():
        with store.lock:
            return {**BrandingSettings.read(store.root).model_dump(),
                    "has_logo": (store.root / "branding-logo.png").is_file()}

    @app.put("/api/branding")
    def put_branding(settings: BrandingSettings):
        with store.lock:
            store.save_branding("branding.json", settings.model_dump_json().encode())
            return get_branding()

    @app.get("/api/branding/logo")
    def get_branding_logo():
        logo = read_logo(store.root)
        if logo is None:
            raise HTTPException(404, "No branding logo has been uploaded.")
        return Response(logo, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.put("/api/branding/logo")
    async def upload_branding_logo(request: Request):
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > LOGO_LIMIT:
                raise HTTPException(413, "Use a PNG file no larger than 2 MB.")
            data.extend(chunk)
        try:
            logo = await run_in_threadpool(normalize_logo, data)
        except (OSError, ValueError, Image.DecompressionBombError):
            raise HTTPException(422, "Upload a visible, static PNG up to 2048 × 2048 pixels.") from None
        await run_in_threadpool(store.save_branding, "branding-logo.png", logo)
        return {"has_logo": True}

    @app.delete("/api/branding/logo")
    def remove_branding_logo():
        with store.lock:
            (store.root / "branding-logo.png").unlink(missing_ok=True)
        return {"has_logo": False}

    @app.get("/api/voices")
    def list_voices():
        return {"voices": [{"id": key, "name": p.name, "description": p.description} for key, p in PERSONAS.items()],
                "deliveries": list(MOODS)}

    @app.post("/api/voices/preview")
    def preview_voice(request: VoicePreview):
        text = " ".join((request.text or SAMPLES[request.language]).split())
        key = hashlib.sha1(json.dumps([text, request.voice, request.delivery, request.language, request.category]).encode())
        folder = store.root / "voice-previews"
        path = folder / f"{key.hexdigest()[:20]}.wav"
        info = path.with_suffix(".json")
        if not (path.is_file() and info.is_file()):
            folder.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.stem + f".{uuid.uuid4().hex[:6]}.wav")
            try:
                spoken = narrate(text, request.language, partial, store.root / "voices", request.voice,
                                 request.category, "preview", request.delivery)
            except Exception as error:
                partial.unlink(missing_ok=True)
                logger.warning("Voice preview failed", exc_info=True)
                raise HTTPException(503, "Voice preview is unavailable right now") from error
            partial.replace(path)
            info.write_text(json.dumps(spoken))
            previews = sorted(folder.glob("*.wav"), key=lambda p: p.stat().st_mtime)
            for stale in previews[:-PREVIEW_CACHE]:
                stale.unlink(missing_ok=True)
                stale.with_suffix(".json").unlink(missing_ok=True)
        spoken = json.loads(info.read_text())
        return FileResponse(path, media_type="audio/wav", headers={
            "X-Voice-Name": spoken["voice_name"], "X-Voice-Mood": spoken["mood"], "X-Voice-Engine": spoken["engine"],
            "Cache-Control": "no-store"})

    @app.post("/api/uploads", status_code=201)
    async def upload_source(request: Request):
        """Raw-body upload of an authorized source file (X-Filename header names it)."""
        if not request.headers.get("x-filename"):
            raise HTTPException(400, "Send the file name in the X-Filename header.")
        filename = Path(unquote(request.headers["x-filename"])).name[:120]
        suffix = Path(filename).suffix.lower()
        if suffix not in UPLOAD_SUFFIXES:
            raise HTTPException(400, "Upload an MP4, MOV, M4V, MKV or WebM video.")
        limit = int(os.environ.get("CLIPPER_UPLOAD_MAX_BYTES", str(8 * 1024 ** 3)))
        uid = uuid.uuid4().hex
        folder = store.root / "uploads" / uid
        folder.mkdir(parents=True)
        path, size = folder / f"source{suffix}", 0
        try:
            with path.open("wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"Uploads are limited to {limit:,} bytes.")
                    handle.write(chunk)
            if not size:
                raise HTTPException(400, "The upload was empty.")
            try:
                info = await run_in_threadpool(probe_source, path)
            except ValueError as error:
                raise HTTPException(400, str(error)) from None
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        record = {"id": uid, "filename": filename, "duration": info["duration"], "bytes": size}
        (folder / "upload.json").write_text(json.dumps(record))
        return record

    @app.get("/api/jobs/{jid}/source")
    def source_preview(jid: str):
        job = store.get(jid)
        try:
            path = upload_path(job)
        except FileNotFoundError:
            path = None
        if path is None:
            raise HTTPException(404, "No uploaded source for this job")
        return FileResponse(path)

    @app.post("/api/jobs/{jid}/rights")
    def confirm_rights(jid: str, request: RightsRequest):
        with store.lock:
            job = store.get(jid)
            job["rights"] = {**request.model_dump(), "confirmed_at": datetime.now(timezone.utc).isoformat()}
            store.save(job)
        return job["rights"]

    @app.get("/api/jobs/{jid}/clips/{cid}/description")
    def clip_description(jid: str, cid: str):
        job = store.get(jid)
        return {"text": export_description(job, find_clip(job, cid))}

    @app.post("/api/jobs", status_code=202)
    def create_job(request: JobRequest):
        jid = uuid.uuid4().hex
        filename = None
        if request.upload:
            info = store.root / "uploads" / request.upload / "upload.json"
            if not info.is_file():
                raise HTTPException(400, "Upload the source file again; it was not found.")
            filename = json.loads(info.read_text()).get("filename")
        job = dict(id=jid, created_at=datetime.now(timezone.utc).isoformat(), url=request.url, request=request.model_dump(), status="queued", stage="Queued",
                   progress=0, metadata=None, activity=[], clips=[], error=None, generation=0, analysis_complete=False,
                   upload=request.upload, filename=filename, rights=None)
        job["request"]["director"] = director_config(request.editing_mode, request.director)
        with store.lock:
            store.save(job)
            pending.put((jid, None, 0), preflight=True)
        return {"id": jid}

    @app.get("/api/jobs")
    def list_jobs(summary: bool = False):
        with store.lock:
            jobs = [queued_job(job) for job in store.jobs()]
        if not summary:
            return jobs
        return [dict(id=job["id"], url=job.get("url"), created_at=job.get("created_at"), status=job["status"],
                     metadata=job.get("metadata"), clip_count=len(job["clips"]),
                     queue_position=job["queue_position"]) for job in jobs]

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        with store.lock:
            return queued_job(store.get(jid))

    @app.post("/api/jobs/{jid}/chat/retry", status_code=202)
    def queue_chat_retry(jid: str):
        with store.lock:
            job = store.get(jid)
            if (job["status"] not in ("completed", "completed_with_errors", "failed") or
                    job.get("chat_retrying") or is_podcast(job) or not job.get("url") or not job.get("metadata")):
                raise HTTPException(409, "Chat analysis can only be retried for a finished YouTube job.")
            job["chat_retrying"] = True
            store.save(job)
            pending.put((jid, None, job.get("generation", 0)))
        return {"id": jid}

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        with store.lock:
            job = store.get(jid)
            if job["status"] not in ("queued", "running", "cancelled"):
                raise HTTPException(409, "This job has already finished")
            job.update(status="cancelled", stage="Cancelled. In-flight work stops at the next stage boundary.",
                       generation=job.get("generation", 0) + 1, chat_retrying=False)
            for clip in job["clips"]:
                if clip["status"] in ("queued", "rendering"):
                    previous = clip.pop("previous_render", None)
                    if previous is not None:
                        clip.update(previous, status="ready", error=None)
                    else:
                        clip.update(status="cancelled", error=None)
            store.save(job)
            pending.cancel(jid)
            for key in list(metadata_cache):
                if key[0] == jid:
                    metadata_cache.pop(key)
        return {"id": jid, "status": "cancelled"}

    @app.patch("/api/jobs/{jid}/clips/{cid}")
    def keep_clip(jid: str, cid: str, request: KeepRequest):
        with store.lock:
            job = store.get(jid)
            clip = find_clip(job, cid)
            clip["kept"] = request.kept
            store.save(job)
        return {"id": cid, "kept": request.kept}

    @app.get("/api/jobs/{jid}/download-kept")
    def download_kept(jid: str):
        with store.lock:
            job = store.get(jid)
            require_rights(job)
            clips = [c for c in job["clips"] if c.get("kept") and c["status"] == "ready"]
            if not clips:
                raise HTTPException(409, "Keep at least one ready clip first")
            archive_dir = Path(tempfile.mkdtemp(dir=store.root))
            archive = archive_dir / "kept-clips.zip"
            try:
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                    for clip in clips:
                        directory = store.root / jid / clip["id"]
                        stem = f"highlight-{clip_number(job, clip):02d}"
                        bundle.write(directory / "clip.mp4", stem + ".mp4")
                        thumbnail = directory / "thumbnail.jpg"
                        if thumbnail.is_file():
                            bundle.write(thumbnail, stem + ".jpg")
            except Exception:
                shutil.rmtree(archive_dir)
                raise
        return FileResponse(archive, media_type="application/zip", filename=f"{title_slug(job)}-kept-clips.zip",
                            background=BackgroundTask(shutil.rmtree, archive_dir))

    def find_moment(job, mid):
        moment = next((m for m in job["moments"] if m["id"] == mid), None)
        if moment is None:
            raise HTTPException(404, "Moment not found")
        return moment

    @app.patch("/api/jobs/{jid}/moments/{mid}")
    def patch_moment(jid: str, mid: str, request: MomentPatch):
        with store.lock:
            job = store.get(jid)
            moment = find_moment(job, mid)
            if moment["status"] == "rendered" and request.status is not None:
                raise HTTPException(409, "Delete the rendered clip before changing this moment's status")
            start = request.start if request.start is not None else moment["start"]
            end = request.end if request.end is not None else moment["end"]
            if request.start is not None or request.end is not None:
                if not 0 <= start < end <= job["metadata"]["duration"] or end - start > 60 + 1e-6:
                    raise HTTPException(422, "Moment must be within the video and at most 60 seconds long")
                if moment.get("sources") == ["ai"]:
                    moment.setdefault("original_ai_range", {k: moment[k] for k in ("start", "end")})
                moment.update(start=start, end=end, duration=end - start)
            if request.status is not None:
                moment["status"] = request.status
            if request.podcast is not None:
                if not is_podcast(job):
                    raise HTTPException(422, "Podcast options apply to podcast jobs only")
                moment["edits"] = request.podcast.model_dump()
            if request.start is not None or request.end is not None or request.status is not None:
                moment["reviewed"] = True
            store.save(job)
        return moment

    @app.post("/api/jobs/{jid}/moments/render", status_code=202)
    def render_moments(jid: str, request: MomentRender = MomentRender()):
        with store.lock:
            job = store.get(jid)
            if request.ids is not None:
                for mid in request.ids:
                    find_moment(job, mid)
            moments = [m for m in job["moments"] if m["status"] != "rendered" and
                       (m["id"] in request.ids if request.ids is not None else m["status"] == "selected")]
            if not moments:
                raise HTTPException(409, "No moments to render")
            number = max((clip_number(job, c) for c in job["clips"]), default=0)
            clips = [moment_clip(job, m, number + i) for i, m in enumerate(moments, 1)]
            for clip in clips:
                clip["render_priority"] = True
            job["clips"].extend(clips)
            job.update(status="queued", stage="Queued for rendering", progress=0, error=None,
                       analysis_complete=True, processing_started=time.time())
            store.save(job)
            for clip in clips:
                pending.put((jid, clip["id"], job.get("generation", 0)), priority=True)
        return {"clips": [c["id"] for c in clips]}

    @app.post("/api/jobs/{jid}/clips/{cid}/render", status_code=202)
    def rerender(jid: str, cid: str, request: RenderRequest):
        with store.lock:
            job = store.get(jid)
            clip = find_clip(job, cid)
            if clip["status"] in ("rendering", "queued"):
                raise HTTPException(409, "Wait for this highlight to finish rendering")
            if request.end <= request.start or request.end - request.start > 60 or request.end > job["metadata"]["duration"]:
                raise HTTPException(422, "Clip must be within the video and between 0 and 60 seconds long")
            if clip["status"] == "ready":
                clip["previous_render"] = {key: clip[key] for key in (
                    "start", "end", "mode", "captions", "file_url", "download_url", "thumbnail_url",
                    "rendered_at", "attempt", "facecam", "facecam_result", "source_still_url", "director",
                    "director_result", "tracking", "tracking_result", "tracking_still_url", "needs_review", "podcast") if key in clip}
                clip["previous_render"].update({key: clip.get(key) for key in ("language", "caption_style", "caption_language",
                                                                             "authenticity")})
                clip["previous_render"]["editing_mode"] = clip.get("editing_mode", "full")
            changes = request.model_dump(exclude={"podcast"})
            current_mode = clip.get("editing_mode", "full")
            editing_mode = request.editing_mode if "editing_mode" in request.model_fields_set else current_mode
            changes["editing_mode"] = editing_mode
            if editing_mode != "full" and editing_mode == current_mode and "director" not in request.model_fields_set:
                changes["director"] = clip["director"]
            else:
                changes["director"] = director_config(editing_mode, request.director)
            if request.podcast is not None and is_podcast(job):
                changes["podcast"] = request.podcast.model_dump()
            clip.update(changes, status="queued", error=None, file_url=None,
                        download_url=None, thumbnail_url=None, render_priority=True)
            clip.pop("authenticity", None)
            if job["status"] not in ACTIVE_STATUSES:
                job.update(status="queued", stage="Queued re-render", progress=0, error=None,
                           processing_started=time.time())
            job["analysis_complete"] = True
            store.save(job)
            pending.put((jid, cid, job.get("generation", 0)), priority=True)
        return {"id": cid, "status": "queued"}

    def file(jid, cid, filename, download=False):
        with store.lock:
            job = store.get(jid)
            clip = find_clip(job, cid)
            if clip["status"] != "ready":
                raise HTTPException(409, "Clip is not ready")
            path = store.root / jid / cid / filename
            if not path.is_file():
                raise HTTPException(404, "File not found")
        extension = path.suffix
        return FileResponse(path, media_type="image/jpeg" if extension == ".jpg" else "video/mp4",
                            filename=f"{title_slug(job)}-highlight-{clip_number(job, clip):02d}{extension}" if download else None,
                            content_disposition_type="attachment" if download else "inline")

    @app.get("/api/jobs/{jid}/clips/{cid}/video")
    def video(jid: str, cid: str):
        return file(jid, cid, "clip.mp4")

    @app.get("/api/jobs/{jid}/clips/{cid}/download")
    def download(jid: str, cid: str):
        require_rights(store.get(jid))
        return file(jid, cid, "clip.mp4", True)

    @app.get("/api/jobs/{jid}/clips/{cid}/thumbnail")
    def thumbnail(jid: str, cid: str, download: bool = False):
        if download:
            require_rights(store.get(jid))
        return file(jid, cid, "thumbnail.jpg", download)

    @app.post("/api/jobs/{jid}/clips/{cid}/thumbnail")
    def regenerate_thumbnail(jid: str, cid: str, request: ThumbnailRequest | None = None):
        custom = ' '.join(((request and request.text) or "").upper().split())[:TEXT_LIMIT] or None
        with tempfile.TemporaryDirectory(dir=store.root, prefix="thumbnail-") as temporary:
            work = Path(temporary)
            video = work / "clip.mp4"
            thumbnail = work / "thumbnail.jpg"
            with store.lock:
                job = store.get(jid)
                clip = find_clip(job, cid)
                if clip["status"] != "ready":
                    raise HTTPException(409, "Clip is not ready")
                directory = store.root / jid / cid
                if not (directory / "clip.mp4").is_file():
                    raise HTTPException(404, "File not found")
                os.link(directory / "clip.mp4", video)
                rendered_at = clip.get("rendered_at")
                version = clip.get("thumbnail_version")
                source_peak = thumbnail_peak(clip, clip["end"] - clip["start"])
                hook = clip.get("thumbnail_hook", "")
                if clip.get("editing_mode", "full") == "full" and not hook:
                    hook = thumbnail_hook(job, clip, source_peak)
                peak = rendered_offset(clip, source_peak)
            try:
                generate_thumbnail(video, thumbnail, custom or hook, peak)
            except (ImageProcessingError, OSError, ValueError) as error:
                logger.warning("Could not regenerate thumbnail for %s/%s: %s", jid, cid, error)
                raise HTTPException(500, "Could not generate a thumbnail for this clip") from None
            with store.lock:
                job = store.get(jid)
                clip = find_clip(job, cid)
                if (clip["status"], clip.get("rendered_at"), clip.get("thumbnail_version")) != ("ready", rendered_at, version):
                    raise HTTPException(409, "This highlight changed. Try again.")
                thumbnail.replace(directory / "thumbnail.jpg")
                clip.update(thumbnail_url=f"/api/jobs/{jid}/clips/{cid}/thumbnail", thumbnail_text=custom,
                            thumbnail_hook=hook, thumbnail_version=str(time.time_ns()))
                report = clip.get("authenticity")
                if report:
                    report["checks"]["thumbnail"] = (
                        {"status": "review", "detail": "Thumbnail text changed after rendering; check it against the source."}
                        if custom or hook else {"status": "skip", "detail": "No text to check."})
                store.save(job)
                return {key: clip[key] for key in ("thumbnail_url", "thumbnail_text", "thumbnail_hook", "thumbnail_version",
                                                  "authenticity") if key in clip}

    @app.get("/api/jobs/{jid}/clips/{cid}/source-still")
    def source_still(jid: str, cid: str):
        with store.lock:
            clip = find_clip(store.get(jid), cid)
            if not clip.get("source_still_url"):
                raise HTTPException(404, "Source still not available")
        return file(jid, cid, "source.jpg")

    @app.get("/api/jobs/{jid}/clips/{cid}/tracking-still")
    def tracking_still(jid: str, cid: str):
        with store.lock:
            clip = find_clip(store.get(jid), cid)
            if not clip.get("tracking_still_url"):
                raise HTTPException(404, "Tracking still not available")
        return file(jid, cid, "tracking.jpg")

    @app.delete("/api/jobs/{jid}/clips/{cid}")
    def delete_clip(jid: str, cid: str):
        with store.lock:
            job = store.get(jid)
            clip = find_clip(job, cid)
            if clip["status"] == "rendering":
                raise HTTPException(409, "Wait for this highlight to finish rendering")
            job["clips"] = [clip for clip in job["clips"] if clip["id"] != cid]
            for moment in job["moments"]:
                if moment.get("clip_id") == cid:
                    moment.update(status="suggested", clip_id=None, reviewed=True)
            store.save(job)
            pending.cancel(jid, cid)
            shutil.rmtree(store.root / jid / cid, ignore_errors=True)
            if job["status"] != "cancelled":
                finish(jid, job.get("generation", 0))
        return {"id": cid, "status": "deleted"}

    app.mount("/caption-fonts", StaticFiles(directory=FONT_DIR), name="caption-fonts")
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="studio")
    return app


app = create_app()
