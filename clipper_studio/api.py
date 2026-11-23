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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import parse_qs, urlsplit

import cv2

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from cv2 import error as ImageProcessingError
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.background import BackgroundTask

from shorts_clipper.captions import FONT_DIR, CaptionStyle, caption_height, to_ass, transcribe
from shorts_clipper.downloader import download_analysis_media, download_section
from shorts_clipper.moments import analyze_media
from shorts_clipper.thumbnails import TEXT_LIMIT, choose_hook, generate_thumbnail
from shorts_clipper.director import (DirectorOptions, cut_source, effects_for, framing_fix, hook_layout,
                                     hook_title_ass, loudness, make_seamless_loop, mix_audio, normalize_loudness,
                                     plan_edit, plan_record, quality_check, remap_captions)
from shorts_clipper.story import game_name, hook_language, streamer_name
from shorts_clipper.voices import DELIVERY_CHOICES, MOODS, PERSONAS, SAMPLES, VOICE_CHOICES, narrate
from shorts_clipper.video import detect_focal_x, make_vertical_clip, video_duration
from shorts_clipper.facecam import (SourceBox, detect_facecam, auto_layout_facecam,
                                    layout_facecam, save_source_still)
from shorts_clipper.youtube import YouTubeAccessError

from .chat import METADATA_KEYS, ChatReplayError, fetch_chat, fetch_metadata
from .scheduler import Scheduler
from .spikes import detect_spikes
from .ranking import chat_moment, merge_moments


logger = logging.getLogger(__name__)
YOUTUBE_URL_ERROR = "Paste a YouTube video link for a finished livestream (watch, live, or youtu.be)."
ACTIVE_STATUSES = ("queued", "running")


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


class VoicePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voice: VoiceChoice = "auto"
    delivery: DeliveryChoice = "auto"
    text: str | None = Field(default=None, max_length=64)
    language: Literal["en", "id"] = "en"
    category: str | None = Field(default=None, max_length=24)


PREVIEW_CACHE = 64
LEGACY_DIRECTOR = {**DirectorSpec().model_dump(), "enabled": False}


class JobRequest(BaseModel):
    url: str
    ai_analysis: bool = True
    review: bool = True
    clips: int = Field(default=8, ge=1, le=50)
    lead: float = Field(default=18, ge=15, le=20)
    mode: Literal["crop", "blur"] = "crop"
    captions: bool = True
    facecam: FacecamSpec = Field(default_factory=FacecamSpec)
    language: str | None = Field(default="auto", pattern=r"^(auto|[a-z]{2,3})$")
    caption_style: dict | None = None
    director: DirectorSpec = Field(default_factory=DirectorSpec)

    @field_validator("url")
    @classmethod
    def youtube_url(cls, value):
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


class RenderRequest(BaseModel):
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)
    mode: Literal["crop", "blur"] = "crop"
    captions: bool = True
    facecam: FacecamSpec = Field(default_factory=FacecamSpec)
    language: str | None = Field(default="auto", pattern=r"^(auto|[a-z]{2,3})$")
    caption_style: dict | None = None
    director: DirectorSpec = Field(default_factory=DirectorSpec)


class KeepRequest(BaseModel):
    kept: bool


class MomentPatch(BaseModel):
    status: Literal["suggested", "selected", "rejected"] | None = None
    start: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end: float | None = Field(default=None, gt=0, allow_inf_nan=False)


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

    def voice(text, language, voice="auto", category=None, delivery="auto"):
        path = work / "hook-voice.wav"
        return path, narrate(text, language, path, voices, voice, category, clip.get("id", ""), delivery)

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
                for clip in job.get("clips", []):
                    clip.setdefault("facecam", FacecamSpec().model_dump())
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

    def stream_metadata(jid, generation, job):
        """Jobs made before channel names were kept get them fetched once, for the storytelling hook."""
        metadata = job.get("metadata") or {}
        if not job.get("url") or "channel" in metadata or "uploader" in metadata:
            return job
        try:
            info = fetch_metadata(job["url"])
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
                update(jid, generation, stage=f"{clip_label} · Downloading")
                source = download_section(job["url"], work, spec["start"], spec["end"])
                duration = min(video_duration(source), spec["end"] - spec["start"])
                subtitles, captions = None, []
                caption_language = None
                director = DirectorOptions.from_dict(spec.get("director") or LEGACY_DIRECTOR)
                style = CaptionStyle.from_dict(spec.get("caption_style"))
                if spec["captions"]:
                    update(jid, generation, stage=f"{clip_label} · Transcribing")
                    captions = transcribe(source, language=spec.get("language"), clip_end=duration)
                    caption_language = getattr(captions, "language", None)
                elif director.enabled:
                    update(jid, generation, stage=f"{clip_label} · Transcribing")
                    try:
                        captions = transcribe(source, language=spec.get("language"), clip_end=duration)
                    except Exception:
                        captions = []
                        logger.debug("Transcription for AI Director unavailable", exc_info=True)
                plan, directed, director_result = None, None, {"status": "disabled"}
                peak = thumbnail_peak(spec, duration)
                hook = thumbnail_hook(job, spec, peak, captions)
                if director.enabled:
                    update(jid, generation, stage=f"{clip_label} · AI Director planning edits")
                    try:
                        plan = plan_edit(loudness(source, duration), captions, duration, director, peak,
                                         spec.get("score"), spec.get("thumbnail_text") or hook,
                                         story_context(stream_metadata(jid, generation, job), spec,
                                                       caption_language or getattr(captions, "language", None), work,
                                                       store.root / "voices"))
                        directed = cut_source(source, work / "directed.mp4", plan)
                    except Exception:
                        plan, directed = None, None
                        director_result = {"status": "unavailable"}
                        logger.warning("AI Director unavailable for %s/%s; rendering unedited", jid, cid, exc_info=True)
                render_source = directed or source
                render_duration = plan.duration if plan else duration
                shown = remap_captions(captions, plan) if plan and spec["captions"] else captions if spec["captions"] else []
                text_height = None
                caption_limit = None
                update(jid, generation, stage=f"{clip_label} · Rendering")
                video = work / "clip.mp4"
                overlay = None
                result = {"status": "disabled", "box": None}
                still = work / "source.jpg"
                try:
                    save_source_still(source, still, duration / 2)
                except Exception:
                    logger.debug("Could not save source still", exc_info=True)
                try:
                    options = FacecamSpec.model_validate(spec.get("facecam", {}))
                    if options.enabled:
                        box = SourceBox(**options.box.model_dump()) if options.box else detect_facecam(render_source, 0, render_duration)
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
                            caption_limit = max(1, rect.y-260-24 if rect.y != 24 else 1500-rect.y-rect.h-24)
                            text_height = caption_height(shown, style, caption_limit) if spec["captions"] else None
                            corner = ("top" if rect.y == 24 else "bottom") + ("-left" if rect.x == 24 else "-right")
                            result = {"status": "used", "box": vars(box), "position": corner,
                                      "size": rect.w / 1080, "origin": "manual" if options.box else "detected"}
                except Exception:
                    overlay = None
                    result = {"status": "unavailable", "box": None}
                    logger.debug("Facecam unavailable; rendering without overlay", exc_info=True)
                fixes = []
                if plan and plan.hook and spec["captions"] and text_height is None:
                    text_height = caption_height(shown, style)
                if spec["captions"]:
                    style, note = framing_fix(style, overlay.rect if overlay else None, text_height)
                    if note:
                        fixes.append(note)
                if plan:
                    if spec["mode"] == "crop" and overlay is None:
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
                    story = plan.hook.get("style") == "story"
                    ass = hook_title_ass(ass, plan.hook["text"], plan.hook["duration"] + (0 if story else .7),
                                         top=band[0],story=story,highlight=plan.hook.get("streamer"),max_height=band[1]-band[0])
                if ass is not None:
                    subtitles = work / "captions.ass"
                    subtitles.write_text(ass, encoding="utf-8")
                effects = {"effects": effects_for(plan)} if plan else {}
                if overlay is None:
                    make_vertical_clip(render_source, video, 0, render_duration, spec["mode"], subtitles, **effects)
                else:
                    try:
                        make_vertical_clip(render_source, video, 0, render_duration, spec["mode"], subtitles, facecam=overlay, **effects)
                    except Exception:
                        result = {"status": "unavailable", "box": None}
                        make_vertical_clip(render_source, video, 0, render_duration, spec["mode"], subtitles, **effects)
                if plan:
                    update(jid, generation, stage=f"{clip_label} · AI Director mixing audio")
                    mixed = work / "mixed.mp4"
                    try:
                        mix_audio(video, mixed, plan)
                        mixed.replace(video)
                    except Exception:
                        fixes.append("Sound effects and music skipped (audio mix failed)")
                        logger.warning("AI Director audio mix failed for %s/%s", jid, cid, exc_info=True)
                    if director.loop:
                        update(jid, generation, stage=f"{clip_label} · AI Director smoothing the loop")
                        looped = work / "looped.mp4"
                        try:
                            plan.loop = make_seamless_loop(video, looped, shown)
                            if plan.loop["status"] == "applied":
                                looped.replace(video)
                        except (OSError, ValueError, KeyError, StopIteration, subprocess.CalledProcessError):
                            plan.loop = {"status": "skipped", "reason": "Could not render loop"}
                            logger.warning("AI Director loop failed for %s/%s", jid, cid, exc_info=True)
                    try:
                        normalize_loudness(video)
                    except Exception:
                        logger.warning("AI Director loudness correction failed for %s/%s", jid, cid, exc_info=True)
                    update(jid, generation, stage=f"{clip_label} · AI Director quality check")
                    director_result = plan_record(plan, director, quality_check(video, plan, shown, fixes, layout))
                    peak = plan.map_time(peak) if peak is not None else None
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
                    clip["director_result"] = director_result
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
                job.update(status="failed", stage="Failed", error=message, generation=generation + 1)
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
            info = (fetch_metadata(job["url"], False) if job["request"].get("ai_analysis", False)
                    else fetch_metadata(job["url"]))
            with store.lock:
                job = active_job(jid, generation)
                metadata_cache[jid, generation] = info
                job.update(status="queued", stage="Queued for chat analysis",
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
                      ("language", "auto"), ("caption_style", None), ("director", dict(LEGACY_DIRECTOR)))})
        clip.update(id=uuid.uuid4().hex, moment_id=moment["id"], number=number, status="queued", error=None,
                    file_url=None, download_url=None, thumbnail_url=None)
        moment.update(status="rendered", clip_id=clip["id"])
        return clip

    def analyze(jid, generation):
        try:
            started = time.monotonic()
            update(jid, generation, status="running", stage="Fetching chat replay")
            with store.lock:
                job = active_job(jid, generation)
                info = metadata_cache.pop((jid, generation))
            chat_error = None
            try:
                metadata, messages = fetch_chat(job["url"], store.root / jid / "chat", info)
            except ChatReplayError as error:
                if not job["request"].get("ai_analysis", False):
                    raise
                chat_error = error
                metadata, messages = {key: info.get(key) for key in METADATA_KEYS}, []
                update(jid, generation, chat_error=str(error))
            update(jid, generation, metadata=metadata, stage="Detecting chat spikes", progress=0.15,
                   chat_seconds=round(time.monotonic() - started, 2), message_count=len(messages))
            spikes, activity = detect_spikes(messages, metadata["duration"],
                                            clips=job["request"]["clips"], lead=job["request"]["lead"])
            count, lead = job["request"]["clips"], job["request"]["lead"]
            chat = [chat_moment(s) for s in sorted(spikes, key=lambda s: -s.get("score", 0))[:count]]
            ai, ai_error = [], None
            if job["request"].get("ai_analysis", False):
                directory = store.root / jid / "analysis"
                try:
                    update(jid, generation, stage="Downloading stream audio for AI analysis")
                    audio, video = download_analysis_media(job["url"], directory)
                    update(jid, generation, stage="AI analyzing video, audio & speech")
                    language = job["request"].get("language")

                    def progress(fraction, text):
                        update(jid, generation, stage="AI analyzing video, audio & speech",
                               progress=.2 + .65 * max(0, min(1, fraction)), analysis_detail=text)

                    ai = analyze_media(audio, video, metadata["duration"], clips=count, lead=lead,
                                       language=None if language == "auto" else language, progress=progress)[:count]
                except WorkCancelled:
                    raise
                except Exception as error:
                    ai_error = error
                    update(jid, generation, ai_error=str(error) if isinstance(error, YouTubeAccessError)
                           else "AI analysis was unavailable. Try this replay again.")
                finally:
                    shutil.rmtree(directory, ignore_errors=True)
            if chat_error and (ai_error or not job["request"].get("ai_analysis", False)):
                raise ai_error or chat_error
            moments = merge_moments(ai, chat, metadata["duration"], lead)
            with store.lock:
                job = active_job(jid, generation)
                clips = [] if job["request"].get("review", False) else [
                    moment_clip(job, m, i) for i, m in enumerate(moments[:count], 1)]
                job.update(clips=clips, moments=moments, activity=activity, analysis_complete=True, stage="Queued for rendering")
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
        analyses, renders = pending.positions()
        for clip in job["clips"]:
            clip["queue_position"] = renders.get((job["id"], clip["id"])) if clip["status"] == "queued" else None
        position = analyses.get(job["id"])
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

    @app.post("/api/jobs", status_code=202)
    def create_job(request: JobRequest):
        jid = uuid.uuid4().hex
        job = dict(id=jid, created_at=datetime.now(timezone.utc).isoformat(), url=request.url, request=request.model_dump(), status="queued", stage="Queued",
                   progress=0, metadata=None, activity=[], clips=[], error=None, generation=0, analysis_complete=False)
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

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        with store.lock:
            job = store.get(jid)
            if job["status"] not in ("queued", "running", "cancelled"):
                raise HTTPException(409, "This job has already finished")
            job.update(status="cancelled", stage="Cancelled. In-flight work stops at the next stage boundary.",
                       generation=job.get("generation", 0) + 1)
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
            if not 0 <= start < end <= job["metadata"]["duration"] or end - start > 60:
                raise HTTPException(422, "Moment must be within the video and at most 60 seconds long")
            moment.update(start=start, end=end, duration=end - start)
            if request.status is not None:
                moment["status"] = request.status
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
                    "director_result") if key in clip}
                clip["previous_render"].update({key: clip.get(key) for key in ("language", "caption_style", "caption_language")})
            clip.update(request.model_dump(), status="queued", error=None, file_url=None,
                        download_url=None, thumbnail_url=None, render_priority=True)
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
        return file(jid, cid, "clip.mp4", True)

    @app.get("/api/jobs/{jid}/clips/{cid}/thumbnail")
    def thumbnail(jid: str, cid: str, download: bool = False):
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
                hook = clip.get("thumbnail_hook") or thumbnail_hook(job, clip, source_peak)
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
                store.save(job)
                return {key: clip[key] for key in ("thumbnail_url", "thumbnail_text", "thumbnail_hook", "thumbnail_version")}

    @app.get("/api/jobs/{jid}/clips/{cid}/source-still")
    def source_still(jid: str, cid: str):
        with store.lock:
            clip = find_clip(store.get(jid), cid)
            if not clip.get("source_still_url"):
                raise HTTPException(404, "Source still not available")
        return file(jid, cid, "source.jpg")

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
                    moment.update(status="suggested", clip_id=None)
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
