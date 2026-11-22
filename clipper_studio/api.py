"""Local, single-process API. Run with uvicorn clipper_studio.api:app."""

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
from typing import Literal
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

from shorts_clipper.captions import to_ass, transcribe
from shorts_clipper.downloader import download_section
from shorts_clipper.video import make_vertical_clip, video_duration

from .chat import ChatReplayError, fetch_chat, fetch_metadata
from .scheduler import Scheduler
from .spikes import detect_spikes


logger = logging.getLogger(__name__)
YOUTUBE_URL_ERROR = "Paste a YouTube video link for a finished livestream (watch, live, or youtu.be)."
ACTIVE_STATUSES = ("queued", "running")


class WorkCancelled(Exception):
    pass


class JobRequest(BaseModel):
    url: str
    clips: int = Field(default=8, ge=1, le=50)
    lead: float = Field(default=18, ge=15, le=20)
    mode: Literal["crop", "blur"] = "crop"
    captions: bool = True

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


class KeepRequest(BaseModel):
    kept: bool


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
                return json.loads((self.root / jid / "job.json").read_text())
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
            store.save(job)
        try:
            output_dir = store.root / jid / cid
            output_dir.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
                work = Path(temporary)
                number = clip_number(job, spec)
                update(jid, generation, stage=f"Downloading highlight {number} of {len(job['clips'])}")
                source = download_section(job["url"], work, spec["start"], spec["end"])
                subtitles = None
                if spec["captions"]:
                    update(jid, generation, stage=f"Transcribing highlight {number}")
                    subtitles = work / "captions.ass"
                    subtitles.write_text(to_ass(transcribe(source)), encoding="utf-8")
                update(jid, generation, stage=f"Rendering highlight {number}")
                video = work / "clip.mp4"
                make_vertical_clip(source, video, 0, min(video_duration(source), spec["end"] - spec["start"]),
                                   spec["mode"], subtitles)
                update(jid, generation, stage=f"Creating thumbnail for highlight {number}")
                thumbnail = work / "thumbnail.jpg"
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-frames:v", "1",
                                "-vf", "scale=270:480", str(thumbnail)], check=True)
                with store.lock:
                    job = active_job(jid, generation)
                    clip = find_clip(job, cid)
                    video.replace(output_dir / "clip.mp4")
                    thumbnail.replace(output_dir / "thumbnail.jpg")
                    base = f"/api/jobs/{jid}/clips/{cid}"
                    clip.update(status="ready", rendered_at=time.time_ns(), file_url=base + "/video", download_url=base + "/download",
                                thumbnail_url=base + "/thumbnail")
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
            except WorkCancelled:
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
                       stage="Finished" if job["clips"] else "No significant chat spikes found",
                       elapsed_seconds=round(time.time() - job.get("processing_started", time.time()), 2))
            store.save(job)

    def fail_job(jid, generation, error):
        if isinstance(error, ChatReplayError):
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
            info = fetch_metadata(job["url"])
            with store.lock:
                job = active_job(jid, generation)
                metadata_cache[jid, generation] = info
                job.update(status="queued", stage="Queued for chat analysis",
                           metadata={key: info.get(key) for key in ("id", "title", "duration", "thumbnail")})
                store.save(job)
                pending.put((jid, None, generation))
        except WorkCancelled:
            pass
        except Exception as error:
            fail_job(jid, generation, error)

    def analyze(jid, generation):
        try:
            started = time.monotonic()
            update(jid, generation, status="running", stage="Fetching chat replay")
            with store.lock:
                job = active_job(jid, generation)
                info = metadata_cache.pop((jid, generation))
            metadata, messages = fetch_chat(job["url"], store.root / jid / "chat", info)
            update(jid, generation, metadata=metadata, stage="Detecting chat spikes", progress=0.15,
                   chat_seconds=round(time.monotonic() - started, 2), message_count=len(messages))
            spikes, activity = detect_spikes(messages, metadata["duration"],
                                            clips=job["request"]["clips"], lead=job["request"]["lead"])
            clips = [dict(spike, number=number, id=uuid.uuid4().hex, status="queued", error=None,
                          mode=job["request"]["mode"], captions=job["request"]["captions"],
                          file_url=None, download_url=None, thumbnail_url=None)
                     for number, spike in enumerate(sorted(spikes, key=lambda s: s["start"]), 1)]
            with store.lock:
                job = active_job(jid, generation)
                job.update(clips=clips, activity=activity, analysis_complete=True, stage="Queued for rendering")
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
                        bundle.write(store.root / jid / clip["id"] / "clip.mp4", f"highlight-{clip_number(job, clip):02d}.mp4")
            except Exception:
                shutil.rmtree(archive_dir)
                raise
        return FileResponse(archive, media_type="application/zip", filename=f"{title_slug(job)}-kept-clips.zip",
                            background=BackgroundTask(shutil.rmtree, archive_dir))

    @app.post("/api/jobs/{jid}/clips/{cid}/render", status_code=202)
    def rerender(jid: str, cid: str, request: RenderRequest):
        with store.lock:
            job = store.get(jid)
            clip = find_clip(job, cid)
            if clip["status"] in ("rendering", "queued"):
                raise HTTPException(409, "Wait for this highlight to finish rendering")
            if request.end <= request.start or request.end - request.start > 60 or request.end > job["metadata"]["duration"]:
                raise HTTPException(422, "Clip must be within the video and between 0 and 60 seconds long")
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
        return FileResponse(path, media_type="image/jpeg" if filename.endswith("jpg") else "video/mp4",
                            filename=f"{title_slug(job)}-highlight-{clip_number(job, clip):02d}.mp4" if download else None,
                            content_disposition_type="attachment" if download else "inline")

    @app.get("/api/jobs/{jid}/clips/{cid}/video")
    def video(jid: str, cid: str):
        return file(jid, cid, "clip.mp4")

    @app.get("/api/jobs/{jid}/clips/{cid}/download")
    def download(jid: str, cid: str):
        return file(jid, cid, "clip.mp4", True)

    @app.get("/api/jobs/{jid}/clips/{cid}/thumbnail")
    def thumbnail(jid: str, cid: str):
        return file(jid, cid, "thumbnail.jpg")

    @app.delete("/api/jobs/{jid}/clips/{cid}")
    def delete_clip(jid: str, cid: str):
        with store.lock:
            job = store.get(jid)
            clip = find_clip(job, cid)
            if clip["status"] == "rendering":
                raise HTTPException(409, "Wait for this highlight to finish rendering")
            job["clips"] = [clip for clip in job["clips"] if clip["id"] != cid]
            store.save(job)
            pending.cancel(jid, cid)
            shutil.rmtree(store.root / jid / cid, ignore_errors=True)
            if job["status"] != "cancelled":
                finish(jid, job.get("generation", 0))
        return {"id": cid, "status": "deleted"}

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="studio")
    return app


app = create_app()
