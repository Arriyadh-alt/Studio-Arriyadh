"""Local, single-process API. Run with uvicorn clipper_studio.api:app."""

import json
import os
import queue
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
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

from shorts_clipper.captions import to_ass, transcribe
from shorts_clipper.downloader import download_section
from shorts_clipper.video import make_vertical_clip, video_duration

from .chat import fetch_chat
from .spikes import detect_spikes


class JobRequest(BaseModel):
    url: str
    clips: int = Field(default=8, ge=1, le=50)
    lead: float = Field(default=18, ge=15, le=20)
    mode: Literal["crop", "blur"] = "crop"
    captions: bool = True

    @field_validator("url")
    @classmethod
    def youtube_url(cls, value):
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname not in (
            "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be") or parsed.username or parsed.password:
            raise ValueError("Paste an HTTPS YouTube livestream VOD URL")
        return value


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
        try:
            return json.loads((self.root / jid / "job.json").read_text())
        except FileNotFoundError:
            raise HTTPException(404, "Job not found") from None

    def save(self, job):
        directory = self.root / job["id"]
        directory.mkdir(exist_ok=True)
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(job, allow_nan=False))
        temporary.replace(directory / "job.json")

    def jobs(self):
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


def create_app(data_dir=None):
    runtime_dir = str(Path(sys.executable).parent)
    if (Path(runtime_dir) / "deno").is_file() and runtime_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = runtime_dir + os.pathsep + os.environ.get("PATH", "")
    store = Store(data_dir or os.environ.get("CLIPPER_DATA_DIR", "~/clipper-data"))
    pending = queue.Queue()

    def update(jid, **fields):
        with store.lock:
            job = store.get(jid)
            job.update(fields)
            store.save(job)

    def render_attempt(jid, cid):
        with store.lock:
            job = store.get(jid)
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
                update(jid, stage=f"Downloading highlight {number} of {len(job['clips'])}")
                source = download_section(job["url"], work, spec["start"], spec["end"])
                subtitles = None
                if spec["captions"]:
                    update(jid, stage=f"Transcribing highlight {number}")
                    subtitles = work / "captions.ass"
                    subtitles.write_text(to_ass(transcribe(source)), encoding="utf-8")
                update(jid, stage=f"Rendering highlight {number}")
                video = work / "clip.mp4"
                make_vertical_clip(source, video, 0, min(video_duration(source), spec["end"] - spec["start"]),
                                   spec["mode"], subtitles)
                thumbnail = work / "thumbnail.jpg"
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-frames:v", "1",
                                "-vf", "scale=270:480", str(thumbnail)], check=True)
                video.replace(output_dir / "clip.mp4")
                thumbnail.replace(output_dir / "thumbnail.jpg")
            with store.lock:
                job = store.get(jid)
                clip = find_clip(job, cid)
                base = f"/api/jobs/{jid}/clips/{cid}"
                clip.update(status="ready", rendered_at=time.time_ns(), file_url=base + "/video", download_url=base + "/download",
                            thumbnail_url=base + "/thumbnail")
                store.save(job)
        except Exception:
            raise

    def render(jid, cid):
        for attempt in range(1, 4):
            with store.lock:
                job = store.get(jid)
                clip = next((c for c in job["clips"] if c["id"] == cid), None)
                if clip is None:
                    return
                clip.update(attempt=attempt, status="rendering", error=None)
                store.save(job)
            try:
                render_attempt(jid, cid)
                return
            except Exception as error:
                with store.lock:
                    job = store.get(jid)
                    clip = next((c for c in job["clips"] if c["id"] == cid), None)
                    if clip is None:
                        return
                    clip.update(status="failed" if attempt == 3 else "rendering",
                                              error=str(error))
                    job["stage"] = "Clip failed" if attempt == 3 else f"Retrying clip in {attempt * 2}s (attempt {attempt + 1}/3)"
                    store.save(job)
                if attempt < 3:
                    time.sleep(attempt * 2)

    def finish(jid):
        with store.lock:
            job = store.get(jid)
            if any(c["status"] in ("queued", "rendering") for c in job["clips"]):
                return
            failed = any(c["status"] == "failed" for c in job["clips"])
            job.update(status="completed_with_errors" if failed else "completed", progress=1,
                       stage="Finished" if job["clips"] else "No significant chat spikes found")
            store.save(job)

    def worker():
        while True:
            item = pending.get()
            if item is None:
                pending.task_done()
                return
            jid, cid = item
            try:
                started = time.monotonic()
                update(jid, status="running", stage="Fetching chat replay" if cid is None else "Re-rendering")
                if cid is None:
                    job = store.get(jid)
                    metadata, messages = fetch_chat(job["url"], store.root / jid / "chat")
                    update(jid, metadata=metadata, stage="Detecting chat spikes", progress=0.15,
                           chat_seconds=round(time.monotonic() - started, 2), message_count=len(messages))
                    spikes, activity = detect_spikes(messages, metadata["duration"],
                                                     clips=job["request"]["clips"], lead=job["request"]["lead"])
                    clips = [dict(spike, number=number, id=uuid.uuid4().hex, status="queued", error=None,
                                  mode=job["request"]["mode"], captions=job["request"]["captions"],
                                  file_url=None, download_url=None, thumbnail_url=None)
                             for number, spike in enumerate(sorted(spikes, key=lambda s: s["start"]), 1)]
                    update(jid, clips=clips, activity=activity)
                    for index, clip in enumerate(clips):
                        render(jid, clip["id"])
                        update(jid, progress=0.15 + 0.85 * (index + 1) / len(clips))
                else:
                    render(jid, cid)
                finish(jid)
                update(jid, elapsed_seconds=round(time.monotonic() - started, 2))
            except Exception as error:
                update(jid, status="failed", stage="Failed", error=str(error))
            finally:
                pending.task_done()

    @asynccontextmanager
    async def lifespan(app):
        # Interrupted work is explicit instead of silently stuck after a restart.
        with store.lock:
            for job in store.jobs():
                if job["status"] in ("queued", "running"):
                    job.update(status="failed", stage="Interrupted", error="Server restarted; submit a new job or re-render clips.")
                    for clip in job["clips"]:
                        if clip["status"] in ("queued", "rendering"):
                            clip.update(status="failed", error="Server restarted")
                    store.save(job)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        yield
        pending.put(None)
        thread.join()

    app = FastAPI(title="Clipper Studio", lifespan=lifespan)
    app.state.store = store
    app.state.pending = pending

    @app.post("/api/jobs", status_code=202)
    def create_job(request: JobRequest):
        jid = uuid.uuid4().hex
        job = dict(id=jid, created_at=datetime.now(timezone.utc).isoformat(), url=request.url, request=request.model_dump(), status="queued", stage="Queued",
                   progress=0, metadata=None, activity=[], clips=[], error=None)
        with store.lock:
            store.save(job)
            pending.put((jid, None))
        return {"id": jid}

    @app.get("/api/jobs")
    def list_jobs():
        with store.lock:
            return store.jobs()

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        with store.lock:
            return store.get(jid)

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
                        download_url=None, thumbnail_url=None)
            if job["status"] not in ("running", "queued"):
                job.update(status="queued", stage="Queued re-render", progress=0, error=None)
            store.save(job)
            pending.put((jid, cid))
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
            shutil.rmtree(store.root / jid / cid, ignore_errors=True)
        return {"id": cid, "status": "deleted"}

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="studio")
    return app


app = create_app()
