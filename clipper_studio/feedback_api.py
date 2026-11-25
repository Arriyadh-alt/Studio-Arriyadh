"""Manage local records of clips already published outside Studio."""

import csv
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from clipper_studio.feedback import (
    ClipId, FeedbackStore, ImportRequest, OUTCOME, Publication, PublicationUpdate,
    calibration_summary, comparison, import_publications, numeric,
)


def clip_snapshot(job, clip):
    metadata = job.get("metadata") or {}
    rights = job.get("rights") or {}
    report = clip.get("authenticity") or {}
    director = clip.get("director_result") or {}
    edits = director.get("edits") or {}
    hook = edits.get("hook") or {}
    operations = report.get("operations") or {}
    source = {
        "url": job.get("url"),
        "channel": metadata.get("channel") or metadata.get("uploader"),
        "title": metadata.get("title") or job.get("filename"),
        "start": clip.get("start"),
        "end": clip.get("end"),
    }
    flags = [{
        "check": "Copyright / permission",
        "detail": "Verify ownership, license or permission. No rights confirmation is recorded." if not rights.get("confirmed")
        else "Rights are self-declared. Verify permission covers the footage, audio and destination.",
    }]
    if not source["channel"] or not source["url"] or not rights.get("attribution", True) or not rights.get("include_link", True):
        flags.append({"check": "Attribution", "detail": "Check the original creator and include source credit and a link where appropriate."})
    flags.append({
        "check": "Reused content",
        "detail": "Little substantive editing is visible in the render metadata. Cropping, subtitles and effects alone may not add editorial value."
        if not any(edits.get(key) for key in ("commentary", "cuts", "removed_seconds"))
        else "Review whether the edits add substantive editorial value and preserve the original setup and payoff.",
    })
    context = (report.get("checks") or {}).get("context") or {}
    if context.get("status") == "review":
        flags.append({"check": "Source context", "detail": context["detail"]})
    options = director.get("options") or {}
    hook_style = None
    if hook:
        if edits.get("commentary") or hook.get("style") == "commentary":
            hook_style = "commentary"
        elif hook.get("style") == "source" and not options.get("hook_text"):
            hook_style = "source quote"
    elif director.get("status") in {"disabled", "used"}:
        hook_style = "no hook"
    voiceover = bool(operations.get("synthetic_voiceover") or hook.get("narration") or edits.get("commentary"))
    ai_hook = bool(hook.get("text") and (hook.get("writer") or hook.get("style") == "story")
                   and (not options.get("hook_text") or hook["text"] in (hook.get("options") or [])))
    flags.append({
        "check": "Synthetic-content disclosure",
        "detail": ("AI voiceover or generated commentary is indicated. Review the platform’s disclosure rules."
                   if voiceover else "A generated text hook is indicated. Verify it against the source and review platform disclosure rules."
                   if ai_hook else "No AI voiceover or generated hook was identified in the recorded edit metadata; verify the published cut."),
    })
    moment = next((item for item in job.get("moments", [])
                   if item.get("id") == clip.get("moment_id") or item.get("clip_id") == clip["id"]), {})
    sources = clip.get("sources") or moment.get("sources") or []
    breakdown = clip.get("score_breakdown") or moment.get("score_breakdown") or {}
    score = clip.get("editorial_score", clip.get("score")) if breakdown or sources == ["ai"] else moment.get("ai_score")
    duration = next((value for value in (clip.get("rendered_duration"), edits.get("duration"))
                     if numeric(value) and value > 0), None)
    duration_basis = "render" if duration is not None else "source timeline"
    if duration is None and numeric(source["start"]) and numeric(source["end"]):
        duration = round(source["end"] - source["start"], 3)
    return {
        "job_id": job["id"], "clip_id": clip["id"],
        "title": clip.get("title") or metadata.get("title") or "Highlight",
        "hook": hook.get("text"), "hook_style": hook_style,
        "duration": duration, "duration_basis": duration_basis, "editing_mode": clip.get("editing_mode"),
        "detection_sources": sources,
        "editorial_score": score if numeric(score) else None,
        "score_breakdown": breakdown,
        "director_score": director.get("score") or {},
        "score_penalties": clip.get("score_penalties", moment.get("score_penalties")),
        "score_explanation": clip.get("score_explanation", moment.get("score_explanation")),
        "source": source, "integrity": flags,
    }


def update_record(record, update):
    fields = update.model_dump(mode="json", exclude_unset=True, exclude={"job_id", "clip_id", "confirmed_published"})
    metrics = fields.pop("metrics", {})
    changed_video = "publish_url" in record and fields.get("publish_url", record["publish_url"]) != record["publish_url"]
    record.update(fields)
    if changed_video:
        record["metrics"] = {}
    else:
        record["metrics"].update(metrics)


ScalarMetric = Literal["views", "engaged_views", "viewed_percent", "swiped_away_percent",
                       "average_view_duration", "average_percentage_viewed",
                       "likes", "comments", "shares", "subscribers_gained"]


def feedback_router(studio):
    router = APIRouter(prefix="/api/feedback")
    data = FeedbackStore(studio.root)

    def snapshot(jid, cid):
        with studio.lock:
            job = studio.get(jid)
        clip = next((clip for clip in job["clips"] if clip["id"] == cid), None)
        if clip is None:
            raise HTTPException(404, "Clip not found")
        if clip["status"] != "ready":
            raise HTTPException(409, "Select a rendered highlight")
        return clip_snapshot(job, clip)

    def find(records, rid):
        record = next((record for record in records if record["id"] == rid), None)
        if record is None:
            raise HTTPException(404, "Performance record not found")
        return record

    def save_entries(records, entries, *, importing=False):
        by_url = {record["publish_url"]: record for record in records}
        saved = []
        for entry in entries:
            record = by_url.get(entry.publish_url)
            if record is not None:
                if not importing or (record["job_id"], record["clip_id"]) != (entry.job_id, entry.clip_id):
                    raise HTTPException(409, "This published video is already linked to a performance record")
            else:
                record = snapshot(entry.job_id, entry.clip_id)
                record.update(id=uuid.uuid4().hex, recorded_at=datetime.now(timezone.utc).isoformat(),
                              publish_date=None, metrics={})
                records.insert(0, record)
            update_record(record, entry)
            saved.append(record)
        data.save(records)
        return saved

    @router.get("")
    def list_records():
        return data.records()

    @router.get("/clips")
    def rendered_clips():
        with studio.lock:
            jobs = studio.jobs()
        return [clip_snapshot(job, clip) for job in jobs for clip in job["clips"] if clip["status"] == "ready"]

    @router.get("/comparison")
    def compare(metric: ScalarMetric = OUTCOME):
        return comparison(data.records(), metric)

    @router.get("/calibration")
    def summarize():
        return calibration_summary(data.records(), data_dir=studio.root)

    @router.post("", status_code=201)
    def record_publication(entry: Publication):
        with data.lock:
            return save_entries(data.records(), [entry])[0]

    @router.post("/import")
    def load_file(request: ImportRequest):
        with data.lock:
            records = data.records()
            try:
                entries = import_publications(request, records)
            except ValidationError as error:
                details = "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                                    for item in error.errors(include_input=False, include_url=False))
                raise HTTPException(422, details) from error
            except (ValueError, csv.Error) as error:
                raise HTTPException(422, str(error)) from error
            return save_entries(records, entries, importing=True)

    @router.get("/{rid}")
    def get_record(rid: ClipId):
        return find(data.records(), rid)

    @router.patch("/{rid}")
    def update(rid: ClipId, changes: PublicationUpdate):
        with data.lock:
            records = data.records()
            record = find(records, rid)
            if "publish_url" in changes.model_fields_set:
                if changes.publish_url is None:
                    raise HTTPException(422, "A published video URL is required")
                if any(row["publish_url"] == changes.publish_url and row["id"] != rid for row in records):
                    raise HTTPException(409, "This published video is already linked to a performance record")
            update_record(record, changes)
            data.save(records)
            return record

    @router.delete("/{rid}")
    def remove(rid: ClipId):
        with data.lock:
            records = data.records()
            records.remove(find(records, rid))
            data.save(records)
            return {"deleted": rid}

    return router
