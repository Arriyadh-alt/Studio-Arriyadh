# Clipper Studio backend

The original `shorts_clipper/` CLI is unchanged. Studio adds chat-based automatic
highlight selection in `clipper_studio/`.

## Run

Python 3.11+ is recommended. Install ffmpeg (including ffprobe and libass), then:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn clipper_studio.api:app --host 127.0.0.1 --port 8000
```

Activate the venv so yt-dlp can find the bundled Deno JavaScript runtime. The
first caption render downloads faster-whisper's `base` model; subsequent renders
use its cache. Caption generation runs on CPU. `/docs` serves OpenAPI docs.

`CLIPPER_DATA_DIR` defaults to `~/clipper-data` and must be outside this repo.
Each job has an atomically replaced `job.json`, downloaded chat, and clip folders
with `clip.mp4` and `thumbnail.jpg`. Intermediate section downloads are removed.
Use **one uvicorn process, without reload**: a single background thread serializes
all jobs and clip renders. Interrupted jobs are marked failed at startup; clips
can be re-rendered or the URL resubmitted. This is a local backend, with no
authentication: do not expose it publicly without access controls and quotas.

## API

- `POST /api/jobs` → 202 `{id}`. Body:
  `{url, clips:8, lead:18, mode:"crop", captions:true}`. HTTPS YouTube VOD URLs
  only; clips 1–50, lead 15–20, mode `crop` or `blur`. No timestamps needed.
- `GET /api/jobs` → array of jobs; `GET /api/jobs/{id}` → job.
- `POST /api/jobs/{id}/clips/{cid}/render` → 202 `{id:cid,status:"queued"}`.
  Body `{start,end,mode:"crop",captions:true}`. Seconds, within the original
  duration, end after start, maximum 60 seconds. Preserves detection metadata.
- `GET /api/jobs/{id}/clips/{cid}/video` → inline MP4, supports byte ranges (206).
- `GET /api/jobs/{id}/clips/{cid}/download` → MP4 attachment.
- `GET /api/jobs/{id}/clips/{cid}/thumbnail` → JPEG.
- `DELETE /api/jobs/{id}/clips/{cid}` → `{id:cid,status:"deleted"}`; removes files.

Job shape: `{id,url,request,status,stage,progress,metadata,activity,clips,error}`.
Status: `queued`, `running`, `completed`, `completed_with_errors`, `failed`.
Progress is 0–1. Metadata: `{id,title,duration,thumbnail}`. Once chat is fetched,
`chat_seconds` and `message_count` are added; on completion `elapsed_seconds`
records pipeline time (or the latest re-render time).
Activity points: `{time,count,baseline,intensity,score}` in 5-second bins.
Clip shape: `{id,onset,peak,score,start,end,message_sample,mode,captions,status,
error,file_url,download_url,thumbnail_url}`; status `queued`, `rendering`, `ready`,
or `failed`. URLs are null until ready. Errors are explicit strings.

Validation errors: 422; missing job/clip: 404; unavailable file or attempting
edits/deletion while a job is active: 409. A video without replay makes the
background job fail with a clear `error`. Fewer than the requested N highlights
(including zero) are valid when there are insufficient separated spikes.

## Detection and checks

Count bins against the trailing 180-second median; combine ratio and Poisson-like
z-score with a 25% reaction/spam boost. Follow rises to a local peak, retain onset,
then score-sort and suppress overlapping windows and peaks within 60 seconds.
Windows begin onset minus lead, target at least 30 seconds, include a 20-second
peak tail where possible, cap at 60 seconds and clamp at video boundaries.
These are heuristic chat highlights, not semantic guarantees of interesting video.

```sh
python -m pytest -q
```

Tests cover replay text/emoji/paid messages and absent timestamps, no replay,
synthetic onset/peak/ranking/NMS/window boundaries/intensity, API job lifecycle,
re-render validation, range streaming, downloads, thumbnail and deletion. API
unit tests mock media/network work; a real YouTube check is separate.
