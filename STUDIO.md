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

For a one-command production start after installation:

```sh
CLIPPER_DATA_DIR=~/clipper-data .venv/bin/uvicorn clipper_studio.api:app --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`, or your laptop's LAN IP on an iPhone on the same
trusted Wi-Fi. The root serves the responsive static UI, including a web manifest
and app icons; no build step or service worker is used. Deno is located alongside
the venv interpreter automatically. The
first caption render downloads faster-whisper's `base` model; subsequent renders
use its cache. Caption generation runs on CPU. `/docs` serves OpenAPI docs.

`CLIPPER_DATA_DIR` defaults to `~/clipper-data` and must be outside this repo.
Each job has an atomically replaced `job.json`, downloaded chat, and clip folders
with `clip.mp4` and `thumbnail.jpg`. Intermediate section downloads are removed.
Use **one uvicorn process, without reload**. Independent bounded thread pools run
metadata preflight (default 3), chat fetch/analysis (3), and rendering (2).
Configure `CLIPPER_METADATA_WORKERS`, `CLIPPER_ANALYSIS_WORKERS`, and
`CLIPPER_RENDER_WORKERS` with integer counts from 1 to 32. Reduce render workers
to 1 on smaller laptops: transcription and ffmpeg are CPU/memory intensive.
Metadata preflight checks for a finished VOD and live-chat replay before any
replay download, independently of busy analysis/render pools. Preflight metadata
is reused rather than fetched again. Actual network latency still varies.

Re-renders have priority over waiting automatic clips; automatic renders are
round-robin across jobs. In-flight renders are not preempted. JSON reads, writes
and read/modify/write operations share a process-local reentrant lock; files are
atomically replaced. Multiple server processes sharing one data directory are
not supported.

On restart, interrupted analysis jobs are re-queued for preflight/analysis;
interrupted queued/rendering clips are re-queued without redoing ready or failed
clips. A completed analysis is persisted even when it found no spikes.
Cancelled jobs remain cancelled. This is a local backend, with no
authentication: do not expose it publicly without access controls and quotas.

## Optional YouTube authentication

Set **`CLIPPER_YTDLP_COOKIES`** to the path of a Netscape cookies.txt export, or,
when that path is unset, inject **`YOUTUBE_COOKIES`** containing its full text.
Export from your own authorized session following
[yt-dlp's instructions](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies);
a spare account is recommended. Keep exports outside the repo and never log or
commit them. See [README.md](README.md#optional-youtube-cookies) for setup/security.

All yt-dlp metadata/chat/section requests use these settings. A configured path
is copied before use, allowing read-only mounts. Text secrets are written under
`CLIPPER_DATA_DIR` (default `~/clipper-data`). Each invocation has a private 0700
temporary directory and writable 0600 cookie file, cleaned up after yt-dlp closes;
the original secret is not modified. Without either setting, cookie behavior is
unchanged. The CLI's `--cookies FILE` overrides both environment settings.

Bot-check/sign-in/HTTP 429 failures produce the friendly job error
`YouTube is blocking this server — add or refresh YouTube cookies (see README)`,
distinguishing missing cookies from configured cookies that may be rejected or
expired. A section download blocked this way fails the job rather than repeatedly
retrying every highlight. Cookies do not guarantee access; respect rate limits
and use only media you are authorized to access.

## API

- `POST /api/jobs` → 202 `{id}`. Body:
  `{url, clips:8, lead:18, mode:"crop", captions:true, language:"auto", caption_style:null}`. Use an HTTPS YouTube
  video link (`watch`, `live`, or `youtu.be`, including mobile) for a finished
  livestream; clips 1–50, lead 15–20, mode `crop` or `blur`. No timestamps needed.
- `GET /api/jobs` → array of jobs; add `?summary=true` for recent-stream cards
  without activity and clips, including `clip_count` and `queue_position`.
  `GET /api/jobs/{id}` → job.
- `POST /api/jobs/{id}/cancel` → `{id,status:"cancelled"}`. Cancels queued work
  and stops running work at the next stage boundary; an active yt-dlp download,
  Whisper transcription or ffmpeg process may finish first (not forcibly killed).
  Ready clips and Keep flags remain available. Cancelled clips can be re-rendered.
  A generation token prevents old work from publishing after cancel/re-render.
  Cancelling a finished job returns 409; repeated cancellation is safe.
- `POST /api/jobs/{id}/clips/{cid}/render` → 202 `{id:cid,status:"queued"}`.
  Body `{start,end,mode:"crop",captions:true,language:"auto",caption_style:null}`. Seconds, within the original
  duration, end after start, maximum 60 seconds. Preserves detection metadata.
- `GET /api/jobs/{id}/clips/{cid}/video` → inline MP4, supports byte ranges (206).
- `GET /api/jobs/{id}/clips/{cid}/download` → MP4 attachment.
- `GET /api/jobs/{id}/clips/{cid}/thumbnail` → JPEG.
- `DELETE /api/jobs/{id}/clips/{cid}` → `{id:cid,status:"deleted"}`; removes files.
- `PATCH /api/jobs/{id}/clips/{cid}` with `{kept:true}` → `{id,kept}`. Persisted
  during processing and across re-renders; defaults to false for older clips.
- `GET /api/jobs/{id}/download-kept` → ZIP attachment of ready kept clips; 409
  if none. The temporary archive is removed after streaming.

Job shape: `{id,url,request,status,stage,progress,metadata,activity,clips,error}`.
Status: `queued`, `running`, `completed`, `completed_with_errors`, `failed`, `cancelled`.
Jobs and clips include `queue_position` (1-based when waiting, otherwise null).
Positions are per pool, exclude active work, and reflect re-render priority and
round-robin render order; they may change as new work arrives. The UI shows
`Queued (position N)` and a Cancel job button for queued/running jobs.
Progress is 0–1. New jobs include UTC `created_at`. Metadata: `{id,title,duration,thumbnail}`. Once chat is fetched,
`chat_seconds` and `message_count` are added; on completion `elapsed_seconds`
records pipeline time (or the latest re-render time).
Activity points: `{time,count,baseline,intensity,score}` in 5-second bins.
Clip shape: `{id,onset,peak,score,start,end,message_sample,mode,captions,status,
error,file_url,download_url,thumbnail_url,language,caption_style,caption_language}`; status `queued`, `rendering`, `ready`,
`failed`, or `cancelled`. URLs are null until ready. Errors are explicit strings.

### Caption options

`captions` remains a boolean. `language` is `"auto"`/null (Whisper auto-detection)
or a Whisper language code such as `"en"` or `"id"`; it is not a translation target.
`caption_language` records Whisper's language result (null when captions are off).
Generated clips inherit language/style from the job. Old clips without these fields
use automatic language detection and the default pop style. Re-renders transcribe
again; cancellation restores the previous render's caption settings and language.
`CLIPPER_WHISPER_MODEL` selects the CPU/int8 model (fallback `base`).

`caption_style` is an optional object; null uses defaults:
```json
{"preset":"pop","font":"Montserrat","size":80,"position":"bottom",
 "text_color":"#FFFFFF","highlight_color":"#FFE34D","emphasis_color":"#67F5B5",
 "uppercase":false,"emphasis":true}
```
Unknown keys/invalid values fall back to defaults; numeric size clamps to 40–130.
Presets: `pop` (word-active color/scale animation, stable chunk wrapping), `classic`
(original static 76px look at default settings), `minimal` (80% size, no emphasis,
fade in/out). Fonts: bundled OFL `Montserrat` (Black, pop default), `Anton`,
`Bebas Neue`, plus system `DejaVu Sans`, `Liberation Sans`, `JetBrains Mono`.
Classic/minimal default to DejaVu Sans; an explicitly stored font remains respected.
Bundled TTFs are loaded directly for wrapping and supplied to libass via `fontsdir`,
and served at `/caption-fonts/` for CSS previews. Licenses/source attribution are in
`shorts_clipper/fonts/`. System faces require installation on other hosts or libass
will substitute. Position: `bottom`, `middle`, `top`; top/bottom
reserve platform UI space and all positions retain the wider right-rail margin.
Colors are `#RRGGBB`. Keyword emphasis picks up to two words using numbers,
punctuation, uppercase and English/Indonesian content-word heuristics, not an LLM.
Studio exposes these controls in collapsible create/per-clip sections. Changes
are drafts until Re-render; detected language is shown separately on the clip.

New clips also have a stable `number`, assigned in start order and preserved on deletion.
Older clips fall back to their list position. Single downloads use
`<title-slug>-highlight-02.mp4`; kept ZIPs use `<title-slug>-kept-clips.zip`
and contain `highlight-02.mp4` entries matching the review numbers. Slugs are
ASCII lowercase hyphenated titles, at most 40 characters, with `clipper-studio`
as the empty-title fallback.
Each clip gets at most three full attempts (initial + two retries) across section
download, transcription, video conversion and thumbnail creation, with 2s/4s
backoff. `attempt` and stage expose retry progress. Final errors remain visible
on the clip with a Re-render action. `rendered_at` changes on successful renders
to avoid stale video caching. Kept clips can be downloaded together as a ZIP.

Validation errors: 422; missing job/clip: 404; unavailable file or attempting
edits while that clip is queued/rendering or deletion while it is rendering: 409.
Ready clips can be re-rendered while Autopilot runs; queued clips may be deleted
and are skipped by the worker. Regular videos, unavailable videos and streams
without chat replay fail with an explanation. Long replays are supported;
replay downloads are checked against a five-minute time budget and a 128 MiB
size budget. Each pool queues in submission order, with re-render priority and
round-robin automatic clips as described above; refreshing the job page reconnects
to the existing job. Fewer than the requested N highlights (including zero)
are valid when there are insufficient separated spikes.

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

Tests cover replay text/emoji/paid messages and absent timestamps, fast no replay,
synthetic onset/peak/ranking/NMS/window boundaries/intensity, API job lifecycle,
independent pools, priority/round-robin ordering, queue positions, cancellation,
restart recovery, re-render validation, range streaming, downloads, thumbnail and deletion. API
unit tests mock media/network work; a real YouTube check is separate.
