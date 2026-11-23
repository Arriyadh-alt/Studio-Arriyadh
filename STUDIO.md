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
- `GET /api/jobs/{id}/clips/{cid}/thumbnail` → JPEG; add `?download=true` for an attachment.
- `POST /api/jobs/{id}/clips/{cid}/thumbnail` → regenerate the ready clip’s
  thumbnail without changing its MP4. Send `{"text":"Custom hook"}` (up to 64
  characters), or omit/clear `text` to restore the automatic hook.
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
and contain `highlight-02.mp4` and, when available, `highlight-02.jpg` entries
matching the review numbers. Slugs are
ASCII lowercase hyphenated titles, at most 40 characters, with `clipper-studio`
as the empty-title fallback.
Each clip gets at most three full attempts (initial + two retries) across section
download, transcription, video conversion and thumbnail creation, with 2s/4s
backoff. `attempt` and stage expose retry progress. Final errors remain visible
on the clip with a Re-render action. `rendered_at` changes on successful renders
to avoid stale video caching. Kept clips can be downloaded together as a ZIP.

Ready clips get a 9:16 thumbnail automatically. Studio samples the rendered
clip locally, favoring visible faces, detected smiles, clear frames, and frames
near the chat peak. It styles the selected still with a short hook from nearby
captions, chat reactions, or the stream title. The clip card previews the JPG
and lets you download it or change its text; custom text is kept on re-render.

### AI Director

Jobs and re-renders accept `director` (`{"enabled": true, "hook": true, "edits": true,
"sfx": true, "music": "auto"|"on"|"off", "story": true, "voiceover": false,
"hook_text": null, "voice": "auto", "delivery": "auto", "loop": true}`; new jobs default to enabled, clips from older jobs stay off until
re-rendered with it). Before the final MP4 it plans the edit locally from Whisper word
timings, audio loudness and the chat peak:

- **Storytelling hook** (`story`, on by default): the clip opens on a ~3 s card, a dimmed
  still of the build-up with a bold title popping in word by word, a hit and a riser (or a
  spring for funny moments), then the story plays as build-up → main event → reaction →
  natural ending. The event is the loudest reaction onset near the chat peak; the
  build-up starts at a sentence 5–14 s earlier and the clip ends once speech and loudness
  have settled after the reaction (never mid-word). The hook names the streamer (YouTube
  channel) and the game when the stream title names one, in Indonesian or English to
  match the transcript, from the event type read off the dialogue and chat (karma, fail,
  scare, win, funny, rage, shock, wholesome). It teases the moment without telling the
  outcome, and clips of one stream get different hooks. `director_result.edits.hook`
  holds the text, up to three `options`, the event type, and the build-up and payoff
  lengths; the clip card previews the hook, offers the options and takes your own text
  as `hook_text` on re-render. Set `CLIPPER_LLM_API_KEY` (plus optional
  `CLIPPER_LLM_BASE_URL`, `CLIPPER_LLM_MODEL`, default `gpt-4o-mini`) to have an
  OpenAI-compatible model write the hooks from the transcript; any error falls back to
  the offline writer. `voiceover` has a creator read the hook: a short reaction
  first ("Wait, what?", "Guys, dengerin."), then the line in the scene's delivery
  (`excited`, `surprise`, `suspense` with a held-back payoff, `humor`, `warm`), with pitch,
  pace and pauses to match. `voice` picks the persona: `hype` (Hype Streamer), `bestie`
  (Bubbly Bestie), `trailer` (Movie Trailer), `chill` (Chill Commentator), `anchor`
  (Breaking News) or `classic` (the offline Piper voice); `auto` matches it to the scene,
  and `delivery: "auto"` reads the mood off the event type. Neural voices come from
  Microsoft Edge read-aloud (edge-tts, no key); if it is unreachable the offline Piper voice
  (downloaded once into `<data dir>/voices`) reads the same script. `GET /api/voices` lists
  the personas and `POST /api/voices/preview` (`{"voice", "delivery", "text", "language",
  "category"}`) returns a WAV to audition before rendering; the start form and every clip
  card have a Preview voice button. `edits.hook.narration` records the voice, mood, engine
  and spoken script.
- **Teaser hook** (`story: false`): copies the strongest 1.5–3 s moment (loud reaction, excited words, near the
  chat peak) to the front, with a punch-in zoom, the thumbnail hook text as a title,
  a whoosh and a flash transition into the clip.
- **Smart editing**: trims quiet lead-in, jump-cuts dead air over 0.8 s (never around
  the peak), alternates punch-ins on cuts, adds a reaction zoom with a short shake and
  hit at the peak, and up to four pop zooms on excited words or loudness spikes.
- **Reframing**: in center crop without a facecam overlay, each shot gets its own focal
  point with a short pan between shots. Zooms only apply to the main picture, so the
  facecam overlay stays fixed.
- **Sound**: synthesized whoosh/hit/pop effects; a soft music bed (`auto` adds it only
  to quiet clips with little speech) ducked under the voice with a sidechain
  compressor; the mix is normalized to −14 LUFS, −1.5 dBTP.
- **Seamless loop** (`loop`, on by default): samples the first and last frames of the
  rendered clip, then blends the ending into the opening picture. The blend is shorter
  over fast motion or captions. A brief audio ramp avoids clicks without moving or
  repeating narration. The opening, clip length and edit timeline stay intact.
  `director_result.edits.loop` records whether the transition was applied or skipped.
  Applied clips replay automatically in the preview; the transition is also in the
  downloaded MP4.
- **Viral score and quality check**: `director_result` has a 0–100 `score` (hook,
  pacing, emotion, dialogue, chat), `edits` counts, `suggestions`, and `quality`
  checks (format 1080×1920 H.264/AAC ≤ 60 s, loudness, caption overlaps/flashes after
  remapping to the edit, framing, black frames and very short shots). Captions are moved
  to the middle when they would cover the facecam.

If any director step fails the clip still renders unedited (`director_result.status`
`"unavailable"`); a failed audio mix keeps the edited video with its original audio.
Edited clips can be up to 60 s including the hook. The clip card shows the score,
edits and checks, and per-clip toggles that apply on Re-render.

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
