# Clipper Studio

A local web workspace that turns YouTube livestream chat spikes into vertical,
captioned highlights. Paste a replay URL, let Autopilot find the moments, then
review, nudge, re-render, keep and download your favorites.

## Run the web app on your laptop

Install Python 3.11+ and ffmpeg (including ffprobe and libass). From this folder:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
CLIPPER_DATA_DIR=~/clipper-data .venv/bin/uvicorn clipper_studio.api:app --host 0.0.0.0 --port 8080
```

Open http://localhost:8080. On an iPhone on the same Wi-Fi, open
`http://YOUR-LAPTOP-LAN-IP:8080` in Safari. Allow the port through your laptop's
firewall only on your trusted local network. This app has no authentication;
do not expose it to the public internet. No frontend build or service worker.

The first caption render downloads the Whisper `base` model. Output, replay chat
and job state are saved outside this repository in `~/clipper-data`. Keep a single
server process; do not use `--workers` or `--reload` during processing.
Separate pools handle metadata checks (3 workers), chat analysis (3) and rendering
(2); set `CLIPPER_METADATA_WORKERS`, `CLIPPER_ANALYSIS_WORKERS` or
`CLIPPER_RENDER_WORKERS` to tune them (1–32). Use one render worker on smaller
laptops. Re-renders take priority and automatic clips rotate across jobs. The UI
shows queue positions and lets you cancel pending work; active work stops at its
next stage boundary. Interrupted work is re-queued on restart, preserving ready
clips and cancelled jobs.
See [STUDIO.md](STUDIO.md) for the API and [E2E.md](E2E.md) for backend evidence.
The original CLI remains available below.

## Optional YouTube cookies

If YouTube responds with a bot check, sign-in prompt or HTTP 429, Studio shows
`YouTube is blocking this server — add or refresh YouTube cookies (see README)`
and says whether no cookies are configured or configured cookies may be rejected
or expired. Cookies are optional and are not guaranteed to bypass an IP block;
wait before retrying rate-limited requests.

Export a **Netscape cookies.txt** from your own authorized browser session using
[yt-dlp's official YouTube cookie export instructions](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies).
Use a **spare account**, not your primary account: automated access carries account
risk. Treat the export like a password. Store it outside the repository, restrict
permissions (for example `chmod 600 /secure/youtube-cookies.txt`), and never commit,
log or share its contents. Only use cookies for videos you are authorized to access.

Set the path in the environment of the server or CLI:

```sh
export CLIPPER_YTDLP_COOKIES=/secure/youtube-cookies.txt
```

Alternatively, inject **`YOUTUBE_COOKIES`** through your secret manager containing
the full cookies.txt text, including newlines. It is used only when no cookie path
is configured. Studio writes it to a private **0600** temporary file under
`CLIPPER_DATA_DIR` (default `~/clipper-data`). Do not paste the secret into command
history or enable shell tracing while handling it.

The CLI also accepts `--cookies /secure/youtube-cookies.txt`. Precedence is:
**CLI `--cookies` > `CLIPPER_YTDLP_COOKIES` path > `YOUTUBE_COOKIES` text**. These
settings cover metadata, chat replay and section downloads. Each yt-dlp invocation
uses its own writable private copy (0600 file in a 0700 temporary directory),
removed afterward even on failure. Refreshed cookies are sanitized and saved back
to a configured **file path** with an atomic replacement, keeping mode **0600**.
Only exception-free calls write back; any failure preserves the source. The source directory must be writable
for a private `.lock` file and same-directory replacement file; read-only secret
mounts are not suitable. Calls sharing a cookie source are serialized across
load, request and save so they cannot overwrite one another's refreshed sessions.
`YOUTUBE_COOKIES` text is never changed or written back; refresh it through your
secret manager. An invalid or unreadable configured path fails rather than
silently falling back to another secret.
With neither variable set (and no CLI option), no cookie option or file is added.
Refresh the export if rejected or expired, then submit/re-render again. Set server
environment variables at your next planned server start; this does not hot-update
an already running process.

## Proxy

YouTube blocks datacenter IPs, including this VM's IP, even with fresh cookies.
A **residential proxy** is needed for requests from this deployment. Configure
`CLIPPER_YTDLP_PROXY` with the full proxy URL (for example
`http://proxy.example:8080`); HTTP URLs with `user:password@host:port` and `socks5://`
URLs are passed through to yt-dlp for metadata and chat. Section downloads require
an `http://` proxy: HTTPS and SOCKS proxies are rejected before any request because
ffmpeg can otherwise bypass them. The CLI accepts `--proxy URL`, overriding the environment.
Metadata, replay and section requests all receive the setting; when unset their
proxy behavior is unchanged. A proxy can also be blocked by YouTube.

Inject credential-bearing URLs through your secret manager; do not paste them
into shell history, enable shell tracing or share them. yt-dlp log messages and
escaping errors redact the configured URL and credentials. Restart the server
at your next planned deploy to pick up environment changes.

## License

Licensed under the [MIT License](LICENSE).

# shorts-clipper

Cut a time range out of a YouTube video and turn it into a 1080x1920 (9:16) vertical MP4 for YouTube Shorts / TikTok.

Only the requested section is downloaded (via yt-dlp), then ffmpeg converts it to vertical.
Speech is transcribed locally with faster-whisper and burned in as synchronized
captions. The default **pop** preset highlights each spoken word with a quick scale
animation, while **classic** preserves the original static look and **minimal** uses
smaller fading text. Short chunks follow word timestamps and split at pauses/punctuation.

Studio's collapsible **Caption style** controls are available on both the create
form and every clip: choose Auto/English/Indonesian, preset, font, size, position,
colors, uppercase and keyword emphasis, then Re-render to apply changes. English
and Indonesian keyword heuristics emphasize at most two words per chunk. The clip
shows the transcription language. Other Whisper language codes are available via
API/CLI; transcription is not translation. Accuracy depends on audio/model quality.
Top/bottom positions reserve platform UI space and all positions keep the right rail clear.
Bundled OFL display fonts: **Montserrat Black** (pop's default), **Anton**, and
**Bebas Neue**. Both libass and line measurement use the bundled TTFs; no system
font installation is needed for these faces. DejaVu Sans (classic/minimal default),
Liberation Sans and JetBrains Mono remain available for existing styles; install
these system fonts on other hosts or libass will substitute. Font licenses and
source links are in `shorts_clipper/fonts/`.

## Requirements

- Python 3.10+ (3.11+ recommended by yt-dlp)
- ffmpeg (with libass subtitle support) and ffprobe on your PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: `winget install ffmpeg`

## Setup

```bash
cd shorts-clipper
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The default `base` model runs on CPU with int8 computation; no API key is needed.
Set `CLIPPER_WHISPER_MODEL` to change the default; CLI `--model` overrides it.
The first captioned run downloads model weights from Hugging Face (internet and
disk space required); later runs use the cache. Larger models improve accuracy
but need more RAM/time. Review automatic transcripts before publishing. Silence
produces no captions; transcription/download failures are reported as errors,
not silently ignored. Use `--no-captions` to skip transcription entirely.

## Usage

```bash
python -m shorts_clipper URL START END [-o OUTPUT] [-m crop|blur] [--keep-download DIR]
```

Times accept `SS`, `MM:SS` or `HH:MM:SS` (decimals allowed, e.g. `1:05.5`).

```bash
# 12-second clip from 1:05 to 1:17, centre-cropped to fill the vertical frame
python -m shorts_clipper "https://www.youtube.com/watch?v=aqz-KE-bpKQ" 1:05 1:17 -o my_short.mp4

# Keep the whole horizontal frame, centred over a blurred background
python -m shorts_clipper "https://www.youtube.com/watch?v=aqz-KE-bpKQ" 1:05 1:17 -m blur -o my_short_blur.mp4
```

| Option | Default | Meaning |
|---|---|---|
| `-o, --output` | `short.mp4` | Output file |
| `-m, --mode` | `crop` | `crop` fills the frame from the centre; `blur` fits the full frame over a blurred copy |
| `--keep-download DIR` | temp folder | Keep the downloaded horizontal section in `DIR` |
| `--cookies FILE` | environment fallback | Netscape cookies.txt; overrides `CLIPPER_YTDLP_COOKIES` and `YOUTUBE_COOKIES` |
| `--no-captions` | off | Disable captions |
| `--model MODEL` | environment or `base` | faster-whisper model name or local model directory |
| `--language CODE` | auto-detect | Speech language code, e.g. `en` (not translation) |
| `--caption-preset` | `pop` | `pop`, `classic`, or `minimal` |
| `--caption-position` | `bottom` | `bottom`, `middle`, or `top` |
| `--caption-size` | `80` | Font pixels at 1080x1920, clamped 40–130 (classic default uses original 76px; minimal scales to 80%) |
| `--caption-font` | preset default | Montserrat (Black), Anton, Bebas Neue, DejaVu Sans, Liberation Sans, JetBrains Mono; classic/minimal default to DejaVu Sans |
| `--srt` | off | Save UTF-8 subtitles next to the MP4; requires captions |
| `--local` | off | Read a local video path instead of downloading a URL |

```bash
python -m shorts_clipper "https://www.youtube.com/watch?v=aqz-KE-bpKQ" 1:05 1:25 --model base --language en --srt -o captioned.mp4
python -m shorts_clipper input.mp4 0 20 --local --srt -o captioned.mp4
python -m shorts_clipper URL 0 20 --no-captions -o plain.mp4
```

## Project layout

| File | Job |
|---|---|
| `shorts_clipper/cli.py` | Argument parsing and the download -> convert flow |
| `shorts_clipper/downloader.py` | Downloads the time range with yt-dlp |
| `shorts_clipper/video.py` | Builds and runs the ffmpeg 9:16 command |
| `shorts_clipper/captions.py` | Local transcription, word chunking and SRT/ASS formatting |
| `shorts_clipper/timeutils.py` | Parses and formats timestamps |

## Tests

```bash
pip install pytest
python -m pytest
```

Only download videos you have the rights to use.
