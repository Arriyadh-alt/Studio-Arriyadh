# Real YouTube API check

URL: https://www.youtube.com/watch?v=LIxBN4AHMEE

Valkyrae, “LIVE! LATE NIGHT DEADLOCK WITH FRIENDS PART 2 (also live on twitch)”,
duration 5,986 seconds (1:39:46). Submitted via `POST /api/jobs` with
`clips=3`, `lead=18`, `mode=crop`, `captions=true`.

Job: `936a1f6f53ab40aaab9b52c953127a09`.
Data: `/home/ubuntu/clipper-data-final/936a1f6f53ab40aaab9b52c953127a09/`.

Chat fetch + parse: **7.57 seconds**, 754 timestamped messages.

| Onset | Peak | Score | Clip window | Chat sample |
|---|---|---:|---|---|
| 34:20 | 34:20 | 9.750 | 34:02–34:40 | “Heyyyyy”, “HEEYYYY”, waving emotes |
| 1:16:35 | 1:16:35 | 5.175 | 1:16:17–1:16:55 | “LMAOOO”, “lol”, reaction emotes |
| 1:39:10 | 1:39:35 | 6.500 | 1:38:52–1:39:46 | “Dude what!?!?”, “goodnight”, repeated raid link |

The sparse replay produces greeting, laughter and goodbye/raid traffic peaks;
these are plausible chat spikes, not verified gameplay highlights.

Initial job finished in **299.89 seconds** with two ready clips and one failed
section download (`ERROR: ffmpeg exited with code 1`). The original downloader
does not include further stderr in its exception; root cause is unverified.
Re-submitting the same first window through the clip `/render` API succeeded in
**71.07 seconds**, without code changes. Final job status: `completed`, all three
clips `ready`. Thus total processing time including the successful retry was
approximately **371 seconds**, excluding manual inspection time. No automatic
retry was added; per-clip failures remain visible and manually re-renderable.

All three files were inspected with ffprobe for 1080×1920 dimensions. JPEG
thumbnails were generated. Captions were enabled using the original faster-whisper
base transcription and ASS burn-in pipeline. Visual caption quality and native
iPhone Safari playback were not checked; HTTP byte-range responses were checked.

An earlier exploratory Ludwig VOD API job also produced three clips in 408.25
seconds, including the first Whisper model download. Its 11h46m replay took about
28 seconds to download, but that job ran the earlier detector revision, so the
shorter Valkyrae job above is the final-code check.

Environment: Python 3.10 venv (the only Python available on this machine).
yt-dlp prints `Deprecated Feature: Support for Python version 3.10 has been deprecated. Please update to Python 3.11 or above`.
This is not an unloaded venv: every command used `.venv/bin/python` or
`.venv/bin/yt-dlp`. Python 3.11+ is recommended for deployment. The initial missing
JavaScript runtime warning was resolved by installing Deno and yt-dlp's default
extras and starting the final API with the activated venv's bin directory on PATH.
