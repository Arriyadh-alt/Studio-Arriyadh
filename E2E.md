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

## Web UI + retry verification

The same-origin frontend is served on port 8080 with `CLIPPER_DATA_DIR=~/clipper-data`.
The completed three-clip demo job above was copied into that directory. Exploratory
servers on 8000 and 8001 were stopped.

Using Python Playwright connected to Chrome CDP at port 29229, submitted the same
Valkyrae URL from the home form, expanded Options and selected two clips (crop,
captions on). UI job: `f3cd8131cc074215ac26dcef95c0176f`.

- Two playable 1080×1920 clips appeared after **140.08 seconds**; chat analysis
  took **6.46 seconds**. Both were played and decoded without video errors.
- Nudged the first start from **34:02 to 34:03** and clicked Re-render. It finished
  in **60.22 seconds**. Played the replacement video for over three seconds,
  spanning a poll refresh, with no video error.
- Kept the first clip and downloaded the resulting ZIP from the UI; it contained
  `highlight-1.mp4`.
- Chart marker navigation was clicked on desktop and mobile. Expanded marker
  hit areas and adapted the chart coordinate width for readable mobile markers.
- Checked 1280×900 desktop, 390×844 mobile, and 360px width. No horizontal overflow
  at either mobile width. Browser page-error list was empty.
- Improved the review-button layout after screenshot inspection to prevent
  wrapping and use the full card width on desktop.

Screenshots saved outside the repository:
`~/screens/home-desktop.png`, `~/screens/home-mobile.png`,
`~/screens/job-desktop.png`, `~/screens/job-mobile.png`,
`~/screens/job-mobile-360.png`.

Backend tests after adding retry, keep/ZIP and static-file coverage:
`39 passed in 0.67s`. The retry test injects a download failure and a render
failure, succeeds on attempt three, then verifies a permanent failure exhausts
all three attempts and exposes its error. Browser viewport testing used Chrome,
not native iPhone Safari; native share/save sheet behavior remains unverified.
