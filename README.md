# shorts-clipper

Cut a time range out of a YouTube video and turn it into a 1080x1920 (9:16) vertical MP4 for YouTube Shorts / TikTok.

Only the requested section is downloaded (via yt-dlp), then ffmpeg converts it to vertical.
Speech is transcribed locally with faster-whisper and burned in as bold white,
black-outlined captions in the lower-middle of the frame. Captions use up to four
words per chunk, following word timestamps and splitting at pauses/punctuation.

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
| `--no-captions` | off | Disable captions |
| `--model MODEL` | `base` | faster-whisper model name or local model directory |
| `--language CODE` | auto-detect | Speech language code, e.g. `en` (not translation) |
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
