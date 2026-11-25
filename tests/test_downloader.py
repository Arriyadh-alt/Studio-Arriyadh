import fcntl
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yt_dlp

from clipper_studio.download_worker import run_download
from shorts_clipper.downloader import download_analysis_media, download_section
from shorts_clipper.youtube import YouTubeAccessError


def test_download_worker_translates_download_errors(tmp_path):
    with pytest.raises(YouTubeAccessError, match="require an http:// proxy"):
        run_download("section", ["https://youtu.be/video", tmp_path, 5, 30],
                     {"proxy": "socks5://localhost"}, timeout=10)


def test_download_worker_kills_and_reaps_a_stalled_process(monkeypatch):
    spawn = subprocess.Popen
    processes = []

    def stalled(*args, **kwargs):
        process = spawn([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("clipper_studio.download_worker.subprocess.Popen", stalled)
    with pytest.raises(TimeoutError, match="exceeded its time limit"):
        run_download("replay", [], timeout=.1)
    assert processes[0].returncode == -signal.SIGKILL


def test_download_worker_returns_completed_audio_when_video_times_out(monkeypatch, tmp_path):
    spawn = subprocess.Popen
    code = """
import sys
import time
from pathlib import Path
from clipper_studio.download_worker import main
from shorts_clipper import downloader

def download(url, output_dir, **kwargs):
    if "on_start" in kwargs:
        kwargs["on_start"]()
    audio = output_dir / "audio.m4a"
    audio.write_bytes(b"completed audio")
    if "on_audio" in kwargs:
        kwargs["on_audio"](audio)
    time.sleep(60)

downloader.download_analysis_media = download
main(Path(sys.argv[1]))
"""

    def stalled(command, **kwargs):
        return spawn([sys.executable, "-c", code, command[-1]], **kwargs)

    monkeypatch.setattr("clipper_studio.download_worker.subprocess.Popen", stalled)
    audio, video = run_download("media", ["https://youtu.be/video", tmp_path], timeout=2)
    assert Path(audio).read_bytes() == b"completed audio"
    assert video is None


def test_section_has_its_own_budget_after_waiting_for_cookie_lock(monkeypatch, tmp_path):
    spawn = subprocess.Popen
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("CLIPPER_YTDLP_COOKIES", str(cookie_file))
    code = """
import sys
from contextlib import contextmanager
from pathlib import Path
from clipper_studio.download_worker import main
from shorts_clipper import downloader

original = downloader.cookie_options

@contextmanager
def cookies(*args, **kwargs):
    (Path(sys.argv[1]) / "waiting").touch()
    with original(*args, **kwargs) as options:
        yield options

class Stub:
    def __init__(self, options):
        self.path = Path(options["outtmpl"]).parent / "section.mp4"
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def extract_info(self, *args, **kwargs):
        return {"duration": 60}
    def process_ie_result(self, *args, **kwargs):
        self.path.touch()
        return {"requested_downloads": [{"filepath": str(self.path)}]}

downloader.cookie_options = cookies
downloader.yt_dlp.YoutubeDL = Stub
main(Path(sys.argv[1]))
"""
    started_waiting = None

    def child(command, **kwargs):
        nonlocal started_waiting
        process = spawn([sys.executable, "-c", code, command[-1]], **kwargs)
        waiting = Path(command[-1]) / "waiting"
        start = time.monotonic()
        while not waiting.exists() and process.poll() is None and time.monotonic() < start + 5:
            time.sleep(.01)
        assert waiting.exists()
        started_waiting = time.monotonic()
        return process

    monkeypatch.setattr("clipper_studio.download_worker.subprocess.Popen", child)
    with cookie_file.with_suffix(".txt.lock").open("wb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        def release_after_download_budget():
            if time.monotonic() - started_waiting > 1:
                fcntl.flock(lock, fcntl.LOCK_UN)

        result = run_download("section", ["https://youtu.be/video", tmp_path, 5, 30],
                              timeout=1, check=release_after_download_budget)
    assert Path(result) == tmp_path / "section.mp4"


def test_fractional_windows_do_not_reuse_another_sections_file(monkeypatch, tmp_path):
    youtube_dl = yt_dlp.YoutubeDL
    downloads = []

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            return {"id": "video", "duration": 100, "ext": "mp4"}

        def process_ie_result(self, info, download=True):
            section = next(iter(self.options["download_ranges"](info, self)))
            info.update(section_start=section["start_time"], section_end=section["end_time"])
            with youtube_dl({"outtmpl": self.options["outtmpl"], "quiet": True}) as formatter:
                path = Path(formatter.prepare_filename(info))
            if not path.exists():
                path.write_text(repr((info["section_start"], info["section_end"])))
                downloads.append(path)
            return {"requested_downloads": [{"filepath": str(path)}]}

    monkeypatch.setattr("shorts_clipper.downloader.yt_dlp.YoutubeDL", Downloader)
    first = download_section("https://youtu.be/video", tmp_path, 10.2, 20.4)
    second = download_section("https://youtu.be/video", tmp_path, 10.7, 20.9)
    precise = download_section("https://youtu.be/video", tmp_path, 10.20001, 20.40001)
    assert len({first, second, precise}) == 3
    assert first.read_text() == "(10.2, 20.4)"
    assert second.read_text() == "(10.7, 20.9)"
    assert precise.read_text() == "(10.20001, 20.40001)"
    assert download_section("https://youtu.be/video", tmp_path, 10.2, 20.4) == first
    assert len(downloads) == 3


@pytest.mark.parametrize("video_failure", [False, True, "missing", "oversized"])
def test_analysis_download(monkeypatch, tmp_path, video_failure):
    options_seen = []

    class Downloader:
        def __init__(self, options):
            self.options = options
            options_seen.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            return {"id": "video"}

        def process_ie_result(self, info, download=True):
            video = self.options["format"].startswith("bv")
            if video and video_failure is True:
                raise yt_dlp.utils.DownloadError("Requested format is not available")
            if video and video_failure == "missing":
                return {}
            if video and video_failure == "oversized":
                self.options["progress_hooks"][0]({"downloaded_bytes": self.options["max_filesize"] + 1})
            path = tmp_path / ("video.mp4" if video else "audio.m4a")
            path.touch()
            return {"requested_downloads": [{"filepath": str(path)}]}

    monkeypatch.setattr("shorts_clipper.downloader.yt_dlp.YoutubeDL", Downloader)
    audio, video = download_analysis_media("url", tmp_path)
    assert audio.exists()
    assert (video is None) == bool(video_failure)
    assert all("download_ranges" not in options for options in options_seen)
    assert options_seen[0]["format"] == "ba[abr<=80]/ba/wa"
    assert options_seen[1]["format"] == "bv[height<=160]/wv"
    assert all(o["max_filesize"] == 1_500_000_000 for o in options_seen)


def test_analysis_download_errors(monkeypatch, tmp_path):
    with pytest.raises(YouTubeAccessError):
        download_analysis_media("url", tmp_path, proxy="socks5://localhost")

    def fail(*args, **kwargs):
        raise yt_dlp.utils.DownloadError("unavailable")

    monkeypatch.setattr("shorts_clipper.downloader.yt_dlp.YoutubeDL", fail)
    with pytest.raises(yt_dlp.utils.DownloadError):
        download_analysis_media("url", tmp_path)
    monkeypatch.setattr("shorts_clipper.downloader.blocking_message", lambda *args: "blocked")
    with pytest.raises(YouTubeAccessError, match="blocked"):
        download_analysis_media("url", tmp_path)


@pytest.mark.parametrize("limit", ["duration", "bytes"])
def test_analysis_limits(monkeypatch, tmp_path, limit):
    monkeypatch.setenv("CLIPPER_ANALYSIS_MAX_BYTES", "100")
    monkeypatch.setenv("CLIPPER_ANALYSIS_MAX_HOURS", "1")

    class Downloader:
        def __init__(self, options):
            self.options = options
            assert options["max_filesize"] == 100

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, *args, **kwargs):
            return {"duration": 3601 if limit == "duration" else 600}

        def process_ie_result(self, *args, **kwargs):
            assert limit == "bytes", "Overlong replay must not start downloading"
            self.options["progress_hooks"][0]({"downloaded_bytes": 101})

    monkeypatch.setattr("shorts_clipper.downloader.yt_dlp.YoutubeDL", Downloader)
    with pytest.raises(YouTubeAccessError, match="1 hours|100-byte"):
        download_analysis_media("url", tmp_path)


def test_analysis_cancellation_stops_before_video(monkeypatch, tmp_path):
    from clipper_studio.api import WorkCancelled

    downloads = []

    class Downloader:
        def __init__(self, options):
            downloads.append(options["format"])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, *args, **kwargs):
            return {"duration": 600}

        def process_ie_result(self, *args, **kwargs):
            path = tmp_path / "audio.m4a"
            path.touch()
            return {"requested_downloads": [{"filepath": str(path)}]}

    def should_continue():
        if downloads:
            raise WorkCancelled

    monkeypatch.setattr("shorts_clipper.downloader.yt_dlp.YoutubeDL", Downloader)
    with pytest.raises(WorkCancelled):
        download_analysis_media("url", tmp_path, should_continue=should_continue)
    assert downloads == ["ba[abr<=80]/ba/wa"]
