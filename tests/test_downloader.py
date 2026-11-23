from pathlib import Path

import pytest
import yt_dlp

from shorts_clipper.downloader import download_analysis_media, download_section
from shorts_clipper.youtube import YouTubeAccessError


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


@pytest.mark.parametrize("video_failure", [False, True, "missing"])
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
