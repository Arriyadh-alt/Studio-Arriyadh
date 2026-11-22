from pathlib import Path

import yt_dlp

from shorts_clipper.downloader import download_section


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
