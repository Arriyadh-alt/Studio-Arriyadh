"""Bound download work and its child processes to the Studio job lifetime."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from yt_dlp.utils import DownloadError

from shorts_clipper.youtube import YouTubeAccessError


def run_download(operation, args, kwargs=None, *, timeout, check=lambda: None):
    deadline = time.monotonic() + timeout
    with TemporaryDirectory(prefix="clipper-download-") as directory:
        request, result = Path(directory) / "request.json", Path(directory) / "result.json"
        request.write_text(json.dumps([operation, args, kwargs or {}], default=os.fspath))
        with subprocess.Popen([sys.executable, "-m", "clipper_studio.download_worker", directory],
                              start_new_session=True,
                              env={**os.environ, "TMPDIR": directory, "CLIPPER_DATA_DIR": directory}) as process:
            try:
                while process.poll() is None:
                    check()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("YouTube download exceeded its time limit.")
                    time.sleep(.2)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise
            if not result.is_file():
                raise DownloadError("The YouTube download worker exited without a result.")
            reply = json.loads(result.read_text())
            if "error" in reply:
                from .chat import ChatReplayError
                error = {"ChatReplayError": ChatReplayError, "YouTubeAccessError": YouTubeAccessError,
                                "ValueError": ValueError, "TimeoutError": TimeoutError}.get(reply["kind"], DownloadError)
                raise error(reply["error"])
            return reply["value"]


def main(directory):
    from yt_dlp import YoutubeDL
    from shorts_clipper.downloader import download_analysis_media, download_section
    from .chat import _download_replay, fetch_metadata

    operation, args, kwargs = json.loads((directory / "request.json").read_text())
    try:
        if operation == "replay":
            value = _download_replay(args[0], Path(args[1]))
        elif operation == "metadata":
            value = YoutubeDL.sanitize_info(fetch_metadata(*args, **kwargs))
        elif operation == "section":
            value = download_section(args[0], Path(args[1]), *args[2:], **kwargs)
        elif operation == "media":
            if kwargs.get("prior_audio"):
                kwargs["prior_audio"] = Path(kwargs["prior_audio"])
            value = download_analysis_media(args[0], Path(args[1]), **kwargs)
        else:
            raise ValueError("Unknown download operation.")
        reply = {"value": value}
    except Exception as error:
        reply = {"kind": type(error).__name__, "error": str(error)}
    (directory / "result.json").write_text(json.dumps(reply, default=os.fspath))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
