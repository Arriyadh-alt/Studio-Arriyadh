"""Independent checks of the actual MP4 delivered by Studio."""

import json
import subprocess
from fractions import Fraction

import cv2
import numpy as np

from .video import EXPORT_SIZES, HEIGHT, WIDTH


def _fast_start(video):
    with video.open("rb") as media:
        while header := media.read(8):
            if len(header) < 8:
                break
            size, tag = int.from_bytes(header[:4], "big"), header[4:]
            if tag in (b"moov", b"mdat"):
                return tag == b"moov"
            head_size = 8
            if size == 1:
                head_size = 16
                size = int.from_bytes(media.read(8), "big")
            if size < head_size:
                break
            media.seek(size - head_size, 1)
    return False


def _watermark_visible(video, mark, size, duration):
    artwork = cv2.imread(str(mark.path), cv2.IMREAD_UNCHANGED)
    if artwork is None or artwork.ndim != 3 or artwork.shape[2] != 4:
        return False
    width, height = size
    x, y = round(mark.x * width / WIDTH), round(mark.y * height / HEIGHT)
    w = max(2, round(artwork.shape[1] * width / WIDTH))
    h = max(2, round(artwork.shape[0] * height / HEIGHT))
    alpha = cv2.resize(artwork[:, :, 3], (w, h), interpolation=cv2.INTER_AREA)
    template = cv2.Canny(alpha, 12, 24)
    if not template.any():
        return False
    # A small search window tolerates chroma subsampling and subpixel layout scaling.
    left, top = max(0, x - 4), max(0, y - 4)
    right, bottom = min(width, x + w + 4), min(height, y + h + 4)
    roi = (right - left, bottom - top)
    scores = []
    for fraction in (.1, .5, .9):
        frame = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-ss", str(duration * fraction),
                                "-i", str(video), "-frames:v", "1", "-filter_threads", "1",
                                "-vf", f"crop={roi[0]}:{roi[1]}:{left}:{top}:exact=1,format=gray",
                                "-f", "rawvideo", "-"], check=True, capture_output=True, timeout=30).stdout
        if len(frame) != roi[0] * roi[1]:
            return False
        edges = cv2.Canny(np.frombuffer(frame, dtype=np.uint8).reshape(roi[1], roi[0]), 12, 24)
        scores.append(float(cv2.matchTemplate(edges, template, cv2.TM_CCORR_NORMED).max()))
    return min(scores) >= .22


def check_export(video, preset, expected_duration, fps, watermark, *, source_size=None, approved=False):
    """Measure technical conformance separately from editorial approval."""
    checks = []

    def check(name, passed, detail, *, manual=False):
        checks.append({"check": name, "status": "pass" if passed else "review" if manual else "fail",
                       "detail": detail})

    size = EXPORT_SIZES[preset]
    result = {"checks": checks, "preset": preset, "width": None, "height": None, "duration": None,
              "expected_duration": round(expected_duration, 3), "source_size": source_size,
              "editorial_status": "needs_review"}
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-show_streams",
                                "-show_format", "-of", "json", str(video)],
                               check=True, capture_output=True, text=True, timeout=60)
        data = json.loads(probe.stdout)
        picture = next(s for s in data["streams"] if s["codec_type"] == "video")
        sound = next(s for s in data["streams"] if s["codec_type"] == "audio")
        duration = float(data["format"]["duration"])
        width, height = picture["width"], picture["height"]
        result.update(width=width, height=height, duration=duration)
        check("Resolution", (width, height) == size and picture.get("sample_aspect_ratio") == "1:1",
              f"{width}×{height}; preset requires {size[0]}×{size[1]} with square pixels")
        frame_seconds = 1 / float(Fraction(fps))
        check("Duration", abs(duration - expected_duration) <= max(.08, frame_seconds * 1.5),
              f"{duration:.3f}s vs {expected_duration:.3f}s {'approved timeline' if approved else 'planned edit'}")
        check("MP4 encoding", picture["codec_name"] == "h264" and picture["pix_fmt"] == "yuv420p"
              and sound["codec_name"] == "aac" and sound["sample_rate"] == "48000"
              and abs(float(Fraction(picture["r_frame_rate"]) - Fraction(fps))) < .01
              and all(picture.get(key) == "bt709" for key in ("color_space", "color_transfer", "color_primaries"))
              and "mp4" in data["format"]["format_name"].split(",") and _fast_start(video),
              f"Preset requires H.264, AAC 48 kHz, {fps} fps, YUV420P, BT.709 and fast start")
        start_delta = abs(float(picture["start_time"]) - float(sound["start_time"]))
        end_delta = abs(float(picture["duration"]) + float(picture["start_time"])
                        - float(sound["duration"]) - float(sound["start_time"]))
        check("Audio alignment", max(start_delta, end_delta) <= max(.12, frame_seconds * 2),
              f"Stream starts differ by {start_delta:.3f}s and ends by {end_delta:.3f}s; confirm lip sync in review")
        packets_ok = not probe.stderr and all(int(track["nb_read_packets"]) > 0 for track in (picture, sound))
        if packets_ok:
            decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-threads", "2", "-i", str(video),
                                      "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                                     capture_output=True, timeout=120)
        integrity = packets_ok and decoded.returncode == 0
        check("File integrity", integrity, "Video and audio packets were scanned and decoded" if integrity
              else "Video or audio could not be decoded end to end")
    except (StopIteration, KeyError, OSError, ValueError, ZeroDivisionError, subprocess.SubprocessError):
        check("File integrity", False, "Could not read a complete MP4 with video and audio")
    if result["width"] == size[0] and result["height"] == size[1]:
        try:
            visible = watermark is not None and _watermark_visible(video, watermark, size, result["duration"])
        except (cv2.error, OSError, subprocess.SubprocessError):
            visible = False
        check("Watermark", visible, "Branding pattern sampled at the start, middle and end" if visible
              else "Could not verify visible branding throughout; check the preview", manual=True)
    if source_size and preset != "preview":
        check("Source detail", source_size[1] >= size[1],
              f"Original section: {source_size[0]}×{source_size[1]}; a smaller original cannot supply native vertical detail",
              manual=True)
    result["status"] = ("failed" if any(c["status"] == "fail" for c in checks)
                        else "needs_review" if any(c["status"] == "review" for c in checks) else "passed")
    return result


def editorial_check(authenticity, director_result, tracking_result):
    """Retain automated evidence without claiming the footage is ready to publish."""
    evidence = authenticity["checks"]
    director = {c["check"]: c for c in director_result.get("quality", [])}
    return [
        {"check": "Moment and trigger", "status": "review",
         "detail": (director.get("Reaction trigger") or {}).get("detail") or
         ("Tracking flagged the event for review." if tracking_result.get("needs_review")
          else "Play the setup, event and reaction to verify the trigger is visible.")},
        {"check": "Hook", **evidence["hook"]},
        {"check": "Story completeness", **evidence["context"]},
        {"check": "Subtitle accuracy", **evidence["subtitles"]},
        {"check": "Framing", "status": "review",
         "detail": (director.get("Framing") or {}).get("detail") or "Check subject visibility and overlay placement."},
        {"check": "Natural ending", "status": "review", "detail": "Play the payoff and listen for an abrupt cut."},
    ]
