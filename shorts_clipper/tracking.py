"""Reaction-aware tracking: find what the streamer reacts to and keep it in the vertical frame.

The original 16:9 footage is sampled a few times per second around the highlight. Each frame gets an
importance map (motion that is not camera shake plus visual saliency, with the facecam and static HUD
overlays suppressed). From those maps we backtrack from the reaction to the visual trigger, track the
subject, and plan a smooth 9:16 camera path that keeps the trigger in frame."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .facecam import FacecamOverlay, OutputRect, SourceBox, _saliency_map, layout_facecam

FPS = 4
SAMPLE_WIDTH = 320
LAYOUTS = ("gameplay", "reaction", "split")
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1080, 1920
LEAD, EXTENDED_LEAD = 15.0, 25.0
MAX_SPEED = 0.25  # fraction of the source width per second
LOOKAHEAD = 0.5  # seconds the camera anticipates the subject


@dataclass
class Footage:
    times: np.ndarray
    frames: np.ndarray  # (n, h, w) uint8 grayscale

    @property
    def aspect(self) -> float:
        return self.frames.shape[2] / self.frames.shape[1]

    def index(self, t: float) -> int:
        return int(np.argmin(np.abs(self.times - t)))


@dataclass
class Track:
    """Per-sample subject position (normalised source coordinates), spread and activity."""
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    spread: np.ndarray
    activity: np.ndarray

    def at(self, t: float) -> tuple[float, float]:
        return float(np.interp(t, self.times, self.x)), float(np.interp(t, self.times, self.y))


def sample_footage(source: Path, start: float, end: float, fps: int = FPS, width: int = SAMPLE_WIDTH) -> Footage | None:
    capture = cv2.VideoCapture(str(source))
    source_w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    source_h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()
    if not source_w or not source_h or end <= start:
        return None
    height = max(2, int(round(width * source_h / source_w / 2)) * 2)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(source),
               "-t", f"{end - start:.3f}", "-an", "-vf", f"fps={fps},scale={width}:{height},format=gray",
               "-f", "rawvideo", "-"]
    try:
        data = subprocess.run(command, capture_output=True, check=True, timeout=600).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    count = len(data) // (width * height)
    if count < 2:
        return None
    frames = np.frombuffer(data[:count * width * height], np.uint8).reshape(count, height, width)
    return Footage(start + np.arange(count) / fps, frames)


def _mask(shape, exclude: SourceBox | None, margin=0.03) -> np.ndarray:
    h, w = shape
    mask = np.ones((h, w), np.float32)
    if exclude is not None:
        x0, y0 = int(max(0, exclude.x - margin) * w), int(max(0, exclude.y - margin) * h)
        x1 = int(min(1, exclude.x + exclude.w + margin) * w)
        y1 = int(min(1, exclude.y + exclude.h + margin) * h)
        mask[y0:y1, x0:x1] = 0
    return mask


def importance_maps(footage: Footage, exclude: SourceBox | None = None) -> np.ndarray:
    frames = footage.frames.astype(np.float32)
    mask = _mask(frames.shape[1:], exclude)
    # HUD, minimap and logos barely change over the clip; damp them without erasing static scenes.
    variation = cv2.GaussianBlur(frames.std(axis=0), (0, 0), 2)
    mask *= 0.3 + 0.7 * np.clip(variation / max(float(np.percentile(variation, 60)), 1.0), 0, 1)
    maps, previous = [], frames[0]
    for frame in frames:
        motion = cv2.GaussianBlur(np.abs(frame - previous), (0, 0), 3)
        motion = np.maximum(motion - np.median(motion), 0)  # remove global camera motion
        motion /= max(float(motion.max()), 8.0)
        previous = frame
        maps.append((0.55 * motion + 0.45 * _saliency_map(frame.astype(np.uint8))) * mask)
    return np.array(maps, np.float32)


def track_subject(footage: Footage, maps: np.ndarray) -> Track:
    h, w = maps.shape[1:]
    ys, xs = np.mgrid[0:h, 0:w]
    xs, ys = (xs + 0.5) / w, (ys + 0.5) / h
    rows = []
    for m in maps:
        weight = np.where(m >= np.percentile(m, 97), m, 0)
        total = float(weight.sum())
        if total <= 1e-6:
            rows.append((0.5, 0.5, 0.0, 0.0))
            continue
        cx, cy = float((weight * xs).sum() / total), float((weight * ys).sum() / total)
        spread = float(np.sqrt((weight * (xs - cx) ** 2).sum() / total))
        rows.append((cx, cy, spread, float(m.mean())))
    x, y, spread, activity = (np.array(c) for c in zip(*rows))
    return Track(footage.times, x, y, spread, activity)


def follow_point(footage: Footage, t: float, x: float, y: float, base: Track | None = None) -> Track:
    """Track a manually chosen object/character from one frame forwards and backwards."""
    frames = footage.frames
    n, h, w = frames.shape
    size = max(8, int(0.14 * h))
    start = footage.index(t)
    positions = {start: (x, y)}
    for step in (1, -1):
        px, py = x * w, y * h
        template = None
        for i in range(start, n if step > 0 else -1, step):
            frame = frames[i]
            if template is not None:
                rx0, ry0 = int(max(0, px - size * 2)), int(max(0, py - size * 2))
                rx1, ry1 = int(min(w, px + size * 2)), int(min(h, py + size * 2))
                region = frame[ry0:ry1, rx0:rx1]
                if region.shape[0] > template.shape[0] and region.shape[1] > template.shape[1]:
                    scores = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
                    _, score, _, loc = cv2.minMaxLoc(scores)
                    if score >= 0.4:
                        px = rx0 + loc[0] + template.shape[1] / 2
                        py = ry0 + loc[1] + template.shape[0] / 2
                positions[i] = (px / w, py / h)
            x0, y0 = int(np.clip(px - size / 2, 0, w - size)), int(np.clip(py - size / 2, 0, h - size))
            template = frame[y0:y0 + size, x0:x0 + size].copy()
    x_track = np.array([positions[i][0] for i in range(n)])
    y_track = np.array([positions[i][1] for i in range(n)])
    zeros = np.zeros(n)
    return Track(footage.times, x_track, y_track, zeros,
                 base.activity if base is not None else zeros)


def _normalise(values: np.ndarray) -> np.ndarray:
    span = float(np.percentile(values, 98) - np.min(values)) if len(values) else 0.0
    return np.clip((values - np.min(values)) / span, 0, 1) if span > 1e-9 else np.zeros_like(values)


def find_trigger(footage: Footage, maps: np.ndarray, track: Track, reaction: float, end: float | None = None,
                 voice: tuple | None = None) -> dict | None:
    """Backtrack from the strongest reaction to the visual event that caused it.

    Returns source times for when the trigger appeared, when the streamer noticed it, the reaction and
    the natural end, plus where in the frame the trigger is."""
    times = footage.times
    if len(times) < 3 or reaction <= times[0]:
        return None
    hists = [cv2.calcHist([f], [0], None, [32], [0, 256]).ravel() for f in footage.frames]
    hists = [hh / max(float(hh.sum()), 1.0) for hh in hists]
    scene = np.array([0.0] + [0.5 * float(np.abs(a - b).sum()) for a, b in zip(hists, hists[1:])])
    activity = track.activity
    window = int(2 * FPS)
    rising = np.array([max(0.0, activity[i] - activity[max(0, i - window):i].mean()) if i else 0.0
                       for i in range(len(activity))])
    novelty = 0.45 * _normalise(scene) + 0.55 * _normalise(rising)
    novelty = np.convolve(novelty, np.ones(3) / 3, mode="same")

    def best(lead):
        lo = int(np.searchsorted(times, reaction - lead))
        hi = int(np.searchsorted(times, reaction - 0.5))
        if hi - lo < 2:
            return None
        return lo, lo + int(np.argmax(novelty[lo:hi]))

    found = best(LEAD)
    if found is None or (novelty[found[1]] < 0.35 and reaction - EXTENDED_LEAD < times[0] + LEAD):
        found = best(EXTENDED_LEAD) or found
    if found is None:
        return None
    lo, peak = found
    onset = peak
    while onset > lo and novelty[onset - 1] >= 0.5 * novelty[peak]:
        onset -= 1
    trigger, event = float(times[onset]), float(times[peak])
    notice = trigger + 0.35 * (reaction - trigger)
    if voice is not None and len(voice[0]):
        vt, vv = np.asarray(voice[0], float), np.asarray(voice[1], float)
        inside = (vt >= trigger) & (vt <= reaction)
        if inside.any():
            threshold = float(vv[inside].mean() + 0.5 * vv[inside].std())
            loud = vt[inside][vv[inside] > threshold]
            if len(loud):
                notice = float(loud[0])
    m = maps[peak]
    hot = m >= np.percentile(m, 97)
    rows, cols = np.nonzero(hot)
    h, w = m.shape
    box = [float(cols.min() / w), float(rows.min() / h), float((cols.max() - cols.min() + 1) / w),
           float((rows.max() - rows.min() + 1) / h)] if len(rows) else [0.4, 0.4, 0.2, 0.2]
    return {"trigger": round(trigger, 2), "event": round(event, 2), "notice": round(min(notice, reaction), 2),
            "reaction": round(float(reaction), 2),
            "end": round(float(end if end is not None else min(times[-1], reaction + 3.0)), 2),
            "confidence": round(float(novelty[peak]), 3),
            "target": [round(float(track.x[peak]), 4), round(float(track.y[peak]), 4)],
            "box": [round(v, 4) for v in box]}


def window_width(layout_height: int, aspect: float) -> float:
    """Fraction of the source width visible in a full-height crop filling OUTPUT_WIDTH x layout_height."""
    return float(min(1.0, (OUTPUT_WIDTH / layout_height) / aspect))


def camera_path(track: Track, width: float, fixed: float | None = None, deadzone: float | None = None,
                hold: tuple | None = None) -> list[tuple[float, float, float]]:
    """Smooth (t, x, y) keyframes in source time. `hold` = (start, end, x) pins the camera on a trigger."""
    times = track.times
    half = width / 2
    if fixed is not None:
        x = np.full(len(times), float(np.clip(fixed, half, 1 - half)))
    else:
        shift = int(LOOKAHEAD * FPS)
        target = np.clip(np.concatenate([track.x[shift:], np.repeat(track.x[-1:], shift)]), half, 1 - half)
        zones = np.full(len(times), width * 0.18 if deadzone is None else deadzone)
        if hold is not None:
            # Arrive on the trigger by the time it appears: start the move early enough for the speed cap.
            pinned = float(np.clip(hold[2], half, 1 - half))
            before = target[times < hold[0]]
            lead = 1.0 + (abs(pinned - float(before[-1])) / MAX_SPEED if len(before) else 0.0)
            inside = (times >= hold[0] - lead) & (times <= hold[1])
            target[inside], zones[inside] = pinned, 0.0
        x, position, limit = [], float(target[0]), MAX_SPEED / FPS
        for value, zone in zip(target, zones):
            gap = value - position
            if abs(gap) > zone:
                position += float(np.clip(gap - np.sign(gap) * zone, -limit, limit))
            x.append(position)
        x = np.array(x)
        sigma = FPS * 0.6
        kernel = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        pad = len(kernel) // 2
        x = np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")[:len(times)]
        x = np.clip(x, half, 1 - half)
    y = np.clip(np.convolve(np.pad(track.y, 4, mode="edge"), np.ones(9) / 9, mode="valid")[:len(times)], 0, 1)
    return [(float(t), float(a), float(b)) for t, a, b in zip(times, x, y)]


def _camera_at(path, t):
    times = [p[0] for p in path]
    return float(np.interp(t, times, [p[1] for p in path]))


def choose_layout(track: Track, trigger: dict | None, has_facecam: bool, aspect: float) -> str:
    """Gameplay with a corner facecam by default; split is only a fallback when that framing fails validation."""
    if not has_facecam:
        return "gameplay"
    if trigger is None or trigger["confidence"] < 0.2:
        return "reaction"  # nothing clear on screen: the streamer's face carries the moment
    return "gameplay"


def validate(track: Track, path, width: float, trigger: dict | None, clip: tuple[float, float],
             cam_rect: OutputRect | None = None, main_top: int = 0, main_height: int = OUTPUT_HEIGHT) -> dict:
    """Check the planned framing before rendering. Times are source times."""
    def visible(x, t, margin=0.02):
        centre = _camera_at(path, t)
        return abs(x - centre) <= width / 2 - margin

    def covered(x, y, t):
        if cam_rect is None:
            return False
        left = _camera_at(path, t) - width / 2
        ox, oy = (x - left) / width * OUTPUT_WIDTH, main_top + y * main_height
        return cam_rect.x <= ox <= cam_rect.x + cam_rect.w and cam_rect.y <= oy <= cam_rect.y + cam_rect.h

    inside = (track.times >= clip[0]) & (track.times <= clip[1])
    times, xs, ys = track.times[inside], track.x[inside], track.y[inside]
    subject = float(np.mean([visible(x, t) for x, t in zip(xs, times)])) if len(times) else 1.0
    offset = float(np.mean([abs(x - _camera_at(path, t)) for x, t in zip(xs, times)]) / (width / 2)) if len(times) else 0.0
    checks = {"subject_in_frame": subject >= 0.75, "camera_follows": offset <= 0.85}
    if trigger is not None:
        tx, ty = trigger["target"]
        span = [t for t in times if trigger["trigger"] <= t <= trigger["reaction"]] or [trigger["event"]]
        seen = float(np.mean([visible(tx if abs(t - trigger["event"]) < 0.5 else track.at(t)[0], t) for t in span]))
        checks["trigger_visible"] = seen >= 0.7 and visible(tx, trigger["event"])
        checks["facecam_clear"] = not covered(tx, ty, trigger["event"])
        checks["reaction_understandable"] = trigger["notice"] - trigger["trigger"] <= 12 and trigger["confidence"] >= 0.15
        checks["story_complete"] = clip[0] <= trigger["trigger"] + 0.05 and clip[1] >= min(trigger["end"], trigger["reaction"] + 1.5)
    else:
        checks["trigger_visible"] = False
    return {"checks": checks, "passed": all(checks.values()), "subject_visible": round(subject, 3)}


def to_output_time(path, segments) -> list[tuple[float, float, float]]:
    """Map source-time keyframes onto the edited timeline (segments with .start/.end/.role)."""
    if not segments:
        return list(path)
    times = [p[0] for p in path]
    result, offset = [], 0.0
    for segment in segments:
        length = segment.end - segment.start
        steps = max(2, int(length * 2) + 1)
        for k in range(steps):
            local = length * k / (steps - 1)
            source_t = segment.start if segment.role == "intro" else segment.start + local
            result.append((offset + local, float(np.interp(source_t, times, [p[1] for p in path])),
                           float(np.interp(source_t, times, [p[2] for p in path]))))
        offset += length
    return result


def keyframe_expression(keys, index: int = 1, epsilon: float = 0.002) -> str | None:
    """Piecewise-linear ffmpeg expression of `t` through (t, value...) keyframes."""
    from .director import _keyframes
    return _keyframes(keys, index, "t", epsilon) if keys else None


def split_band(box: SourceBox, aspect: float) -> int:
    """Height of the facecam band in the split-screen layout (camera on top, wider gameplay below)."""
    camera_aspect = box.w * aspect / box.h
    return int(np.clip(OUTPUT_WIDTH / camera_aspect, 560, 820)) // 2 * 2


def is_split(overlay: FacecamOverlay | None) -> bool:
    return overlay is not None and overlay.rect.x == 0 and overlay.rect.y == 0 and overlay.rect.w == OUTPUT_WIDTH


def layout_options(overlay: FacecamOverlay | None, source_size: tuple[int, int]) -> dict:
    """layout -> (facecam overlay, main picture top, main picture height)."""
    if overlay is None:
        return {"gameplay": (None, 0, OUTPUT_HEIGHT)}
    rect = overlay.rect
    corner = ("top" if rect.y == 24 else "bottom") + ("-left" if rect.x == 24 else "-right")
    big = layout_facecam(overlay.box, source_size, corner, 0.6)
    band = split_band(overlay.box, source_size[0] / source_size[1])
    return {"gameplay": (overlay, 0, OUTPUT_HEIGHT), "reaction": (big, 0, OUTPUT_HEIGHT),
            "split": (FacecamOverlay(overlay.box, OutputRect(0, 0, OUTPUT_WIDTH, band)), band, OUTPUT_HEIGHT - band)}


@dataclass
class Reaction:
    footage: Footage
    track: Track
    trigger: dict | None
    maps: np.ndarray


def analyze_reaction(source: Path, duration: float, reaction: float | None, exclude: SourceBox | None = None,
                     voice: tuple | None = None) -> Reaction | None:
    footage = sample_footage(source, 0, duration)
    if footage is None:
        return None
    maps = importance_maps(footage, exclude)
    track = track_subject(footage, maps)
    trigger = find_trigger(footage, maps, track, reaction, voice=voice) if reaction is not None else None
    return Reaction(footage, track, trigger, maps)


def plan_reframe(reaction: Reaction, clip: tuple[float, float], layouts: dict, layout: str = "auto",
                 follow: str = "auto", x: float | None = None, target: dict | None = None) -> dict:
    """Pick a layout and camera path, validate it, and retry with stronger framing when the trigger is lost."""
    footage, track, trigger = reaction.footage, reaction.track, reaction.trigger
    if target is not None:
        track = follow_point(footage, target["t"], target["x"], target["y"], track)
        if trigger is not None:
            tx, ty = track.at(trigger["event"])
            trigger = {**trigger, "target": [round(tx, 4), round(ty, 4)]}
    automatic = layout not in layouts
    first = choose_layout(track, trigger, len(layouts) > 1, footage.aspect) if automatic else layout
    attempts = [(first, False), (first, True)]
    if automatic and "split" in layouts and first != "split":
        attempts.append(("split", True))
    if follow == "fixed":
        attempts = attempts[:1]
    best = None
    for number, (name, hold) in enumerate(attempts, 1):
        overlay, top, height = layouts[name]
        width = window_width(height, footage.aspect)
        pin = (trigger["trigger"], trigger["reaction"] + 0.5, trigger["target"][0]) if hold and trigger else None
        path = camera_path(track, width, fixed=x if follow == "fixed" else None, hold=pin)
        report = validate(track, path, width, trigger, clip, overlay.rect if overlay else None, top, height)
        checks = report["checks"]
        rank = (report["passed"], checks.get("trigger_visible", False), checks.get("facecam_clear", True),
                sum(checks.values()))
        if best is None or rank > best[0]:
            best = (rank, {"layout": name, "path": path, "width": width, "validation": report,
                           "overlay": overlay, "attempt": number})
        if report["passed"]:
            break
    result = best[1]
    checks = result["validation"]["checks"]
    result.update(attempts=min(number, len(attempts)), auto_layout=automatic, trigger=trigger,
                  needs_review=not (checks.get("trigger_visible") and checks.get("facecam_clear", True)),
                  target=[round(v, 4) for v in (track.at(trigger["event"]) if trigger else track.at(clip[0]))])
    return result
