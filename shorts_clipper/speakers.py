"""Active-speaker framing for podcasts: find each participant's seat, match voices to faces and plan the camera.

Layouts: `speaker` (single speaker focus), `split` (two-person split screen), `switch` (dynamic speaker
switching with reaction cutaways) and `wide` (full conversation view on a blurred background).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .facecam import FacecamOverlay, OutputRect, SourceBox
from .moments import _frames
from .tracking import OUTPUT_HEIGHT, OUTPUT_WIDTH, window_width

LAYOUTS = ("auto", "speaker", "split", "switch", "wide")
SAMPLE_FPS = 5
SAMPLE_WIDTH = 320
MIN_SHOT = 2.5  # seconds on one speaker before the camera may move again
TAKEOVER = 1.0  # a new voice must hold the floor this long to earn a cut
EASE = 0.2
SPLIT_BAND = 960


def sample_faces(source: Path, duration: float, fps=SAMPLE_FPS):
    """Grey thumbnails and frontal-face boxes (fractions of the frame) at `fps`."""
    capture = cv2.VideoCapture(str(source))
    try:
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    finally:
        capture.release()
    if not width or not height:
        return None
    h = max(2, int(round(SAMPLE_WIDTH * height / width / 2)) * 2)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frames, faces = [], []
    options = ["-an", "-vf", f"fps={fps},scale={SAMPLE_WIDTH}:{h},format=gray", "-f", "rawvideo"]
    for data in _frames(source, options, SAMPLE_WIDTH * h, duration):
        if len(data) < SAMPLE_WIDTH * h:
            break
        frame = np.frombuffer(data, dtype=np.uint8).reshape(h, SAMPLE_WIDTH)
        found = cascade.detectMultiScale(frame, 1.1, 4, minSize=(max(12, h // 14),) * 2)
        frames.append(frame)
        faces.append([(x / SAMPLE_WIDTH, y / h, w / SAMPLE_WIDTH, fh / h) for x, y, w, fh in found])
    if not frames:
        return None
    return {"times": np.arange(len(frames)) / fps, "frames": np.asarray(frames), "faces": faces,
            "aspect": width / height}


def find_seats(faces, minimum=0.15):
    """Group face detections by horizontal position into up to three seats, left to right."""
    points = sorted((x + w / 2, y + h / 2, w, h) for frame in faces for x, y, w, h in frame)
    if not points:
        return []
    groups, current = [], [points[0]]
    for point in points[1:]:
        if point[0] - current[-1][0] > max(0.08, 0.6 * float(np.median([p[2] for p in current]))):
            groups.append(current)
            current = []
        current.append(point)
    groups.append(current)
    seats = [{"x": float(np.median([p[0] for p in g])), "y": float(np.median([p[1] for p in g])),
              "w": float(np.median([p[2] for p in g])), "h": float(np.median([p[3] for p in g])),
              "presence": len(g) / max(1, len(faces))} for g in groups]
    seats = [s for s in seats if s["presence"] >= minimum]
    return sorted(sorted(seats, key=lambda s: -s["presence"])[:3], key=lambda s: s["x"])


def seat_motion(frames, seats):
    """Mouth-and-jaw motion per seat (lower half of the face box), robustly normalised."""
    if len(frames) < 2:
        return np.zeros((len(seats), len(frames)))
    h, w = frames.shape[1:]
    result = []
    for seat in seats:
        x0, x1 = int((seat["x"] - seat["w"] / 2) * w), int((seat["x"] + seat["w"] / 2) * w)
        y0, y1 = int(seat["y"] * h), int((seat["y"] + seat["h"] * 0.7) * h)
        region = frames[:, max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)].astype(np.float32)
        diff = np.abs(np.diff(region, axis=0)).mean(axis=(1, 2))
        diff = np.concatenate([[diff[0]], diff])
        median = np.median(diff)
        mad = np.median(np.abs(diff - median)) + 1e-3
        result.append((diff - median) / (1.4826 * mad))
    return np.asarray(result)


def activity(turns, times, speakers):
    grid = np.zeros((speakers, len(times)))
    for start, end, speaker in turns:
        if speaker < speakers:
            grid[speaker, (times >= start) & (times <= end)] = 1
    return grid


def match_speakers(turns, times, motion, seats):
    """speaker id -> seat index, by correlating each voice's talk time with each face's mouth motion."""
    speakers = sorted({t[2] for t in turns})
    if not seats or not speakers:
        return {}, 0.0
    voices = activity(turns, times, max(speakers) + 1)
    scores = np.zeros((len(speakers), len(seats)))
    for i, s in enumerate(speakers):
        for j in range(len(seats)):
            if voices[s].std() > 0 and motion.shape[1] and motion[j].std() > 0:
                scores[i, j] = float(np.corrcoef(voices[s], motion[j])[0, 1])
    mapping, confidence = {}, []
    free_speakers, free_seats = set(range(len(speakers))), set(range(len(seats)))
    while free_speakers and free_seats:
        i, j = max(((i, j) for i in free_speakers for j in free_seats), key=lambda p: scores[p])
        if scores[i, j] < 0.05:
            break
        mapping[speakers[i]] = j
        confidence.append(scores[i, j])
        free_speakers.discard(i)
        free_seats.discard(j)
    for i in sorted(free_speakers):  # unmatched voices take the remaining seats left to right
        if free_seats:
            j = min(free_seats)
            mapping[speakers[i]] = j
            free_seats.discard(j)
        else:
            mapping[speakers[i]] = int(np.argmax(scores[i])) if len(seats) else 0
    return mapping, float(np.mean(confidence)) if confidence else 0.0


def plan_shots(turns, duration, mapping, motion=None, times=None, step=0.1):
    """Speaker shots with hysteresis: hold each shot at least MIN_SHOT and only cut when a new voice holds the
    floor; a listener's strong reaction during a long shot earns one short cutaway."""
    if not mapping:
        return []
    grid = np.arange(0, duration, step)
    active = np.full(len(grid), -1)
    for start, end, speaker in sorted(turns):
        active[(grid >= start) & (grid < end)] = mapping.get(speaker, -1)
    current = int(active[active >= 0][0]) if (active >= 0).any() else min(mapping.values())
    shots, began, candidate, since = [], 0.0, None, 0.0
    for t, seat in zip(grid, active):
        if seat < 0 or seat == current:
            candidate = None
            continue
        if candidate != seat:
            candidate, since = int(seat), float(t)
        if t - since >= TAKEOVER and since - began >= MIN_SHOT * 0.5:
            cut = max(since, began + MIN_SHOT)
            if cut < duration - 1.0:
                shots.append([began, cut, current])
                began, current, candidate = cut, int(seat), None
    shots.append([began, duration, current])
    cutaways = 0
    if motion is not None and times is not None and motion.size:
        result = []
        last = -10.0
        for start, end, seat in shots:
            pieces, cursor = [], start
            if end - start > 7.0:
                for other in range(len(motion)):
                    if other == seat:
                        continue
                    strong = (times > start + 1.5) & (times < end - 3.0) & (motion[other] > 3.0)
                    hits = times[strong]
                    for t in hits:
                        sustained = ((times >= t) & (times <= t + 0.5) & (motion[other] > 2.0)).sum() >= 3
                        if sustained and t - last >= 10.0 and t > cursor + MIN_SHOT:
                            pieces += [[cursor, float(t), seat], [float(t), float(t) + 1.5, other]]
                            cursor, last = float(t) + 1.5, float(t)
                            cutaways += 1
                            break
            pieces.append([cursor, end, seat])
            result += [p for p in pieces if p[1] - p[0] > 0.05]
        shots = result
    return shots, cutaways


def _x(seat, width):
    half = width / 2
    return float(np.clip(seat["x"], half, 1 - half))


def shot_path(shots, seats, width):
    if not shots:
        return []
    keys = [(0.0, _x(seats[shots[0][2]], width), seats[shots[0][2]]["y"])]
    for (_, _, before), (t, _, after) in zip(shots, shots[1:]):
        if before == after:
            continue
        a, b = seats[before], seats[after]
        keys += [(max(keys[-1][0] + 0.01, t - EASE), _x(a, width), a["y"]), (t + EASE, _x(b, width), b["y"])]
    end = shots[-1][1]
    last = seats[shots[-1][2]]
    keys.append((max(keys[-1][0] + 0.01, end), _x(last, width), last["y"]))
    return keys


def split_box(seat, aspect, band=SPLIT_BAND):
    """Source box around a seat with the output band's aspect, so the second speaker fills the top half."""
    target = OUTPUT_WIDTH / band
    h = min(1.0, max(seat["h"] * 3.0, 0.35))
    w = target * h / aspect
    if w > 1.0:
        w, h = 1.0, aspect / target
    x = float(np.clip(seat["x"] - w / 2, 0, 1 - w))
    y = float(np.clip(seat["y"] - h * 0.4, 0, 1 - h))
    return SourceBox(x, y, w, h)


def frame_conversation(source: Path, duration: float, turns, layout="auto", sample=None) -> dict | None:
    """Plan podcast framing for a clip; turns are clip-relative (start, end, speaker). None when no faces."""
    sample = sample or sample_faces(source, duration)
    if sample is None:
        return None
    if layout == "wide":
        return {"layout": "wide", "mode": "blur", "path": None, "overlay": None, "seats": 0, "shots": [],
                "switches": 0, "cutaways": 0, "needs_review": False, "auto_layout": False}
    seats = find_seats(sample["faces"])
    if not seats:
        return None
    motion = seat_motion(sample["frames"], seats)
    mapping, confidence = match_speakers(turns, sample["times"], motion, seats)
    talk = {}
    for start, end, speaker in turns:
        talk[speaker] = talk.get(speaker, 0) + max(0.0, min(end, duration) - max(start, 0))
    order = [mapping[s] for s in sorted(talk, key=lambda s: -talk[s]) if s in mapping]
    order += [i for i in range(len(seats)) if i not in order]
    automatic = layout not in LAYOUTS[1:]
    aspect = sample["aspect"]
    full = window_width(OUTPUT_HEIGHT, aspect)
    if automatic:
        spread = seats[-1]["x"] + seats[-1]["w"] / 2 - (seats[0]["x"] - seats[0]["w"] / 2)
        layout = "speaker" if len(seats) == 1 or spread <= full * 0.9 else "switch"
    if layout == "split" and len(seats) < 2:
        layout = "speaker"
    overlay, shots, cutaways = None, [], 0
    if layout == "speaker":
        width = full
        if len(seats) > 1 and seats[-1]["x"] - seats[0]["x"] <= full * 0.7:
            centre = {"x": float(np.mean([s["x"] for s in seats])), "y": float(np.mean([s["y"] for s in seats]))}
        else:
            centre = seats[order[0]]
        path = [(0.0, _x(centre, width), centre["y"]), (duration, _x(centre, width), centre["y"])]
        shots = [[0.0, duration, order[0]]]
    elif layout == "split":
        width = window_width(OUTPUT_HEIGHT - SPLIT_BAND, aspect)
        main, top = seats[order[0]], seats[order[1]]
        overlay = FacecamOverlay(split_box(top, aspect), OutputRect(0, 0, OUTPUT_WIDTH, SPLIT_BAND))
        path = [(0.0, _x(main, width), main["y"]), (duration, _x(main, width), main["y"])]
        shots = [[0.0, duration, order[0]]]
    else:
        layout, width = "switch", full
        shots, cutaways = plan_shots(turns, duration, mapping, motion, sample["times"])
        path = shot_path(shots, seats, width)
    speakers = len({t[2] for t in turns})
    return {"layout": layout, "mode": "crop", "path": path, "overlay": overlay, "width": width,
            "seats": len(seats), "speaker_seats": {str(k): v for k, v in mapping.items()},
            "match_confidence": round(confidence, 3), "auto_layout": automatic,
            "shots": [[round(a, 2), round(b, 2), s] for a, b, s in shots],
            "switches": max(0, sum(1 for a, b in zip(shots, shots[1:]) if a[2] != b[2])), "cutaways": cutaways,
            "needs_review": speakers > len(seats) or (speakers > 1 and confidence < 0.05 and layout == "switch")}
