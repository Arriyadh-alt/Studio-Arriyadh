"""Conservative Haar facecam detection and vertical overlay geometry."""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from .captions import MARGIN_BOTTOM, MARGIN_RIGHT


@dataclass(frozen=True)
class SourceBox:
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.w, self.h)):
            raise ValueError('Box coordinates must be finite')
        x, y = min(max(self.x, 0), 1), min(max(self.y, 0), 1)
        w, h = min(self.w, 1-x), min(self.h, 1-y)
        if min(w, h) <= 0:
            raise ValueError('Box must have positive area')
        for name, value in zip(('x', 'y', 'w', 'h'), (x, y, w, h)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class OutputRect:
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self):
        if (any(not isinstance(v, int) for v in (self.x, self.y, self.w, self.h))
                or min(self.x, self.y) < 0 or min(self.w, self.h) < 2
                or self.x+self.w > 1080 or self.y+self.h > 1920):
            raise ValueError('Rectangle must fit inside 1080x1920')


@dataclass(frozen=True)
class FacecamOverlay:
    box: SourceBox
    rect: OutputRect


def _sample_frames(source, start, end):
    if not Path(source).is_file() or not 0 <= start < end:
        return []
    capture = cv2.VideoCapture(str(source))
    frames = []
    try:
        if capture.isOpened():
            for fraction in np.linspace(.08, .92, 8):
                capture.set(cv2.CAP_PROP_POS_MSEC, (start+fraction*(end-start))*1000)
                ok, frame = capture.read()
                if ok:
                    h, w = frame.shape[:2]
                    frames.append(cv2.resize(frame, (640, max(1, round(h*640/w)))))
    finally:
        capture.release()
    return frames


def find_facecam_border(frame: np.ndarray, face: SourceBox) -> SourceBox | None:
    height, width = frame.shape[:2]
    contours, _ = cv2.findContours(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 160),
                                 cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, .02*cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, w, h = cv2.boundingRect(polygon)
        b = SourceBox(x/width, y/height, w/width, h/height)
        ratio = b.w*b.h/(face.w*face.h)
        if (b.x <= face.x and b.y <= face.y and b.x+b.w >= face.x+face.w
                and b.y+b.h >= face.y+face.h and 3 <= ratio <= 30
                and b.w*b.h < .35 and .5 <= w/h <= 2.5):
            candidates.append(b)
    return min(candidates, key=lambda b: b.w*b.h, default=None)


def detect_facecam(source: Path, start: float, end: float) -> SourceBox | None:
    try:
        frames = _sample_frames(source, start, end)
        if len(frames) < 5:
            return None
        frontal = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        profile = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_profileface.xml')
        if frontal.empty():
            return None
        detections = []
        for frame in frames:
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = list(frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)))
            if not profile.empty():
                faces.extend(profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)))
                faces.extend((width-x-w, y, w, h) for x, y, w, h in profile.detectMultiScale(
                    cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)))
            if any(w/width >= .25 for x, y, w, h in faces):
                return None
            detections.append([SourceBox(x/width, y/height, w/width, h/height) for x, y, w, h in faces])
        tracks = []
        for seed in (b for boxes in detections for b in boxes):
            track = []
            for boxes in detections:
                nearby = [b for b in boxes if abs(b.x+b.w/2-seed.x-seed.w/2) < .04
                          and abs(b.y+b.h/2-seed.y-seed.h/2) < .05]
                if nearby:
                    track.append(min(nearby, key=lambda b: abs(b.x-seed.x)+abs(b.y-seed.y)))
            if len(track) > 4:
                tracks.append(track)
        if not tracks:
            return None
        track = max(tracks, key=lambda t: (len(t), np.median([b.w*b.h for b in t])))
        face = SourceBox(*np.median([[b.x, b.y, b.w, b.h] for b in track], axis=0))
        borders = [find_facecam_border(frame, face) for frame in frames]
        borders = [b for b in borders if b is not None]
        if len(borders) > len(frames)/2:
            return SourceBox(*np.median([[b.x, b.y, b.w, b.h] for b in borders], axis=0))
        x, y = max(0, face.x-face.w), max(0, face.y-face.h*.65)
        return SourceBox(x, y, min(1, face.x+face.w*2)-x, min(1, face.y+face.h*2.7)-y)
    except Exception:
        return None


def layout_facecam(box: SourceBox, source_size: tuple[int, int], position: str = 'top-left',
                   width: float = .42) -> FacecamOverlay:
    if position not in ('top-left', 'top-right', 'bottom-left', 'bottom-right'):
        raise ValueError('Unknown facecam corner')
    if not math.isfinite(width) or min(source_size) <= 0:
        raise ValueError('Invalid facecam size')
    aspect = box.w*source_size[0]/(box.h*source_size[1])
    w = round(1080*min(.6, max(.25, width)))
    h = round(w/aspect)
    if h > 640:
        h, w = 640, max(2, round(640*aspect))
    w, h = max(2, w//2*2), max(2, h//2*2)
    x = 1080-MARGIN_RIGHT-24-w if position.endswith('right') else 24
    # Reserve multi-line caption height above the caption baseline.
    y = 1920-MARGIN_BOTTOM-240-24-h if position.startswith('bottom') else 24
    return FacecamOverlay(box, OutputRect(x, y, w, h))


def _main_preview(frame, mode):
    h, w = frame.shape[:2]
    scale = max(270/w, 480/h)
    resized = cv2.resize(frame, (math.ceil(w*scale), math.ceil(h*scale)))
    y, x = (resized.shape[0]-480)//2, (resized.shape[1]-270)//2
    background = resized[y:y+480, x:x+270].copy()
    if mode == 'crop':
        return background
    background = cv2.convertScaleAbs(cv2.GaussianBlur(background, (31, 31), 0), alpha=.65)
    if w/h > 2:
        cw = int(h*16/9)//2*2
        frame = frame[:, (w-cw)//2:(w+cw)//2]
        h, w = frame.shape[:2]
    scale = min(270/w, 480/h)
    fg = cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale))))
    y, x = (480-fg.shape[0])//2, (270-fg.shape[1])//2
    background[y:y+fg.shape[0], x:x+fg.shape[1]] = fg
    return background


def auto_layout_facecam(source: Path, start: float, end: float, box: SourceBox,
                        mode: str = 'crop', width: float = .42) -> FacecamOverlay:
    if mode not in ('crop', 'blur'):
        raise ValueError('Unknown main layout')
    frames = _sample_frames(source, start, end)
    if not frames:
        raise ValueError('No readable source frames')
    size = (frames[0].shape[1], frames[0].shape[0])
    previews = [cv2.cvtColor(_main_preview(f, mode), cv2.COLOR_BGR2GRAY) for f in frames]
    candidates = []
    for corner in ('top-left', 'top-right', 'bottom-left', 'bottom-right'):
        spec = layout_facecam(box, size, corner, width)
        r = spec.rect
        regions = [p[r.y//4:(r.y+r.h)//4, r.x//4:(r.x+r.w)//4] for p in previews]
        edges = np.mean([np.mean(cv2.Canny(p, 60, 160))/255 for p in regions])
        motion = np.mean([np.mean(cv2.absdiff(a, b))/255 for a, b in zip(regions, regions[1:])]) if len(regions)>1 else 0
        candidates.append((float(edges+motion), corner, spec))
    score, corner, spec = min(candidates, key=lambda c: c[0])
    return layout_facecam(box, size, corner, width*.85) if score > .20 else spec


def save_source_still(source: Path, output: Path, timestamp: float = 0) -> Path:
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError('Timestamp must be finite and nonnegative')
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
        ok, frame = capture.read()
        if not ok:
            raise ValueError('Cannot decode source still')
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError('Cannot save source still')
        return output
    finally:
        capture.release()
