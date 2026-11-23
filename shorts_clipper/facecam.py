"""Conservative Haar facecam detection and vertical overlay geometry."""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from .captions import FONT_SIZE, MARGIN_BOTTOM, MARGIN_RIGHT


CAPTION_BAND_TOP = 1920 - MARGIN_BOTTOM - 2 * FONT_SIZE - 10
BOTTOM_SAFE_EDGE = 1920 - 320


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
    hide_source: bool = True  # split screen: paint this region out of the main picture (off when it's a shared camera)


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


def find_persistent_border(frames: list[np.ndarray], face: SourceBox) -> SourceBox | None:
    if not frames:
        return None
    height, width = frames[0].shape[:2]
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) for frame in frames]
    vertical = np.percentile([np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0))/4 for g in grays], 20, axis=0)
    horizontal = np.percentile([np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))/4 for g in grays], 20, axis=0)
    fx, fy, fw, fh = face.x*width, face.y*height, face.w*width, face.h*height

    def boundary(edges, near, extent, other, other_extent, direction, bounds=None):
        limit, span = edges.shape[1], edges.shape[0]
        if direction < 0 and near < extent:
            return 0
        if direction > 0 and limit-near < extent:
            return limit
        lo, hi = bounds if bounds is not None else (max(0, int(other-.6*other_extent)), min(span, int(other+1.7*other_extent)))
        candidates = []
        for distance in range(max(2, round(.25*extent)), round(1.25*extent)+1):
            index = round(near+direction*distance)
            if not 1 <= index < limit-1:
                continue
            line = edges[lo:hi, index]
            if (np.mean(line > 6) >= .48 and np.mean(line) >= 8
                    and np.mean(line) >= np.mean(edges[lo:hi, index-direction])):
                candidates.append((index, float(np.mean(line))))
        if not candidates:
            return None
        strongest = max(score for _, score in candidates)
        return next(index for index, score in candidates if score >= strongest*.85)

    left = boundary(vertical, fx, fw, fy, fh, -1)
    right = boundary(vertical, fx+fw, fw, fy, fh, 1)
    bounds = (left, right) if left is not None and right is not None else None
    top = boundary(horizontal.T, fy, fh, fx, fw, -1, bounds)
    bottom = boundary(horizontal.T, fy+fh, fh, fx, fw, 1, bounds)
    if any(v is None for v in (left, right, top, bottom)):
        return None
    w, h = right-left, bottom-top
    if not (1.25*fw <= w <= 4*fw and 1.25*fh <= h <= 4*fh
            and .7 <= w/h <= 2.2 and w*h < width*height*.35):
        return None
    return SourceBox(left/width, top/height, w/width, h/height)


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
        left, top = min(b.x for b in track), min(b.y for b in track)
        right, bottom = max(b.x+b.w for b in track), max(b.y+b.h for b in track)
        safe = SourceBox(max(0, left-face.w*.1), max(0, top-face.h*.1),
                         min(1, right+face.w*.1)-max(0, left-face.w*.1),
                         min(1, bottom+face.h*.1)-max(0, top-face.h*.1))
        def contains_face(border):
            return (border is not None and border.x <= safe.x and border.y <= safe.y
                    and border.x+border.w >= safe.x+safe.w
                    and border.y+border.h >= safe.y+safe.h)
        persistent = find_persistent_border(frames, face)
        if contains_face(persistent):
            return persistent
        borders = [find_facecam_border(frame, face) for frame in frames]
        borders = [b for b in borders if contains_face(b)]
        if len(borders) > len(frames)/2:
            return SourceBox(*np.median([[b.x, b.y, b.w, b.h] for b in borders], axis=0))
        height, width = frames[0].shape[:2]
        face = SourceBox(left, top, right-left, bottom-top)
        h = min(1, face.h*1.7)
        w = min(1, max(face.w*1.65, h*height/width*4/3))
        x = max(0, min(1-w, face.x+face.w/2-w/2))
        y = max(0, min(1-h, face.y-face.h*.35))
        return SourceBox(x, y, w, h)
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
    h = max(2, round(w/aspect/2)*2)
    max_height = min(CAPTION_BAND_TOP-40, 1920-MARGIN_BOTTOM-2.6*130-48)
    if position.startswith('bottom'):
        max_height = min(max_height, BOTTOM_SAFE_EDGE-260-2.6*130-24)
    if h > max_height:
        h = int(max_height)//2*2
        w = max(2, round(h*aspect/2)*2)
    w, h = max(2, w//2*2), max(2, h//2*2)
    x = 1080-MARGIN_RIGHT-24-w if position.endswith('right') else 24
    y = BOTTOM_SAFE_EDGE-h if position.startswith('bottom') else 24
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
    importance = []
    for index, preview in enumerate(previews):
        motion = cv2.absdiff(preview, previews[index-1])/255 if index else np.zeros_like(preview)
        importance.append(cv2.Canny(preview, 60, 160)/255 + motion
                          + .35*_saliency_map(preview) + .8*_hud_mask(preview))
    candidates = []
    for corner in ('top-left', 'top-right'):
        spec = layout_facecam(box, size, corner, width)
        r = spec.rect
        scores = [np.mean(p[r.y//4:math.ceil((r.y+r.h)/4), r.x//4:math.ceil((r.x+r.w)/4)])
                  for p in importance]
        candidates.append((float((np.mean(scores)+max(scores))/2), corner, spec))
    score, corner, spec = min(candidates, key=lambda c: c[0])
    if score > .20 and width > .38:
        return layout_facecam(box, size, corner, max(.38, width*.92))
    return spec


def _saliency_map(gray):
    if np.std(gray) < 1:
        return np.zeros_like(gray, dtype=float)
    small = cv2.resize(gray, (64, 112)).astype(float)
    spectrum = np.fft.fft2(small)
    amplitude = np.log(np.abs(spectrum)+1e-6)
    residual = amplitude-cv2.blur(amplitude, (3, 3))
    saliency = np.abs(np.fft.ifft2(np.exp(residual+1j*np.angle(spectrum))))**2
    saliency = cv2.GaussianBlur(saliency, (9, 9), 0)
    saliency /= max(float(saliency.max()), 1e-6)
    return cv2.resize(saliency, (gray.shape[1], gray.shape[0]))


def _hud_mask(gray):
    mask = np.zeros_like(gray)
    detector = cv2.MSER_create()
    detector.setMinArea(4)
    detector.setMaxArea(300)
    _, boxes = detector.detectRegions(gray)
    for x, y, w, h in boxes:
        if 3 <= h <= 22 and .12 <= w/h <= 1.5:
            mask[y:y+h, x:x+w] = 255
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    result = np.zeros_like(gray, dtype=float)
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 15 and w/h >= 2.5 and h <= 28:
            result[y:y+h, x:x+w] = 1
    return result


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
