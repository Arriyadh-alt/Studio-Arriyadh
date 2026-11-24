"""Pick and style a vertical cover from a rendered highlight, offline."""

import re
from pathlib import Path
from textwrap import wrap

import cv2
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 540, 960
TEXT_LIMIT = 64


def choose_hook(captions, peak, messages, title):
    def excerpt(value):
        words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", value.upper())
        return ' '.join(words[:5]) if len(words) >= 2 and len(' '.join(words[:5])) <= TEXT_LIMIT else None

    if peak is not None:
        for caption in sorted(captions, key=lambda c: abs((c.start + c.end) / 2 - peak)):
            if caption.start - 4 <= peak <= caption.end + 4 and (phrase := excerpt(caption.text)):
                return phrase
    for message in messages:
        if not re.search(r"https?://", message, re.I) and (phrase := excerpt(message)):
            return phrase
    if any(re.search(r"\b(lol|lmao|kekw|wkwk|haha+)\b", message, re.I) for message in messages):
        return "CHAT LOST IT"
    if any(re.fullmatch(r"(?:w+|gg+)[! ]*", message, re.I) for message in messages):
        return "WHAT A PLAY"
    return excerpt(title or "") or "THE BIG MOMENT"


def frame_score(frame, faces, smiles):
    gray = cv2.cvtColor(cv2.resize(frame, (270, 480)), cv2.COLOR_BGR2GRAY)
    sharpness = min(1, cv2.Laplacian(gray, cv2.CV_64F).var() / 350)
    contrast = min(1, gray.std() / 55)
    exposure = 1 - abs(gray.mean() - 128) / 128
    score = (sharpness + contrast + exposure) / 3
    face_center = None
    equalized = cv2.equalizeHist(gray)
    boxes = faces.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=5, minSize=(22, 22))
    if len(boxes):
        x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
        face_center = (y + h / 2) / gray.shape[0]
        mouth = equalized[y + h // 2:y + h, x:x + w]
        expressions = smiles.detectMultiScale(mouth, scaleFactor=1.15, minNeighbors=12, minSize=(14, 8))
        score += 0.8 + min(0.6, 5 * w * h / gray.size) + (0.7 if len(expressions) else 0)
    return score, face_center


def select_frame(video: Path, peak):
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smiles = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    capture = cv2.VideoCapture(str(video))
    best = None
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        indices = {round((frame_count - 1) * i / 10) for i in range(11)} if frame_count > 1 else {0}
        if peak is not None and fps > 0 and frame_count > 1:
            indices.update(round(second * fps) for second in (peak - 1, peak, peak + 1)
                           if 0 <= second * fps <= frame_count - 1)
        for index in sorted(indices):
            if index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            score, center = frame_score(frame, faces, smiles)
            if peak is not None and fps > 0:
                score += 0.3 * max(0, 1 - abs(index / fps - peak) / 5)
            if best is None or score > best[0]:
                best = score, frame, center
    finally:
        capture.release()
    if best is None:
        raise ValueError("No frames could be read from the clip")
    return best[1], best[2]


def draw_cover(frame, face_center, text, output):
    image = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB))
    if not text:
        image.save(output, quality=92, optimize=True)
        return
    draw = ImageDraw.Draw(image)
    lines = '\n'.join(wrap(text, width=16))
    for size in range(84, 15, -2):
        try:
            font = ImageFont.truetype('DejaVuSans-Bold.ttf', size)
        except OSError:
            font = ImageFont.load_default(size=size)
        spacing = size // 8
        left, top, right, bottom = draw.multiline_textbbox((0, 0), lines, font=font, spacing=spacing)
        if right - left <= WIDTH - 104 and bottom - top <= 235:
            break
    width, height = right - left, bottom - top
    x = (WIDTH - width) // 2
    y = round(HEIGHT * (0.23 if face_center is not None and face_center > 0.52 else 0.72)) - height // 2
    draw.rounded_rectangle((x - 24, y - 19, x + width + 24, y + height + 19), radius=18, fill='#161b22')
    draw.rounded_rectangle((WIDTH // 2 - 27, y - 22, WIDTH // 2 + 27, y - 16), radius=3, fill='#b8f477')
    draw.multiline_text((x - left, y - top), lines, font=font, align="center", spacing=spacing,
                        fill='#ffffff', stroke_width=2, stroke_fill='#000000')
    image.save(output, quality=92, optimize=True)


def generate_thumbnail(video: Path, output: Path, text: str, peak=None):
    frame, face_center = select_frame(video, peak)
    draw_cover(frame, face_center, text, output)
