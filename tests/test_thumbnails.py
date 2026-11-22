import cv2
import numpy as np
from PIL import Image

from shorts_clipper.captions import Caption
from shorts_clipper.thumbnails import choose_hook, frame_score, generate_thumbnail, select_frame


def test_hook_prefers_captions_near_peak_then_chat_then_title():
    captions = [Caption(0, 1, "An old quote"), Caption(11, 12, "Serius ini baru saja terjadi")]
    assert choose_hook(captions, 11, ["nice win"], "Stream title") == "SERIUS INI BARU SAJA TERJADI"
    assert choose_hook(captions, 25, ["nice win"], "Stream title") == "NICE WIN"
    assert choose_hook([], None, ["wkwk"], "Stream title") == "CHAT LOST IT"
    assert choose_hook([], None, ["https://example.org"], "Stream title") == "STREAM TITLE"
    assert choose_hook([], None, [], "🔥") == "THE BIG MOMENT"


def test_frame_score_rewards_detected_expression():
    class Cascade:
        def __init__(self, boxes):
            self.boxes = boxes

        def detectMultiScale(self, *args, **kwargs):
            return self.boxes

    frame = np.full((320, 180, 3), 128, dtype=np.uint8)
    absent = Cascade([])
    face = Cascade([(80, 80, 75, 75)])
    smile = Cascade([(5, 5, 20, 10)])
    no_face, _ = frame_score(frame, absent, absent)
    neutral, center = frame_score(frame, face, absent)
    expressive, _ = frame_score(frame, face, smile)
    assert no_face < neutral < expressive
    assert center == (80 + 75 / 2) / 480


def test_thumbnail_uses_clear_later_frame_and_overlays_text(tmp_path):
    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 6, (180, 320))
    assert writer.isOpened()
    flat = np.full((320, 180, 3), 24, dtype=np.uint8)
    marked = flat.copy()
    marked[:, :90] = (68, 144, 220)
    marked[::16, :] = 230
    try:
        for frame in [flat] * 6 + [marked] * 6:
            writer.write(frame)
    finally:
        writer.release()
    frame, _ = select_frame(video, 1.5)
    assert frame[10, 10, 2] > 180
    output = tmp_path / "cover.jpg"
    generate_thumbnail(video, output, "WHAT A PLAY", peak=1.5)
    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.size == (540, 960)
        r, g, b = image.getpixel((270, 650))
        assert g > r + 30 and g > b + 50
        assert (np.asarray(image.crop((170, 670, 370, 715))).min(axis=2) > 243).any()
        assert image.getpixel((15, 30))[0] > 180
