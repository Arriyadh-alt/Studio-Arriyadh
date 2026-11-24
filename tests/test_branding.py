from io import BytesIO
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFont

from clipper_studio.api import create_app
from shorts_clipper.branding import BOTTOM, LOGO_LIMIT, BrandingSettings, prepare_watermark
from shorts_clipper.facecam import FacecamOverlay, OutputRect, SourceBox
from shorts_clipper.video import make_vertical_clip

LOGO_COLOR = (22, 184, 104)


def png_logo():
    buffer = BytesIO()
    Image.new("RGBA", (60, 48), (*LOGO_COLOR, 255)).save(buffer, "PNG")
    return buffer.getvalue()


def install_test_logo(client):
    assert client.put("/api/branding", json={"position": "top-left", "opacity": 100}).status_code == 200
    assert client.put("/api/branding/logo", content=png_logo()).status_code == 200


def decoded_frame(path, at=0):
    frame = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(path),
                                     "-frames:v", "1", "-c:v", "png", "-threads", "1",
                                     "-f", "image2pipe", "pipe:1"])
    return Image.open(BytesIO(frame))


def assert_logo_burned(path, at=0):
    pixel = decoded_frame(path, at).getpixel((130, 116))
    assert pixel == pytest.approx(LOGO_COLOR, abs=7)


def test_settings_and_logo_survive_restart(tmp_path):
    path = "/api/branding"
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(path).json() == {"text": "@ARLSTUDIOZ", "opacity": 50, "size": 48,
                                          "position": "auto", "has_logo": False}
        settings = {"text": "Arriyadh", "opacity": 75, "size": 62, "position": "bottom-right"}
        assert client.put(path, json=settings).json() == {**settings, "has_logo": False}
        assert client.put(path, json={**settings, "text": "  "}).status_code == 422
        assert client.put(path + "/logo", content=b"not a PNG").status_code == 422
        assert client.put(path + "/logo", content=b"x" * (LOGO_LIMIT + 1)).status_code == 413
        assert client.put(path + "/logo", content=png_logo()).json() == {"has_logo": True}
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(path).json() == {**settings, "has_logo": True}
        response = client.get(path + "/logo")
        assert response.headers["content-type"] == "image/png"
        assert Image.open(BytesIO(response.content)).getpixel((0, 0)) == (*LOGO_COLOR, 255)
        assert client.delete(path + "/logo").json() == {"has_logo": False}
        assert client.get(path + "/logo").status_code == 404
        assert client.get(path).json() == {**settings, "has_logo": False}


def test_auto_watermark_keeps_clear_of_overlays_and_busy_regions(tmp_path, monkeypatch):
    cam = FacecamOverlay(SourceBox(0, 0, .25, .25), OutputRect(0, 0, 1080, 680))
    grid = np.zeros((480, 270))
    grid[390:, :135] = 1
    monkeypatch.setattr("shorts_clipper.branding.picture_importance", lambda *args: [grid])
    path = tmp_path / "branding.png"
    mark = prepare_watermark(path, BrandingSettings(), None, tmp_path / "source.mp4", 0, 2, "crop",
                             facecam=cam, bands=((680, 1040), (1200, 1510)))
    with Image.open(path) as artwork:
        assert mark.x > 540 and mark.y + artwork.height == pytest.approx(BOTTOM, abs=1)
        assert artwork.getchannel("A").getextrema()[1] == 128


def test_non_ascii_watermark_loads_a_charset_matched_font(tmp_path):
    settings = BrandingSettings(text="@日本語", position="top-left")
    font = ImageFont.load_default(size=settings.size)
    result = subprocess.CompletedProcess([], 0, stdout="/fonts/NotoSansCJK-Regular.ttc\t2")
    with patch("shorts_clipper.branding.subprocess.run", return_value=result) as match, \
         patch("shorts_clipper.branding.ImageFont.truetype", return_value=font) as load:
        prepare_watermark(tmp_path / "mark.png", settings, None, tmp_path / "source.mp4", 0, 1, "crop")
    assert match.call_args.args[0] == ["fc-match", "-f", "%{file}\t%{index}",
                                      "sans-serif:style=Regular:charset=40 65e5 672c 8a9e"]
    load.assert_called_once_with("/fonts/NotoSansCJK-Regular.ttc", settings.size, index=2)


@pytest.mark.parametrize("mode", ["crop", "blur"])
def test_default_watermark_is_burned_into_first_and_last_frames(tmp_path, mode):
    source, output = tmp_path / "source.mp4", tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=black:s=320x180:r=10:d=1", "-c:v", "libx264", str(source)], check=True)
    mark = prepare_watermark(tmp_path / "mark.png", BrandingSettings(), None, source, 0, 1, mode)
    make_vertical_clip(source, output, 0, 1, mode, watermark=mark)
    for timestamp in (0, .9):
        frame = decoded_frame(output, timestamp)
        roi = frame.crop((mark.x, mark.y, mark.x + 360, mark.y + 68)).convert("L")
        assert 110 <= roi.getextrema()[1] <= 140
        luma = np.asarray(roi)
        assert np.count_nonzero((95 <= luma) & (luma <= 140)) > 500
        assert frame.getpixel((540, 970)) == (0, 0, 0)
