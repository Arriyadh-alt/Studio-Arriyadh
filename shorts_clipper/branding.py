"""Workspace branding settings and the artwork burned into vertical exports."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import subprocess
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel, Field, field_validator

from .facecam import picture_importance

LOGO_LIMIT = 2 * 1024 * 1024
MARGIN = 64
BOTTOM = 1728
WIDTH, HEIGHT = 1080, 1920


class BrandingSettings(BaseModel):
    text: str = Field(default="@ARLSTUDIOZ", min_length=1, max_length=60)
    opacity: int = Field(default=50, ge=1, le=100)
    size: int = Field(default=48, ge=24, le=96)
    position: Literal["auto", "top-left", "top", "top-right", "bottom-left", "bottom", "bottom-right"] = "auto"

    @field_validator("text", mode="before")
    @classmethod
    def clean_text(cls, value):
        return " ".join(value.split()) if isinstance(value, str) else value

    @classmethod
    def read(cls, root):
        try:
            return cls.model_validate_json((root / "branding.json").read_bytes())
        except FileNotFoundError:
            return cls()


def normalize_logo(data):
    with Image.open(BytesIO(data), formats=["PNG"]) as image:
        if max(image.size) > 2048 or getattr(image, "is_animated", False):
            raise ValueError("Use a static PNG up to 2048 × 2048 pixels.")
        rgba = image.convert("RGBA")
        bounds = rgba.getbbox()
        if bounds is None:
            raise ValueError("The logo needs visible pixels.")
        result = BytesIO()
        rgba.crop(bounds).save(result, format="PNG")
        return result.getvalue()


def read_logo(root):
    try:
        return (root / "branding-logo.png").read_bytes()
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class Watermark:
    path: Path
    x: int
    y: int


def _font_face(text):
    if not text.isascii():
        charset = " ".join(f"{ord(char):x}" for char in sorted(set(text)))
        try:
            match = subprocess.run(["fc-match", "-f", "%{file}\t%{index}",
                                    f"sans-serif:style=Regular:charset={charset}"],
                                   capture_output=True, check=True, text=True)
            path, index = match.stdout.split("\t")
            return path, int(index)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return "DejaVuSans.ttf", 0


def _artwork(settings, logo):
    if logo:
        with Image.open(BytesIO(logo)) as image:
            scale = min(2 * settings.size / image.height, 6 * settings.size / image.width, 2)
            mark = image.resize(tuple(max(1, round(dimension * scale)) for dimension in image.size),
                                Image.Resampling.LANCZOS)
    else:
        size = settings.size
        face, index = _font_face(settings.text)
        while True:
            try:
                font = ImageFont.truetype(face, size, index=index)
            except OSError:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", size)
                except OSError:
                    font = ImageFont.load_default(size=size)
            left, top, right, bottom = font.getbbox(settings.text)
            if right - left <= WIDTH // 2 or size <= 1:
                break
            size = max(1, min(size - 1, int(size * (WIDTH // 2) / (right - left))))
        mark = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)))
        ImageDraw.Draw(mark).text((-left, -top), settings.text, font=font, fill="white")
    alpha = mark.getchannel("A")
    size = (mark.width + 12, mark.height + 12)
    shadow = Image.new("L", size)
    shadow.paste(alpha.point(lambda a: round(a / 2)), (7, 8))
    layer = Image.new("RGBA", size)
    layer.putalpha(shadow.filter(ImageFilter.GaussianBlur(2)))
    layer.alpha_composite(mark, (6, 6))
    layer.putalpha(layer.getchannel("A").point(lambda a: round(a * settings.opacity / 100)))
    return layer


def _position(settings, size, facecam, bands, source, start, end, mode):
    w, h = size
    left, center, right = MARGIN, (WIDTH - w) // 2, WIDTH - MARGIN - w
    top, bottom = MARGIN, BOTTOM - h
    positions = {"top-left": (left, top), "top": (center, top), "top-right": (right, top),
                 "bottom-left": (left, bottom), "bottom": (center, bottom), "bottom-right": (right, bottom)}
    if settings.position != "auto":
        return positions[settings.position]

    blocked = [(0, band[0], WIDTH, band[1]) for band in bands if band is not None]
    if facecam:
        r = facecam.rect
        blocked.append((r.x, r.y, r.x + r.w, r.y + r.h))

    def overlap(point):
        x, y = point
        return sum(max(0, min(x + w, x2 + 16) - max(x, x1 - 16))
                   * max(0, min(y + h, y2 + 16) - max(y, y1 - 16)) for x1, y1, x2, y2 in blocked)

    corners = [positions[key] for key in ("top-left", "bottom-left", "top-right", "bottom-right")]
    free = [point for point in corners if not overlap(point)]
    if not free:
        rows = [max(top, min(bottom, y)) for _, a, _, b in blocked for y in (a - 16 - h, b + 16)]
        candidates = [*corners, *((x, y) for y in rows for x in (left, right))]
        free = [point for point in candidates if not overlap(point)]
        if not free:
            return min(candidates, key=overlap)
    try:
        maps = picture_importance(source, start, end, mode)
    except cv2.error:
        maps = []
    if not maps:
        return free[0]

    def importance(point):
        x, y = point
        scores = [np.mean(grid[y//4:(y+h+3)//4, x//4:(x+w+3)//4]) for grid in maps]
        return np.mean(scores) + max(scores)

    return min(free, key=importance)


def prepare_watermark(path, settings, logo, source, start, end, mode, *, facecam=None, bands=()):
    artwork = _artwork(settings, logo)
    x, y = _position(settings, artwork.size, facecam, bands, source, start, end, mode)
    artwork.save(path)
    return Watermark(path, x // 2 * 2, y // 2 * 2)
