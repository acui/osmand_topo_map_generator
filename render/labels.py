"""Label layout: multi-language name selection, font rendering and
collision avoidance for map labels.

Text is rendered with Pillow (CJK capable) onto small RGBA bitmaps that are
alpha-composited onto the OpenCV canvas; collision tracking uses a simple
rectangle list with a uniform grid index.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from obf.map_index import MapObject

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansSC-Regular.otf",
    "/usr/share/fonts/google-noto-sans-cjk/NotoSansCJK-Regular.ttc",
]


def find_font(preferred: str | None = None) -> str | None:
    candidates = ([preferred] if preferred else []) + _FONT_CANDIDATES
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


def _parse_color(value: str) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    value = value.strip()
    if not value.startswith("#"):
        return None
    try:
        raw = int(value[1:], 16)
    except ValueError:
        return None
    if len(value) == 7:  # #RRGGBB
        return ((raw >> 16) & 255, (raw >> 8) & 255, raw & 255, 255)
    if len(value) == 9:  # #AARRGGBB
        return ((raw >> 16) & 255, (raw >> 8) & 255, raw & 255, (raw >> 24) & 255)
    return None


def select_name(obj: MapObject, language: str, user_lang: str = "zh") -> str | None:
    """Pick the label text according to the language mode."""
    tags = obj.tags
    if language == "local":
        return tags.get("name")
    primary_local = tags.get("name")
    localized = tags.get(f"name:{user_lang}") or tags.get(f"name:{user_lang.lower()}")
    if language == "user":
        return localized or primary_local
    # bilingual: local + user (if different)
    if localized and primary_local and localized != primary_local:
        return f"{primary_local}\n{localized}"
    return localized or primary_local


class LabelEngine:
    """Draws labels onto the canvas with rectangle based collision avoidance."""

    def __init__(self, width: int, height: int, font_path: str | None,
                 text_scale: float = 1.0) -> None:
        self.width = width
        self.height = height
        self.text_scale = text_scale
        self.font_path = font_path or find_font()
        self._font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont | None] = {}
        self._occupied: list[tuple[float, float, float, float]] = []
        self._grid: dict[tuple[int, int], list[int]] = {}
        self._cell = 96

    # -- collision -------------------------------------------------------

    def _cells(self, rect: tuple[float, float, float, float]) -> list[tuple[int, int]]:
        x0, y0, x1, y1 = rect
        return [(int(x0 // self._cell), int(y0 // self._cell)),
                (int(x1 // self._cell), int(y0 // self._cell)),
                (int(x0 // self._cell), int(y1 // self._cell)),
                (int(x1 // self._cell), int(y1 // self._cell))]

    def overlaps(self, rect: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = rect
        checked: set[int] = set()
        for cell in self._cells(rect):
            for idx in self._grid.get(cell, ()):
                if idx in checked:
                    continue
                checked.add(idx)
                ox0, oy0, ox1, oy1 = self._occupied[idx]
                if x0 < ox1 and ox0 < x1 and y0 < oy1 and oy0 < y1:
                    return True
        return False

    def reserve(self, rect: tuple[float, float, float, float]) -> None:
        idx = len(self._occupied)
        self._occupied.append(rect)
        for cell in self._cells(rect):
            self._grid.setdefault(cell, []).append(idx)

    # -- fonts -----------------------------------------------------------

    def font(self, size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont | None:
        key = (int(size_px), bold)
        if key in self._font_cache:
            return self._font_cache[key]
        font = None
        if self.font_path:
            try:
                font = ImageFont.truetype(self.font_path, int(size_px), index=2 if bold and ".ttc" in self.font_path else 0)
            except Exception:
                try:
                    font = ImageFont.truetype(self.font_path, int(size_px))
                except Exception:
                    font = None
        self._font_cache[key] = font
        return font

    # -- drawing ---------------------------------------------------------

    def _render_text_bitmap(self, text: str, font: ImageFont.FreeTypeFont,
                            color: tuple[int, int, int, int],
                            halo_color: tuple[int, int, int, int],
                            halo_radius: float,
                            angle: float = 0.0) -> np.ndarray:
        """Render one text line into an RGBA numpy bitmap."""
        # measure
        probe = Image.new("RGBA", (8, 8))
        probe_draw = ImageDraw.Draw(probe)
        try:
            bbox = probe_draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
        except TypeError:
            bbox = probe_draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = int(math.ceil(halo_radius)) + 2
        img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        origin = (pad - bbox[0], pad - bbox[1])
        if halo_radius > 0:
            hr = halo_radius
            steps = max(4, int(hr * 4))
            for i in range(steps):
                a = 2 * math.pi * i / steps
                draw.multiline_text((origin[0] + hr * math.cos(a),
                                     origin[1] + hr * math.sin(a)),
                                    text, font=font, fill=halo_color, spacing=2)
        draw.multiline_text(origin, text, font=font, fill=color, spacing=2)
        if abs(angle) > 0.5:
            img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
        return np.asarray(img)

    def _composite(self, canvas: np.ndarray, bitmap: np.ndarray, x: float, y: float) -> None:
        h, w = bitmap.shape[:2]
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = x0 + w, y0 + h
        cx0, cy0 = max(0, -x0), max(0, -y0)
        cx1, cy1 = min(w, self.width - x0), min(h, self.height - y0)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        region = bitmap[cy0:cy1, cx0:cx1].astype(np.float32)
        target = canvas[max(0, y0):y1, max(0, x0):x1].astype(np.float32)
        alpha = region[:, :, 3:4] / 255.0
        # RGBA bitmap -> BGR canvas
        blended = region[:, :, 2::-1] * alpha + target * (1.0 - alpha)
        canvas[max(0, y0):y1, max(0, x0):x1] = blended.astype(np.uint8)

    def draw_label(self, canvas: np.ndarray, text: str, x: float, y: float,
                   size_dp: float, color: str | None, halo_radius_dp: float,
                   bold: bool = False, angle: float = 0.0,
                   anchor: str = "center", text_scale: float = 1.0,
                   min_size_px: float = 5.0) -> bool:
        """Draw one label; returns False if it was rejected by collision."""
        size_px = max(min_size_px, size_dp * text_scale * self.text_scale)
        font = self.font(size_px, bold)
        if font is None:
            return False
        fg = _parse_color(color or "#000000") or (0, 0, 0, 255)
        halo = _parse_color("#ffffff") or (255, 255, 255, 255)
        halo_r = halo_radius_dp * text_scale * self.text_scale
        bitmap = self._render_text_bitmap(text, font, fg, halo, halo_r, angle)
        h, w = bitmap.shape[:2]
        if anchor == "center":
            ox, oy = x - w / 2, y - h / 2
        elif anchor == "start":
            ox, oy = x, y - h / 2
        else:
            ox, oy = x - w, y - h / 2
        rect = (ox, oy, ox + w, oy + h)
        if self.overlaps(rect):
            return False
        self.reserve(rect)
        self._composite(canvas, bitmap, ox, oy)
        return True

    def draw_road_label(self, canvas: np.ndarray, text: str,
                        polyline_px: np.ndarray, size_dp: float,
                        color: str | None, halo_radius_dp: float,
                        bold: bool = False, text_scale: float = 1.0) -> bool:
        """Draw a label along the flattest middle part of a polyline."""
        if len(polyline_px) < 2:
            return False
        # pick the segment near the middle with the lowest curvature proxy:
        # use the middle segment of the longest overall run
        pts = np.asarray(polyline_px)
        seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if seg_len.sum() == 0:
            return False
        cum = np.cumsum(seg_len)
        mid = cum[-1] / 2.0
        idx = int(np.searchsorted(cum, mid))
        idx = min(idx, len(pts) - 2)
        p0, p1 = pts[idx], pts[idx + 1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        seg = math.hypot(dx, dy)
        if seg < 8:
            # segment too short: search the longest segment instead
            j = int(np.argmax(seg_len))
            p0, p1 = pts[j], pts[j + 1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            seg = math.hypot(dx, dy)
            if seg < 8:
                return False
        angle = math.degrees(math.atan2(-dy, dx))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
        return self.draw_label(canvas, text, mx, my, size_dp, color,
                               halo_radius_dp, bold=bold, angle=-angle,
                               text_scale=text_scale)
