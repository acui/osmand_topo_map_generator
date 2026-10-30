"""Offline map renderer: OBF vector data + render.xml style -> high-res PNG.

Rendering pipeline (mirroring OsmAnd's native order):

1. query vector objects from one or more ``.obf`` files for the map region
2. resolve the draw order per object (``order`` rules)
3. draw polygons (ascending order), then lines (ascending order/layer),
   then points
4. draw labels (ascending ``textOrder``) with collision avoidance
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import cv2 as cv
import numpy as np

from obf.map_index import MapObject
from obf.obf_file import ObfFile
from render.labels import LabelEngine, find_font, select_name
from render.project import Projector
from render.style import DrawSpec, RenderStyle


@dataclass
class RenderOptions:
    dpi: int = 300
    zoom: int = 15
    language: str = "local"            # local | user | bilingual
    user_lang: str = "zh"
    style_props: dict[str, str | bool] = field(default_factory=dict)
    stroke_scale: float = 1.0          # multiplier on OsmAnd dp line widths
    text_scale: float = 1.0
    font_path: str | None = None
    draw_icons: bool = True
    margin_deg: float = 0.02           # query bbox expansion


@dataclass
class RenderResult:
    png_path: Path
    width_px: int
    height_px: int
    corners_px: np.ndarray
    stats: dict[str, int] = field(default_factory=dict)


def _parse_color(value: str | None) -> tuple[int, int, int, float] | None:
    """#RRGGBB / #AARRGGBB -> (r, g, b, alpha 0..1)."""
    if not value or not value.startswith("#"):
        return None
    try:
        raw = int(value[1:], 16)
    except ValueError:
        return None
    if len(value) == 7:
        return ((raw >> 16) & 255, (raw >> 8) & 255, raw & 255, 1.0)
    if len(value) == 9:
        return ((raw >> 16) & 255, (raw >> 8) & 255, raw & 255, ((raw >> 24) & 255) / 255.0)
    return None


def _float_first(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value.split(":")[0])
    except ValueError:
        return default


def _bbox_slice(canvas: np.ndarray, pts: np.ndarray, pad: int):
    """Return (y0, y1, x0, x1) bounding the points with padding, or None."""
    h, w = canvas.shape[:2]
    x0 = max(0, int(pts[:, 0].min()) - pad)
    x1 = min(w, int(np.ceil(pts[:, 0].max())) + pad + 1)
    y0 = max(0, int(pts[:, 1].min()) - pad)
    y1 = min(h, int(np.ceil(pts[:, 1].max())) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return y0, y1, x0, x1


def _blend_poly(canvas: np.ndarray, pts: np.ndarray, rgb: tuple[int, int, int],
                alpha: float) -> None:
    if len(pts) < 3 or alpha <= 0:
        return
    region_box = _bbox_slice(canvas, pts, pad=2)
    if region_box is None:
        return
    y0, y1, x0, x1 = region_box
    sub = canvas[y0:y1, x0:x1]
    mask = np.zeros(sub.shape[:2], dtype=np.uint8)
    shifted = np.round(pts - np.array([x0, y0])).astype(np.int32)
    cv.fillPoly(mask, [shifted], int(round(alpha * 255)))
    sel = mask > 0
    if not sel.any():
        return
    color = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
    m = (mask[sel] / 255.0).astype(np.float32)[:, None]
    region = sub[sel].astype(np.float32)
    sub[sel] = (color * m + region * (1.0 - m)).astype(np.uint8)


def _blend_stroke(canvas: np.ndarray, pts: np.ndarray, rgb: tuple[int, int, int],
                  alpha: float, width_px: float, dash: tuple[float, float] | None,
                  cap_round: bool) -> None:
    if len(pts) < 2 or width_px <= 0 or alpha <= 0:
        return
    pad = int(width_px) + 4
    region_box = _bbox_slice(canvas, pts, pad=pad)
    if region_box is None:
        return
    y0, y1, x0, x1 = region_box
    sub = canvas[y0:y1, x0:x1]
    shifted = pts - np.array([x0, y0])

    intensity = max(1, int(round(alpha * 255)))
    color = (intensity, intensity, intensity)
    color_bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)

    def stroke(target: np.ndarray, poly: np.ndarray) -> None:
        ipts = np.round(poly).astype(np.int32)
        cv.polylines(target, [ipts], False, color,
                     max(1, int(round(width_px))), lineType=cv.LINE_AA)
        if cap_round:
            r = width_px / 2.0
            for p in (ipts[0], ipts[-1]):
                cv.circle(target, tuple(p), max(1, int(round(r))), color, -1,
                          lineType=cv.LINE_AA)

    mask = np.zeros(sub.shape[:2], dtype=np.uint8)
    if dash is None:
        stroke(mask, shifted)
    else:
        # dashed polyline: walk by arc length
        on, off = dash
        on, off = max(0.5, on), max(0.5, off)
        dist = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        total = dist[-1]
        pos = 0.0
        drawing = True
        while pos < total:
            seg_len = on if drawing else off
            end = min(pos + seg_len, total)
            i0 = int(np.searchsorted(dist, pos, "right")) - 1
            i1 = int(np.searchsorted(dist, end, "left"))
            i1 = max(i1, i0 + 1)
            i0, i1 = min(i0, len(pts) - 1), min(i1, len(pts) - 1)
            if drawing and i1 > i0:
                stroke(mask, shifted[i0:i1 + 1])
            pos = end
            drawing = not drawing

    sel = mask > 0
    if not sel.any():
        return
    m = (mask[sel] / 255.0).astype(np.float32)[:, None]
    region = sub[sel].astype(np.float32)
    sub[sel] = (color_bgr * m + region * (1.0 - m)).astype(np.uint8)


def _object_layer(obj: MapObject) -> int:
    layer = 0
    for tvp in obj.additional_types:
        if tvp.tag == "tunnel":
            layer = min(layer, -1)
        elif tvp.tag == "bridge":
            layer = max(layer, 1)
        elif tvp.tag == "layer" and tvp.value:
            try:
                layer = int(tvp.value)
            except ValueError:
                pass
    return layer


def _centroid(pts: np.ndarray) -> tuple[float, float]:
    x = pts[:, 0]
    y = pts[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return float(x.mean()), float(y.mean())
    cx = ((x + x1) * cross).sum() / (6.0 * area)
    cy = ((y + y1) * cross).sum() / (6.0 * area)
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return float(x.mean()), float(y.mean())
    return float(cx), float(cy)


def _in_canvas(pts: np.ndarray, width: int, height: int, margin: float = 32.0) -> bool:
    if len(pts) == 0:
        return False
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return not (x1 < -margin or x0 > width + margin or y1 < -margin or y0 > height + margin)


class Renderer:
    def __init__(self, obf_paths: list[Path], style: RenderStyle,
                 options: RenderOptions) -> None:
        self.obf_paths = [Path(p) for p in obf_paths]
        self.style = style
        self.options = options

    def render(self, bbox, scale: Fraction, out_png: Path) -> RenderResult:
        opts = self.options
        projector = Projector(bbox, scale, opts.dpi)
        width, height = projector.width_px, projector.height_px
        canvas = np.full((height, width, 3), 235, dtype=np.uint8)

        # background color from style attribute
        default_color = None
        state_color = self.style.attributes.get("defaultColor")
        if state_color:
            from render.style import MatchState
            ms = MatchState()
            ms.zoom = opts.zoom
            ms.point = False
            ms.area = True
            for prop, spec in self.style.properties.items():
                ms.props[prop] = spec["default"]
            if opts.style_props:
                ms.props.update(opts.style_props)
            ms.attr_values = {}
            for rule in self.style.attributes.get("defaultColor", []):
                self.style._visit(rule, ms)
            default_color = _parse_color(ms.attr_values.get("color"))
        if default_color:
            canvas[:, :] = (default_color[2], default_color[1], default_color[0])

        # ---- query data -------------------------------------------------
        south, west, north, east = _bbox_to_latlon(bbox)
        margin = opts.margin_deg
        query_bbox = (south - margin, west - margin, north + margin, east + margin)
        objects: list[MapObject] = []
        seen: set[tuple[int, str]] = set()
        files = [ObfFile(p) for p in self.obf_paths]
        try:
            for obf in files:
                if not _obf_intersects(obf, query_bbox, opts.zoom):
                    continue
                for obj in obf.query_map_objects(query_bbox, zoom=opts.zoom):
                    key = (obj.osm_id, str(obj.types))
                    if key in seen:
                        continue
                    seen.add(key)
                    objects.append(obj)
        finally:
            for obf in files:
                obf.close()

        # ---- project geometry -------------------------------------------
        for obj in objects:
            obj._px = projector.project_points(obj.points)  # type: ignore[attr-defined]
            obj._layer = _object_layer(obj)  # type: ignore[attr-defined]

        # ---- classify & resolve specs ------------------------------------
        polygons: list[tuple[float, int, MapObject, DrawSpec]] = []
        lines: list[tuple[float, int, MapObject, DrawSpec]] = []
        points: list[tuple[float, MapObject, DrawSpec]] = []
        stats = {"polygons": 0, "lines": 0, "points": 0, "skipped": 0}
        for obj in objects:
            pts: np.ndarray = obj._px  # type: ignore[attr-defined]
            if not _in_canvas(pts, width, height):
                stats["skipped"] += 1
                continue
            order, _obj_type = self.style.resolve_order(obj, opts.zoom, props=opts.style_props)
            if order >= (1 << 30):
                stats["skipped"] += 1
                continue
            if obj.node:
                spec = self.style.resolve("point", obj, opts.zoom, props=opts.style_props)
                if spec is not None:
                    points.append((order, obj, spec))
                    stats["points"] += 1
                continue
            if obj.polygon:
                spec = self.style.resolve("polygon", obj, opts.zoom, props=opts.style_props)
                if spec is None:
                    spec = self.style.resolve("line", obj, opts.zoom, props=opts.style_props)
                if spec is not None:
                    polygons.append((order, obj._layer, obj, spec))  # type: ignore[attr-defined]
                    stats["polygons"] += 1
            else:
                spec = self.style.resolve("line", obj, opts.zoom, props=opts.style_props)
                if spec is not None:
                    lines.append((order, obj._layer, obj, spec))  # type: ignore[attr-defined]
                    stats["lines"] += 1

        # ---- draw polygons ------------------------------------------------
        polygons.sort(key=lambda item: (item[0], item[1]))
        for _order, _layer, obj, spec in polygons:
            pts: np.ndarray = obj._px  # type: ignore[attr-defined]
            color = _parse_color(spec.color)
            if color:
                _blend_poly(canvas, pts, color[:3], color[3])
            outline_w = _float_first(spec.outputs.get("strokeWidth_2"), 0.0)
            outline_c = _parse_color(spec.outputs.get("color_2"))
            if outline_w > 0 and outline_c:
                _blend_stroke(canvas, np.vstack([pts, pts[:1]]), outline_c[:3],
                              outline_c[3], outline_w * opts.stroke_scale, None, False)

        # ---- draw lines ----------------------------------------------------
        lines.sort(key=lambda item: (item[1], item[0]))
        for _order, _layer, obj, spec in lines:
            pts: np.ndarray = obj._px  # type: ignore[attr-defined]
            dash = None
            dash_raw = spec.outputs.get("pathEffect") or spec.outputs.get("pathEffect_5")
            if dash_raw and "_" in dash_raw:
                parts = dash_raw.replace(",", "_").split("_")
                if len(parts) == 2:
                    dash = (_float_first(parts[0], 4) * opts.stroke_scale,
                            _float_first(parts[1], 2) * opts.stroke_scale)
            cap_round = (spec.cap or "BUTT").upper() == "ROUND"

            main_color = _parse_color(spec.color)
            main_width = _float_first(spec.outputs.get("strokeWidth"), 0.0)
            # '5' variant doubles as the visible style when the main color
            # is absent (footway-style dashed paths)
            c5 = _parse_color(spec.outputs.get("color_5"))
            w5 = _float_first(spec.outputs.get("strokeWidth_5"), 0.0)
            d5 = spec.outputs.get("pathEffect_5")

            casing_color = _parse_color(spec.outputs.get("color__2"))
            casing_width = _float_first(spec.outputs.get("strokeWidth__2"), 0.0)
            if casing_color and casing_width > main_width:
                _blend_stroke(canvas, pts, casing_color[:3], casing_color[3],
                              casing_width * opts.stroke_scale, None, cap_round)

            if main_color and main_width > 0:
                _blend_stroke(canvas, pts, main_color[:3], main_color[3],
                              main_width * opts.stroke_scale, dash, cap_round)
            elif c5 and w5 > 0:
                dash5 = None
                if d5 and "_" in d5:
                    parts = d5.replace(",", "_").split("_")
                    if len(parts) == 2:
                        dash5 = (_float_first(parts[0], 4) * opts.stroke_scale,
                                 _float_first(parts[1], 2) * opts.stroke_scale)
                _blend_stroke(canvas, pts, c5[:3], c5[3], w5 * opts.stroke_scale,
                              dash5, cap_round)

        # ---- draw points ---------------------------------------------------
        for order, obj, spec in points:
            pts: np.ndarray = obj._px  # type: ignore[attr-defined]
            if len(pts) == 0:
                continue
            color = _parse_color(spec.color) or (0, 0, 0, 1.0)
            radius = max(2.0, 2.5 * opts.stroke_scale)
            cv.circle(canvas, tuple(np.round(pts[0]).astype(int)),
                      int(round(radius)), (color[2], color[1], color[0]), -1, cv.LINE_AA)

        # ---- labels ----------------------------------------------------------
        self._draw_labels(canvas, objects, projector, opts)

        out_png.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(out_png), canvas)
        stats["objects"] = len(objects)
        return RenderResult(
            png_path=out_png,
            width_px=width,
            height_px=height,
            corners_px=projector.corners_px(),
            stats=stats,
        )

    # ------------------------------------------------------------------

    def _draw_labels(self, canvas: np.ndarray, objects: list[MapObject],
                     projector: Projector, opts: RenderOptions) -> None:
        label_specs: list[tuple[int, MapObject, DrawSpec]] = []
        for obj in objects:
            if obj.node or len(obj.points) == 0:
                spec = self.style.resolve("point", obj, opts.zoom, props=opts.style_props)
                text_spec = None
                if spec is not None:
                    text_spec = spec
                spec = self.style.resolve("text", obj, opts.zoom, props=opts.style_props)
                if spec is None and text_spec is not None and text_spec.text_order:
                    spec = text_spec
            else:
                # try the generic nameTag="" rules first, then the specific
                # string attributes the object carries (e.g. elevation)
                spec = None
                for name_tag in ["", *[k for k in obj.tags if k != "name"]]:
                    spec = self.style.resolve(
                        "text", obj, opts.zoom, name_tag=name_tag,
                        text_length=len(obj.tags.get(name_tag, "") or ""),
                        props=opts.style_props)
                    if spec is not None:
                        break
            if spec is None or spec.text_order <= 0:
                continue
            label_specs.append((spec.text_order, obj, spec))
        label_specs.sort(key=lambda item: item[0])

        font_path = self.options.font_path or find_font()
        engine = LabelEngine(canvas.shape[1], canvas.shape[0], font_path,
                             text_scale=1.0)
        for _order, obj, spec in label_specs:
            name_tag = spec.name_tag
            if name_tag:
                text = obj.tags.get(name_tag)
            else:
                text = select_name(obj, opts.language, opts.user_lang)
            if not text:
                continue
            size = spec.text_size or 12.0
            color = spec.text_color or "#000000"
            halo = _float_first(spec.outputs.get("textHaloRadius"), 0.0)
            bold = spec.outputs.get("textBold") == "true"
            pts: np.ndarray = obj._px  # type: ignore[attr-defined]
            if obj.node or len(pts) == 1:
                if obj.label is not None:
                    lx, ly = projector.project_points([obj.label])[0][0], \
                        projector.project_points([obj.label])[1][0]
                else:
                    lx, ly = float(pts[0][0]), float(pts[0][1])
                engine.draw_label(canvas, text, lx, ly, size, color, halo,
                                  bold=bold, text_scale=opts.text_scale)
            elif obj.polygon:
                cx, cy = _centroid(pts)
                engine.draw_label(canvas, text, cx, cy, size, color, halo,
                                  bold=bold, text_scale=opts.text_scale)
            else:
                engine.draw_road_label(canvas, text, pts, size, color, halo,
                                       bold=bold, text_scale=opts.text_scale)


def _bbox_to_latlon(bbox) -> tuple[float, float, float, float]:
    """UTMBoundingBox -> (south, west, north, east)."""
    return (bbox.bottom_left.to_latlon()[0], bbox.bottom_left.to_latlon()[1],
            bbox.top_right.to_latlon()[0], bbox.top_right.to_latlon()[1])


def _obf_intersects(obf: ObfFile, query_bbox: tuple[float, float, float, float],
                    zoom: int) -> bool:
    """Quick reject: file has no root covering the query bbox at this zoom."""
    south, west, north, east = query_bbox
    from obf.primitives import get31_x_from_lon, get31_y_from_lat
    left = get31_x_from_lon(west)
    right = get31_x_from_lon(east)
    top = get31_y_from_lat(north)
    bottom = get31_y_from_lat(south)
    for index in obf.map_indexes:
        for root in index.select_roots(zoom):
            if not (root.right < left or root.left > right or
                    root.top > bottom or root.bottom < top):
                return True
    return False
