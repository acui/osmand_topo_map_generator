#!/usr/bin/env python
"""Debug tool: dump OBF map objects around a UTM point.

Examples::

    python tools/obf_dump.py ~/osmand-data/China_beijing_asia.obf \\
        --zone 50T --easting 420500 --northing 4437185 --radius 800
    python tools/obf_dump.py ~/osmand-data/srtm/China_beijing_asia.srtmf.obf \\
        --zone 50T --easting 420500 --northing 4437185 --radius 800 --contours
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utm  # noqa: E402

from obf.obf_file import ObfFile  # noqa: E402


def parse_zone(text: str) -> tuple[int, str]:
    return int(text[:-1]), text[-1].upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obf_files", nargs="+", type=Path)
    parser.add_argument("--zone", required=True, help="UTM zone, e.g. 50T")
    parser.add_argument("--easting", type=int, required=True)
    parser.add_argument("--northing", type=int, required=True)
    parser.add_argument("--radius", type=float, default=500.0, help="radius in meters")
    parser.add_argument("--zoom", type=int, default=15)
    parser.add_argument("--limit", type=int, default=40, help="max objects printed")
    parser.add_argument("--filter", default=None, help="substring filter on tag=value")
    parser.add_argument("--contours", action="store_true", help="show only contour lines")
    parser.add_argument("--names", action="store_true", help="only objects with names")
    args = parser.parse_args()

    zone_number, zone_letter = parse_zone(args.zone)
    lat, lon = utm.to_latlon(args.easting, args.northing, zone_number, zone_letter)
    print(f"center: lat={lat:.6f} lon={lon:.6f} radius={args.radius}m zoom={args.zoom}")
    d = args.radius / 111_320.0
    bbox = (lat - d, lon - d / max(0.2, __import__("math").cos(__import__("math").radians(lat))),
            lat + d, lon + d / max(0.2, __import__("math").cos(__import__("math").radians(lat))))

    total = 0
    for path in args.obf_files:
        t0 = time.time()
        obf = ObfFile(path)
        print(f"\n=== {path.name} version={obf.version} "
              f"({time.time() - t0:.2f}s to open) ===")
        for index in obf.map_indexes:
            roots = ", ".join(f"[{r.min_zoom}-{r.max_zoom} trees={len(r.trees)}]" for r in index.roots)
            print(f"  map section: name={index.name!r} rules={len(index.rules)} roots: {roots}")
        t0 = time.time()
        objects = obf.query_map_objects(bbox, zoom=args.zoom)
        print(f"  query: {len(objects)} objects in {time.time() - t0:.2f}s")

        shown = 0
        type_counter: dict[str, int] = {}
        named = 0
        for obj in objects:
            tag_strs = [f"{tvp.tag}={tvp.value}" for tvp in obj.types]
            if args.contours and not any(t.startswith("contour") or t.startswith("ele") for t in tag_strs):
                continue
            if args.filter and not any(args.filter in t for t in tag_strs):
                continue
            if obj.tags.get("name"):
                named += 1
            for t in tag_strs:
                type_counter[t] = type_counter.get(t, 0) + 1
            total += 1
            if shown < args.limit:
                shown += 1
                kind = "poly" if obj.polygon else ("node" if obj.node else "line")
                pts = f"pts={len(obj.points)}"
                print(f"  [{kind}] {';'.join(tag_strs)} {pts}")
                if obj.tags:
                    items = list(obj.tags.items())[:6]
                    print(f"        tags: {items}")
        if args.contours or args.filter:
            print(f"  type histogram (filtered):")
        else:
            print(f"  type histogram (all):")
        for tag, count in sorted(type_counter.items(), key=lambda kv: -kv[1])[:25]:
            print(f"    {tag}: {count}")
        print(f"  named objects: {named}")
        obf.close()
    print(f"\ntotal printed: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
