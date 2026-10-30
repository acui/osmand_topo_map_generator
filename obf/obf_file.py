"""File-level access to OBF (OsmAnd binary map) files.

An OBF file is a sequence of top-level protobuf fields (``OsmAndStructure``)::

    VERSION=1  TRANSPORTINDEX=4  MAPINDEX=6  ADDRESSINDEX=7  POIINDEX=8
    ROUTINGINDEX=9  HHROUTINGINDEX=10  DATECREATED=18  VERSIONCONFIRM=32
    OWNER=33

All top-level section fields use OsmAnd's custom big-endian section length.
Only map sections (MAPINDEX) are decoded by this module; everything else is
skipped.
"""
from __future__ import annotations

import mmap
from pathlib import Path

from obf.map_index import MapIndex, MapObject
from obf.primitives import (
    ProtoReader,
    get31_x_from_lon,
    get31_y_from_lat,
)

# OsmAndStructure field numbers
_FIELD_VERSION = 1
_FIELD_TRANSPORT_INDEX = 4
_FIELD_MAP_INDEX = 6
_FIELD_ADDRESS_INDEX = 7
_FIELD_POI_INDEX = 8
_FIELD_ROUTING_INDEX = 9
_FIELD_HH_ROUTING_INDEX = 10
_FIELD_DATE_CREATED = 18
_FIELD_VERSION_CONFIRM = 32
_FIELD_OWNER = 33

_SECTION_FIELDS = {
    _FIELD_TRANSPORT_INDEX,
    _FIELD_MAP_INDEX,
    _FIELD_ADDRESS_INDEX,
    _FIELD_POI_INDEX,
    _FIELD_ROUTING_INDEX,
    _FIELD_HH_ROUTING_INDEX,
    _FIELD_OWNER,
}


class ObfFile:
    """Reader for one ``.obf`` file (map sections only)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._file = open(self.path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, prot=mmap.PROT_READ)
        self.buf: memoryview = memoryview(self._mmap)
        self.version: int | None = None
        self.date_created: int | None = None
        self.map_indexes: list[MapIndex] = []
        self._parse_structure()

    # ------------------------------------------------------------------

    def _parse_structure(self) -> None:
        reader = ProtoReader(self.buf)
        while not reader.eof:
            tag = reader.read_tag()
            if tag == 0:
                break
            f = ProtoReader.tag_field(tag)
            if f == _FIELD_VERSION:
                self.version = reader.read_uint32()
            elif f == _FIELD_DATE_CREATED:
                reader.read_uint64()
            elif f in _SECTION_FIELDS:
                length = reader.read_section_length()
                body_start = reader.pos
                if f == _FIELD_MAP_INDEX:
                    index = MapIndex(source=self.path.name)
                    index.parse_header(reader.sub_reader(length))
                    index.attach_buffer(self.buf)
                    self.map_indexes.append(index)
                reader.pos = body_start + length
            elif f == _FIELD_VERSION_CONFIRM:
                reader.read_uint32()
            else:
                reader.skip_unknown_field(tag)

    # ------------------------------------------------------------------

    def query_map_objects(self, bbox: tuple[float, float, float, float],
                          zoom: int = 15) -> list[MapObject]:
        """Query map objects inside a lat/lon bbox ``(south, west, north, east)``."""
        south, west, north, east = bbox
        left = get31_x_from_lon(west)
        right = get31_x_from_lon(east)
        top = get31_y_from_lat(north)
        bottom = get31_y_from_lat(south)
        objects: list[MapObject] = []
        for index in self.map_indexes:
            objects.extend(index.query(left, top, right, bottom, zoom))
        return objects

    def may_cover(self, bbox: tuple[float, float, float, float],
                  zoom: int = 15) -> bool:
        """Fast check: does any map root cover the lat/lon bbox at ``zoom``?"""
        south, west, north, east = bbox
        left = get31_x_from_lon(west)
        right = get31_x_from_lon(east)
        top = get31_y_from_lat(north)
        bottom = get31_y_from_lat(south)
        for index in self.map_indexes:
            for root in index.select_roots(zoom):
                if not (root.right < left or root.left > right or
                        root.top > bottom or root.bottom < top):
                    return True
        return False

    def close(self) -> None:
        try:
            try:
                self.buf.release()
            except (BufferError, ValueError):
                pass
            self._mmap.close()
        except BufferError:
            pass
        finally:
            self._file.close()

    def __enter__(self) -> "ObfFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"<ObfFile {self.path.name} version={self.version} "
                f"map_sections={len(self.map_indexes)}>")


def open_obf_files(paths: list[str | Path]) -> list[ObfFile]:
    """Open multiple OBF files, skipping ones without map sections."""
    files: list[ObfFile] = []
    for path in paths:
        obf = ObfFile(path)
        if obf.map_indexes:
            files.append(obf)
    return files
