"""Decoding of OBF map sections: roots, tile trees, data blocks and objects.

Layout of a map section (``OsmAndMapIndex``)::

    field 2  NAME       string
    field 4  RULES      repeated MapEncodingRule {TAG=3, VALUE=5, ID=7, TYPE=10}
    field 5  LEVELS     MapRootLevel {MAXZOOM=1, MINZOOM=2, LEFT=3, RIGHT=4,
                                    TOP=5, BOTTOM=6, BOXES=7}
    MapDataBox (quad tree node) {LEFT=1, RIGHT=2, TOP=3, BOTTOM=4,
                                 SHIFTTOMAPDATA=5, OCEAN=6, BOXES=7}
    MapDataBlock {BASEID=10, DATAOBJECTS=12, STRINGTABLE=15}
    MapData      {COORDINATES=1, AREACOORDINATES=2, POLYGONINNERCOORDINATES=4,
                  ADDITIONALTYPES=6, TYPES=7, LABELCOORDINATES=8,
                  STRINGNAMES=10, ID=12}

All section/level/box lengths use OsmAnd's custom big-endian ``readInt``
framing; sub-messages inside data blocks use regular protobuf varint
lengths.  Coordinates are deltas on the 31-bit mercator grid, shifted left
by 5 bits (1/32 unit precision) and relative to the enclosing tile origin.

Reference: ``BinaryMapIndexReader.readMapIndex / readMapLevel /
searchMapTreeBounds / readMapDataBlocks / readMapDataObject``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from obf.primitives import ProtoReader, get31_latitude_y, get31_longitude_x, to_int32
from obf.rules import EncodingRules, TagValuePair

_COORDINATE_SHIFT = 5
_MASK_TO_READ = ~((1 << _COORDINATE_SHIFT) - 1)

# OsmAndMapIndex field numbers
_FIELD_NAME = 2
_FIELD_RULES = 4
_FIELD_LEVELS = 5
# MapRootLevel
_FIELD_MAXZOOM = 1
_FIELD_MINZOOM = 2
_FIELD_LEFT = 3
_FIELD_RIGHT = 4
_FIELD_TOP = 5
_FIELD_BOTTOM = 6
_FIELD_BOXES = 7
_FIELD_BLOCKS = 15
# MapDataBox
_FIELD_SHIFT_TO_DATA = 5
_FIELD_OCEAN = 6
# MapDataBlock
_FIELD_BASE_ID = 10
_FIELD_DATA_OBJECTS = 12
_FIELD_STRING_TABLE = 15
# MapData
_FIELD_COORDINATES = 1
_FIELD_AREA_COORDINATES = 2
_FIELD_INNER_COORDINATES = 4
_FIELD_ADDITIONAL_TYPES = 6
_FIELD_TYPES = 7
_FIELD_LABEL_COORDINATES = 8
_FIELD_STRING_NAMES = 10
_FIELD_ID = 12


@dataclass
class MapObject:
    """A decoded vector map object with geographic coordinates."""

    points: list[tuple[float, float]]  # (lat, lon), outer ring for polygons
    polygon: bool
    types: list[TagValuePair]
    inner_rings: list[list[tuple[float, float]]] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    additional_types: list[TagValuePair] = field(default_factory=list)
    osm_id: int = 0
    label: tuple[float, float] | None = None  # (lat, lon)
    node: bool = False
    source: str = ""
    root_min_zoom: int = 0
    root_max_zoom: int = 0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(min_lat, min_lon, max_lat, max_lon)"""
        lats = [p[0] for p in self.points]
        lons = [p[1] for p in self.points]
        return (min(lats), min(lons), max(lats), max(lons))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        kinds = "polygon" if self.polygon else ("node" if self.node else "line")
        type_str = ",".join(str(t) for t in self.types)
        name = self.tags.get("name", "")
        return f"<MapObject {kinds} [{type_str}] {name!r} pts={len(self.points)}>"


@dataclass
class _Box:
    """Quad tree node (MapDataBox) or root tree."""

    file_pointer: int = 0
    length: int = 0
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    ocean: bool | None = None
    map_data_block: int = 0
    children: list["_Box"] = field(default_factory=list)


@dataclass
class MapRoot:
    min_zoom: int = 0
    max_zoom: int = 0
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    trees: list[_Box] = field(default_factory=list)
    # lazily parsed boxes: (absolute offset, length)
    pending_boxes: list[tuple[int, int]] = field(default_factory=list)


class MapIndex:
    """One map section of an OBF file."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.name: str = ""
        self.rules = EncodingRules()
        self.roots: list[MapRoot] = []
        self.file_pointer: int = 0
        self.length: int = 0
        self._buf: memoryview | None = None
        self._rules_loaded = False

    # ------------------------------------------------------------------
    # index header parsing
    # ------------------------------------------------------------------

    def parse_header(self, reader: ProtoReader) -> None:
        """Parse name, encoding rules and zoom level/tree structure.

        ``reader`` must be limited to the map section body; positions of the
        stored tree file pointers are absolute offsets in the whole file.
        """
        self.file_pointer = reader.pos
        self.length = reader.remaining()
        default_id = 1
        while True:
            tag = reader.read_tag()
            if tag == 0:
                break
            f = ProtoReader.tag_field(tag)
            if f == _FIELD_NAME:
                self.name = reader.read_string()
            elif f == _FIELD_RULES:
                sub = reader.sub_reader(reader.read_length())
                rule_id, tvp = self._parse_rule(sub, default_id)
                default_id += 1
                self.rules.add(rule_id, tvp.tag, tvp.value, tvp.rule_type)
            elif f == _FIELD_LEVELS:
                length = reader.read_section_length()
                body = reader.sub_reader(length)
                root = self._parse_root_level(body)
                self.roots.append(root)
            else:
                reader.skip_unknown_field(tag)
        self.rules.finish_initializing_tags()
        self._rules_loaded = True

    @staticmethod
    def _parse_rule(reader: ProtoReader, default_id: int) -> tuple[int, TagValuePair]:
        rule_id = default_id
        tag_str = ""
        value_str = ""
        rule_type = 0
        while True:
            tag = reader.read_tag()
            if tag == 0:
                break
            f = ProtoReader.tag_field(tag)
            if f == 3:  # TAG
                tag_str = reader.read_string()
            elif f == 5:  # VALUE
                value_str = reader.read_string()
            elif f == 7:  # ID
                rule_id = reader.read_uint32()
            elif f == 10:  # TYPE
                rule_type = reader.read_uint32()
            else:
                reader.skip_unknown_field(tag)
        return rule_id, TagValuePair(tag_str, value_str, rule_type)

    def _parse_root_level(self, reader: ProtoReader) -> MapRoot:
        root = MapRoot()
        while True:
            tag = reader.read_tag()
            if tag == 0:
                break
            f = ProtoReader.tag_field(tag)
            if f == _FIELD_MAXZOOM:
                root.max_zoom = reader.read_int32()
            elif f == _FIELD_MINZOOM:
                root.min_zoom = reader.read_int32()
            elif f == _FIELD_LEFT:
                root.left = reader.read_int32()
            elif f == _FIELD_RIGHT:
                root.right = reader.read_int32()
            elif f == _FIELD_TOP:
                root.top = reader.read_int32()
            elif f == _FIELD_BOTTOM:
                root.bottom = reader.read_int32()
            elif f == _FIELD_BOXES:
                length = reader.read_section_length()
                # tree parsing is deferred until a query touches this root
                root.pending_boxes.append((reader.pos, length))
                reader.skip(length)
            elif f == _FIELD_BLOCKS:
                # object blocks stored inline are an old-format leftover; skip
                reader.skip(reader.remaining())
                break
            else:
                reader.skip_unknown_field(tag)
        return root

    def _parse_box(self, reader: ProtoReader, box: _Box, root: MapRoot,
                   parent: _Box | None = None) -> None:
        """Recursively parse one quad tree box (Java ``readMapTreeBounds``)."""
        # children are stored relative to the root bounds for root trees,
        # relative to the parent box otherwise
        anchor = root if parent is None else parent
        box.left = anchor.left
        box.right = anchor.right
        box.top = anchor.top
        box.bottom = anchor.bottom
        while True:
            tag = reader.read_tag()
            if tag == 0:
                return
            f = ProtoReader.tag_field(tag)
            if f == 1:  # LEFT
                box.left = to_int32(reader.read_sint32() + anchor.left)
            elif f == 2:  # RIGHT
                box.right = to_int32(reader.read_sint32() + anchor.right)
            elif f == 3:  # TOP
                box.top = to_int32(reader.read_sint32() + anchor.top)
            elif f == 4:  # BOTTOM
                box.bottom = to_int32(reader.read_sint32() + anchor.bottom)
            elif f == _FIELD_OCEAN:
                box.ocean = reader.read_bool()
            elif f == _FIELD_SHIFT_TO_DATA:
                box.map_data_block = reader.read_section_length() + box.file_pointer
            elif f == _FIELD_BOXES:
                length = reader.read_section_length()
                child = _Box(file_pointer=reader.pos, length=length)
                if box.ocean is not None:
                    child.ocean = box.ocean
                sub = reader.sub_reader(length)
                self._parse_box(sub, child, root, parent=box)
                box.children.append(child)
            else:
                reader.skip_unknown_field(tag)

    # ------------------------------------------------------------------
    # searching / object decoding
    # ------------------------------------------------------------------

    def attach_buffer(self, buf: memoryview) -> None:
        """Provide the whole-file buffer so data blocks can be read lazily."""
        self._buf = buf

    def _ensure_rules(self) -> None:
        if self._rules_loaded and self._buf is not None:
            return
        if self._buf is None:
            raise RuntimeError("buffer not attached")
        reader = ProtoReader(self._buf, self.file_pointer, self.file_pointer + self.length)
        self.roots.clear()
        self.parse_header(reader)

    def _ensure_trees(self, root: MapRoot) -> None:
        if root.trees or not root.pending_boxes:
            return
        assert self._buf is not None
        for offset, length in root.pending_boxes:
            box = _Box(file_pointer=offset, length=length)
            sub = ProtoReader(self._buf, offset, offset + length)
            self._parse_box(sub, box, root)
            root.trees.append(box)
        root.pending_boxes.clear()

    def select_roots(self, zoom: int) -> list[MapRoot]:
        return [r for r in self.roots if r.min_zoom <= zoom <= r.max_zoom]

    def query(self, left: int, top: int, right: int, bottom: int, zoom: int) -> list[MapObject]:
        """Query objects intersecting the given 31-bit bbox at ``zoom``."""
        self._ensure_rules()
        req = (left, top, right, bottom)
        objects: list[MapObject] = []
        for root in self.select_roots(zoom):
            if not self._root_intersects(root, req):
                continue
            self._ensure_trees(root)
            leaves = self._collect_leaves(root, req)
            leaves.sort(key=lambda b: b.map_data_block)
            assert self._buf is not None
            for leaf in leaves:
                reader = ProtoReader(self._buf, leaf.map_data_block)
                length = reader.read_length()
                self._read_data_block(reader.sub_reader(length), leaf, req, zoom, objects)
        return objects

    @staticmethod
    def _root_intersects(root: MapRoot, req: tuple[int, int, int, int]) -> bool:
        left, top, right, bottom = req
        return not (root.right < left or root.left > right or
                    root.top > bottom or root.bottom < top)

    @staticmethod
    def _intersects(box: _Box, req: tuple[int, int, int, int]) -> bool:
        left, top, right, bottom = req
        return not (box.right < left or box.left > right or box.top > bottom or box.bottom < top)

    def _collect_leaves(self, root: MapRoot, req: tuple[int, int, int, int]) -> list[_Box]:
        leaves: list[_Box] = []

        def walk(box: _Box) -> None:
            if not self._intersects(box, req):
                return
            if box.map_data_block:
                leaves.append(box)
            for child in box.children:
                walk(child)

        for tree in root.trees:
            walk(tree)
        return leaves

    # ------------------------------------------------------------------
    # data block / object decoding
    # ------------------------------------------------------------------

    def _read_data_block(self, reader: ProtoReader, leaf: _Box,
                         req: tuple[int, int, int, int], zoom: int,
                         out: list[MapObject]) -> None:
        base_id = 0
        pending: list[MapObject] = []
        pending_names: list[list[tuple[int, int]]] = []
        while True:
            tag = reader.read_tag()
            if tag == 0:
                break
            f = ProtoReader.tag_field(tag)
            if f == _FIELD_BASE_ID:
                base_id = reader.read_uint64()
            elif f == _FIELD_DATA_OBJECTS:
                sub = reader.sub_reader(reader.read_length())
                obj, names = self._read_map_data_object(sub, leaf, req)
                if obj is not None:
                    obj.osm_id = (obj.osm_id + base_id) >> 1
                    obj.source = self.source
                    obj.root_min_zoom = zoom
                    obj.root_max_zoom = zoom
                    pending.append(obj)
                    pending_names.append(names)
            elif f == _FIELD_STRING_TABLE:
                sub = reader.sub_reader(reader.read_length())
                table = self._read_string_table(sub)
                for obj, names in zip(pending, pending_names):
                    for rule_id, str_index in names:
                        tag_name = self.rules.name_tag_for_id(rule_id)
                        if tag_name is None or str_index >= len(table):
                            continue
                        obj.tags[tag_name] = table[str_index]
            else:
                reader.skip_unknown_field(tag)
        out.extend(pending)

    @staticmethod
    def _read_string_table(reader: ProtoReader) -> list[str]:
        strings: list[str] = []
        while True:
            tag = reader.read_tag()
            if tag == 0:
                break
            if ProtoReader.tag_field(tag) == 1:  # S
                strings.append(reader.read_string())
            else:
                reader.skip_unknown_field(tag)
        return strings

    def _read_map_data_object(self, reader: ProtoReader, leaf: _Box,
                              req: tuple[int, int, int, int]
                              ) -> tuple[MapObject | None, list[tuple[int, int]]]:
        tag = reader.read_tag()
        f = ProtoReader.tag_field(tag)
        polygon = f == _FIELD_AREA_COORDINATES
        if not polygon and f != _FIELD_COORDINATES:
            raise ValueError(f"MapData must start with coordinates, got field {f}")

        coords = self._read_coordinate_pairs(reader.sub_reader(reader.read_length()),
                                             leaf.left, leaf.top)
        # bbox acceptance, mirroring the Java reader: any vertex inside the
        # request bbox, or (if none inside) the vertex bounding box intersects
        left, top, right, bottom = req
        contains = any(left <= x <= right and top <= y <= bottom for x, y in coords)
        if not contains and coords:
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            contains = max(xs) >= left and min(xs) <= right and \
                min(ys) <= bottom and max(ys) >= top
        if not contains:
            reader.skip(reader.remaining())
            return None, []

        inner_rings: list[list[tuple[float, float]]] = []
        additional_types: list[int] = []
        types: list[int] = []
        names: list[tuple[int, int]] = []
        raw_id = 0
        label: tuple[int, int] | None = None

        while True:
            tag = reader.read_tag()
            if tag == 0:
                break
            f = ProtoReader.tag_field(tag)
            if f == _FIELD_INNER_COORDINATES:
                ring = self._read_coordinate_pairs(reader.sub_reader(reader.read_length()),
                                                   leaf.left, leaf.top)
                inner_rings.append(ring)
            elif f == _FIELD_ADDITIONAL_TYPES:
                sub = reader.sub_reader(reader.read_length())
                while not sub.eof:
                    additional_types.append(sub.read_uint32())
            elif f == _FIELD_TYPES:
                sub = reader.sub_reader(reader.read_length())
                while not sub.eof:
                    types.append(sub.read_uint32())
            elif f == _FIELD_LABEL_COORDINATES:
                sub = reader.sub_reader(reader.read_length())
                values = []
                while not sub.eof:
                    values.append(sub.read_sint32())
                if len(values) >= 2:
                    label = (values[0], values[1])
            elif f == _FIELD_STRING_NAMES:
                sub = reader.sub_reader(reader.read_length())
                while not sub.eof:
                    names.append((sub.read_uint32(), sub.read_uint32()))
            elif f == _FIELD_ID:
                raw_id = reader.read_sint64()
            else:
                reader.skip_unknown_field(tag)

        decoded_types = [t for t in (self.rules.decode_type(v) for v in types) if t]
        decoded_additional = [t for t in (self.rules.decode_type(v) for v in additional_types) if t]
        points = [(get31_latitude_y(y), get31_longitude_x(x)) for x, y in coords]
        decoded_inner_rings = [
            [(get31_latitude_y(y), get31_longitude_x(x)) for x, y in ring]
            for ring in inner_rings
        ]
        label_ll = None
        if label is not None:
            label_ll = (get31_latitude_y(label[1]), get31_longitude_x(label[0]))

        obj = MapObject(
            points=points,
            polygon=polygon,
            types=decoded_types,
            inner_rings=decoded_inner_rings,
            additional_types=decoded_additional,
            osm_id=raw_id,
            label=label_ll,
            node=len(points) <= 1,
            source=self.source,
        )
        return obj, names

    @staticmethod
    def _read_coordinate_pairs(reader: ProtoReader, tile_left: int, tile_top: int
                               ) -> list[tuple[int, int]]:
        px = tile_left & _MASK_TO_READ
        py = tile_top & _MASK_TO_READ
        coords: list[tuple[int, int]] = []
        while reader.remaining() > 0:
            x = to_int32(to_int32(reader.read_sint32() << _COORDINATE_SHIFT) + px)
            y = to_int32(to_int32(reader.read_sint32() << _COORDINATE_SHIFT) + py)
            coords.append((x, y))
            px, py = x, y
        return coords
