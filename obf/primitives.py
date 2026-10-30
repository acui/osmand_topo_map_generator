"""Low-level binary primitives for parsing OBF (OsmAnd binary map) files.

The OBF format combines protobuf-style field framing with OsmAnd's custom
4/8-byte big-endian section length encoding.  This module implements the
byte-level primitives only; message-level decoding lives in the sibling
modules.

Reference: osmandapp/OsmAnd ``BinaryMapIndexReader.java`` and the generated
``OsmandOdb`` protobuf constants.
"""
from __future__ import annotations

import math

_UINT32 = 0xFFFFFFFF
_POW31 = float(1 << 31)


def to_int32(value: int) -> int:
    """Wrap a Python int to a signed 32-bit integer (Java ``int`` semantics)."""
    value &= _UINT32
    return value - (1 << 32) if value >= (1 << 31) else value


def zigzag_decode(value: int) -> int:
    """Decode a protobuf signed varint (sint32/sint64 zigzag encoding)."""
    return (value >> 1) ^ -(value & 1)


# ---------------------------------------------------------------------------
# 31-bit map coordinates (Web Mercator on a 2^31 grid)
# ---------------------------------------------------------------------------

def get31_longitude_x(x: int) -> float:
    """Convert a 31-bit map X coordinate to longitude in degrees."""
    return (x * 360.0) / _POW31 - 180.0


def get31_latitude_y(y: int) -> float:
    """Convert a 31-bit map Y coordinate to latitude in degrees (Web Mercator)."""
    t = math.pi * (1.0 - 2.0 * y / _POW31)
    return math.degrees(math.atan(math.sinh(t)))


def get31_x_from_lon(lon: float) -> int:
    """Convert longitude in degrees to a 31-bit map X coordinate."""
    return int(round((lon + 180.0) / 360.0 * _POW31))


def get31_y_from_lat(lat: float) -> int:
    """Convert latitude in degrees to a 31-bit map Y coordinate (Web Mercator)."""
    t = math.radians(max(-85.05112878, min(85.05112878, lat)))
    merc = math.asinh(math.tan(t))  # = ln(tan + sec)
    y = (1.0 - merc / math.pi) / 2.0
    return int(round(y * _POW31))


class ProtoReader:
    """Sequential reader over a byte buffer with protobuf-style primitives.

    ``pos`` advances from ``pos`` up to ``end`` (exclusive).  ``end`` acts as
    the protobuf ``pushLimit`` boundary: ``read_tag`` returns ``0`` once the
    limit is reached, which terminates every message loop.
    """

    __slots__ = ("buf", "pos", "end")

    def __init__(self, buf: bytes | bytearray | memoryview, pos: int = 0, end: int | None = None):
        self.buf: memoryview = memoryview(buf)
        self.pos = pos
        self.end = len(self.buf) if end is None else end
        if self.end > len(self.buf):
            raise ValueError(f"limit {self.end} beyond buffer size {len(self.buf)}")

    # -- basic access ------------------------------------------------------

    @property
    def eof(self) -> bool:
        return self.pos >= self.end

    def remaining(self) -> int:
        return self.end - self.pos

    def read_raw(self, n: int) -> bytes:
        if self.pos + n > self.end:
            raise EOFError(f"cannot read {n} bytes, only {self.remaining()} remain")
        data = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return data

    def skip(self, n: int) -> None:
        if self.pos + n > self.end:
            raise EOFError(f"cannot skip {n} bytes, only {self.remaining()} remain")
        self.pos += n

    def sub_reader(self, length: int) -> "ProtoReader":
        """Return a reader limited to the next ``length`` bytes; self.pos advances."""
        if self.pos + length > self.end:
            raise EOFError(f"submessage of {length} bytes exceeds limit ({self.remaining()} remain)")
        sub = ProtoReader(self.buf, self.pos, self.pos + length)
        self.pos += length
        return sub

    # -- protobuf primitives -------------------------------------------------

    def read_raw_varint(self) -> int:
        """Read an unsigned base-128 varint."""
        result = 0
        shift = 0
        buf = self.buf
        pos = self.pos
        while True:
            if pos >= self.end:
                raise EOFError("varint exceeds limit")
            b = buf[pos]
            pos += 1
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
            if shift > 63:
                raise ValueError("varint too long")
        self.pos = pos
        return result

    def read_tag(self) -> int:
        """Read the next field tag. Returns 0 at the end of the current limit."""
        if self.pos >= self.end:
            return 0
        first = self.buf[self.pos]
        if first < 0x80:
            self.pos += 1
            return first
        return self.read_raw_varint()

    @staticmethod
    def tag_field(tag: int) -> int:
        return tag >> 3

    @staticmethod
    def tag_wire_type(tag: int) -> int:
        return tag & 7

    def read_uint32(self) -> int:
        return self.read_raw_varint()

    def read_uint64(self) -> int:
        return self.read_raw_varint()

    def read_int32(self) -> int:
        """Read a protobuf int32: unsigned varint wrapped to signed 32-bit."""
        return to_int32(self.read_raw_varint())

    def read_sint32(self) -> int:
        """Read a protobuf sint32: zigzag encoded varint."""
        return zigzag_decode(self.read_raw_varint())

    def read_sint64(self) -> int:
        return zigzag_decode(self.read_raw_varint())

    def read_bool(self) -> bool:
        return self.read_raw_varint() != 0

    def read_string(self) -> str:
        """Read a protobuf length-delimited UTF-8 string."""
        length = self.read_raw_varint()
        return self.read_raw(length).decode("utf-8")

    def read_length(self) -> int:
        """Read the length prefix of a normal (wire type 2) field."""
        return self.read_raw_varint()

    def read_section_length(self) -> int:
        """Read OsmAnd's custom section length (Java ``readInt``).

        The first byte signals the width: values above 0x7F mean an 8-byte
        big-endian integer (with the high bit of the first byte stripped),
        otherwise a 4-byte big-endian integer.
        """
        first = self.read_raw(1)[0]
        wide = first > 0x7F
        value = (first & 0x7F) if wide else first
        for _ in range(7 if wide else 3):
            value = (value << 8) | self.read_raw(1)[0]
        return value

    def skip_unknown_field(self, tag: int) -> None:
        """Skip a field the parser does not know about."""
        wire_type = self.tag_wire_type(tag)
        if wire_type == 0:  # varint
            self.read_raw_varint()
        elif wire_type == 1:  # fixed 64
            self.skip(8)
        elif wire_type == 2:  # length delimited
            self.skip(self.read_length())
        elif wire_type == 4:  # OsmAnd custom: big-endian section length
            self.skip(self.read_section_length())
        elif wire_type == 5:  # fixed 32
            self.skip(4)
        elif wire_type == 3:  # start group: skip until matching end group
            while True:
                t = self.read_tag()
                if t == 0:
                    return
                if self.tag_wire_type(t) == 4:
                    return
                self.skip_unknown_field(t)
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
