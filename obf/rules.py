"""MapEncodingRules: the tag/value -> rule-id dictionary of an OBF map section.

Every vector object inside a map section stores its OSM-like attributes as
integer rule ids.  The id -> (tag, value) mapping is defined once per map
section by a list of ``MapEncodingRule`` messages and must be used to decode
``TYPES``/``ADDITIONALTYPES``/``STRINGNAMES`` fields.

Reference: ``BinaryMapIndexReader.MapIndex.initMapEncodingRule``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TagValuePair:
    tag: str
    value: str
    rule_type: int = 0  # additionalAttribute: 0 normal, 1 additional, 2 name-ish

    def __str__(self) -> str:
        return self.tag if not self.value else f"{self.tag}={self.value}"


class EncodingRules:
    """Bidirectional tag/value <-> rule-id table for one map section."""

    __slots__ = ("decode", "encode", "rule_types", "name_rule_ids")

    def __init__(self) -> None:
        self.decode: dict[int, TagValuePair] = {}
        self.encode: dict[tuple[str, str], int] = {}
        self.rule_types: dict[int, int] = {}
        self.name_rule_ids: dict[str, int] = {}  # e.g. {"name": 5, "name:en": 312}

    def __len__(self) -> int:
        return len(self.decode)

    @property
    def empty(self) -> bool:
        return not self.decode

    def add(self, rule_id: int, tag: str, value: str, rule_type: int = 0) -> None:
        """Register a rule (Java ``initMapEncodingRule``)."""
        self.encode[(tag, value)] = rule_id
        if rule_id not in self.decode:
            self.decode[rule_id] = TagValuePair(tag, value, rule_type)
        self.rule_types[rule_id] = rule_type
        if tag.startswith("name"):
            self.name_rule_ids[tag] = rule_id

    def decode_type(self, rule_id: int) -> TagValuePair | None:
        return self.decode.get(rule_id)

    def get_rule(self, tag: str, value: str) -> int | None:
        return self.encode.get((tag, value))

    def finish_initializing_tags(self) -> None:
        """Add synthetic rules every map section relies on.

        Mirrors ``MapIndex.finishInitializingTags``: broken coastlines and a
        fallback ``natural=land`` rule are referenced by renderers even when
        absent from the file's rule table.
        """
        free = len(self.decode)
        for tag, value in (("natural", "coastline_broken"), ("natural", "land")):
            if (tag, value) not in self.encode:
                self.add(free, tag, value)
            free = max(free + 1, len(self.decode))

    # convenience lookups ----------------------------------------------------

    @property
    def name_id(self) -> int | None:
        return self.name_rule_ids.get("name")

    def name_tag_for_id(self, rule_id: int) -> str | None:
        """Return the tag name for a string-name rule id.

        String names are not limited to ``name*`` tags: contour lines carry
        their elevation as a string attribute (``elevation``, ``gpx``...).
        """
        tvp = self.decode.get(rule_id)
        return tvp.tag if tvp is not None else None
