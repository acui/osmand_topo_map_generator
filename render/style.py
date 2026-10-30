"""OsmAnd render.xml style engine (practical subset).

Compiles ``*.render.xml`` styles into rule trees and resolves draw
specifications for map objects, mirroring the semantics of OsmAnd's Java
implementation (``RenderingRulesStorage`` / ``RenderingRuleSearchRequest``):

* ``switch``/``case`` build the if/else tree, ``apply``/``apply_if`` are
  additional attribute setters visited after the matched case chain.
* Rule attributes classified as *input* (``tag``, ``value``, ``additional``,
  ``minzoom``, ``maxzoom``, ``point``, ``area``, ``cycle``, ``layer``,
  ``nightMode``, ``nameTag``, ``textLength`` plus declared
  ``renderingProperty`` attributes) are match conditions; every other
  attribute is an *output* collected into the result state.
* Lookup key falls back ``(tag, value) -> (tag, "") -> ("", "")``.
* ``$name`` output values reference ``renderingAttribute`` rule trees whose
  cases set ``attrColorValue``/``attrIntValue``/``attrBoolValue``/
  ``attrStringValue``.
* ``seq="a:b"`` attributes expand rules with a ``#SEQ`` placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from obf.map_index import MapObject
from obf.rules import EncodingRules

SEQ_ATTR = "seq"
SEQ_PLACEHOLDER = "#SEQ"

SECTIONS = ("point", "line", "polygon", "text", "order")

# attributes that act as match conditions
_INPUT_STRING = {"tag", "value", "nameTag"}
_INPUT_BOOL = {"point", "area", "cycle", "nightMode", "test"}
_INPUT_INT = {"layer", "textLength"}
_GREATER_INT = {"minzoom"}   # rule value must be <= state zoom
_LESS_INT = {"maxzoom"}      # rule value must be >= state zoom
# outputs used by renderingAttribute trees
_ATTR_VALUE_KEYS = {"attrColorValue", "attrIntValue", "attrBoolValue", "attrStringValue"}


@dataclass
class Rule:
    """One rendering rule node (case/switch/apply)."""

    attrs: dict[str, str]
    group: bool
    input: dict[str, str] = field(default_factory=dict)
    output: dict[str, str] = field(default_factory=dict)
    if_else: list["Rule"] = field(default_factory=list)   # case children
    if_children: list["Rule"] = field(default_factory=list)  # apply children


@dataclass
class DrawSpec:
    """Resolved draw specification for one object/zoom."""

    outputs: dict[str, str]
    order: int = 0
    object_type: int = 0
    layer: int = 0

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.outputs.get(key, default)

    @property
    def color(self) -> str | None:
        return self.outputs.get("color")

    @property
    def stroke_width(self) -> float | None:
        raw = self.outputs.get("strokeWidth")
        if raw is None:
            return None
        return float(raw.split(":")[0] or 0)

    @property
    def path_effect(self) -> list[float] | None:
        raw = self.outputs.get("pathEffect")
        if not raw:
            return None
        return [float(v) for v in raw.split(",")]

    @property
    def cap(self) -> str | None:
        return self.outputs.get("cap")

    @property
    def text_order(self) -> int:
        raw = self.outputs.get("textOrder")
        return int(float(raw)) if raw else 0

    @property
    def text_size(self) -> float | None:
        raw = self.outputs.get("textSize")
        return float(raw) if raw else None

    @property
    def text_color(self) -> str | None:
        return self.outputs.get("textColor")

    @property
    def text_halo_radius(self) -> float:
        raw = self.outputs.get("textHaloRadius")
        return float(raw) if raw else 0.0

    @property
    def name_tag(self) -> str | None:
        return self.outputs.get("nameTag") or None

    @property
    def name_tag2(self) -> str | None:
        return self.outputs.get("nameTag2")

    @property
    def text_on_path(self) -> bool:
        return self.outputs.get("textOnPath", "false") == "true"

    @property
    def icon(self) -> str | None:
        return self.outputs.get("icon")


class MatchState:
    """Mutable matching state (Java ``RenderingRulesState`` equivalent)."""

    __slots__ = (
        "tag", "value", "zoom", "layer", "additional", "additional_rules",
        "point", "area", "cycle", "name_tag", "text_length", "props",
        "outputs", "disable", "attr_values",
    )

    def __init__(self) -> None:
        self.tag = ""
        self.value = ""
        self.zoom = -1
        self.layer = 0
        self.additional: list[tuple[str, str]] = []       # decoded additional types
        self.additional_rules: EncodingRules | None = None
        self.point = False
        self.area = False
        self.cycle = False
        self.name_tag = ""
        self.text_length = 0
        self.props: dict[str, str | bool | float] = {}    # renderingProperty inputs
        self.outputs: dict[str, str] = {}
        self.disable = False
        self.attr_values: dict[str, str] | None = None

    def set_object(self, obj: MapObject, zoom: int) -> None:
        self.zoom = zoom
        self.layer = 0
        self.point = obj.node or (not obj.polygon and len(obj.points) <= 1)
        self.area = obj.polygon
        self.cycle = False
        self.additional = [(t.tag, t.value) for t in obj.additional_types]
        self.disable = False


class RenderStyle:
    """Compiled rendering style."""

    def __init__(self) -> None:
        self.constants: dict[str, str] = {}
        self.properties: dict[str, dict] = {}   # renderingProperty declarations
        self.attributes: dict[str, list[Rule]] = {}  # renderingAttribute trees
        self.sections: dict[str, dict[tuple[str, str], Rule]] = {s: {} for s in SECTIONS}

    # ------------------------------------------------------------------
    # parsing
    # ------------------------------------------------------------------

    def load(self, path: str | Path, depends: "RenderStyle | None" = None,
             addon: bool = False) -> None:
        """Parse a render.xml file.

        ``depends`` provides base constants; ``addon=True`` merges an
        ``*.addon.render.xml`` into this storage without overriding existing
        properties (Java parses addon styles into the same storage).
        """
        if depends is not None:
            self.constants.update(depends.constants)
        root = etree.parse(str(path)).getroot()
        for element in root:
            if isinstance(element, etree._Comment):
                continue
            tag = etree.QName(element).localname
            if tag == "renderingConstant":
                name = element.get("name")
                if name and name not in self.constants:
                    self.constants[name] = element.get("value", "")
            elif tag == "renderingProperty":
                attr = element.get("attr", "")
                if addon and attr in self.properties:
                    continue
                self._parse_property(element)
            elif tag == "renderingAttribute":
                name = element.get("name", "")
                rules = [self._parse_child(e) for e in element if not isinstance(e, etree._Comment)]
                rules = [r for r in rules if r is not None]
                self.attributes.setdefault(name, []).extend(rules)
            elif tag in SECTIONS:
                self._register_top(element, tag)
            # renderingClass / legend elements: not needed for rendering

    def _parse_property(self, element: etree._Element) -> None:
        attr = element.get("attr", "")
        type_name = (element.get("type") or "").lower()
        possible = [v.strip() for v in (element.get("possibleValues") or "").split(",") if v.strip()]
        if type_name == "boolean":
            default: str | bool = False
        elif type_name == "string":
            default = possible[0] if possible else ""
        elif possible:
            default = possible[0]
        else:
            default = ""
        self.properties[attr] = {
            "type": type_name or ("boolean" if not possible else "string"),
            "possible": possible,
            "default": default,
        }

    def _substitute(self, attrs: dict[str, str]) -> dict[str, str]:
        out = {}
        for key, value in attrs.items():
            if value.startswith("$"):
                name = value[1:]
                if name in self.constants:
                    value = self.constants[name]
                # otherwise keep "$name": resolved against attributes at match time
            out[key] = value
        return out

    def _parse_child(self, element: etree._Element,
                     seq_context: dict[str, str] | None = None) -> Rule | None:
        if isinstance(element, etree._Comment):
            return None
        name = etree.QName(element).localname
        if name not in ("switch", "group", "case", "filter", "apply", "apply_if", "groupFilter"):
            return None
        attrs = dict(element.attrib)
        if seq_context is not None:
            attrs = {k: v.replace(SEQ_PLACEHOLDER, seq_context["value"]) for k, v in attrs.items()}
        attrs = self._substitute(attrs)
        rule = Rule(attrs=attrs, group=name in ("switch", "group"))
        self._classify(attrs, rule)
        for child in element:
            if isinstance(child, etree._Comment):
                continue
            seq = child.get(SEQ_ATTR)
            if seq:
                start, end = seq.split(":")
                for i in range(int(start), int(end)):
                    cattrs = dict(child.attrib)
                    cattrs.pop(SEQ_ATTR, None)
                    fake = etree.Element(child.tag, attrib=cattrs)
                    sub = self._parse_child(fake, seq_context={"value": str(i)})
                    if sub is not None:
                        rule.if_else.append(sub)
            else:
                sub = self._parse_child(child)
                if sub is None:
                    continue
                if etree.QName(child).localname in ("apply", "apply_if", "groupFilter"):
                    rule.if_children.append(sub)
                else:
                    rule.if_else.append(sub)
        return rule

    def _classify(self, attrs: dict[str, str], rule: Rule) -> None:
        prop_inputs = set(self.properties)
        for key, value in attrs.items():
            if key in _INPUT_STRING or key in _INPUT_BOOL or key in _INPUT_INT \
                    or key in _GREATER_INT or key in _LESS_INT or key == "additional" \
                    or key in prop_inputs:
                rule.input[key] = value
            else:
                rule.output[key] = value

    def _register_top(self, element: etree._Element, section: str) -> None:
        # the section element itself (line/polygon/order/...) only selects the
        # rule group; its direct children are the top-level rules
        for child in element:
            if isinstance(child, etree._Comment):
                continue
            rule = self._parse_child(child)
            if rule is None:
                continue
            self._register_top_level(rule, [], {}, section)

    def _register_top_level(self, rule: Rule, apply_rules: list[Rule],
                            attrs: dict[str, str], section: str) -> None:
        """Flatten top-level tagless groups (Java ``registerTopLevel``)."""
        if rule.group and ("tag" not in rule.attrs or "value" not in rule.attrs):
            merged_apply = list(rule.if_children) + list(apply_rules)
            for child in rule.if_else:
                child_attrs = dict(attrs)
                child_attrs.update(rule.attrs)
                self._register_top_level(child, merged_apply, child_attrs, section)
            return
        merged = dict(attrs)
        merged.update(rule.attrs)
        tag = merged.pop("tag", "")
        value = merged.pop("value", "")
        rule.attrs = merged
        rule.input, rule.output = {}, {}
        self._classify(merged, rule)
        for ap in apply_rules:
            rule.if_children.append(ap)
        existing = self.sections[section].get((tag, value))
        if existing is not None:
            # multiple top rules may share one key: chain them, earlier first
            wrapper = Rule(attrs={}, group=True)
            wrapper.if_else = [existing, rule]
            self.sections[section][(tag, value)] = wrapper
        else:
            self.sections[section][(tag, value)] = rule

    # ------------------------------------------------------------------
    # matching
    # ------------------------------------------------------------------

    def _check_input(self, rule: Rule, state: MatchState) -> bool:
        for key, expected in rule.input.items():
            if key == "tag":
                if expected != state.tag:
                    return False
            elif key == "value":
                if expected != state.value:
                    return False
            elif key == "nameTag":
                if expected != state.name_tag:
                    return False
            elif key == "minzoom":
                if int(float(expected)) > state.zoom:
                    return False
            elif key == "maxzoom":
                if int(float(expected)) < state.zoom:
                    return False
            elif key == "layer":
                if int(float(expected)) != state.layer:
                    return False
            elif key == "textLength":
                if int(float(expected)) != state.text_length:
                    return False
            elif key == "additional":
                if not self._match_additional(expected, state):
                    return False
            elif key in ("point", "area", "cycle", "nightMode", "test"):
                if key in ("point", "area", "cycle"):
                    actual = getattr(state, key)
                else:
                    actual = bool(state.props.get(key, False))
                if (expected == "true") != bool(actual):
                    return False
            elif key in state.props:
                actual = state.props[key]
                if isinstance(actual, bool):
                    if (str(expected).lower() == "true") != actual:
                        return False
                elif str(actual) != str(expected):
                    return False
            else:
                return False
        return True

    @staticmethod
    def _match_additional(expected: str, state: MatchState) -> bool:
        negate = expected.startswith("!")
        if negate:
            expected = expected[1:]
        if "=" in expected and not negate:
            tag, value = expected.split("=", 1)
            if state.additional_rules is not None:
                rule_id = state.additional_rules.get_rule(tag, value)
                if rule_id is not None and any(
                        state.additional_rules.get_rule(t, v) == rule_id
                        for t, v in state.additional):
                    return True
            return (tag, value) in state.additional
        found = any(t == expected for t, _ in state.additional)
        return (not found) if negate else found

    def _load_output(self, rule: Rule, state: MatchState, override: bool) -> None:
        for key, raw in rule.output.items():
            if raw.startswith("$"):
                evaluated = self._eval_attribute(raw[1:], state)
                if evaluated is not None and (override or key not in state.outputs):
                    state.outputs[key] = str(evaluated)
                    if state.attr_values is not None:
                        state.attr_values[key] = str(evaluated)
            else:
                if override or key not in state.outputs:
                    state.outputs[key] = raw
                    if state.attr_values is not None:
                        state.attr_values[key] = raw
            if key == "disable":
                state.disable = raw == "true"

    def _eval_attribute(self, name: str, state: MatchState) -> str | None:
        rules = self.attributes.get(name)
        if not rules:
            return None
        saved_attr_values = state.attr_values
        state.attr_values = {}
        for root in rules:
            self._visit(root, state)
        values = state.attr_values or {}
        state.attr_values = saved_attr_values
        for key in ("attrColorValue", "attrIntValue", "attrBoolValue", "attrStringValue"):
            if key in values:
                return values[key]
        return None

    def _visit(self, rule: Rule, state: MatchState) -> bool:
        if not self._check_input(rule, state):
            return False
        if not rule.group:
            self._load_output(rule, state, override=True)
        matched = False
        for child in rule.if_else:
            if self._visit(child, state):
                matched = True
                break
        fit = matched or not rule.group
        if fit:
            if rule.group:
                self._load_output(rule, state, override=False)
            for ap in rule.if_children:
                self._visit(ap, state)
        return fit

    def _search(self, section: str, state: MatchState) -> bool:
        table = self.sections[section]
        for key in ((state.tag, state.value), (state.tag, ""), ("", "")):
            rule = table.get(key)
            if rule is not None and self._visit(rule, state) and not state.disable:
                return True
            state.disable = False
        return False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def resolve(self, section: str, obj: MapObject, zoom: int,
                layer: int = 0, name_tag: str = "", text_length: int = 0,
                tag: str | None = None, value: str | None = None,
                props: dict[str, str | bool] | None = None) -> DrawSpec | None:
        """Resolve the draw specification for one object in a section."""
        if tag is None:
            if not obj.types:
                return None
            main = obj.types[0]
            tag, value = main.tag, main.value
        # text rules are declared with nameTag="" (any name) or a specific tag;
        # try the generic form first, then the explicit one
        name_tag_candidates = ("", name_tag) if name_tag else ("",)
        for candidate in name_tag_candidates:
            state = MatchState()
            state.zoom = zoom
            state.layer = layer
            state.point = obj.node or (not obj.polygon and len(obj.points) <= 1)
            state.area = obj.polygon
            state.cycle = obj.polygon
            state.additional = [(t.tag, t.value) for t in obj.additional_types]
            state.name_tag = candidate
            state.text_length = text_length
            for prop, spec in self.properties.items():
                state.props[prop] = spec["default"]
            if props:
                state.props.update(props)
            state.tag, state.value = tag, value
            if self._search(section, state):
                spec = DrawSpec(outputs=dict(state.outputs))
                spec.order = int(float(state.outputs.get("order", "0") or 0))
                spec.object_type = int(float(state.outputs.get("objectType", "0") or 0))
                spec.layer = layer
                if candidate:
                    # remember which name tag matched so the renderer can
                    # fetch the label text (e.g. elevation on contour lines)
                    spec.outputs.setdefault("nameTag", candidate)
                return spec
        return None

    def resolve_order(self, obj: MapObject, zoom: int,
                      props: dict[str, str | bool] | None = None) -> tuple[int, int]:
        """Return ``(min_order, object_type)`` over all object types."""
        best_order = 1 << 30
        best_type = 0
        for tvp in obj.types:
            state = MatchState()
            state.zoom = zoom
            state.point = obj.node or (not obj.polygon and len(obj.points) <= 1)
            state.area = obj.polygon
            state.cycle = obj.polygon
            state.additional = [(t.tag, t.value) for t in obj.additional_types]
            for prop, spec in self.properties.items():
                state.props[prop] = spec["default"]
            if props:
                state.props.update(props)
            state.tag, state.value = tvp.tag, tvp.value
            if self._search("order", state):
                order = int(float(state.outputs.get("order", "128") or 128))
                otype = int(float(state.outputs.get("objectType", "0") or 0))
                if order < best_order:
                    best_order = order
                    best_type = otype
        return best_order, best_type
