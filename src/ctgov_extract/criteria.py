"""Deterministic splitting of free-text eligibility criteria.

The splitter works line by line and keeps character offsets into the original
text, so that every :class:`Criterion` satisfies ``text == source[start:end]``.

Rules, in order of precedence for each line:

1. Blank lines end the current paragraph.
2. Section headers (``Inclusion Criteria:``, ``Core Study: Exclusion Criteria``,
   ``Key inclusion criteria that must be met by all participants:``) switch the
   current kind. Text after a header colon on the same line becomes a criterion.
3. Lines starting with a list marker (``*``, ``-``, ``•``, ``1.``, ``a)``,
   ``(ii)``...) start a new criterion. Nesting is derived from indentation, and,
   at equal indentation, from a change of marker style after a line ending in ``:``.
4. Other lines ending in ``:`` are treated as sub-headings ("group") and passed on
   as context to the criteria that follow.
5. Other lines continue the current criterion when they directly follow a list
   item (markdown "lazy" continuation), are indented under it, or clearly
   continue a sentence (previous line ends with ``,`` ``;`` ``-``, or the line
   starts in lower case). Otherwise they start a new free-standing criterion.

Text before the first header is kept as ``preamble``. If the text has no header
at all, every criterion is assumed to be an inclusion criterion and a warning
is recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ctgov_extract.models import CriteriaSplit, Criterion, CriterionKind

_MARKER = re.compile(
    r"""
    ^(?P<indent>[ \t]*)
    (?P<marker>
        [*\u2022\u00b7\u25aa\u25e6\u2023\u25cf\u25cb\u25a0\u25a1\u25ba\u27a2\u2013-]
        (?![\d*])                         # symbol bullets (not '-5', not '**bold')
      | \+(?=\s)                           # '+ item'
      | \(?\d{1,3}[.)](?!\d)               # 1.  1)  (1)
      | \(?(?:[ivx]{2,5}|[IVX]{2,5})[.)](?=\s)  # ii.  (iv)  III)
      | \(?[A-Za-z][.)](?=\s)              # a.  b)  (c)  A.
    )
    (?P<space>[ \t]*)
    """,
    re.VERBOSE,
)

_HEADER_KEY = re.compile(r"\b(?P<kind>inclusion|exclusion)(?:\s+criteri(?:a|on|as))?\b", re.I)
_DECORATION = " \t*_#>"
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!<>=~|^])")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove markdown backslash escapes and collapse whitespace."""
    return _WS.sub(" ", _ESCAPE.sub(r"\1", text)).strip()


@dataclass
class _Line:
    text: str  # without line terminator
    start: int  # offset of text[0] in the source

    @property
    def blank(self) -> bool:
        return not self.text.strip()

    @property
    def indent(self) -> int:
        stripped = self.text.lstrip(" \t")
        return len(self.text[: len(self.text) - len(stripped)].expandtabs(4))


@dataclass
class _Header:
    kind: CriterionKind
    label: str
    rest_start: int | None  # offset (within the line) of same-line content, if any


@dataclass
class _Open:
    """A criterion being built."""

    kind: CriterionKind
    start: int
    end: int
    indent: int
    style: str | None
    marker: str | None
    level: int
    id: str
    parent_id: str | None
    section: str | None
    group: str | None
    children: int = 0


@dataclass
class _State:
    source: str
    kind: CriterionKind | None = None
    section: str | None = None
    group: str | None = None
    counters: dict[str, int] = field(default_factory=lambda: {"inclusion": 0, "exclusion": 0})
    stack: list[_Open] = field(default_factory=list)
    current: _Open | None = None
    done: list[_Open] = field(default_factory=list)
    prev_blank: bool = True


def _lines(text: str) -> list[_Line]:
    lines = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        content = raw.rstrip("\r\n")
        lines.append(_Line(content, offset))
        offset += len(raw)
    return lines


def detect_header(line: str) -> _Header | None:
    """Return header info if ``line`` is an inclusion/exclusion section header."""
    if _MARKER.match(line):
        return None
    match = _HEADER_KEY.search(line)
    if not match:
        return None
    prefix = line[: match.start()].strip(_DECORATION)
    if len(prefix) > 60:
        return None
    after = line[match.end() :]
    after_stripped = after.lstrip(_DECORATION)
    kind: CriterionKind = "inclusion" if match.group("kind").lower() == "inclusion" else "exclusion"
    label = line.strip(_DECORATION).rstrip(":").strip(_DECORATION)

    if not after_stripped.strip():
        return _Header(kind, label, None)
    if after_stripped.startswith(":"):
        rest = after_stripped[1:]
        label = line[: match.end()].strip(_DECORATION)
        if not rest.strip(_DECORATION):
            return _Header(kind, label, None)
        rest_offset = len(line) - len(rest)
        rest_offset += len(rest) - len(rest.lstrip(_DECORATION))
        return _Header(kind, label, rest_offset)
    if line.rstrip(_DECORATION).endswith(":") and len(after) <= 80:
        return _Header(kind, label, None)
    return None


def _marker_style(marker: str, stack: list[_Open], indent: int) -> str:
    core = marker.strip("()")
    core = core.rstrip(".)") if len(core) > 1 else core
    if core.isdigit():
        style = "1"
    elif (len(core) > 1 and set(core.lower()) <= set("ivx")) or (
        core.lower() in {"i", "v", "x"}
        and any(o.indent == indent and (o.style or "").lstrip("(") == "i" for o in stack)
    ):
        style = "i"  # roman numerals; a lone i/v/x only when it continues a roman list
    elif core.isalpha() and len(core) == 1:
        style = "a" if core.islower() else "A"
    else:
        style = core  # the bullet symbol itself
    return ("(" if marker.startswith("(") else "") + style


def _close_current(state: _State) -> None:
    if state.current is not None:
        state.done.append(state.current)
        state.current = None


def _new_id(state: _State, kind: CriterionKind, parent: _Open | None) -> str:
    if parent is not None:
        parent.children += 1
        return f"{parent.id}.{parent.children}"
    state.counters[kind] += 1
    prefix = "inc" if kind == "inclusion" else "exc"
    return f"{prefix}-{state.counters[kind]}"


def _content_bounds(line: _Line, offset_in_line: int) -> tuple[int, int] | None:
    text = line.text
    seg = text[offset_in_line:]
    stripped = seg.strip()
    if not stripped:
        return None
    lead = len(seg) - len(seg.lstrip())
    start = line.start + offset_in_line + lead
    end = line.start + len(text.rstrip())
    return start, end


def _start_item(
    state: _State,
    *,
    start: int,
    end: int,
    indent: int,
    style: str | None,
    marker: str | None,
) -> None:
    kind = state.kind or "inclusion"
    _close_current(state)

    parent: _Open | None = None
    if style is None:
        state.stack.clear()
    else:
        stack = state.stack
        while stack:
            top = stack[-1]
            if indent > top.indent:
                break
            if indent < top.indent:
                stack.pop()
                continue
            if style == top.style:
                stack.pop()
                break
            if any(o.indent == indent and o.style == style for o in stack[:-1]):
                stack.pop()
                continue
            if state.source[top.start : top.end].rstrip().endswith(":"):
                break
            stack.pop()
            break
        parent = stack[-1] if stack else None

    item = _Open(
        kind=kind,
        start=start,
        end=end,
        indent=indent,
        style=style,
        marker=marker,
        level=0 if parent is None else parent.level + 1,
        id=_new_id(state, kind, parent),
        parent_id=parent.id if parent else None,
        section=state.section,
        group=state.group,
    )
    if style is not None:
        state.stack.append(item)
    state.current = item


def split_criteria(text: str, nct_id: str | None = None) -> CriteriaSplit:
    """Split eligibility text into inclusion and exclusion criteria with offsets."""
    lines = _lines(text)
    has_header = any(detect_header(line.text) for line in lines)
    state = _State(source=text, kind=None if has_header else "inclusion")
    preamble_parts: list[str] = []
    warnings: list[str] = []
    if text.strip() and not has_header:
        warnings.append("No inclusion/exclusion headers found; all criteria assumed inclusion.")

    for line in lines:
        if line.blank:
            state.prev_blank = True
            continue

        header = detect_header(line.text)
        if header is not None:
            _close_current(state)
            state.stack.clear()
            state.kind = header.kind
            state.section = header.label
            state.group = None
            if header.rest_start is not None:
                bounds = _content_bounds(line, header.rest_start)
                if bounds:
                    _start_item(
                        state, start=bounds[0], end=bounds[1], indent=0, style=None, marker=None
                    )
            state.prev_blank = False
            continue

        if state.kind is None:  # before the first header
            preamble_parts.append(line.text.strip())
            state.prev_blank = False
            continue

        marker_match = _MARKER.match(line.text)
        if marker_match:
            bounds = _content_bounds(line, marker_match.end())
            if bounds is None:  # a lone bullet character
                state.prev_blank = False
                continue
            marker = marker_match.group("marker")
            indent = len(marker_match.group("indent").expandtabs(4))
            style = _marker_style(marker, state.stack, indent)
            _start_item(
                state, start=bounds[0], end=bounds[1], indent=indent, style=style, marker=marker
            )
            state.prev_blank = False
            continue

        bounds = _content_bounds(line, 0)
        assert bounds is not None  # non-blank line
        current = state.current
        if current is None:
            continues = False
        elif state.prev_blank or current.marker is None:
            # After a blank line, or after an unmarked line: ClinicalTrials.gov text is not
            # hard-wrapped, so a new line is a new criterion unless it is indented or
            # clearly continues a sentence.
            previous = text[current.start : current.end].rstrip()
            continues = line.indent > current.indent or (
                not state.prev_blank
                and (previous.endswith((",", "-", ";", "(", "/")) or line.text.strip()[0].islower())
            )
        else:
            continues = True  # markdown "lazy" continuation of a list item
        if continues and current is not None:
            current.end = bounds[1]
        elif line.text.rstrip().endswith(":") and len(line.text.strip()) <= 160:
            _close_current(state)
            state.stack.clear()
            state.group = clean_text(line.text).rstrip(":").strip()
        else:
            _start_item(
                state, start=bounds[0], end=bounds[1], indent=line.indent, style=None, marker=None
            )
        state.prev_blank = False

    _close_current(state)

    criteria = [
        Criterion(
            id=o.id,
            kind=o.kind,
            text=text[o.start : o.end],
            clean_text=clean_text(text[o.start : o.end]),
            start=o.start,
            end=o.end,
            level=o.level,
            marker=o.marker,
            parent_id=o.parent_id,
            section=o.section,
            group=o.group,
        )
        for o in sorted(state.done, key=lambda o: o.start)
    ]
    preamble = " ".join(p for p in preamble_parts if p) or None
    return CriteriaSplit(nct_id=nct_id, criteria=criteria, preamble=preamble, warnings=warnings)
