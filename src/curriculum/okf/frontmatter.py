"""A tiny, dependency-free reader/writer for OKF YAML frontmatter.

WHY this exists instead of pyyaml: the curriculum CORE is stdlib-only (no
third-party imports), yet OKF documents are markdown files prefixed with a
``---``-delimited YAML block (see docs/okf-spec.md sections 4.1 and 4.2). We do
not need the whole YAML language -- only the small, predictable subset that OKF
frontmatter actually uses -- so we parse/emit that subset by hand and keep the
behaviour fully deterministic and round-trippable.

Supported subset
----------------
* A leading block delimited by a line that is exactly ``---`` (optionally with
  trailing spaces/tabs) at the very start of the document, closed by another
  such line. A document without that opening block is returned verbatim as the
  body with empty metadata (OKF consumers must tolerate missing frontmatter).
* One ``key: value`` mapping per line. The key is everything before the first
  colon; the value is the remainder. Blank lines and ``#`` comment lines inside
  the block are ignored.
* Scalar value types: ``str``, ``int``, ``float`` and ``bool`` (``true`` /
  ``false``, case-insensitive on read; always lower-case on write).
* Flat inline lists written ``[a, b, c]``. Elements are scalars only -- nested
  lists/mappings are NOT supported.
* Double- and single-quoted strings. A value is quoted on write whenever it
  would otherwise be misread (e.g. the literal string ``"true"`` or ``"42"``,
  empty strings, or strings that begin with a YAML indicator character).

Round-trip guarantee
--------------------
For metadata built from the supported types, ``parse(dump(meta, body))`` returns
``(meta, body)`` unchanged (lists round-trip as Python ``list`` objects). Values
outside the subset (``None``, nested containers, non-finite floats) are emitted
best-effort via ``str()`` and are NOT guaranteed to round-trip.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = ["parse", "dump"]


# A frontmatter block: an opening "---" line, a (possibly empty) metadata
# region, a closing "---" line, then an optional body. ``meta`` is non-greedy so
# the FIRST closing delimiter wins; ``body`` is DOTALL so it captures verbatim.
_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(?P<meta>.*?)\r?\n---[ \t]*(?:\r?\n(?P<body>.*))?\Z",
    re.DOTALL,
)

# Integers and floats. Integer is checked first; the float pattern therefore
# only needs to recognise forms an int would reject (a dot or an exponent).
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(
    r"^[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?$|^[+-]?\d+[eE][+-]?\d+$"
)

# Characters that carry special meaning to a YAML parser when they LEAD a
# scalar; a string starting with one of these must be quoted to stay a string.
_INDICATOR_CHARS = frozenset("\"'[]{}#&*!|>%@`,?:-")


def parse(text: str) -> tuple[dict[str, Any], str]:
    """Split *text* into ``(metadata, body)``.

    A document with no leading ``---`` block is not an error: we return an empty
    mapping and the original text untouched, because OKF requires consumers to
    tolerate missing frontmatter rather than reject the document.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    body = match.group("body")
    return _parse_block(match.group("meta")), body if body is not None else ""


def dump(meta: dict[str, Any], body: str) -> str:
    """Render *meta* as a frontmatter block followed by *body*.

    With empty metadata we emit the body verbatim (no empty ``---`` fence): that
    keeps ``dump`` the exact inverse of ``parse`` for the missing-frontmatter
    case, where ``parse`` returns an empty mapping for body-only documents.
    """
    if not meta:
        return body
    lines = ["---"]
    lines.extend(f"{key}: {_dump_value(value)}" for key, value in meta.items())
    lines.append("---")
    return "\n".join(lines) + "\n" + body


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _parse_block(block: str) -> dict[str, Any]:
    """Parse the metadata region (text between the two ``---`` fences)."""
    meta: dict[str, Any] = {}
    for raw in block.split("\n"):
        line = raw.strip()
        # Blank lines and full-line comments carry no mapping data.
        if not line or line.startswith("#"):
            continue
        # Lines without a colon are malformed for our subset; OKF asks consumers
        # to be permissive, so we skip them rather than raise.
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = _parse_value(value.strip())
    return meta


def _parse_value(text: str) -> Any:
    """Parse a single value, dispatching inline lists to scalar parsing."""
    if len(text) >= 2 and text[0] == "[" and text[-1] == "]":
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in _split_items(inner)]
    return _parse_scalar(text)


def _parse_scalar(text: str) -> Any:
    """Coerce a single, already-stripped scalar token to its Python type."""
    if text == "":
        return ""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return _unescape_double(text[1:-1])
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        # Single quotes are treated as a literal wrapper (no escape sequences),
        # mirroring how we only emit escapes inside double quotes.
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _split_items(inner: str) -> list[str]:
    """Split an inline list body on commas that are not inside quotes.

    A naive ``split(",")`` would break elements that legitimately contain a
    comma (which we quote on write), so we track quote state and only treat a
    top-level comma as a separator."""
    items: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    n = len(inner)
    while i < n:
        ch = inner[i]
        if quote is not None:
            # Preserve backslash escapes verbatim inside double quotes so the
            # closing-quote detector is not fooled by an escaped quote.
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(ch)
                buf.append(inner[i + 1])
                i += 2
                continue
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
            buf.append(ch)
        elif ch == ",":
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    items.append("".join(buf).strip())
    return items


def _unescape_double(text: str) -> str:
    """Resolve the backslash escapes that :func:`_escape_double` produces."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            out.append(
                {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt, nxt)
            )
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# Dumping helpers
# --------------------------------------------------------------------------- #
def _dump_value(value: Any) -> str:
    """Render a top-level value (scalar or flat list) as YAML-subset text."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_dump_item(item) for item in value) + "]"
    return _dump_scalar(value, in_list=False)


def _dump_item(value: Any) -> str:
    """Render one list element (nested lists are not part of the subset)."""
    return _dump_scalar(value, in_list=True)


def _dump_scalar(value: Any, *, in_list: bool) -> str:
    """Render a scalar, quoting strings only when needed to survive a round-trip."""
    # bool must be checked before int: ``bool`` is a subclass of ``int``.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    text = str(value)
    if _needs_quote(text, in_list):
        return '"' + _escape_double(text) + '"'
    return text


def _needs_quote(text: str, in_list: bool) -> bool:
    """Decide whether a string must be quoted to read back as the same string."""
    if text == "":
        return True
    if text != text.strip():
        # Leading/trailing whitespace would be lost to the strip on read.
        return True
    if text[0] in _INDICATOR_CHARS:
        return True
    if _parse_scalar(text) != text:
        # Would otherwise be read as a bool/int/float instead of a string.
        return True
    if "\n" in text or "\t" in text:
        return True
    if in_list and ("," in text or "[" in text or "]" in text):
        return True
    return False


def _escape_double(text: str) -> str:
    """Escape a string for placement inside double quotes."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
