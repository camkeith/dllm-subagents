"""Exact-match multi-edit: unique, non-overlapping replacements against the original.

In a file that mixes CRLF and LF, old_text may match with either line ending at each line break
(an exact match and its twin with other line endings both count for the uniqueness rule)."""
from __future__ import annotations

import re
from typing import Any, List, Tuple

BOM = "\ufeff"


class EditError(Exception):
    pass


def apply_edits(content: str, edits: List[dict]) -> str:
    """Apply edits (each {"old_text", "new_text"}) matched against the original content.

    Pure: raises EditError before returning anything if any edit is invalid.
    """
    if not isinstance(edits, list) or not edits:
        raise EditError("edits must be a non-empty list of {old_text, new_text} objects")

    bom = ""
    text = content
    if text.startswith(BOM):
        bom, text = BOM, text[1:]
    # Normalize only a uniformly CRLF file; a mixed one is matched and edited as is, so its
    # LF lines are not turned into CRLF.
    crlf = "\r\n" in text and text.count("\n") == text.count("\r\n")
    mixed = "\r\n" in text and not crlf  # both CRLF and lone LF line breaks
    if crlf:
        text = text.replace("\r\n", "\n")

    spans = []  # (start, end, new_text, edit number)
    for i, edit in enumerate(edits, start=1):
        old, new = _validate(edit, i)
        if not mixed:  # LF line breaks only (a CRLF file was normalized above); a lone CR is not one
            old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        if not old:
            raise EditError("edit %d: old_text is empty; provide the exact text to replace" % i)
        if mixed and "\n" in old:
            matches = _find_any_line_endings(text, old, new)
        else:
            matches = [(s, s + len(old), new) for s in _find_all(text, old)]
        count = len(matches)
        if count == 0:
            raise EditError(
                "edit %d: old_text not found in the file; re-read the file and copy the text "
                "exactly, including whitespace" % i
            )
        if count > 1:
            raise EditError(
                "edit %d: old_text matches %d times; add more surrounding context to make it unique"
                % (i, count)
            )
        start, end, replacement = matches[0]
        spans.append((start, end, replacement, i))

    ordered = sorted(spans)
    for prev, cur in zip(ordered, ordered[1:]):
        if cur[0] < prev[1]:
            raise EditError(
                "edit %d and edit %d overlap; merge them into one edit or choose "
                "non-overlapping old_text" % (prev[3], cur[3])
            )

    parts = []
    pos = 0
    for start, end, new, _ in ordered:
        parts.append(text[pos:start])
        parts.append(new)
        pos = end
    parts.append(text[pos:])
    result = "".join(parts)
    if crlf:
        result = result.replace("\n", "\r\n")
    return bom + result


def _find_any_line_endings(text: str, old: str, new: str) -> List[Tuple[int, int, str]]:
    """Matches of old in a mixed-ending text where each line break may be LF or CRLF, overlapping
    ones included (the exact match among them).

    A match equal to old gets new as given. Otherwise new's line breaks take the matched ones
    one for one, or, when their numbers differ, all take the match's first line ending.
    """
    lines = old.replace("\r\n", "\n").split("\n")
    # A match must not start at the \n of a CRLF: that is the line break the \r before it starts.
    start = "(?<!\r)" if not lines[0] else ""
    pattern = re.compile("(?=(%s%s))" % (start, "(\r?\n)".join(re.escape(line) for line in lines)))
    plain = new.replace("\r\n", "\n").split("\n")
    found = []
    for m in pattern.finditer(text):
        matched, breaks = m.group(1), m.groups()[1:]
        if matched == old:
            replacement = new
        elif len(plain) == len(breaks) + 1:
            replacement = "".join(part + eol for part, eol in zip(plain, breaks + ("",)))
        else:
            replacement = breaks[0].join(plain)
        found.append((m.start(1), m.end(1), replacement))
    return found


def _find_all(text: str, old: str) -> List[int]:
    """All start indexes of old in text, including self-overlapping matches."""
    starts = []
    i = text.find(old)
    while i != -1:
        starts.append(i)
        i = text.find(old, i + 1)
    return starts


def _validate(edit: Any, i: int):
    if not isinstance(edit, dict):
        raise EditError("edit %d: must be an object with old_text and new_text" % i)
    for key in ("old_text", "new_text"):
        if not isinstance(edit.get(key), str):
            raise EditError("edit %d: %s is required and must be a string" % (i, key))
    return edit["old_text"], edit["new_text"]
