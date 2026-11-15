"""Exact-match multi-edit: unique, non-overlapping replacements against the original."""
from __future__ import annotations

from typing import Any, List

BOM = "﻿"


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
    crlf = "\r\n" in text
    if crlf:
        text = text.replace("\r\n", "\n")

    spans = []  # (start, end, new_text, edit number)
    for i, edit in enumerate(edits, start=1):
        old, new = _validate(edit, i)
        old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        if not old:
            raise EditError("edit %d: old_text is empty; provide the exact text to replace" % i)
        count = text.count(old)
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
        start = text.index(old)
        spans.append((start, start + len(old), new, i))

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


def _validate(edit: Any, i: int):
    if not isinstance(edit, dict):
        raise EditError("edit %d: must be an object with old_text and new_text" % i)
    for key in ("old_text", "new_text"):
        if not isinstance(edit.get(key), str):
            raise EditError("edit %d: %s is required and must be a string" % (i, key))
    return edit["old_text"], edit["new_text"]
