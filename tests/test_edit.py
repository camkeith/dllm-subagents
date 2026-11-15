from __future__ import annotations

import copy
import unittest

from dllm.edit import EditError, apply_edits


def e(old, new):
    return {"old_text": old, "new_text": new}


class ApplyEditsTests(unittest.TestCase):
    def test_single_replacement(self):
        self.assertEqual(apply_edits("a\nb\nc\n", [e("b", "B")]), "a\nB\nc\n")

    def test_multiple_disjoint_edits_match_original(self):
        out = apply_edits("x = 1\ny = 2\n", [e("x = 1", "y = 2"), e("y = 2", "z = 3")])
        self.assertEqual(out, "y = 2\nz = 3\n")

    def test_not_found(self):
        with self.assertRaises(EditError) as cm:
            apply_edits("abc", [e("zzz", "y")])
        self.assertIn("edit 1", str(cm.exception))
        self.assertIn("not found", str(cm.exception))

    def test_not_unique(self):
        with self.assertRaises(EditError) as cm:
            apply_edits("a a", [e("a", "b")])
        self.assertIn("matches 2 times", str(cm.exception))
        self.assertIn("edit 1", str(cm.exception))

    def test_self_overlapping_matches_are_not_unique(self):
        for content, old, n in (("aaa", "aa", 2), ("a\n\n\nb", "\n\n", 2), ("}}}", "}}", 2)):
            with self.assertRaises(EditError) as cm:
                apply_edits(content, [e(old, "X")])
            self.assertIn("matches %d times" % n, str(cm.exception))

    def test_overlap(self):
        with self.assertRaises(EditError) as cm:
            apply_edits("abcde", [e("abc", "X"), e("bcd", "Y")])
        self.assertIn("overlap", str(cm.exception))
        self.assertIn("edit 1", str(cm.exception))
        self.assertIn("edit 2", str(cm.exception))

    def test_empty_old_text(self):
        with self.assertRaises(EditError) as cm:
            apply_edits("abc", [e("", "x")])
        self.assertIn("empty", str(cm.exception))

    def test_all_or_nothing(self):
        content = "a\nb\n"
        edits = [e("a", "A"), e("missing", "M")]
        saved = copy.deepcopy(edits)
        with self.assertRaises(EditError) as cm:
            apply_edits(content, edits)
        self.assertIn("edit 2", str(cm.exception))
        self.assertEqual(content, "a\nb\n")
        self.assertEqual(edits, saved)

    def test_crlf_preserved(self):
        self.assertEqual(apply_edits("a\r\nb\r\n", [e("a\nb", "A\nB")]), "A\r\nB\r\n")

    def test_crlf_in_old_text_also_matches(self):
        self.assertEqual(apply_edits("a\r\nb\r\n", [e("a\r\nb", "A\r\nB")]), "A\r\nB\r\n")

    def test_bom_preserved(self):
        self.assertEqual(apply_edits("\ufeffhello", [e("hello", "bye")]), "\ufeffbye")

    def test_edits_list_required(self):
        for bad in ([], None, "x"):
            with self.assertRaises(EditError):
                apply_edits("abc", bad)

    def test_malformed_edit(self):
        for bad in ([{"old_text": "a"}], [{"new_text": "a"}], ["a"], [{"old_text": 1, "new_text": "x"}]):
            with self.assertRaises(EditError) as cm:
                apply_edits("abc", bad)
            self.assertIn("edit 1", str(cm.exception))

    def test_deletion_with_empty_new_text(self):
        self.assertEqual(apply_edits("abc", [e("b", "")]), "ac")

    def test_edits_applied_regardless_of_order(self):
        self.assertEqual(apply_edits("a b c", [e("c", "C"), e("a", "A")]), "A b C")


if __name__ == "__main__":
    unittest.main()
