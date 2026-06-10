from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kb_agent.utils import excerpt, read_json, string_list, unique_strings


class UtilsTest(unittest.TestCase):
    def test_unique_strings_strips_skips_empty_and_preserves_order(self) -> None:
        self.assertEqual(
            unique_strings([" a ", "", "b", "a", " b ", "c"]),
            ["a", "b", "c"],
        )

    def test_string_list_normalizes_scalar_and_list_values(self) -> None:
        self.assertEqual(string_list(None), [])
        self.assertEqual(string_list("  topic  "), ["topic"])
        self.assertEqual(string_list([" a ", "", 7]), ["a", "7"])
        self.assertEqual(string_list(("x", " y ")), ["x", "y"])

    def test_excerpt_compacts_and_truncates_text(self) -> None:
        self.assertEqual(excerpt("  hello   world  ", 20), "hello world")
        self.assertEqual(excerpt("alpha beta gamma", 10), "alpha beta...")

    def test_read_json_returns_default_for_missing_or_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.json"
            invalid = root / "invalid.json"
            valid.write_text('{"a": 1}', encoding="utf-8")
            invalid.write_text("{not json", encoding="utf-8")

            self.assertEqual(read_json(valid, {}), {"a": 1})
            self.assertEqual(read_json(root / "missing.json", {"fallback": True}), {"fallback": True})
            self.assertEqual(read_json(invalid, []), [])


if __name__ == "__main__":
    unittest.main()
