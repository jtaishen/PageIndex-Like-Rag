from __future__ import annotations

import unittest

from kb_agent.fact_utils import confidence, excerpt, is_table_source, json_value, normalize_key, query_terms


class FactUtilsTest(unittest.TestCase):
    def test_query_terms_expands_long_chinese_terms(self) -> None:
        terms = query_terms("动态角色任务规划")

        self.assertIn("动态角色任务规划", terms)
        self.assertIn("动态", terms)
        self.assertIn("角色", terms)
        self.assertLessEqual(len(terms), 12)

    def test_json_value_returns_default_for_invalid_json(self) -> None:
        self.assertEqual(json_value("[1, 2]", []), [1, 2])
        self.assertEqual(json_value("not-json", {"fallback": True}), {"fallback": True})

    def test_confidence_clamps_and_rounds(self) -> None:
        self.assertEqual(confidence("1.5", 0.2), 1.0)
        self.assertEqual(confidence("-0.2", 0.2), 0.0)
        self.assertEqual(confidence("0.8766", 0.2), 0.877)
        self.assertEqual(confidence("bad", 0.42), 0.42)

    def test_text_helpers_normalize_table_source_and_excerpt(self) -> None:
        self.assertTrue(is_table_source("llm_table"))
        self.assertFalse(is_table_source("llm_text"))
        self.assertEqual(normalize_key("  Foo， Bar! "), "foo，bar")
        self.assertEqual(excerpt("a  b  c", 20), "a b c")
        self.assertEqual(excerpt("abcdefghij", 5), "abcde ...")


if __name__ == "__main__":
    unittest.main()
