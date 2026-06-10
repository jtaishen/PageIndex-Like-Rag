from __future__ import annotations

import unittest

from kb_agent import cli
from kb_agent.cli_parser import build_parser


class CliStructureTest(unittest.TestCase):
    def test_build_parser_parses_core_search_command_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["search", "agent memory"])

        self.assertEqual(args.command, "search")
        self.assertEqual(args.query, "agent memory")
        self.assertEqual(args.top_k, 8)
        self.assertEqual(args.search_mode, "hybrid")
        self.assertIsNone(args.db)

    def test_build_parser_parses_nested_search_profile_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["search-profile", "show", "active"])

        self.assertEqual(args.command, "search-profile")
        self.assertEqual(args.profile_command, "show")
        self.assertEqual(args.name, "active")

    def test_cli_entrypoint_reuses_parser_and_handler_modules(self) -> None:
        self.assertIs(cli.build_parser, build_parser)
        self.assertEqual(cli.dispatch_command.__module__, "kb_agent.cli_handlers")


if __name__ == "__main__":
    unittest.main()
