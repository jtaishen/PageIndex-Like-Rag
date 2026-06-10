from __future__ import annotations

from typing import Any

from .cli_handlers import dispatch_command
from .cli_parser import build_parser
from .config import resolve_db_path


def main(argv: Any = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = resolve_db_path(args.db)
    dispatch_command(args, db_path)


if __name__ == "__main__":
    main()
