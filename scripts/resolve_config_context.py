#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolve wrapper-facing config context via ConfigManager.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve config context for shell wrappers.")
    parser.add_argument("--config", required=True, type=str, help="Config path")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT), help="Project root")
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        help="Specific field(s) to print, one per line. If omitted, emit JSON.",
    )
    args = parser.parse_args()

    from oglans.utils.pathing import build_runtime_context_from_config_path

    ctx = build_runtime_context_from_config_path(args.config, project_root=args.project_root)
    if args.field:
        for field in args.field:
            value = ctx.get(field, "")
            if value is None:
                print("")
            else:
                print(value)
        return

    print(json.dumps(ctx, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
