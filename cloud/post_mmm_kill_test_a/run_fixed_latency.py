from __future__ import annotations

import argparse

from common import add_common_arguments, print_plan, validate_common


def main() -> int:
    parser = argparse.ArgumentParser(description="POSTMMM-R1 fixed latency plan/runner")
    add_common_arguments(parser)
    args = parser.parse_args()
    manifest = validate_common(args)
    print_plan("fixed_latency", args, manifest)
    if args.execute:
        raise SystemExit("old Round10 latency entrypoints are intentionally rejected; bind a target-device runner first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
