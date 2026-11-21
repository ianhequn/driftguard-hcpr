from __future__ import annotations

import argparse

from common import add_common_arguments, print_plan, validate_common


def main() -> int:
    parser = argparse.ArgumentParser(description="POSTMMM-R1 HCPR quality plan/runner")
    add_common_arguments(parser)
    args = parser.parse_args()
    manifest = validate_common(args)
    print_plan("hcpr_quality", args, manifest)
    if args.execute:
        raise SystemExit("calibration artifact and target-device controller binding are required before launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
