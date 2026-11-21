from __future__ import annotations

import argparse

from common import add_common_arguments, print_plan, validate_common


def main() -> int:
    parser = argparse.ArgumentParser(description="POSTMMM-R1 HCPR latency plan/runner")
    add_common_arguments(parser)
    args = parser.parse_args()
    manifest = validate_common(args)
    print_plan("hcpr_latency", args, manifest)
    if args.execute:
        raise SystemExit("target-device HCPR timing binding is required; historical latency is forbidden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
