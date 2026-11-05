#!/usr/bin/env python3
"""Print Hugging Face upload commands.

This script does not call the Hub. ``--execute`` is refused on purpose so a
curious flag cannot publish the synthetic fixture by accident.
"""

from __future__ import annotations

import argparse
import sys

SPACE_REPO = "Ki-Asif/inference-benchmark-lab"
DATASET_REPO = "Ki-Asif/inference-benchmark-results"

COMMANDS = (
    f"hf upload {SPACE_REPO} space . --repo-type space",
    f"hf upload {DATASET_REPO} dataset_card/README.md README.md --repo-type dataset",
    f"hf upload {DATASET_REPO} results/example-synthetic example-synthetic --repo-type dataset",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print hf upload commands for the Space and the results dataset. Does not upload."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="refused; this script never contacts the Hugging Face Hub",
    )
    args = parser.parse_args(argv)
    if args.execute:
        print(
            "refusing to upload: scripts/publish_hf.py only prints commands.",
            file=sys.stderr,
        )
        return 2
    print("Run these from the repository root, after the Space and dataset exist.")
    print("Do not describe results/example-synthetic as a hardware measurement.")
    print("Upload a real run directory in its place when you have one.")
    print()
    for command in COMMANDS:
        print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
