#!/usr/bin/env python3
"""Pull and independently validate completed HaluEval TPU artifacts."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from run_halueval_tpu import RUNS


def run(command: list[str], *, dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default="GwendalTsang/mistral-7b-hidden-states-tpu8",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=Path("downloaded_outputs/halueval"),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[spec.key for spec in RUNS],
        help="Pull only selected split/view keys (default: all four).",
    )
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and shutil.which("hf") is None:
        raise RuntimeError("the `hf` CLI is required; install huggingface_hub")
    if not args.dry_run:
        args.destination_root.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent
    validator = root / "validate_output.py"
    reports = []
    selected = [
        spec for spec in RUNS if args.only is None or spec.key in args.only
    ]
    for spec in selected:
        source = f"hf://buckets/{args.bucket}/{spec.bucket_prefix}"
        destination = args.destination_root / spec.key
        run(
            ["hf", "buckets", "sync", source, str(destination)],
            dry_run=args.dry_run,
        )
        report = destination / "validation-local.json"
        command = [
            sys.executable,
            str(validator),
            str(destination),
            "--report",
            str(report),
        ]
        if args.skip_sha256:
            command.append("--skip-sha256")
        run(command, dry_run=args.dry_run)
        reports.append(
            {
                "key": spec.key,
                "records": spec.records,
                "path": str(destination),
                "report": str(report),
            }
        )

    summary = {
        "status": "dry-run" if args.dry_run else "valid",
        "source": f"https://huggingface.co/buckets/{args.bucket}",
        "artifacts": reports,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
