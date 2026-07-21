#!/usr/bin/env python3
"""Run and upload all paper-scale HaluEval Mistral TPU extractions.

This is a Kaggle-friendly sequential orchestrator.  It pulls the validated
A100 handoff, resumes local runs when possible, validates every artifact, and
syncs the validation report to the output Hugging Face Bucket.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    key: str
    input_name: str
    output_name: str
    bucket_prefix: str
    answer_view: str
    records: int


RUNS = (
    RunSpec(
        "train-full",
        "halueval_train_full.jsonl",
        "halueval-train-full-spmd",
        "halueval/train/full",
        "full",
        8_000,
    ),
    RunSpec(
        "test-full",
        "halueval_test_full.jsonl",
        "halueval-test-full-spmd",
        "halueval/test/full",
        "full",
        2_000,
    ),
    RunSpec(
        "train-fst",
        "halueval_train_fst.jsonl",
        "halueval-train-fst-spmd",
        "halueval/train/fst",
        "first_sentence",
        8_000,
    ),
    RunSpec(
        "test-fst",
        "halueval_test_fst.jsonl",
        "halueval-test-fst-spmd",
        "halueval/test/fst",
        "first_sentence",
        2_000,
    ),
)


def render(command: list[str]) -> str:
    return shlex.join(command)


def run(command: list[str], *, dry_run: bool = False) -> None:
    print(f"+ {render(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=os.environ.copy())


def completed_and_valid(output: Path, expected_records: int) -> bool:
    manifest_path = output / "manifest.json"
    validation_path = output / "validation.json"
    if not manifest_path.exists() or not validation_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("status") == "complete"
        and int(manifest.get("records", -1)) == expected_records
        and validation.get("valid") is True
        and int(validation.get("records", -1)) == expected_records
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket",
        default="GwendalTsang/mistral-7b-hidden-states-tpu8",
        help="OWNER/BUCKET receiving completed artifacts.",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("prepared_data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[spec.key for spec in RUNS],
        help="Run only selected split/view keys (default: all four).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    extractor = root / "extract_hidden_states.py"
    validator = root / "validate_output.py"
    preparer = root / "prepare_halueval_inputs.py"
    selected = [
        spec for spec in RUNS if args.only is None or spec.key in args.only
    ]

    if not args.skip_pull:
        prepare_command = [
            sys.executable,
            str(preparer),
            "--output-dir",
            str(args.input_dir),
        ]
        manifest_path = args.input_dir / "halueval_manifest.json"
        if manifest_path.exists():
            prepare_command.append("--overwrite")
        run(prepare_command, dry_run=args.dry_run)

    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is required for Bucket upload; load it from Kaggle Secrets"
        )
    if not args.dry_run and shutil.which("hf") is None:
        raise RuntimeError("the `hf` CLI is required; install huggingface_hub")

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for spec in selected:
        input_path = args.input_dir / spec.input_name
        output = args.output_root / spec.output_name
        if not args.dry_run and not input_path.exists():
            raise FileNotFoundError(f"missing prepared input: {input_path}")

        if completed_and_valid(output, spec.records) and not args.overwrite:
            print(f"already complete and valid: {spec.key}", flush=True)
        else:
            command = [
                sys.executable,
                str(extractor),
                "--input-jsonl",
                str(input_path),
                "--output-dir",
                str(output),
                "--answer-column",
                "best_answer",
                "--answer-view",
                spec.answer_view,
                "--batch-size",
                "1",
                "--shard-size",
                "64",
                "--buckets",
                "128",
                "256",
                "512",
                "1024",
                "2048",
                "--expected-world-size",
                "8",
                "--execution-mode",
                "spmd_fsdp",
                "--push-to-bucket",
                args.bucket,
                "--bucket-prefix",
                spec.bucket_prefix,
            ]
            if args.overwrite:
                command.append("--overwrite")
            elif output.exists():
                command.append("--resume")
            run(command, dry_run=args.dry_run)

            validation_command = [
                sys.executable,
                str(validator),
                str(output),
                "--report",
                str(output / "validation.json"),
            ]
            run(validation_command, dry_run=args.dry_run)

        # The extractor uploads before the external validator runs.  Sync once
        # more so the public artifact includes validation.json as evidence.
        destination = f"hf://buckets/{args.bucket}/{spec.bucket_prefix}"
        run(
            ["hf", "buckets", "sync", str(output), destination],
            dry_run=args.dry_run,
        )
        completed.append(spec.key)

    print(
        json.dumps(
            {
                "status": "dry-run" if args.dry_run else "complete",
                "runs": completed,
                "bucket": f"https://huggingface.co/buckets/{args.bucket}",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
