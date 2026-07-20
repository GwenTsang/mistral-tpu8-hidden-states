#!/usr/bin/env python3
"""Download and validate the scaled CoQA inputs used by the TPU extractor.

These records were generated with Llama-3.1-8B-Instruct.  Passing them through
Mistral is a cross-model representation control, not a reproduction of the
paper's Mistral-generated-answer experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE = (
    "hf://buckets/GwendalTsang/fepoid-llama-hidden-states/"
    "coqa-2k-llama-20260720/prepared"
)


@dataclass(frozen=True)
class InputSpec:
    key: str
    remote_name: str
    local_name: str
    source_split: str
    sha256: str


INPUTS = (
    InputSpec(
        "train-full",
        "llama_train_full.jsonl",
        "coqa_train.jsonl",
        "train",
        "b00ca027cc5687b3841f1ba7f6d5cf68aa5b7aa6be2b68933008ddade0174fac",
    ),
    InputSpec(
        "validation-full",
        "llama_test_full.jsonl",
        "coqa_validation.jsonl",
        "validation",
        "5d334f62e101842661279189490b52f08f3df3322ff0d97e49404820cbc28829",
    ),
    InputSpec(
        "train-fst",
        "llama_train_fst.jsonl",
        "coqa_train_fst.jsonl",
        "train",
        "27d90e342ebc8d9f8f5da68dc0aeaac0805b1e39351e4bead10133455b92d576",
    ),
    InputSpec(
        "validation-fst",
        "llama_test_fst.jsonl",
        "coqa_validation_fst.jsonl",
        "validation",
        "730894a979064098353cef5d28027fd6f19dbf80655b8e3b82d92b3ddf1d2df3",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, spec: InputSpec) -> dict[str, object]:
    labels: dict[int, int] = {0: 0, 1: 0}
    rows = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {
                key
                for key in ("context", "question", "best_answer", "label")
                if row.get(key) in (None, "")
            }
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing {sorted(missing)}"
                )
            if row.get("source_split") != spec.source_split:
                raise ValueError(
                    f"{path}:{line_number}: expected source_split="
                    f"{spec.source_split!r}, observed {row.get('source_split')!r}"
                )
            label = int(row["label"])
            if label not in labels:
                raise ValueError(f"{path}:{line_number}: invalid label={label}")
            labels[label] += 1
            rows += 1

    if rows != 1_000:
        raise ValueError(f"{path}: expected 1000 records, observed {rows}")
    if not all(labels.values()):
        raise ValueError(f"{path}: both binary labels are required, observed {labels}")

    digest = file_sha256(path)
    if digest != spec.sha256:
        raise ValueError(
            f"{path}: SHA-256 mismatch; expected {spec.sha256}, observed {digest}"
        )

    return {
        "path": str(path),
        "records": rows,
        "labels": labels,
        "sha256": digest,
        "source_split": spec.source_split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_data"))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[spec.key for spec in INPUTS],
        help="Download only selected split/view keys (default: all four).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if shutil.which("hf") is None:
        raise RuntimeError("the `hf` CLI is required; install huggingface_hub")

    selected = [
        spec for spec in INPUTS if args.only is None or spec.key in args.only
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for spec in selected:
        destination = args.output_dir / spec.local_name
        if destination.exists() and not args.overwrite:
            reports.append(validate(destination, spec))
            print(f"verified existing {destination}", flush=True)
            continue

        subprocess.run(
            [
                "hf",
                "buckets",
                "cp",
                f"{args.source.rstrip('/')}/{spec.remote_name}",
                str(destination),
            ],
            check=True,
        )
        reports.append(validate(destination, spec))
        print(f"downloaded and verified {destination}", flush=True)

    manifest = {
        "scope": (
            "scaled CoQA cross-model control using Llama-generated answers; "
            "not the paper's full Mistral protocol"
        ),
        "paper_protocol": (
            "per QA dataset: 9000 train + 1000 validation + fixed test split"
        ),
        "source": args.source,
        "files": reports,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
