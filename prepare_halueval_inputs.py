#!/usr/bin/env python3
"""Pull, validate, and merge the two A100 HaluEval preparation parts.

The A100 stage produces two disjoint 5,000-record parts.  Each part contains
an ordinary (full) answer view and an FST view with independently assigned
TrueTeacher labels.  This script refuses partial/checkpoint-only artifacts and
writes the four paper-shaped TPU inputs only after cross-view validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE = (
    "hf://buckets/Rachidaaa/fepoid-halueval-mistral-prepared/"
    "halueval-mistral-20260721"
)


@dataclass(frozen=True)
class PartSpec:
    name: str
    start: int
    end: int


PARTS = (
    PartSpec("part-00000-05000", 0, 5_000),
    PartSpec("part-05000-10000", 5_000, 10_000),
)
VIEWS = ("full", "fst")
EXPECTED_SPLIT_COUNTS = {"train": 8_000, "test": 2_000}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def summary_prompt(context: str) -> str:
    return (
        "Summarize the following document in one or two concise sentences.\n"
        f"Document:{context.strip()}\n"
        "Summary:"
    )


def bucket_cp(source: str, destination: Path) -> None:
    try:
        subprocess.run(
            ["hf", "buckets", "cp", source, str(destination)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "The A100 artifact is not complete or is not readable yet: "
            f"{source}. Wait for both preparation Jobs to finish successfully "
            "and rerun this command."
        ) from exc


def pull_part(source: str, spec: PartSpec, destination: Path) -> dict[str, Path]:
    part_dir = destination / spec.name
    part_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "full": part_dir / "part_full.jsonl",
        "fst": part_dir / "part_fst.jsonl",
        "summary": part_dir / "summary.json",
        "success": part_dir / "SUCCESS",
    }
    for key, path in files.items():
        remote_name = {
            "full": "part_full.jsonl",
            "fst": "part_fst.jsonl",
            "summary": "summary.json",
            "success": "SUCCESS",
        }[key]
        bucket_cp(f"{source.rstrip('/')}/{spec.name}/{remote_name}", path)
    return files


def validate_part(
    spec: PartSpec,
    files: dict[str, Path],
) -> tuple[dict[str, list[dict]], dict]:
    if files["success"].read_text(encoding="utf-8").strip() != "ok":
        raise ValueError(f"{spec.name}: invalid SUCCESS marker")
    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"{spec.name}: preparation status is not complete")
    if list(summary.get("source_range", [])) != [spec.start, spec.end]:
        raise ValueError(
            f"{spec.name}: expected source range {[spec.start, spec.end]}, "
            f"observed {summary.get('source_range')}"
        )

    by_view: dict[str, list[dict]] = {}
    expected_indices = set(range(spec.start, spec.end))
    for view in VIEWS:
        rows = read_jsonl(files[view])
        if len(rows) != spec.end - spec.start:
            raise ValueError(
                f"{files[view]}: expected {spec.end - spec.start} rows, "
                f"observed {len(rows)}"
            )
        observed_indices: set[int] = set()
        for line_number, row in enumerate(rows, start=1):
            missing = {
                field
                for field in (
                    "source_index",
                    "paper_split",
                    "context",
                    "question",
                    "best_answer",
                    "label",
                    "answer_view",
                    "text",
                )
                if row.get(field) in (None, "")
            }
            if missing:
                raise ValueError(
                    f"{files[view]}:{line_number}: missing {sorted(missing)}"
                )
            source_index = int(row["source_index"])
            if source_index in observed_indices:
                raise ValueError(f"{files[view]}: duplicate source_index={source_index}")
            observed_indices.add(source_index)
            if row["paper_split"] not in EXPECTED_SPLIT_COUNTS:
                raise ValueError(
                    f"{files[view]}:{line_number}: invalid paper_split"
                )
            if int(row["label"]) not in (0, 1):
                raise ValueError(f"{files[view]}:{line_number}: invalid label")
            if row["answer_view"] != view:
                raise ValueError(
                    f"{files[view]}:{line_number}: answer_view mismatch"
                )
            expected_text = (
                f"{summary_prompt(str(row['context']))} "
                f"{str(row['best_answer']).strip()}"
            )
            if row["text"] != expected_text:
                raise ValueError(
                    f"{files[view]}:{line_number}: prejoined prompt mismatch"
                )
        if observed_indices != expected_indices:
            raise ValueError(f"{files[view]}: source-index coverage mismatch")
        by_view[view] = rows

    full_by_index = {int(row["source_index"]): row for row in by_view["full"]}
    fst_by_index = {int(row["source_index"]): row for row in by_view["fst"]}
    for source_index in expected_indices:
        full_row = full_by_index[source_index]
        fst_row = fst_by_index[source_index]
        if full_row["context"] != fst_row["context"]:
            raise ValueError(f"{spec.name}: context mismatch at {source_index}")
        if full_row["paper_split"] != fst_row["paper_split"]:
            raise ValueError(f"{spec.name}: split mismatch at {source_index}")

    return by_view, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_data"))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep the downloaded two-part staging directory after validation.",
    )
    args = parser.parse_args()

    if shutil.which("hf") is None:
        raise RuntimeError("the `hf` CLI is required; install huggingface_hub")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        args.output_dir / f"halueval_{split}_{view}.jsonl"
        for view in VIEWS
        for split in EXPECTED_SPLIT_COUNTS
    ] + [args.output_dir / "halueval_manifest.json"]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        rendered = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"prepared HaluEval outputs already exist ({rendered}); "
            "pass --overwrite to re-download and replace them"
        )

    with tempfile.TemporaryDirectory(
        prefix="halueval-parts-",
        dir=args.output_dir,
    ) as temporary_name:
        temporary = Path(temporary_name)
        all_rows = {view: [] for view in VIEWS}
        source_files = []
        preparation_summaries = []
        for spec in PARTS:
            files = pull_part(args.source, spec, temporary)
            views, summary = validate_part(spec, files)
            preparation_summaries.append(summary)
            for view in VIEWS:
                all_rows[view].extend(views[view])
                source_files.append(
                    {
                        "part": spec.name,
                        "view": view,
                        "sha256": sha256(files[view]),
                        "records": len(views[view]),
                    }
                )

        split_maps = []
        outputs = []
        for view in VIEWS:
            rows = sorted(all_rows[view], key=lambda row: int(row["source_index"]))
            split_maps.append(
                {int(row["source_index"]): row["paper_split"] for row in rows}
            )
            for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
                selected = [row for row in rows if row["paper_split"] == split]
                if len(selected) != expected_count:
                    raise ValueError(
                        f"{view}/{split}: expected {expected_count}, "
                        f"observed {len(selected)}"
                    )
                labels = Counter(int(row["label"]) for row in selected)
                if set(labels) != {0, 1}:
                    raise ValueError(f"{view}/{split}: both labels are required")
                destination = args.output_dir / f"halueval_{split}_{view}.jsonl"
                write_jsonl(destination, selected)
                outputs.append(
                    {
                        "path": str(destination),
                        "split": split,
                        "view": view,
                        "records": len(selected),
                        "labels": dict(sorted(labels.items())),
                        "sha256": sha256(destination),
                    }
                )
        if split_maps[0] != split_maps[1]:
            raise ValueError("full and FST views disagree on the paper split")

        if args.keep_parts:
            kept = args.output_dir / "halueval_parts"
            if kept.exists():
                if not args.overwrite:
                    raise FileExistsError(f"{kept} exists; pass --overwrite")
                shutil.rmtree(kept)
            shutil.copytree(temporary, kept)

    manifest = {
        "status": "complete",
        "scope": "Mistral-generated HaluEval summaries labeled by TrueTeacher",
        "source": args.source,
        "paper_split": "8000 train pool and 2000 fixed test (seed 2024)",
        "probe_validation": (
            "the paper's 10% validation subset is selected later from the "
            "8000-record train pool"
        ),
        "source_files": source_files,
        "preparation_summaries": preparation_summaries,
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "halueval_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
