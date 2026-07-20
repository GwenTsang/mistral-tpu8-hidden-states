#!/usr/bin/env python3
"""Validate a completed multi-worker Mistral hidden-state artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    root = args.artifact.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    world_size = int(manifest["world_size"])
    problems: list[str] = []
    metadata: list[dict] = []
    for rank in range(world_size):
        path = root / f"metadata-rank-{rank:03d}.jsonl"
        if not path.exists():
            problems.append(f"missing {path.name}")
            continue
        rows = read_jsonl(path)
        if any(int(row["rank"]) != rank for row in rows):
            problems.append(f"{path.name} contains another rank")
        metadata.extend(rows)

    rows_by_shard: dict[str, int] = {}
    observed_files = []
    tensor_rows = 0
    for file_info in manifest["files"]:
        relative = file_info["path"]
        path = root / relative
        if not path.exists():
            problems.append(f"missing shard {relative}")
            continue
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != {"embedding", "hidden_states"}:
                problems.append(f"{relative}: keys={sorted(keys)}")
                continue
            embedding = handle.get_tensor("embedding")
            hidden = handle.get_tensor("hidden_states")
        if embedding.dtype != torch.bfloat16 or hidden.dtype != torch.bfloat16:
            problems.append(f"{relative}: expected BF16 tensors")
        if tuple(embedding.shape[1:]) != (4096,):
            problems.append(f"{relative}: embedding shape={tuple(embedding.shape)}")
        if tuple(hidden.shape[1:]) != (32, 4096):
            problems.append(f"{relative}: hidden shape={tuple(hidden.shape)}")
        if embedding.shape[0] != hidden.shape[0]:
            problems.append(f"{relative}: row count differs between tensors")
        if not torch.isfinite(embedding.float()).all():
            problems.append(f"{relative}: non-finite embedding")
        if not torch.isfinite(hidden.float()).all():
            problems.append(f"{relative}: non-finite hidden state")
        rows = int(hidden.shape[0])
        rows_by_shard[relative] = rows
        tensor_rows += rows
        digest = None if args.skip_sha256 else sha256(path)
        if digest is not None and digest != file_info["sha256"]:
            problems.append(f"{relative}: SHA-256 mismatch")
        observed_files.append({"path": relative, "rows": rows, "sha256": digest})

    metadata_by_shard: dict[str, list[dict]] = {}
    for row in metadata:
        metadata_by_shard.setdefault(row["shard"], []).append(row)
    for shard, rows in metadata_by_shard.items():
        expected = rows_by_shard.get(shard)
        offsets = sorted(int(row["offset"]) for row in rows)
        if expected is None:
            problems.append(f"metadata references missing shard {shard}")
        elif offsets != list(range(expected)):
            problems.append(f"{shard}: incomplete or duplicate offsets")

    source_indices = [int(row["source_index"]) for row in metadata]
    if len(source_indices) != len(set(source_indices)):
        problems.append("duplicate source_index values")
    expected_records = int(manifest["records"])
    if len(metadata) != expected_records:
        problems.append(f"metadata rows: expected {expected_records}, observed {len(metadata)}")
    if tensor_rows != expected_records:
        problems.append(f"tensor rows: expected {expected_records}, observed {tensor_rows}")
    if int(manifest["tensor_rows"]) != tensor_rows:
        problems.append("manifest tensor_rows does not match shards")

    report = {
        "valid": not problems,
        "artifact": str(root),
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "world_size": world_size,
        "records": tensor_rows,
        "shards": len(observed_files),
        "dtype": "bfloat16",
        "embedding_shape_per_record": [4096],
        "hidden_states_shape_per_record": [32, 4096],
        "records_per_rank": dict(sorted(Counter(int(row["rank"]) for row in metadata).items())),
        "bucket_counts": dict(sorted(Counter(int(row["bucket_length"]) for row in metadata).items())),
        "files": observed_files,
        "problems": problems,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
