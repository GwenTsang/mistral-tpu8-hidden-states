#!/usr/bin/env python3
"""Extract Mistral-7B last-token hidden states across an eight-device TPU slice.

The default SPMD/FSDP mode loads one BF16 ``MistralModel`` and shards it across
all physical devices. A legacy replicated mode assigns a full model and a
disjoint input subset to each worker. With the default per-device batch size of
one, either mode has an effective global batch size of eight.

Only ``[:, -1, :]`` is retained from the embedding and every transformer
block. The script never enables ``output_hidden_states=True`` and disables the
KV cache, keeping per-worker HBM use bounded.
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# These must be set before importing torch_xla.
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_NO_SPECIAL_SCALARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from safetensors.torch import save_file
from transformers import AutoTokenizer, MistralModel
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from mistral_tpu8.core import (  # noqa: E402
    InputRecord,
    assign_bucket,
    record_from_mapping,
    worker_source_indices,
)


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
DEFAULT_BUCKETS = (128, 256, 512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/huggingface"),
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--answer-column", default="best_answer")
    parser.add_argument(
        "--answer-view",
        choices=("full", "first_sentence"),
        default="full",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUCKETS),
    )
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument(
        "--execution-mode",
        choices=("spmd_fsdp", "replicated"),
        default="spmd_fsdp",
        help=(
            "spmd_fsdp loads one host model and shards it over all TPU devices; "
            "replicated keeps one complete model per device and needs much more "
            "memory"
        ),
    )
    parser.add_argument("--parallel-model-load", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-single-process", action="store_true")
    parser.add_argument(
        "--push-to-bucket",
        metavar="OWNER/BUCKET",
        help=(
            "After extraction, create/sync to this Hugging Face Bucket using "
            "HF_TOKEN."
        ),
    )
    parser.add_argument(
        "--bucket-prefix",
        help="Destination prefix inside the bucket (default: output directory name).",
    )
    return parser.parse_args()


def json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_records(args: argparse.Namespace) -> list[InputRecord]:
    stop = (
        args.start_index + args.max_samples
        if args.max_samples
        else None
    )
    records: list[InputRecord] = []

    with args.input_jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < args.start_index:
                continue

            if stop is not None and index >= stop:
                break

            line = line.strip()
            if line:
                records.append(
                    record_from_mapping(
                        json.loads(line),
                        source_index=index,
                        text_column=args.text_column,
                        answer_column=args.answer_column,
                        answer_view=args.answer_view,
                    )
                )

    if not records:
        raise ValueError("input selection produced no records")

    return records


def planned_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": args.model_id,
        "revision": args.revision,
        "input_jsonl": str(args.input_jsonl.resolve()),
        "input_sha256": sha256(args.input_jsonl),
        "text_column": args.text_column,
        "answer_column": args.answer_column,
        "answer_view": args.answer_view,
        "max_samples": args.max_samples,
        "start_index": args.start_index,
        "batch_size_per_worker": args.batch_size,
        "shard_size_per_worker": args.shard_size,
        "buckets": list(args.buckets),
        "expected_world_size": args.expected_world_size,
        "execution_mode": args.execution_mode,
        "push_to_bucket": args.push_to_bucket,
        "bucket_prefix": args.bucket_prefix,
        "hidden_state_semantics": {
            "0..30": "post-transformer-block, pre-final-RMSNorm",
            "31": "post-transformer-block-31 and post-final-RMSNorm",
        },
    }


def prepare_output(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    forbidden = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }

    if output in forbidden:
        raise ValueError(
            f"refusing to use broad output directory: {output}"
        )

    if args.overwrite and output.exists():
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)

    existing = output / "run_config.json"
    config = planned_config(args)
    config["config_sha256"] = json_hash(config)

    if existing.exists():
        previous = json.loads(
            existing.read_text(encoding="utf-8")
        )
        if previous != config:
            raise ValueError(
                f"{output} belongs to a different run; "
                "use a new directory or --overwrite"
            )
        if not args.resume:
            raise FileExistsError(
                f"{output} already contains a run; "
                "pass --resume or --overwrite"
            )
    else:
        write_json(existing, config)


def unwrap_xla_sharded_tensor(
    tensor: torch.Tensor,
    *,
    source: str,
) -> torch.Tensor:
    """Return the base XLA tensor inside Python SPMD wrappers."""

    from torch_xla.distributed.spmd.xla_sharded_tensor import (
        XLAShardedTensor,
    )

    wrapper_depth = 0

    while isinstance(tensor, XLAShardedTensor):
        if not hasattr(tensor, "global_tensor"):
            raise RuntimeError(
                f"{source} became an XLAShardedTensor without global_tensor "
                "before it could be unwrapped"
            )

        tensor = tensor.global_tensor
        wrapper_depth += 1

        if wrapper_depth > 16:
            raise RuntimeError(
                "cyclic or excessively nested XLAShardedTensor wrappers"
            )

    return tensor


class LastTokenCapture:
    """Hooks that retain only one token vector from each required stage."""

    def __init__(self, model: MistralModel):
        self.embedding: torch.Tensor | None = None
        self.layers: dict[int, torch.Tensor] = {}

        self.handles = [
            model.embed_tokens.register_forward_hook(
                self._embedding_hook
            )
        ]

        for index, layer in enumerate(model.layers[:-1]):
            self.handles.append(
                layer.register_forward_hook(
                    self._layer_hook(index)
                )
            )

        self.handles.append(
            model.norm.register_forward_hook(
                self._norm_hook
            )
        )

    def clear(self) -> None:
        self.embedding = None
        self.layers.clear()

    def _embedding_hook(
        self,
        _module: Any,
        _inputs: Any,
        output: torch.Tensor,
    ) -> None:
        output = unwrap_xla_sharded_tensor(
            output,
            source="embedding hook output",
        )

        captured = output[:, -1, :].clone()

        self.embedding = unwrap_xla_sharded_tensor(
            captured,
            source="embedding hook capture",
        )

    def _layer_hook(self, index: int):
        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
        ) -> None:
            hidden = (
                output[0]
                if isinstance(output, tuple)
                else output
            )

            hidden = unwrap_xla_sharded_tensor(
                hidden,
                source=f"layer {index} hook output",
            )

            captured = hidden[:, -1, :].clone()

            self.layers[index] = unwrap_xla_sharded_tensor(
                captured,
                source=f"layer {index} hook capture",
            )

        return hook

    def _norm_hook(
        self,
        _module: Any,
        _inputs: Any,
        output: torch.Tensor,
    ) -> None:
        output = unwrap_xla_sharded_tensor(
            output,
            source="norm hook output",
        )

        captured = output[:, -1, :].clone()

        self.layers[31] = unwrap_xla_sharded_tensor(
            captured,
            source="norm hook capture",
        )

    def ordered(self) -> list[torch.Tensor]:
        if (
            self.embedding is None
            or set(self.layers) != set(range(32))
        ):
            raise RuntimeError(
                f"incomplete hook capture: layers={sorted(self.layers)}"
            )

        return [
            self.embedding,
            *[
                self.layers[index]
                for index in range(32)
            ],
        ]

    def stacked(self) -> torch.Tensor:
        # Used by replicated mode only. SPMD must call
        # spmd_captures_to_cpu() instead.
        return torch.stack(
            self.ordered(),
            dim=1,
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def spmd_captures_to_cpu(
    capture: LastTokenCapture,
) -> torch.Tensor:
    """Batch-transfer captures and stack them only after reaching CPU."""

    tensors: list[torch.Tensor] = []

    for index, tensor in enumerate(capture.ordered()):
        tensor = unwrap_xla_sharded_tensor(
            tensor,
            source=f"ordered capture {index}",
        )

        if (
            type(tensor) is not torch.Tensor
            or tensor.device.type != "xla"
        ):
            raise RuntimeError(
                f"ordered capture {index} is not a base XLA tensor: "
                f"type={type(tensor).__name__} "
                f"device={tensor.device}"
            )

        tensors.append(tensor)

    dispatch_guard = torch._C._DisableTorchDispatch()

    try:
        torch_xla._XLAC._xla_sync_multi(
            tensors,
            devices=[],
            wait=True,
            sync_xla_data=False,
        )

        host_tensors = torch_xla._XLAC._xla_get_cpu_tensors(
            tensors
        )
    finally:
        del dispatch_guard

    if len(host_tensors) != 33:
        raise RuntimeError(
            f"expected 33 host tensors, received {len(host_tensors)}"
        )

    for index, host in enumerate(host_tensors):
        if (
            type(host) is not torch.Tensor
            or host.device.type != "cpu"
        ):
            raise RuntimeError(
                f"XLA host transfer {index} is not a base CPU tensor: "
                f"type={type(host).__name__} "
                f"device={host.device}"
            )

        host.untyped_storage().data_ptr()

    return torch.stack(
        host_tensors,
        dim=1,
    )


def load_model_for_rank(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[MistralModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load() -> MistralModel:
        model = MistralModel.from_pretrained(
            args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        model.eval()
        model.to(device)
        torch_xla.sync(wait=True)
        gc.collect()
        return model

    model: MistralModel | None = None

    if args.parallel_model_load:
        model = load()
        xm.rendezvous("all-models-loaded")
    else:
        # Avoid eight simultaneous ~14 GB host-memory peaks.
        for loader_rank in range(world_size):
            if rank == loader_rank:
                model = load()
            xm.rendezvous(
                f"model-loaded-rank-{loader_rank}"
            )

    if model is None:
        raise RuntimeError("model was not initialized")

    return model, tokenizer


def load_spmd_model(
    args: argparse.Namespace,
    device: torch.device,
    physical_device_count: int,
):
    """Load one host model and shard its parameters over the TPU slice."""

    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
    )
    from torch_xla.distributed.spmd.xla_sharded_tensor import (
        XLAShardedTensor,
    )
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel as FSDPv2,
    )

    mesh = xs.Mesh(
        np.arange(physical_device_count),
        (physical_device_count, 1),
        ("fsdp", "model"),
    )
    xs.set_global_mesh(mesh)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"SPMD: loading one host copy of {args.model_id} "
        f"and sharding it over {physical_device_count} devices",
        flush=True,
    )

    base_model = MistralModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    base_model.eval()

    capture = LastTokenCapture(base_model)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MistralDecoderLayer},
    )

    def shard_output(
        output: Any,
        output_mesh,
    ) -> None:
        """Keep already-sharded Transformer outputs from being re-wrapped.

        PyTorch/XLA 2.9 can propagate an XLAShardedTensor through the
        Transformers output-capture decorator without its host-facing
        global_tensor slot. Calling mark_sharding on that wrapper raises
        AttributeError even though the underlying XLA value is already
        sharded. Native XLA tensors still receive the explicit annotation.
        """

        real_output = (
            output[0]
            if isinstance(output, tuple)
            else output
        )

        if not isinstance(real_output, torch.Tensor):
            raise TypeError(
                f"unsupported FSDP output type: {type(output)}"
            )

        if isinstance(real_output, XLAShardedTensor):
            return

        partition_spec = (
            ("fsdp",)
            + (None,) * (real_output.ndim - 1)
        )
        xs.mark_sharding(
            real_output,
            output_mesh,
            partition_spec,
        )

    # FSDPv2 moves the module to the SPMD virtual XLA device and shards every
    # decoder layer separately. return_dict=False is used during extraction so
    # its default output sharder sees a tuple whose first item is the activation.
    model = FSDPv2(
        base_model,
        mesh=mesh,
        auto_wrap_policy=auto_wrap_policy,
        shard_output=shard_output,
    )

    torch_xla.sync(wait=True)
    gc.collect()

    print(
        "SPMD: model parameters materialized and sharded",
        flush=True,
    )

    return model, tokenizer, capture, mesh


def completed_for_rank(
    metadata_path: Path,
) -> set[int]:
    if not metadata_path.exists():
        return set()

    return {
        int(row["source_index"])
        for row in read_jsonl(metadata_path)
    }


def next_shard_number(
    rank_dir: Path,
) -> int:
    numbers = []

    for path in rank_dir.glob("shard-*.safetensors"):
        numbers.append(
            int(path.stem.split("-")[-1])
        )

    return max(numbers, default=-1) + 1


def flush_shard(
    output: Path,
    rank: int,
    world_size: int,
    shard_number: int,
    captured: list[torch.Tensor],
    pending_metadata: list[dict[str, Any]],
) -> None:
    if not captured:
        return

    rank_dir = (
        output
        / "states"
        / f"rank-{rank:03d}"
    )
    rank_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"shard-{shard_number:05d}.safetensors"
    )
    path = rank_dir / filename

    combined = torch.cat(
        captured,
        dim=0,
    )

    embedding = combined[:, 0, :].contiguous()
    hidden_states = combined[:, 1:, :].contiguous()

    if (
        type(embedding) is not torch.Tensor
        or embedding.device.type != "cpu"
    ):
        raise RuntimeError(
            "embedding is not a base CPU tensor before serialization: "
            f"type={type(embedding).__name__} "
            f"device={embedding.device}"
        )

    if (
        type(hidden_states) is not torch.Tensor
        or hidden_states.device.type != "cpu"
    ):
        raise RuntimeError(
            "hidden_states is not a base CPU tensor before serialization: "
            f"type={type(hidden_states).__name__} "
            f"device={hidden_states.device}"
        )

    embedding.untyped_storage().data_ptr()
    hidden_states.untyped_storage().data_ptr()

    save_file(
        {
            "embedding": embedding,
            "hidden_states": hidden_states,
        },
        path,
    )

    relative = path.relative_to(output).as_posix()

    for offset, row in enumerate(pending_metadata):
        row.update(
            {
                "shard": relative,
                "offset": offset,
                "rank": rank,
                "world_size": world_size,
            }
        )

    append_jsonl(
        output / f"metadata-rank-{rank:03d}.jsonl",
        pending_metadata,
    )


def extract_rank(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    model: MistralModel,
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    all_records = load_records(args)

    source_to_record = {
        record.source_index: record
        for record in all_records
    }

    selected_indices = worker_source_indices(
        len(all_records),
        rank,
        world_size,
    )

    # Source indices may start after zero because --start-index applies to the
    # input file; map the worker position to the actual input source index.
    actual_sources = [
        all_records[position].source_index
        for position in selected_indices
    ]

    metadata_path = (
        args.output_dir
        / f"metadata-rank-{rank:03d}.jsonl"
    )
    done = (
        completed_for_rank(metadata_path)
        if args.resume
        else set()
    )

    records = [
        source_to_record[index]
        for index in actual_sources
        if index not in done
    ]

    capture = LastTokenCapture(model)
    buckets = tuple(args.buckets)

    tokenized_lengths = (
        tokenizer(
            [record.text for record in records],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_length=True,
        )["length"]
        if records
        else []
    )

    grouped: dict[
        int,
        list[tuple[InputRecord, int, int]],
    ] = defaultdict(list)

    for record, token_count in zip(
        records,
        tokenized_lengths,
        strict=True,
    ):
        bucket, truncated = assign_bucket(
            int(token_count),
            buckets,
        )
        grouped[bucket].append(
            (
                record,
                int(token_count),
                truncated,
            )
        )

    captured: list[torch.Tensor] = []
    pending_metadata: list[dict[str, Any]] = []

    shard_number = next_shard_number(
        args.output_dir
        / "states"
        / f"rank-{rank:03d}"
    )

    processed = 0
    started = time.perf_counter()

    for bucket in buckets:
        bucket_records = grouped.get(
            bucket,
            [],
        )

        for start in range(
            0,
            len(bucket_records),
            args.batch_size,
        ):
            batch = bucket_records[
                start : start + args.batch_size
            ]
            real_count = len(batch)

            # Pad the final batch by repeating its last record so every graph
            # for a bucket has the same static [batch, sequence] shape.
            if real_count < args.batch_size:
                batch = (
                    batch
                    + [batch[-1]]
                    * (args.batch_size - real_count)
                )

            texts = [
                item[0].text
                for item in batch
            ]

            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=bucket,
                add_special_tokens=False,
            )

            capture.clear()

            # inference_mode produces version-counter-free tensors.
            # Transformers' rotary embedding path moves one such tensor to XLA
            # and fails. no_grad has the same memory goal without that issue.
            with torch.no_grad():
                output = model(
                    input_ids=encoded[
                        "input_ids"
                    ].to(device),
                    attention_mask=encoded[
                        "attention_mask"
                    ].to(device),
                    use_cache=False,
                    return_dict=True,
                )

            del output

            stacked = capture.stacked()[:real_count]
            torch_xla.sync(wait=True)
            host = stacked.cpu()
            captured.append(host)

            for (
                record,
                token_count,
                truncated,
            ) in batch[:real_count]:
                pending_metadata.append(
                    {
                        "source_index": record.source_index,
                        "input_sha256": hashlib.sha256(
                            record.text.encode()
                        ).hexdigest(),
                        "token_count": token_count,
                        "bucket_length": bucket,
                        "truncated_tokens": truncated,
                        "label": record.label,
                    }
                )

            processed += real_count

            if (
                sum(
                    tensor.shape[0]
                    for tensor in captured
                )
                >= args.shard_size
            ):
                flush_shard(
                    args.output_dir,
                    rank,
                    world_size,
                    shard_number,
                    captured,
                    pending_metadata,
                )
                shard_number += 1
                captured.clear()
                pending_metadata.clear()

            if processed % 32 == 0:
                print(
                    f"rank={rank}/{world_size} "
                    f"processed={processed}/{len(records)} "
                    f"bucket={bucket}",
                    flush=True,
                )

    flush_shard(
        args.output_dir,
        rank,
        world_size,
        shard_number,
        captured,
        pending_metadata,
    )

    capture.close()

    return {
        "rank": rank,
        "assigned_records": len(actual_sources),
        "previously_completed": len(done),
        "processed_this_run": processed,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }


def extract_spmd(
    args: argparse.Namespace,
    physical_device_count: int,
    device: torch.device,
    model,
    tokenizer: AutoTokenizer,
    capture: LastTokenCapture,
    mesh,
) -> dict[str, Any]:
    """Extract with one process and an FSDP-sharded model on all devices."""

    import torch_xla.distributed.spmd as xs

    records = load_records(args)

    metadata_path = (
        args.output_dir
        / "metadata-rank-000.jsonl"
    )

    done = (
        completed_for_rank(metadata_path)
        if args.resume
        else set()
    )

    records = [
        record
        for record in records
        if record.source_index not in done
    ]

    buckets = tuple(args.buckets)

    tokenized_lengths = (
        tokenizer(
            [record.text for record in records],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_length=True,
        )["length"]
        if records
        else []
    )

    grouped: dict[
        int,
        list[tuple[InputRecord, int, int]],
    ] = defaultdict(list)

    for record, token_count in zip(
        records,
        tokenized_lengths,
        strict=True,
    ):
        bucket, truncated = assign_bucket(
            int(token_count),
            buckets,
        )
        grouped[bucket].append(
            (
                record,
                int(token_count),
                truncated,
            )
        )

    # --batch-size remains the per-device value. The global SPMD batch is
    # padded so its leading dimension is divisible by the FSDP mesh.
    global_batch_size = (
        args.batch_size
        * physical_device_count
    )

    captured: list[torch.Tensor] = []
    pending_metadata: list[dict[str, Any]] = []

    shard_number = next_shard_number(
        args.output_dir
        / "states"
        / "rank-000"
    )

    processed = 0
    started = time.perf_counter()

    for bucket in buckets:
        bucket_records = grouped.get(
            bucket,
            [],
        )

        for start in range(
            0,
            len(bucket_records),
            global_batch_size,
        ):
            batch = bucket_records[
                start : start + global_batch_size
            ]
            real_count = len(batch)

            if real_count < global_batch_size:
                batch = (
                    batch
                    + [batch[-1]]
                    * (global_batch_size - real_count)
                )

            encoded = tokenizer(
                [item[0].text for item in batch],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=bucket,
                add_special_tokens=False,
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded[
                "attention_mask"
            ].to(device)

            input_ids = xs.mark_sharding(
                input_ids,
                mesh,
                ("fsdp", None),
            )
            attention_mask = xs.mark_sharding(
                attention_mask,
                mesh,
                ("fsdp", None),
            )

            capture.clear()

            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=False,
                )

            del output
            del input_ids
            del attention_mask

            # Compose captures with normal subclass dispatch, then transfer the
            # underlying XLA TensorImpl through the guarded C++ conversion path.
            host = spmd_captures_to_cpu(capture)[:real_count]

            captured.append(host)

            for (
                record,
                token_count,
                truncated,
            ) in batch[:real_count]:
                pending_metadata.append(
                    {
                        "source_index": record.source_index,
                        "input_sha256": hashlib.sha256(
                            record.text.encode()
                        ).hexdigest(),
                        "token_count": token_count,
                        "bucket_length": bucket,
                        "truncated_tokens": truncated,
                        "label": record.label,
                    }
                )

            processed += real_count

            if (
                sum(
                    tensor.shape[0]
                    for tensor in captured
                )
                >= args.shard_size
            ):
                flush_shard(
                    args.output_dir,
                    0,
                    physical_device_count,
                    shard_number,
                    captured,
                    pending_metadata,
                )
                shard_number += 1
                captured.clear()
                pending_metadata.clear()

            print(
                f"spmd processed={processed}/{len(records)} "
                f"bucket={bucket} "
                f"global_batch={global_batch_size}",
                flush=True,
            )

    flush_shard(
        args.output_dir,
        0,
        physical_device_count,
        shard_number,
        captured,
        pending_metadata,
    )

    capture.close()

    return {
        "rank": 0,
        "assigned_records": len(records) + len(done),
        "previously_completed": len(done),
        "processed_this_run": processed,
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }


def finalize_manifest(
    args: argparse.Namespace,
    world_size: int,
    rank_reports: list[bytes],
    *,
    writer_count: int | None = None,
) -> None:
    writer_count = (
        world_size
        if writer_count is None
        else writer_count
    )

    metadata = []

    for rank in range(writer_count):
        path = (
            args.output_dir
            / f"metadata-rank-{rank:03d}.jsonl"
        )
        if path.exists():
            metadata.extend(read_jsonl(path))

    files = []
    total_tensor_rows = 0

    for path in sorted(
        (args.output_dir / "states").glob(
            "rank-*/shard-*.safetensors"
        )
    ):
        from safetensors import safe_open

        with safe_open(
            path,
            framework="pt",
            device="cpu",
        ) as handle:
            rows = int(
                handle.get_slice(
                    "hidden_states"
                ).get_shape()[0]
            )

        total_tensor_rows += rows

        files.append(
            {
                "path": path.relative_to(
                    args.output_dir
                ).as_posix(),
                "bytes": path.stat().st_size,
                "rows": rows,
                "sha256": sha256(path),
            }
        )

    reports = [
        json.loads(payload.decode())
        for payload in rank_reports
    ]

    manifest = planned_config(args) | {
        "status": "complete",
        "world_size": world_size,
        "writer_count": writer_count,
        "effective_batch_size": (
            args.batch_size * world_size
        ),
        "records": len(metadata),
        "tensor_rows": total_tensor_rows,
        "num_shards": len(files),
        "dtype": "bfloat16",
        "embedding_shape_per_record": [4096],
        "hidden_states_shape_per_record": [
            32,
            4096,
        ],
        "architecture": (
            "MistralModel (language-model head omitted)"
        ),
        "use_cache": False,
        "output_hidden_states": False,
        "token_position": (
            "last non-padding token; tokenizer uses "
            "left padding and left truncation"
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_xla": torch_xla.__version__,
        "rank_reports": sorted(
            reports,
            key=lambda report: report["rank"],
        ),
        "files": files,
    }

    write_json(
        args.output_dir / "manifest.json",
        manifest,
    )


def upload_to_hf_bucket(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.push_to_bucket:
        return None

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "--push-to-bucket requires an exported HF_TOKEN"
        )

    parts = (
        args.push_to_bucket
        .strip("/")
        .split("/")
    )

    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "--push-to-bucket must be OWNER/BUCKET"
        )

    prefix = (
        args.bucket_prefix
        or args.output_dir.name
    ).strip("/")

    if not prefix:
        raise ValueError(
            "bucket prefix cannot be empty"
        )

    destination = (
        f"{args.push_to_bucket}/{prefix}"
    )
    destination_uri = (
        f"hf://buckets/{destination}"
    )

    started = time.perf_counter()

    subprocess.run(
        [
            "hf",
            "buckets",
            "create",
            args.push_to_bucket,
            "--exist-ok",
        ],
        check=True,
        env=os.environ.copy(),
    )

    subprocess.run(
        [
            "hf",
            "buckets",
            "sync",
            str(args.output_dir),
            destination_uri,
        ],
        check=True,
        env=os.environ.copy(),
    )

    receipt = {
        "status": "complete",
        "bucket": args.push_to_bucket,
        "prefix": prefix,
        "url": (
            "https://huggingface.co/buckets/"
            f"{args.push_to_bucket}#{prefix}"
        ),
        "elapsed_seconds": (
            time.perf_counter() - started
        ),
    }

    write_json(
        args.output_dir / "upload.json",
        receipt,
    )

    # The first sync necessarily precedes upload.json; copy the receipt last.
    subprocess.run(
        [
            "hf",
            "buckets",
            "cp",
            str(args.output_dir / "upload.json"),
            f"{destination_uri}/upload.json",
        ],
        check=True,
        env=os.environ.copy(),
    )

    return receipt


def worker(
    _index: int,
    args: argparse.Namespace,
) -> None:
    rank = xr.global_ordinal()
    world_size = xr.world_size()

    expected = (
        1
        if args.debug_single_process
        else args.expected_world_size
    )

    if world_size != expected:
        raise RuntimeError(
            f"expected {expected} XLA workers, "
            f"observed {world_size}; "
            "check TPU topology or pass "
            "--expected-world-size"
        )

    device = torch_xla.device()

    model, tokenizer = load_model_for_rank(
        args,
        rank,
        world_size,
        device,
    )

    report = extract_rank(
        args,
        rank,
        world_size,
        device,
        model,
        tokenizer,
    )

    payloads = xm.rendezvous(
        "extraction-finished",
        json.dumps(report).encode(),
    )

    if rank == 0:
        finalize_manifest(
            args,
            world_size,
            payloads,
        )

    xm.rendezvous("manifest-written")

    if rank == 0:
        receipt = upload_to_hf_bucket(args)
        if receipt:
            print(
                json.dumps(
                    receipt,
                    sort_keys=True,
                ),
                flush=True,
            )

    xm.rendezvous(
        "optional-upload-finished"
    )

    print(
        json.dumps(
            report,
            sort_keys=True,
        ),
        flush=True,
    )


def run_spmd(
    args: argparse.Namespace,
) -> None:
    physical_device_count = (
        xr.global_runtime_device_count()
    )

    if (
        physical_device_count
        != args.expected_world_size
    ):
        raise RuntimeError(
            f"expected {args.expected_world_size} "
            "physical TPU devices, observed "
            f"{physical_device_count}; "
            "check the Kaggle accelerator/session"
        )

    device = torch_xla.device()

    model, tokenizer, capture, mesh = (
        load_spmd_model(
            args,
            device,
            physical_device_count,
        )
    )

    report = extract_spmd(
        args,
        physical_device_count,
        device,
        model,
        tokenizer,
        capture,
        mesh,
    )

    finalize_manifest(
        args,
        physical_device_count,
        [json.dumps(report).encode()],
        writer_count=1,
    )

    receipt = upload_to_hf_bucket(args)

    if receipt:
        print(
            json.dumps(
                receipt,
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            report,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()

    args.input_jsonl = (
        args.input_jsonl.resolve()
    )
    args.output_dir = (
        args.output_dir.resolve()
    )
    args.cache_dir = (
        args.cache_dir.resolve()
    )
    args.buckets = sorted(
        set(args.buckets)
    )

    if (
        args.batch_size < 1
        or args.shard_size < 1
    ):
        raise ValueError(
            "batch size and shard size must be positive"
        )

    if (
        args.push_to_bucket
        and not os.environ.get("HF_TOKEN")
    ):
        raise RuntimeError(
            "--push-to-bucket requires an exported HF_TOKEN"
        )

    if (
        args.execution_mode == "spmd_fsdp"
        and args.debug_single_process
    ):
        raise ValueError(
            "--debug-single-process is only valid "
            "with --execution-mode replicated"
        )

    if (
        args.execution_mode == "spmd_fsdp"
        and args.parallel_model_load
    ):
        raise ValueError(
            "--parallel-model-load is only valid "
            "with --execution-mode replicated"
        )

    prepare_output(args)

    if args.execution_mode == "spmd_fsdp":
        # SPMD is deliberately a single host process controlling all devices.
        # It must be enabled before the first XLA tensor/device is created.
        xr.use_spmd()
        run_spmd(args)
    else:
        torch_xla.launch(
            worker,
            args=(args,),
            debug_single_process=(
                args.debug_single_process
            ),
        )


if __name__ == "__main__":
    main()