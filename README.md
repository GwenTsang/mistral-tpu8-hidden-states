# Mistral hidden states on an 8-device TPU v5e slice

Memory-bounded extraction of last-token embeddings and all 32 hidden states
from [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
using all eight devices in a single-host TPU v5e-8 slice.

The implementation is intentionally narrow: it extracts representations from
already prepared prompt/answer records. It does **not** generate answers or
assign correctness labels.

## Design

- `torch_xla.launch` starts one process on every visible TPU device.
- Input row positions are partitioned by `position % world_size`, so eight
  workers process disjoint data concurrently.
- Each worker holds one BF16 `MistralModel` replica. The language-model head is
  omitted, `use_cache=False`, and `output_hidden_states=True` is never used.
- Forward hooks retain only `[:, -1, :]` from the embedding and each layer.
- Static sequence buckets avoid recompilation for every input length.
- Per-rank SafeTensors shards and metadata avoid write collisions.
- Model loading is staggered by default to avoid eight simultaneous ~14 GB
  host-memory peaks. Use `--parallel-model-load` only on a high-memory VM.

At the safe default `--batch-size 1`, an eight-device slice has an effective
batch size of eight.

## Install on the TPU VM

Python 3.12 and PyTorch/XLA 2.9 are the tested environment.

```bash
git clone https://github.com/GwenTsang/mistral-tpu8-hidden-states.git
cd mistral-tpu8-hidden-states

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optionally pre-download the pinned model once before spawning workers:

```bash
hf download mistralai/Mistral-7B-Instruct-v0.3 \
  --revision c170c708c41dac9275d15a8fff4eca08d52bab71
```

## Input format

The input is JSONL. Either provide paper-shaped records:

```json
{"context":"France has Paris as its capital.","question":"What is the capital of France?","best_answer":"Paris.","label":1}
```

or a fully joined `text` field:

```json
{"text":"Complete prompt and answer text"}
```

Structured records use the context-aware QA prompt from the released paper
code. `label` is optional and is copied into metadata without affecting the
forward pass.

## Run on all eight TPU devices

No `torchrun` wrapper is needed. PyTorch/XLA launches one worker on every
visible device:

```bash
python extract_hidden_states.py \
  --input-jsonl prepared_data/coqa_train.jsonl \
  --output-dir outputs/coqa-full \
  --answer-column best_answer \
  --answer-view full \
  --batch-size 1 \
  --shard-size 64 \
  --buckets 128 256 512 1024 2048 \
  --expected-world-size 8
```

For first-sentence truncation compatible with the released scanner, run a
separate output directory with:

```bash
python extract_hidden_states.py \
  --input-jsonl prepared_data/coqa_train.jsonl \
  --output-dir outputs/coqa-fst \
  --answer-view first_sentence \
  --expected-world-size 8
```

Use `--resume` to continue completed rank shards, or `--overwrite` to replace a
run whose configuration matches no longer matters. Overlength inputs are
left-truncated to the largest bucket and their removed-token count is recorded.

## Automatic Hugging Face Bucket upload

Install-time `huggingface_hub` provides the `hf` CLI. Export a write-capable
token; never pass the token on the command line:

```bash
export HF_TOKEN=hf_...

python extract_hidden_states.py \
  --input-jsonl prepared_data/coqa_train.jsonl \
  --output-dir outputs/coqa-full \
  --expected-world-size 8 \
  --push-to-bucket YOUR_USERNAME/mistral-hidden-states \
  --bucket-prefix coqa/full
```

Rank 0 creates the bucket if needed, syncs the completed output, and writes an
`upload.json` receipt with the Hub URL. Extraction fails before model loading
when `--push-to-bucket` is requested without `HF_TOKEN`.

## Output

Each `states/rank-NNN/shard-NNNNN.safetensors` contains:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `embedding` | `[N, 4096]` | Last-position input embedding |
| `hidden_states` | `[N, 32, 4096]` | Last-position state for all Mistral layers |

Layers 0–30 are post-transformer-block states before the final RMSNorm. Layer
31 is the post-layer-31, post-final-RMSNorm state. This matches
`outputs.hidden_states[1:]` in Transformers while avoiding retention of every
`[batch, sequence, 4096]` activation.

Validate counts, rank coverage, tensor shapes, finiteness, metadata offsets and
SHA-256 hashes:

```bash
python validate_output.py outputs/coqa-full \
  --report outputs/coqa-full/validation.json
```

## Scope and reproducibility

This repository makes hidden-state extraction reproducible; it cannot by
itself reproduce the paper's reported AUROCs. Exact paper reproduction is
virtually impossible from the public release because the authors did not
publish their generated `best_answer` records, judge labels, pinned software
snapshots, selected layers or probe outputs.

PyTorch/XLA documents that [`torch_xla.launch`](https://docs.pytorch.org/xla/release/r2.9/learn/xla-examples.html#running-on-multiple-tpu-devices)
uses all available TPU devices by default.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The checked-in version passed five unit tests and a real one-record,
single-worker TPU v5e smoke test, including output validation of BF16
`[1, 32, 4096]` hidden states. The eight-worker orchestration cannot be
exercised on a one-device slice; `--expected-world-size 8` makes a topology
mismatch fail immediately instead of silently using fewer devices.
