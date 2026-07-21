import json

from prepare_halueval_inputs import PartSpec, summary_prompt, validate_part
from run_halueval_tpu import (
    REGRESSION_LINE_INDICES,
    completed_and_valid,
    write_regression_sample,
)


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validate_complete_a100_part(tmp_path):
    spec = PartSpec("part-test", 10, 12)
    files = {
        "full": tmp_path / "part_full.jsonl",
        "fst": tmp_path / "part_fst.jsonl",
        "summary": tmp_path / "summary.json",
        "success": tmp_path / "SUCCESS",
    }
    files["success"].write_text("ok\n", encoding="utf-8")
    files["summary"].write_text(
        json.dumps(
            {
                "status": "complete",
                "source_range": [10, 12],
            }
        ),
        encoding="utf-8",
    )
    for view in ("full", "fst"):
        rows = []
        for source_index in range(10, 12):
            context = f"Document {source_index}."
            answer = f"Summary {source_index}."
            rows.append(
                {
                    "source_index": source_index,
                    "paper_split": "train" if source_index == 10 else "test",
                    "context": context,
                    "question": "Summarize the document.",
                    "best_answer": answer,
                    "label": source_index % 2,
                    "answer_view": view,
                    "text": f"{summary_prompt(context)} {answer}",
                }
            )
        write_jsonl(files[view], rows)

    views, summary = validate_part(spec, files)
    assert len(views["full"]) == len(views["fst"]) == 2
    assert summary["status"] == "complete"


def test_completed_and_valid_requires_matching_counts(tmp_path):
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"status": "complete", "records": 8_000}),
        encoding="utf-8",
    )
    (output / "validation.json").write_text(
        json.dumps({"valid": True, "records": 8_000}),
        encoding="utf-8",
    )
    assert completed_and_valid(output, 8_000)
    assert not completed_and_valid(output, 2_000)


def test_regression_sample_uses_known_problematic_line_positions(tmp_path):
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "sample.jsonl"
    rows = [{"line": index} for index in range(100)]
    write_jsonl(source, rows)
    write_regression_sample(source, destination)
    observed = [
        json.loads(line)["line"]
        for line in destination.read_text().splitlines()
    ]
    assert observed == list(REGRESSION_LINE_INDICES)
