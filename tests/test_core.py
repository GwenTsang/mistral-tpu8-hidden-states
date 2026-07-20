import pytest

from mistral_tpu8.core import (
    assign_bucket,
    extract_first_sentence,
    first_sentence_truncation,
    record_from_mapping,
    worker_source_indices,
)


def test_fst_matches_released_examples():
    assert extract_first_sentence("The answer is Paris. More text follows.") == "The answer is Paris."
    assert extract_first_sentence("Dr. Smith gave 3.14 units. More.") == "Dr. Smith gave 3.14 units."
    assert first_sentence_truncation("Paris.\nQuestion: another") == "Paris."
    assert first_sentence_truncation("https://example.com") == "https://example."


def test_structured_record_full_and_fst():
    row = {
        "context": "France has Paris as its capital.",
        "question": "What is the capital of France?",
        "best_answer": "Paris. It is France's largest city.",
        "label": 1,
    }
    full = record_from_mapping(row, 7, answer_view="full")
    fst = record_from_mapping(row, 7, answer_view="first_sentence")
    assert full.answer == "Paris. It is France's largest city."
    assert fst.answer == "Paris."
    assert full.label == fst.label == 1
    assert full.text.endswith(" Paris. It is France's largest city.")
    assert fst.text.endswith(" Paris.")


def test_prejoined_text_rejects_fst():
    with pytest.raises(ValueError, match="structured"):
        record_from_mapping({"text": "already joined"}, 0, answer_view="first_sentence")


def test_bucket_assignment_and_truncation_count():
    buckets = (128, 256, 512)
    assert assign_bucket(1, buckets) == (128, 0)
    assert assign_bucket(128, buckets) == (128, 0)
    assert assign_bucket(129, buckets) == (256, 0)
    assert assign_bucket(700, buckets) == (512, 188)


def test_eight_way_strided_partition_is_complete_and_disjoint():
    partitions = [worker_source_indices(101, rank, 8) for rank in range(8)]
    flattened = [index for part in partitions for index in part]
    assert sorted(flattened) == list(range(101))
    assert len(flattened) == len(set(flattened))
    for rank, indices in enumerate(partitions):
        assert all(index % 8 == rank for index in indices)
