"""Pure-Python input and first-sentence handling for the TPU extractor."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InputRecord:
    source_index: int
    text: str
    question: str = ""
    answer: str = ""
    context: str = ""
    label: int | None = None
    original_answer: str = ""


def qa_prompt(context: str, question: str) -> str:
    """The context-aware QA prompt used by the paper's released code."""

    return (
        "Answer the question as briefly as possible, based only on the context:\n"
        f" Context:{context.strip()}\n Question:{question.strip()}\n Answer:"
    )


# Kept behavior-compatible with the paper's released FST scanner.
FST_FILTERS = (
    "\n",
    "Q:",
    "A:",
    "question:",
    "answer:",
    "Question:",
    "Answer:",
    "Questions:",
    "questions:",
    "QUESTION:",
    "ANSWER:",
    "REF",
    ".Forms",
    "http",
    "php",
    "Question",
    "Answer",
)
FST_WORD_ABBREVIATIONS = {
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Prof",
    "Sr",
    "Jr",
    "Gen",
    "Brig",
    "Adm",
    "Rear",
    "Lt",
    "Col",
    "Maj",
    "Capt",
    "St",
    "vs",
    "etc",
    "Fig",
    "Eq",
    "No",
}
FST_MULTI_DOT_ABBREVIATION = re.compile(r"(?:[A-Za-z]\.){2,}$")
FST_SINGLE_INITIAL = re.compile(r"^[A-Za-z]$")


def extract_first_sentence(text: str) -> str:
    text = text.strip()
    length = len(text)
    cursor = 0
    while cursor < length:
        char = text[cursor]
        if char not in ".!?":
            cursor += 1
            continue
        if char == "." and text[cursor : cursor + 3] == "...":
            cursor += 3
            continue
        if (
            char == "."
            and 0 < cursor < length - 1
            and text[cursor - 1].isdigit()
            and text[cursor + 1].isdigit()
        ):
            cursor += 1
            continue
        left = cursor - 1
        while left >= 0 and (text[left].isalpha() or text[left] == "."):
            left -= 1
        token = text[left + 1 : cursor].strip()
        if char == ".":
            right_is_letter_dot = (
                cursor + 2 < length
                and text[cursor + 1].isalpha()
                and text[cursor + 2] == "."
            )
            if cursor > 0 and text[cursor - 1].isalpha() and right_is_letter_dot:
                cursor += 1
                continue
        if "." in token and FST_MULTI_DOT_ABBREVIATION.match(token + "."):
            cursor += 1
            continue
        if token in FST_WORD_ABBREVIATIONS:
            if token == "No":
                right = cursor + 1
                while right < length and text[right].isspace():
                    right += 1
                if right < length and text[right].isdigit():
                    cursor += 1
                    continue
            else:
                cursor += 1
                continue
        if FST_SINGLE_INITIAL.match(token):
            right = cursor + 1
            while right < length and text[right].isspace():
                right += 1
            if right < length and text[right].isupper():
                cursor += 1
                continue
        return text[: cursor + 1].strip()
    return text


def first_sentence_truncation(answer: str) -> str:
    original = answer.strip()
    cut_position = len(answer)
    for marker in FST_FILTERS:
        marker_position = answer.find(marker)
        if 0 <= marker_position < cut_position:
            cut_position = marker_position
    filtered = answer[:cut_position].strip() or original
    return extract_first_sentence(filtered)


def record_from_mapping(
    row: dict[str, Any],
    source_index: int,
    *,
    text_column: str = "text",
    answer_column: str = "best_answer",
    answer_view: str = "full",
) -> InputRecord:
    """Normalize either prejoined text or paper-shaped QA JSON."""

    if answer_view not in {"full", "first_sentence"}:
        raise ValueError(f"unsupported answer_view={answer_view!r}")

    if row.get(text_column) not in (None, ""):
        if answer_view != "full":
            raise ValueError(
                "first-sentence truncation requires structured context/question/answer fields"
            )
        return InputRecord(
            source_index=source_index,
            text=str(row[text_column]),
            question=str(row.get("question", "")),
            answer=str(row.get(answer_column, row.get("answer", ""))),
            context=str(row.get("context", row.get("story", ""))),
            label=int(row["label"]) if row.get("label") is not None else None,
            original_answer=str(row.get(answer_column, row.get("answer", ""))),
        )

    context = str(row.get("context", row.get("story", "")))
    question = str(row.get("question", ""))
    answer_value = row.get(answer_column, row.get("answer", row.get("answers", "")))
    if isinstance(answer_value, dict):
        answer_value = answer_value.get("input_text", answer_value.get("text", ""))
    if isinstance(answer_value, (list, tuple)):
        answer_value = answer_value[0] if answer_value else ""
    original_answer = str(answer_value)
    answer = (
        first_sentence_truncation(original_answer)
        if answer_view == "first_sentence"
        else original_answer
    )
    if not question.strip() or not answer.strip():
        raise ValueError(
            f"row {source_index} needs non-empty `{text_column}`, or question plus `{answer_column}`"
        )
    return InputRecord(
        source_index=source_index,
        text=f"{qa_prompt(context, question)} {answer}",
        question=question,
        answer=answer,
        context=context,
        label=int(row["label"]) if row.get("label") is not None else None,
        original_answer=original_answer,
    )


def assign_bucket(token_count: int, buckets: tuple[int, ...]) -> tuple[int, int]:
    if token_count < 1:
        raise ValueError("token_count must be positive")
    if not buckets or any(value < 1 for value in buckets):
        raise ValueError("buckets must contain positive lengths")
    if tuple(sorted(set(buckets))) != buckets:
        raise ValueError("buckets must be sorted and unique")
    for bucket in buckets:
        if token_count <= bucket:
            return bucket, 0
    return buckets[-1], token_count - buckets[-1]


def worker_source_indices(total: int, rank: int, world_size: int) -> list[int]:
    if not 0 <= rank < world_size:
        raise ValueError("rank must be within world_size")
    return list(range(rank, total, world_size))
