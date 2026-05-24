"""Shared utilities: data formatting, I/O, spot-checking."""

import json
import random
from typing import Optional

import pandas as pd

from src.schemas import SYSTEM_PROMPT


def to_training_format(example: dict) -> dict:
    """Convert a validated raw example into Mistral chat training format.

    Transforms a generation-stage example (with "user_message" and
    "assistant_response" keys) into the messages-based structure expected
    by the Mistral fine-tuning API. For "no_tool" examples the assistant
    response is kept as plain text; for all other categories it is
    JSON-serialised.

    Args:
        example: A validated example dict containing at minimum
            "user_message" and "assistant_response" keys, plus a
            "_category" metadata key set during generation.

    Returns:
        A dict with two keys:
          - "messages": A list of role/content dicts (system, user,
            assistant) ready for chat fine-tuning.
          - "category": The example category string (e.g. "simple",
            "complex", "no_tool").
    """
    category = example.get("_category", "unknown")
    user_msg = example["user_message"]
    response = example["assistant_response"]

    assistant_content = (
        response if category == "no_tool"
        else json.dumps(response, ensure_ascii=False)
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_content},
        ],
        "category": category,
    }


def deduplicate_examples(examples: list[dict]) -> list[dict]:
    """Remove examples with duplicate user messages (case-insensitive).

    Handles both raw format (with "user_message" key) and training format
    (with "messages" list). Comparison is performed on the lowercased,
    stripped user message text.

    Args:
        examples: List of example dicts in either raw or training format.

    Returns:
        A new list containing only the first occurrence of each unique
        user message. Prints the number of duplicates removed and the
        remaining count.
    """
    seen = set()
    unique = []
    dupes = 0

    for ex in examples:
        # Handle both raw and training format
        if "messages" in ex:
            msg = ex["messages"][1]["content"].strip().lower()
        else:
            msg = ex["user_message"].strip().lower()

        if msg not in seen:
            seen.add(msg)
            unique.append(ex)
        else:
            dupes += 1

    print(f"Removed {dupes} duplicates | Remaining: {len(unique)}")
    return unique



def save_jsonl(
    examples: list[dict], filepath: str, strip_category: bool = False
):
    """Persist a list of example dicts to a JSON Lines file.

    Each example is written as a single JSON object per line. Optionally
    strips internal category metadata before writing, producing a clean
    file suitable for upload to a fine-tuning API.

    Args:
        examples: List of example dicts to save.
        filepath: Destination file path (e.g. "data/train.jsonl").
        strip_category: If True, removes "category" and "_category" keys
            from each record before writing. Defaults to False.
    """
    with open(filepath, "w") as f:
        for ex in examples:
            record = ex.copy()
            if strip_category:
                record.pop("category", None)
                record.pop("_category", None)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples -> {filepath}")


def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSON Lines file into a list of dicts.

    Args:
        filepath: Path to the .jsonl file to read.

    Returns:
        A list of dicts, one per line in the file.
    """
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


def spot_check(
    examples: list[dict], n: int = 5, category: Optional[str] = None
):
    """Print randomly sampled examples for manual quality review.

    Displays the user message and assistant response for each sampled
    example, formatted with separators for readability. Supports filtering
    by category and handles both raw and training formats.

    Args:
        examples: List of example dicts in either raw or training format.
        n: Number of examples to sample and display. If the pool is
            smaller than n, all available examples are shown. Defaults to 5.
        category: If provided, restricts sampling to examples matching
            this category string. Defaults to None (sample from all).
    """
    pool = examples
    if category:
        pool = [
            ex for ex in examples
            if ex.get("category", ex.get("_category")) == category
        ]

    sample = random.sample(pool, min(n, len(pool)))

    for i, ex in enumerate(sample):
        cat = ex.get("category", ex.get("_category", "unknown"))

        if "messages" in ex:
            user_msg = ex["messages"][1]["content"]
            assistant_msg = ex["messages"][2]["content"]
        else:
            user_msg = ex["user_message"]
            resp = ex["assistant_response"]
            assistant_msg = resp if isinstance(resp, str) else json.dumps(resp, indent=2)

        print(f"\n{'='*70}")
        print(f"Example {i+1} | Category: {cat}")
        print(f"{'='*70}")
        print(f"USER: {user_msg}")
        print(f"\nASSISTANT: {assistant_msg}")


def dataset_stats(examples: list[dict]):
    """Print count and category distribution for a dataset.

    Args:
        examples: List of example dicts in either raw or training format.
            Must be non-empty.
    """
    print(f"Total examples: {len(examples)}")

    cats = pd.Series([
        ex.get("category", ex.get("_category", "unknown")) for ex in examples
    ]).value_counts()
    print(f"\nBy category:\n{cats.to_string()}")
