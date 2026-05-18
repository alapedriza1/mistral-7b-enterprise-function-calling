"""Shared utilities: data formatting, I/O, spot-checking."""

import json
import random
from typing import Optional

import pandas as pd

from src.schemas import SYSTEM_PROMPT


def to_training_format(example: dict) -> dict:
    """
    Converts a validated raw example into Mistral chat training format.
    Expects the example to have '_category' metadata from generation.
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
    """Remove examples with duplicate user messages."""
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
    """Save examples to a .jsonl file."""
    with open(filepath, "w") as f:
        for ex in examples:
            record = ex.copy()
            if strip_category:
                record.pop("category", None)
                record.pop("_category", None)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples → {filepath}")


def load_jsonl(filepath: str) -> list[dict]:
    """Load a .jsonl file into a list of dicts."""
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


def spot_check(
    examples: list[dict], n: int = 5, category: Optional[str] = None
):
    """Print n random examples for manual review."""
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
    """Print summary statistics for a dataset."""
    print(f"Total examples: {len(examples)}")

    cats = pd.Series([
        ex.get("category", ex.get("_category", "unknown")) for ex in examples
    ]).value_counts()
    print(f"\nBy category:\n{cats.to_string()}")

    def estimate_tokens(ex):
        if "messages" in ex:
            return sum(len(m["content"]) for m in ex["messages"]) // 4
        total = len(ex.get("user_message", ""))
        resp = ex.get("assistant_response", "")
        total += len(resp) if isinstance(resp, str) else len(json.dumps(resp))
        return total // 4

    tokens = [estimate_tokens(ex) for ex in examples]
    print(f"\nApprox token lengths:")
    print(f"  Mean: {sum(tokens)/len(tokens):.0f}")
    print(f"  Min:  {min(tokens)} | Max: {max(tokens)}")
    print(f"  P95:  {sorted(tokens)[int(len(tokens)*0.95)]}")

    long = [t for t in tokens if t > 1800]
    if long:
        print(f"\n⚠️  {len(long)} examples may approach 2048 token limit")
    else:
        print(f"\n✅ All examples well within 2048 token limit")
