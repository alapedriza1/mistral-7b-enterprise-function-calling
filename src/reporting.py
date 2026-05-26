"""Reporting and presentation utilities for evaluation results."""

import json
import random

import pandas as pd


# Tool-calling metrics used across all reporting functions.
# Update this list if metrics are added/removed in ExampleScore.
_TOOL_METRICS = [
    "json_valid",
    "func_name_correct",
    "required_fields",
    "type_correct",
    "enum_compliant",
    "hallucinated_params",
    "argument_match",
]

_ALL_METRICS = _TOOL_METRICS + ["no_tool_restraint"]

_BUCKET_LABELS = ["FAILURES", "PARTIAL SUCCESSES", "CLEAN SUCCESSES", "NO-TOOL EXAMPLES"]


def _classify_bucket(row: pd.Series) -> str:
    """Classify a single eval row into an outcome bucket.

    Args:
        row: A single row from the eval DataFrame.

    Returns:
        One of the bucket label strings.
    """
    if row["category"] == "no_tool":
        return "NO-TOOL EXAMPLES"
    if not row["json_valid"] or not row["func_name_correct"]:
        return "FAILURES"
    if row["argument_match"] < 1.0:
        return "PARTIAL SUCCESSES"
    return "CLEAN SUCCESSES"


def _extract_tool_name(expected_str: str) -> str | None:
    """Extract the primary tool name from a ground-truth expected string.

    Args:
        expected_str: JSON string of the expected function call(s).

    Returns:
        The tool name string, or None if parsing fails.
    """
    try:
        parsed = json.loads(expected_str)
    except (json.JSONDecodeError, TypeError):
        return None

    calls = parsed if isinstance(parsed, list) else [parsed]
    if calls and isinstance(calls[0], dict):
        return calls[0].get("name")
    return None


def overall_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Overall summary metrics. Returns a single-column DataFrame of raw floats.

    Args:
        eval_df: DataFrame from evaluate_results (one row per example).

    Returns:
        A single-column DataFrame with metric names as index and scores as values.
    """
    tool_rows = eval_df[eval_df["category"] != "no_tool"]
    no_tool_rows = eval_df[eval_df["category"] == "no_tool"]

    summary = {
        "JSON Validity Rate": tool_rows["json_valid"].mean(),
        "Function Name Accuracy": tool_rows["func_name_correct"].mean(),
        "Required Fields Completeness": tool_rows["required_fields"].mean(),
        "Type Correctness Rate": tool_rows["type_correct"].mean(),
        "Enum Compliance Rate": tool_rows["enum_compliant"].mean(),
        "Avg Hallucinated Params": tool_rows["hallucinated_params"].mean(),
        "Argument Value Match": tool_rows["argument_match"].mean(),
        "No-Tool Restraint Accuracy": (
            no_tool_rows["no_tool_restraint"].mean()
            if len(no_tool_rows) > 0 else None
        ),
    }

    return pd.DataFrame({"score": summary})


def breakdown_by_category(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Metrics broken down by category.

    Tool categories will show NaN for no_tool_restraint, and no_tool will
    show NaN for tool metrics - pandas mean() skips NaN by default.

    Args:
        eval_df: DataFrame from evaluate_results (one row per example).

    Returns:
        A DataFrame with categories as index and metric means as columns,
        plus an 'n' column with the count per category.
    """
    summary = eval_df.groupby("category")[_ALL_METRICS].mean()
    summary["n"] = eval_df.groupby("category").size()
    return summary.sort_index()


def breakdown_by_tool(eval_df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    """Metrics broken down by primary tool name. Returns raw numbers.

    Args:
        eval_df: DataFrame from evaluate_results (one row per example).
        results: The same results list passed to evaluate_results, used to
            extract tool names from the expected output.

    Returns:
        A DataFrame with tool names as index and metric means as columns,
        plus an 'n' column with the count per tool.
    """
    assert len(eval_df) == len(results), (
        f"eval_df ({len(eval_df)}) and results ({len(results)}) must have the same length and order"
    )

    tool_names = [
        None if r["category"] == "no_tool" else _extract_tool_name(r["expected"])
        for r in results
    ]

    df = eval_df.copy()
    df["tool_name"] = tool_names
    df = df[df["tool_name"].notna()]

    summary = df.groupby("tool_name")[_TOOL_METRICS].mean()
    summary["n"] = df.groupby("tool_name").size()

    return summary.sort_index()


def select_examples(
    eval_df: pd.DataFrame,
    n_per_bucket: int,
    shuffle: bool = False,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Select representative example indices from each outcome bucket.

    Args:
        eval_df: DataFrame from evaluate_results.
        n_per_bucket: Number of examples to select from each bucket.
        shuffle: If True, randomly sample from each bucket. If False,
            take the first N indices by position.
        seed: Random seed for reproducibility when shuffle=True.

    Returns:
        A dict mapping bucket labels to lists of positional indices.
    """
    buckets = eval_df.apply(_classify_bucket, axis=1)
    rng = random.Random(seed)
    selected: dict[str, list[int]] = {}

    for label in _BUCKET_LABELS:
        indices = eval_df.index[buckets == label].tolist()

        if shuffle and len(indices) > n_per_bucket:
            selected[label] = rng.sample(indices, n_per_bucket)
        else:
            selected[label] = indices[:n_per_bucket]

    return selected


def print_example(results: list[dict], eval_df: pd.DataFrame, idx: int):
    """Print a single example with its metrics.

    Args:
        results: List of result dicts from run_inference_on_test_set.
        eval_df: DataFrame from evaluate_results.
        idx: Positional index of the example to print.
    """
    r = results[idx]
    cat = r["category"]

    print(f"\n{'='*70}")
    print(f"  Category: {cat}")
    print(f"{'='*70}")
    print(f"\n  USER:\n{r['input_messages'][1]['content']}")
    print(f"\n  EXPECTED:\n{r['expected']}")
    print(f"\n  PREDICTED:\n{r['predicted']}")

    if cat != "no_tool":
        row = eval_df.loc[idx]
        print(f"\n  JSON={row['json_valid']} | Func={row['func_name_correct']} | "
              f"ReqFields={row['required_fields']} | ArgMatch={row['argument_match']}")


def print_qualitative_examples(
    results: list[dict],
    eval_df: pd.DataFrame,
    n_per_bucket: int,
    shuffle: bool = False,
    seed: int = 42,
):
    """Select and print representative examples from each outcome bucket.

    Args:
        results: List of result dicts from run_inference_on_test_set.
        eval_df: DataFrame from evaluate_results.
        n_per_bucket: Number of examples to print from each bucket.
        shuffle: If True, randomly sample from each bucket.
        seed: Random seed for reproducibility when shuffle=True.
    """
    selected = select_examples(eval_df, n_per_bucket=n_per_bucket, shuffle=shuffle, seed=seed)

    for label, indices in selected.items():
        print(f"\n\n{'#'*70}")
        print(f"# {label}")
        print(f"{'#'*70}")
        for idx in indices:
            print_example(results, eval_df, idx)
