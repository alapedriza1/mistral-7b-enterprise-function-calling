"""Evaluation metrics for function-calling fine-tuning."""

import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.schemas import TOOL_SCHEMA_MAP

def _try_parse_json(text: str):
    """Attempt to parse text as JSON, handling markdown fences.

    Args:
        text: Raw string that may contain JSON, optionally wrapped in
            markdown code fences.

    Returns:
        A tuple of (parsed_object, True) on success, or (None, False)
        on failure.
    """
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text), True
    except (json.JSONDecodeError, TypeError):
        return None, False

def _normalise_to_list(obj) -> list[dict]:
    """Wrap a single tool call dict into a list for uniform handling.

    Args:
        obj: A parsed JSON object - either a single dict or a list of dicts.

    Returns:
        A list of dicts. Single dicts are wrapped in a list, lists are
        returned as-is, and anything else returns an empty list.
    """
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return obj
    return []

@dataclass
class ExampleScore:
    """All metric scores for a single test example."""
    category: str
    json_valid: Optional[bool] = None
    func_name_correct: Optional[bool] = None
    required_fields: Optional[float] = None
    type_correct: Optional[float] = None
    enum_compliant: Optional[float] = None
    hallucinated_params: Optional[int] = None
    argument_match: Optional[float] = None
    no_tool_restraint: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "json_valid": self.json_valid,
            "func_name_correct": self.func_name_correct,
            "required_fields": self.required_fields,
            "type_correct": self.type_correct,
            "enum_compliant": self.enum_compliant,
            "hallucinated_params": self.hallucinated_params,
            "argument_match": self.argument_match,
            "no_tool_restraint": self.no_tool_restraint,
        }

_TYPE_MAP = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "array": list, "object": dict,
}


def _score_single_call(exp_call: dict, pred_calls: list[dict], schema: dict) -> dict:
    """Score one expected call against matching predicted call using its schema.

    Handles alignment (finding the matching predicted call by name) and
    all metric computation for a single tool invocation.

    Two sources of "expected" are used here:
      - exp_call (from the test set): the correct argument VALUES for this
        specific example. Used to check argument_match (did the model produce
        the right account_id, right dates, etc.).
      - schema (from schemas.py): the tool's structural DEFINITION (valid
        parameter names, types, enums, required fields). Used to check
        structural validity (required_fields, type_correct, enum_compliant,
        hallucinated_params).

    Args:
        exp_call: The ground-truth expected call dict with "name" and "arguments".
        pred_calls: The full list of predicted calls to search for a match.
        schema: The tool schema dict (with "parameters.properties" and
            "parameters.required").

    Returns:
        A dict with counts for each metric dimension:
          - required: (present, total)
          - typed: (correct, total)
          - enum: (correct, total)
          - hallucinated: int
          - args: (matched, total)
    """
    func_name = exp_call.get("name")
    exp_args = exp_call.get("arguments", {})

    # Find matching predicted call
    pred_call = next(
        (c for c in pred_calls if isinstance(c, dict) and c.get("name") == func_name),
        None,
    )
    pred_args = pred_call.get("arguments", {}) if pred_call else {}

    # Schema-defined constraints (structural validation)
    schema_properties = schema["parameters"].get("properties", {})
    schema_required = schema["parameters"].get("required", [])
    schema_valid_keys = set(schema_properties.keys())

    # Required fields: does pred include all mandatory params?
    required_present = sum(1 for f in schema_required if f in pred_args)

    # Hallucinated params: did pred invent keys not in the schema?
    hallucinated = sum(1 for k in pred_args if k not in schema_valid_keys)

    # Type correctness + enum compliance (pred values vs schema types/enums)
    typed_correct, typed_total = 0, 0
    enum_correct, enum_total = 0, 0

    for key, value in pred_args.items():
        if key not in schema_properties:
            continue

        prop = schema_properties[key]

        expected_type = prop.get("type")
        if expected_type in _TYPE_MAP:
            typed_total += 1
            if isinstance(value, _TYPE_MAP[expected_type]):
                typed_correct += 1

        if "enum" in prop:
            enum_total += 1
            if value in prop["enum"]:
                enum_correct += 1

    # Argument value match (pred values vs ground-truth expected values)
    args_matched = sum(
        1 for key, exp_val in exp_args.items()
        if key in pred_args and pred_args[key] == exp_val
    )

    return {
        "required": (required_present, len(schema_required)),
        "typed": (typed_correct, typed_total),
        "enum": (enum_correct, enum_total),
        "hallucinated": hallucinated,
        "args": (args_matched, len(exp_args)),
    }


def score_no_tool_example(predicted: str) -> ExampleScore:
    """Score a no-tool example: only restraint matters.

    Args:
        predicted: The model's raw predicted output string.

    Returns:
        An ExampleScore with only no_tool_restraint populated.
    """
    parsed, valid = _try_parse_json(predicted)

    showed_restraint = True
    if valid:
        if isinstance(parsed, dict) and "name" in parsed:
            showed_restraint = False
        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "name" in parsed[0]:
            showed_restraint = False

    return ExampleScore(category="no_tool", no_tool_restraint=showed_restraint)


def score_tool_example(predicted: str, expected: str, category: str) -> ExampleScore:
    """Score a tool-calling example by parsing, aligning, and delegating
    per-call scoring to _score_single_call.

    Uses two sources of truth:
      - expected (str): the ground-truth function call(s) from the test set,
        used for value-level comparison (argument_match).
      - TOOL_SCHEMA_MAP: the tool definitions from schemas.py, used for
        structural validation (required fields, types, enums, hallucination).

    Args:
        predicted: The model's raw predicted output string.
        expected: The ground-truth expected output string (JSON of function
            call(s) from the test set).
        category: The example category (e.g. "simple", "complex",
            "multi_tool", "ambiguous").

    Returns:
        An ExampleScore with all tool-calling metrics populated.
    """
    score = ExampleScore(category=category)

    # Parse model output
    pred_parsed, pred_ok = _try_parse_json(predicted)
    score.json_valid = pred_ok
    if not pred_ok:
        return score

    # Parse ground-truth expected output (from test set)
    exp_parsed, exp_ok = _try_parse_json(expected)
    if not exp_ok:
        return score

    pred_calls = _normalise_to_list(pred_parsed)
    exp_calls = _normalise_to_list(exp_parsed)

    # Function name accuracy
    exp_names = {c.get("name") for c in exp_calls}
    pred_names = {c.get("name") for c in pred_calls if isinstance(c, dict)}
    score.func_name_correct = (pred_names == exp_names)

    # Accumulate per-call counts
    total_required, present_required = 0, 0
    total_typed, correct_typed = 0, 0
    total_enum, correct_enum = 0, 0
    total_hallucinated = 0
    total_args, matched_args = 0, 0

    for exp_call in exp_calls:
        func_name = exp_call.get("name")
        if func_name not in TOOL_SCHEMA_MAP:
            continue

        counts = _score_single_call(exp_call, pred_calls, TOOL_SCHEMA_MAP[func_name])

        present_required += counts["required"][0]
        total_required += counts["required"][1]
        correct_typed += counts["typed"][0]
        total_typed += counts["typed"][1]
        correct_enum += counts["enum"][0]
        total_enum += counts["enum"][1]
        total_hallucinated += counts["hallucinated"]
        matched_args += counts["args"][0]
        total_args += counts["args"][1]

    score.required_fields = present_required / total_required if total_required > 0 else 1.0
    score.type_correct = correct_typed / total_typed if total_typed > 0 else 1.0
    score.enum_compliant = correct_enum / total_enum if total_enum > 0 else 1.0
    score.hallucinated_params = total_hallucinated
    score.argument_match = matched_args / total_args if total_args > 0 else 1.0

    return score

def evaluate_results(results: list[dict]) -> pd.DataFrame:
    """Evaluate all results. Returns a DataFrame with one row per example.

    Args:
        results: List of dicts with 'predicted', 'expected', and 'category' keys.

    Returns:
        A DataFrame with one row per example and columns for each metric
        in ExampleScore.
    """
    scores = []
    for r in results:
        if r["category"] == "no_tool":
            s = score_no_tool_example(r["predicted"])
        else:
            s = score_tool_example(r["predicted"], r["expected"], r["category"])
        scores.append(s.to_dict())

    return pd.DataFrame(scores)
