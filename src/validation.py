"""Validation functions for function-calling examples and model outputs."""

import json
import re

_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _check_structure(example: dict) -> list[str]:
    """Verify required top-level keys are present."""
    return [
        f"Missing '{k}'"
        for k in ("user_message", "assistant_response")
        if k not in example
    ]


def parse_model_response(response_text: str) -> list[dict]:
    """Parse a raw text response (possibly with markdown fences) into a list of dicts."""
    text = re.sub(r"```(?:json)?\s*", "", response_text).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in response")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}")
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list, got {type(parsed)}")
    return parsed


def validate_single_tool_example(example: dict, tool_schema: dict) -> list[str]:
    """Validate a single-tool example against its schema. Returns error list (empty = valid)."""
    errors = _check_structure(example)
    if errors:
        return errors

    response = example["assistant_response"]
    if isinstance(response, str):
        return ["assistant_response is a string, expected dict for tool call"]

    if "name" not in response:
        errors.append("Missing 'name' in assistant_response")
    elif response["name"] != tool_schema["name"]:
        errors.append(
            f"Wrong function name: '{response['name']}' (expected '{tool_schema['name']}')"
        )

    if "arguments" not in response:
        errors.append("Missing 'arguments' in assistant_response")
        return errors

    args = response["arguments"]
    properties = tool_schema["parameters"].get("properties", {})
    required_fields = tool_schema["parameters"].get("required", [])

    for field in required_fields:
        if field not in args:
            errors.append(f"Missing required field: '{field}'")

    for field, value in args.items():
        if field not in properties:
            errors.append(f"Hallucinated field: '{field}' (not in schema)")
            continue
        prop = properties[field]
        if "enum" in prop and value not in prop["enum"]:
            errors.append(
                f"Invalid enum value for '{field}': '{value}' (valid: {prop['enum']})"
            )
        expected_type = _TYPE_MAP.get(prop.get("type"))
        if expected_type and not isinstance(value, expected_type):
            errors.append(
                f"Wrong type for '{field}': expected {prop['type']}, got {type(value).__name__}"
            )

    return errors


def validate_multi_tool_example(example: dict, *tool_schemas: dict) -> list[str]:
    """Validate a multi-tool example against N schemas. Returns error list (empty = valid)."""
    errors = _check_structure(example)
    if errors:
        return errors

    response = example["assistant_response"]
    if not isinstance(response, list):
        return [f"Expected list response for multi-tool, got {type(response)}"]
    if len(response) != len(tool_schemas):
        return [f"Expected {len(tool_schemas)} tool calls, got {len(response)}"]

    expected_names = {s["name"] for s in tool_schemas}
    actual_names = {call.get("name") for call in response}
    if actual_names != expected_names:
        return [f"Expected tools {expected_names}, got {actual_names}"]

    schema_by_name = {s["name"]: s for s in tool_schemas}
    for call in response:
        sub_example = {
            "user_message": example["user_message"],
            "assistant_response": call,
        }
        errors.extend(
            validate_single_tool_example(sub_example, schema_by_name[call["name"]])
        )

    return errors


def validate_no_tool_example(example: dict) -> list[str]:
    """Validate a no-tool example. Returns error list (empty = valid)."""
    errors = _check_structure(example)
    if errors:
        return errors

    response = example["assistant_response"]
    if not isinstance(response, str):
        return [f"No-tool response should be a string, got {type(response)}"]

    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "name" in parsed:
            errors.append("No-tool response contains a JSON function call")
    except (json.JSONDecodeError, TypeError):
        pass

    if len(response.strip()) < 20:
        errors.append(f"Response too short ({len(response.strip())} chars)")

    return errors
