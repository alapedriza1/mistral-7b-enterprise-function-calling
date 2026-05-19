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
    """Verify that required top-level keys are present in an example dict.

    Checks for the existence of "user_message" and "assistant_response"
    keys, which are mandatory in every generated example regardless of
    category.

    Args:
        example: A candidate example dict to validate structurally.

    Returns:
        A list of error message strings, one for each missing key.
        An empty list indicates both required keys are present.
    """
    return [
        f"Missing '{k}'"
        for k in ("user_message", "assistant_response")
        if k not in example
    ]


def parse_model_response(response_text: str) -> list[dict]:
    """Parse a raw LLM text response into a list of example dicts.

    Handles responses that may be wrapped in markdown code fences
    (```json ... ```) by stripping them first. Locates the outermost
    JSON array brackets and parses the content between them.

    Args:
        response_text: The raw string returned by the generator LLM,
            potentially containing markdown fences, leading/trailing
            whitespace, or explanatory text surrounding the JSON array.

    Returns:
        A list of dicts parsed from the JSON array found in the response.

    Raises:
        ValueError: If no JSON array brackets are found, if the content
            between brackets is not valid JSON, or if the parsed result
            is not a list.
    """
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
    """Validate a single-tool function-calling example against its schema.

    Performs the following checks in order:
      1. Structural check (required top-level keys).
      2. Response type check (must be a dict, not a string).
      3. Function name matches the expected tool schema name.
      4. "arguments" key is present in the response.
      5. All required parameters (per schema) are provided.
      6. No hallucinated fields (arguments not defined in the schema).
      7. Enum constraints are respected for constrained fields.
      8. Value types match the JSON Schema type declarations.

    Args:
        example: A candidate example dict with "user_message" and
            "assistant_response" keys. The "assistant_response" should
            be a dict with "name" and "arguments" keys.
        tool_schema: The tool's JSON Schema definition dict containing
            at minimum "name" and "parameters" (with "properties" and
            optionally "required" sub-keys).

    Returns:
        A list of error message strings describing validation failures.
        An empty list indicates the example is fully valid.
    """
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
    """Validate a multi-tool function-calling example against N tool schemas.

    Checks that the assistant response is a list of tool calls matching
    the expected set of tools (by name), then delegates per-call validation
    to validate_single_tool_example for argument-level checks.

    Performs the following checks:
      1. Structural check (required top-level keys).
      2. Response is a list with exactly len(tool_schemas) entries.
      3. The set of called tool names matches the expected schema names.
      4. Each individual tool call passes single-tool validation.

    Args:
        example: A candidate example dict with "user_message" and
            "assistant_response" keys. The "assistant_response" should
            be a list of dicts, each with "name" and "arguments" keys.
        *tool_schemas: Variable number of tool schema dicts (one per
            expected tool call). Each must contain "name" and "parameters"
            keys. Order does not need to match the response order.

    Returns:
        A list of error message strings describing validation failures.
        An empty list indicates the example is fully valid.
    """
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
    """Validate a no-tool example where no function call should be triggered.

    Ensures the assistant response is a plain conversational string that
    does not accidentally contain a serialised function call. Also enforces
    a minimum length to catch degenerate or empty responses.

    Performs the following checks:
      1. Structural check (required top-level keys).
      2. Response is a string (not a dict or list).
      3. Response does not parse as a JSON object with a "name" key
         (which would indicate an accidental function call).
      4. Response is at least 20 characters long after stripping whitespace.

    Args:
        example: A candidate example dict with "user_message" and
            "assistant_response" keys. The "assistant_response" should
            be a plain-text conversational string.

    Returns:
        A list of error message strings describing validation failures.
        An empty list indicates the example is fully valid.
    """
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
