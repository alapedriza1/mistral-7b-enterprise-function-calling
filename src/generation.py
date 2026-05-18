"""Prompt templates and generation orchestration for synthetic dataset creation."""

import json
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.schemas import TOOL_SCHEMAS, TOOL_NAMES
from src.validation import (
    parse_model_response,
    validate_single_tool_example,
    validate_multi_tool_example,
    validate_no_tool_example,
)

# Name-keyed lookup from the schema list
TOOL_SCHEMAS_BY_NAME: dict[str, dict] = {t["name"]: t for t in TOOL_SCHEMAS}

# ---------------------------------------------------------------------------
# Prompt templates — shared skeleton + category-specific rules
# ---------------------------------------------------------------------------

_PREAMBLE = "You are generating synthetic training data for fine-tuning an LLM on function calling."
_FOOTER = "Return ONLY the JSON array. No markdown, no explanation, no code fences."

_CATEGORY_RULES = {
    "simple": """\
RULES FOR "SIMPLE" EXAMPLES:
- The user message must explicitly mention ALL required parameters (values are clearly stated)
- The user message should sound natural and conversational — how a real enterprise user would speak
- Vary the writing style: some formal, some casual, some terse, some verbose
- Use realistic enterprise data: real-sounding company names, plausible IDs (e.g., ACC-29571, CUST-4420), realistic dates, amounts, etc.
- Some examples should use ONLY required parameters; others should also include 1-2 optional parameters
- Do NOT include any optional parameters that the user didn't mention or imply
- The function call must be valid JSON with correct types matching the schema""",

    "complex": """\
RULES FOR "COMPLEX" EXAMPLES — these should be HARDER than basic examples:
- The user may use INDIRECT or AMBIGUOUS language that requires inference
- Parameters may be IMPLIED rather than explicitly stated. For example:
  - "last quarter" instead of specific dates (resolve to actual YYYY-MM-DD dates, using 2025 as current year — Q1=Jan-Mar, Q2=Apr-Jun, etc.)
  - "the Johnson account" instead of giving an ID (you should still generate a plausible ID in the response)
  - "send it in pounds" instead of saying "GBP"
  - Abbreviations, slang, or domain shorthand
- The user should mention or imply at least 1-2 OPTIONAL parameters that the model needs to include
- Some examples should have slightly messy or run-on user messages (realistic enterprise chat)
- The function call must still be valid JSON with correct types matching the schema
- Be creative with how users might phrase requests — think of how busy professionals actually type""",

    "multi_tool": """\
RULES FOR "MULTI-TOOL" EXAMPLES:
- The user message must be a single natural request that logically requires BOTH tools
- The request should feel natural — a real user would say this in one sentence or two
- Do NOT make it feel forced or artificial (e.g., don't say "and also run tool X")
- The response must be a JSON ARRAY with both function calls
- Each function call must have correct arguments matching its schema
- Use realistic enterprise data""",

    "ambiguous": """\
RULES FOR "AMBIGUOUS" EXAMPLES:
- The user message should use language that is somewhat vague or could apply to both tools
- But there must be enough context that Tool A is clearly the right choice upon careful reading
- This tests the model's ability to disambiguate between similar tools
- The response should ONLY call Tool A, NOT Tool B
- Use realistic enterprise data""",

    "no_tool": """\
RULES FOR "NO-TOOL" EXAMPLES:
- The user message should be related to the enterprise/financial domain (so it's thematically relevant) but should NOT require any of the available tools
- Examples of good no-tool queries:
  - Asking for explanations of concepts ("What does MiFID II regulate?")
  - Asking for advice ("How should I structure a client presentation?")
  - General knowledge questions ("What's the difference between IFRS and GAAP?")
  - Greetings or small talk ("Good morning, how are you?")
  - Asking about things outside the tool capabilities ("Can you translate this to French?")
- The assistant response should be a helpful, natural conversational answer (2-4 sentences)
- The assistant response must NOT contain any JSON or function call syntax""",
}


def _output_format(category: str, tool_name: str = "", tool_name_2: str = "") -> str:
    """Returns the OUTPUT FORMAT block for the given category."""
    if category == "no_tool":
        example = (
            '  {{\n'
            '    "user_message": "the natural language user question",\n'
            '    "assistant_response": "A helpful conversational response without any JSON."\n'
            '  }}'
        )
    elif category == "multi_tool":
        example = (
            '  {{\n'
            '    "user_message": "the natural language user request",\n'
            f'    "assistant_response": [{{"name": "{tool_name}", "arguments": {{...}}}}, '
            f'{{"name": "{tool_name_2}", "arguments": {{...}}}}]\n'
            '  }}'
        )
    else:
        name = tool_name
        example = (
            '  {{\n'
            '    "user_message": "the natural language user request",\n'
            f'    "assistant_response": {{"name": "{name}", "arguments": {{...}}}}\n'
            '  }}'
        )
    return f"OUTPUT FORMAT — return ONLY a JSON array, no other text:\n[\n{example},\n  ...\n]"


def build_prompt(
    category: str,
    num_examples: int,
    schema_1: Optional[dict] = None,
    schema_2: Optional[dict] = None,
) -> str:
    """
    Single prompt builder for all categories.

    Args:
        category: one of simple, complex, multi_tool, ambiguous, no_tool
        num_examples: how many examples to request
        schema_1: primary tool schema (None for no_tool)
        schema_2: secondary tool schema (multi_tool / ambiguous only)
    """
    parts = [_PREAMBLE, ""]

    # Schema section
    if category == "no_tool":
        parts.append(f"The LLM has access to these tools: {', '.join(TOOL_NAMES)}")
    elif category in ("multi_tool", "ambiguous"):
        label_1 = "TOOL SCHEMA 1" if category == "multi_tool" else "TOOL A (the CORRECT tool to use)"
        label_2 = "TOOL SCHEMA 2" if category == "multi_tool" else "TOOL B (a similar but INCORRECT tool for these requests)"
        parts.append(f"{label_1}:\n{json.dumps(schema_1, indent=2)}")
        parts.append(f"{label_2}:\n{json.dumps(schema_2, indent=2)}")
    else:
        parts.append(f"TOOL SCHEMA:\n{json.dumps(schema_1, indent=2)}")

    # Task description
    if category == "multi_tool":
        parts.append(
            f"Generate exactly {num_examples} examples where the user's request "
            f"naturally requires BOTH tools to be called."
        )
    elif category == "ambiguous":
        parts.append(
            f'Generate exactly {num_examples} examples where the user\'s request could '
            f'superficially seem to match EITHER tool, but "{schema_1["name"]}" is the correct choice.'
        )
    elif category == "no_tool":
        parts.append(
            f"Generate exactly {num_examples} examples where the user asks a question or "
            f"makes a request that should NOT trigger any tool call. The assistant should respond conversationally."
        )
    else:
        parts.append(
            f"Generate exactly {num_examples} examples. Each example is a pair: "
            f"a natural user message and the correct function call response."
        )

    # Category rules
    parts.append(_CATEGORY_RULES[category])

    # Output format
    tool_name = schema_1["name"] if schema_1 else ""
    tool_name_2 = schema_2["name"] if schema_2 else ""
    parts.append(_output_format(category, tool_name, tool_name_2))
    parts.append(_FOOTER)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool pairings
# ---------------------------------------------------------------------------

MULTI_TOOL_PAIRS = [
    ("search_customers", "get_customer_risk_profile"),
    ("run_compliance_check", "get_customer_risk_profile"),
    ("get_transaction_history", "generate_report"),
    ("extract_document_data", "create_audit_task"),
    ("search_customers", "schedule_client_meeting"),
    ("get_portfolio_summary", "generate_report"),
    ("get_workflow_status", "escalate_issue"),
    ("run_compliance_check", "send_notification"),
    ("create_invoice", "send_notification"),
    ("get_transaction_history", "run_compliance_check"),
]

AMBIGUOUS_PAIRS = [
    ("search_customers", "get_customer_risk_profile"),
    ("get_customer_risk_profile", "run_compliance_check"),
    ("search_customers", "update_customer_record"),
    ("get_workflow_status", "get_transaction_history"),
    ("extract_document_data", "run_database_query"),
    ("generate_report", "get_portfolio_summary"),
    ("create_audit_task", "log_time_entry"),
    ("escalate_issue", "create_audit_task"),
    ("send_notification", "schedule_client_meeting"),
    ("run_database_query", "get_transaction_history"),
]

# ---------------------------------------------------------------------------
# Declarative generation plan config
# ---------------------------------------------------------------------------

# (category, batches_per_item, batch_size)
PLAN_CONFIG = [
    ("simple", 4, 10),       # 4 batches × 10 = 40 per tool × 16 tools = 640
    ("complex", 3, 8),       # 3 batches × 8  = 24 per tool × 16 tools = 384  (close enough to ~400)
    ("multi_tool", 3, 8),    # 3 batches × 8  = 24 per pair × 10 pairs = 240
    ("ambiguous", 2, 8),     # 2 batches × 8  = 16 per pair × 10 pairs = 160
    ("no_tool", 16, 10),     # 16 batches × 10 = 160
]


@dataclass
class GenerationTask:
    """A single generation task to send to the LLM."""
    category: str
    prompt: str
    expected_count: int
    tool_name: Optional[str] = None
    tool_name_2: Optional[str] = None


def build_generation_plan() -> list[GenerationTask]:
    """Builds the full task list from PLAN_CONFIG (~1,600 examples)."""
    tasks: list[GenerationTask] = []

    for category, batches, batch_size in PLAN_CONFIG:
        if category in ("simple", "complex"):
            for schema in TOOL_SCHEMAS:
                for _ in range(batches):
                    tasks.append(GenerationTask(
                        category=category,
                        prompt=build_prompt(category, batch_size, schema_1=schema),
                        expected_count=batch_size,
                        tool_name=schema["name"],
                    ))

        elif category == "multi_tool":
            for t1, t2 in MULTI_TOOL_PAIRS:
                for _ in range(batches):
                    tasks.append(GenerationTask(
                        category=category,
                        prompt=build_prompt(
                            category, batch_size,
                            schema_1=TOOL_SCHEMAS_BY_NAME[t1],
                            schema_2=TOOL_SCHEMAS_BY_NAME[t2],
                        ),
                        expected_count=batch_size,
                        tool_name=t1,
                        tool_name_2=t2,
                    ))

        elif category == "ambiguous":
            for correct, distractor in AMBIGUOUS_PAIRS:
                for _ in range(batches):
                    tasks.append(GenerationTask(
                        category=category,
                        prompt=build_prompt(
                            category, batch_size,
                            schema_1=TOOL_SCHEMAS_BY_NAME[correct],
                            schema_2=TOOL_SCHEMAS_BY_NAME[distractor],
                        ),
                        expected_count=batch_size,
                        tool_name=correct,
                        tool_name_2=distractor,
                    ))

        elif category == "no_tool":
            for _ in range(batches):
                tasks.append(GenerationTask(
                    category=category,
                    prompt=build_prompt(category, batch_size),
                    expected_count=batch_size,
                ))

    return tasks


# ---------------------------------------------------------------------------
# Generation runner
# ---------------------------------------------------------------------------

def run_generation(
    tasks: list[GenerationTask],
    llm_call_fn,
    max_retries: int = 2,
    delay_between_calls: float = 1.5,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Executes all generation tasks, validates outputs, and returns results + log.

    Args:
        tasks: list of GenerationTask objects (from build_generation_plan)
        llm_call_fn: callable(prompt: str) -> str
        max_retries: retries per failed task
        delay_between_calls: seconds between API calls

    Returns:
        (all_examples, log_df)
    """
    all_examples: list[dict] = []
    log: list[dict] = []

    for i, task in enumerate(tasks, 1):
        task_id = f"[{i}/{len(tasks)}] {task.category}"
        if task.tool_name:
            task_id += f" | {task.tool_name}"
        if task.tool_name_2:
            task_id += f" + {task.tool_name_2}"

        for attempt in range(1, max_retries + 2):
            try:
                parsed = parse_model_response(llm_call_fn(task.prompt))
                valid, invalid = [], 0

                for ex in parsed:
                    errors = _validate(task, ex)
                    if errors:
                        invalid += 1
                    else:
                        ex["_category"] = task.category
                        valid.append(ex)

                all_examples.extend(valid)
                log.append(_log_entry(task_id, task, attempt, len(parsed), len(valid), invalid, "success"))
                print(f"  ✅ {task_id} | attempt {attempt} | {len(valid)}/{len(parsed)} valid")
                break

            except Exception as e:
                print(f"  ❌ {task_id} | attempt {attempt} | Error: {e}")
                if attempt == max_retries + 1:
                    log.append(_log_entry(task_id, task, attempt, 0, 0, 0, f"failed: {e}"))

            time.sleep(delay_between_calls)

        time.sleep(delay_between_calls)

        if i % 20 == 0:
            print(f"\n{'='*60}\nPROGRESS: {i}/{len(tasks)} tasks | "
                  f"{len(all_examples)} valid examples so far\n{'='*60}\n")

    return all_examples, pd.DataFrame(log)


def _validate(task: GenerationTask, example: dict) -> list[str]:
    """Route validation based on category."""
    if task.category == "no_tool":
        return validate_no_tool_example(example)
    if task.category == "multi_tool":
        return validate_multi_tool_example(
            example,
            TOOL_SCHEMAS_BY_NAME[task.tool_name],
            TOOL_SCHEMAS_BY_NAME[task.tool_name_2],
        )
    return validate_single_tool_example(example, TOOL_SCHEMAS_BY_NAME[task.tool_name])


def _log_entry(task_id, task, attempt, generated, valid, invalid, status) -> dict:
    return {
        "task_id": task_id, "category": task.category,
        "tool": task.tool_name, "attempt": attempt,
        "generated": generated, "valid": valid,
        "invalid": invalid, "status": status,
    }
