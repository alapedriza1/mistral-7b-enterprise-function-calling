"""Tool schemas and system prompt for the enterprise function calling project."""

import json

TOOL_SCHEMAS = [
    {
        "name": "get_transaction_history",
        "description": "Retrieves transaction history for a specific account within a date range.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "The unique account identifier",
                },
                "start_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Start date in YYYY-MM-DD format",
                },
                "end_date": {
                    "type": "string",
                    "format": "date",
                    "description": "End date in YYYY-MM-DD format",
                },
                "transaction_type": {
                    "type": "string",
                    "enum": ["credit", "debit", "all"],
                    "description": "Filter by transaction type",
                    "default": "all",
                },
                "min_amount": {
                    "type": "number",
                    "description": "Minimum transaction amount filter",
                    "default": 0,
                },
            },
            "required": ["account_id", "start_date", "end_date"],
        },
    },
    {
        "name": "create_invoice",
        "description": "Creates a new invoice for a client.",
        "parameters": {
            "type": "object",
            "properties": {
                "client_name": {
                    "type": "string",
                    "description": "Full legal name of the client",
                },
                "client_id": {
                    "type": "string",
                    "description": "Client identifier in the system",
                },
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "number"},
                        },
                        "required": ["description", "quantity", "unit_price"],
                    },
                    "description": "List of line items on the invoice",
                },
                "currency": {
                    "type": "string",
                    "enum": ["GBP", "EUR", "USD", "CHF"],
                    "description": "Invoice currency",
                },
                "due_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Payment due date in YYYY-MM-DD format",
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes for the invoice",
                },
            },
            "required": [
                "client_name",
                "client_id",
                "line_items",
                "currency",
                "due_date",
            ],
        },
    },
    {
        "name": "run_compliance_check",
        "description": "Runs a compliance check against a specific regulatory framework for a given entity.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Name of the entity to check",
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["individual", "company", "fund"],
                    "description": "Type of entity",
                },
                "regulation": {
                    "type": "string",
                    "enum": ["AML", "KYC", "MiFID_II", "GDPR", "SOX"],
                    "description": "Regulatory framework to check against",
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "Legal jurisdiction (e.g., 'UK', 'EU', 'US')",
                },
                "check_depth": {
                    "type": "string",
                    "enum": ["basic", "enhanced", "full"],
                    "description": "Depth of the compliance check",
                    "default": "basic",
                },
            },
            "required": ["entity_name", "entity_type", "regulation", "jurisdiction"],
        },
    },
    {
        "name": "get_portfolio_summary",
        "description": "Retrieves a summary of an investment portfolio including asset allocation and performance metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "portfolio_id": {
                    "type": "string",
                    "description": "Unique portfolio identifier",
                },
                "as_of_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Valuation date in YYYY-MM-DD format",
                },
                "include_benchmarks": {
                    "type": "boolean",
                    "description": "Whether to include benchmark comparison",
                    "default": False,
                },
                "group_by": {
                    "type": "string",
                    "enum": ["asset_class", "sector", "geography", "currency"],
                    "description": "How to group the holdings",
                    "default": "asset_class",
                },
            },
            "required": ["portfolio_id", "as_of_date"],
        },
    },
    {
        "name": "search_customers",
        "description": "Searches the customer database using various filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search query"},
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "suspended", "all"],
                    "default": "all",
                },
                "segment": {
                    "type": "string",
                    "enum": ["retail", "corporate", "institutional", "high_net_worth"],
                    "description": "Customer segment filter",
                },
                "country": {
                    "type": "string",
                    "description": "ISO 2-letter country code",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "update_customer_record",
        "description": "Updates fields on an existing customer record.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Unique customer identifier",
                },
                "updates": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "format": "email"},
                        "phone": {"type": "string"},
                        "address": {"type": "string"},
                        "segment": {
                            "type": "string",
                            "enum": [
                                "retail",
                                "corporate",
                                "institutional",
                                "high_net_worth",
                            ],
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "inactive", "suspended"],
                        },
                    },
                    "description": "Key-value pairs of fields to update",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the update (for audit trail)",
                },
            },
            "required": ["customer_id", "updates", "reason"],
        },
    },
    {
        "name": "get_customer_risk_profile",
        "description": "Retrieves the risk profile and risk score for a customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Unique customer identifier",
                },
                "include_history": {
                    "type": "boolean",
                    "description": "Include historical risk score changes",
                    "default": False,
                },
                "risk_model_version": {
                    "type": "string",
                    "description": "Specific risk model version to use (defaults to latest)",
                    "default": "latest",
                },
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "schedule_client_meeting",
        "description": "Schedules a meeting with a client, checking availability and sending invitations.",
        "parameters": {
            "type": "object",
            "properties": {
                "client_id": {"type": "string", "description": "Client identifier"},
                "meeting_type": {
                    "type": "string",
                    "enum": ["review", "onboarding", "escalation", "general"],
                    "description": "Type of meeting",
                },
                "preferred_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Preferred date in YYYY-MM-DD format",
                },
                "duration_minutes": {
                    "type": "integer",
                    "enum": [15, 30, 45, 60, 90],
                    "description": "Meeting duration",
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of additional attendee email addresses",
                },
                "agenda": {
                    "type": "string",
                    "description": "Meeting agenda or description",
                },
            },
            "required": [
                "client_id",
                "meeting_type",
                "preferred_date",
                "duration_minutes",
            ],
        },
    },
    {
        "name": "extract_document_data",
        "description": "Extracts structured data from an uploaded document using OCR and NLP.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "ID of the uploaded document",
                },
                "document_type": {
                    "type": "string",
                    "enum": [
                        "invoice",
                        "contract",
                        "annual_report",
                        "bank_statement",
                        "tax_return",
                    ],
                    "description": "Expected document type",
                },
                "extract_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific fields to extract (e.g., ['total_amount', 'date', 'counterparty'])",
                },
                "language": {
                    "type": "string",
                    "description": "Document language (ISO 639-1 code)",
                    "default": "en",
                },
            },
            "required": ["document_id", "document_type", "extract_fields"],
        },
    },
    {
        "name": "run_database_query",
        "description": "Executes a read-only SQL query against the specified database.",
        "parameters": {
            "type": "object",
            "properties": {
                "database": {
                    "type": "string",
                    "enum": ["customers", "transactions", "audit_log", "products"],
                    "description": "Target database",
                },
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query to execute",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum number of rows to return",
                    "default": 100,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Query timeout in seconds",
                    "default": 30,
                },
            },
            "required": ["database", "query"],
        },
    },
    {
        "name": "generate_report",
        "description": "Generates a formatted report from specified data sources.",
        "parameters": {
            "type": "object",
            "properties": {
                "report_type": {
                    "type": "string",
                    "enum": [
                        "financial_summary",
                        "risk_assessment",
                        "audit_findings",
                        "client_portfolio",
                        "regulatory_filing",
                    ],
                    "description": "Type of report to generate",
                },
                "entity_id": {
                    "type": "string",
                    "description": "ID of the entity the report is about",
                },
                "period_start": {
                    "type": "string",
                    "format": "date",
                    "description": "Reporting period start date",
                },
                "period_end": {
                    "type": "string",
                    "format": "date",
                    "description": "Reporting period end date",
                },
                "format": {
                    "type": "string",
                    "enum": ["pdf", "xlsx", "html"],
                    "description": "Output format",
                    "default": "pdf",
                },
                "include_charts": {
                    "type": "boolean",
                    "description": "Whether to include data visualisations",
                    "default": True,
                },
            },
            "required": ["report_type", "entity_id", "period_start", "period_end"],
        },
    },
    {
        "name": "send_notification",
        "description": "Sends a notification to one or more recipients via the specified channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of recipient identifiers (email addresses or user IDs)",
                },
                "channel": {
                    "type": "string",
                    "enum": ["email", "sms", "in_app", "slack"],
                    "description": "Notification delivery channel",
                },
                "subject": {
                    "type": "string",
                    "description": "Notification subject line",
                },
                "body": {"type": "string", "description": "Notification body text"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Notification priority level",
                    "default": "normal",
                },
            },
            "required": ["recipients", "channel", "subject", "body"],
        },
    },
    {
        "name": "create_audit_task",
        "description": "Creates a new task in the audit workflow management system.",
        "parameters": {
            "type": "object",
            "properties": {
                "engagement_id": {
                    "type": "string",
                    "description": "Audit engagement identifier",
                },
                "task_title": {
                    "type": "string",
                    "description": "Title of the audit task",
                },
                "assigned_to": {
                    "type": "string",
                    "description": "Email of the assigned team member",
                },
                "due_date": {
                    "type": "string",
                    "format": "date",
                    "description": "Task due date in YYYY-MM-DD format",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Task priority",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "testing",
                        "review",
                        "documentation",
                        "sign_off",
                        "inquiry",
                    ],
                    "description": "Task category",
                },
                "description": {
                    "type": "string",
                    "description": "Detailed task description",
                },
            },
            "required": [
                "engagement_id",
                "task_title",
                "assigned_to",
                "due_date",
                "priority",
                "category",
            ],
        },
    },
    {
        "name": "escalate_issue",
        "description": "Escalates an issue to a higher authority or specialist team.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_id": {
                    "type": "string",
                    "description": "Existing issue or ticket identifier",
                },
                "escalation_level": {
                    "type": "string",
                    "enum": ["team_lead", "manager", "director", "partner"],
                    "description": "Level to escalate to",
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed reason for escalation",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["standard", "urgent", "critical"],
                    "description": "Urgency of the escalation",
                },
                "supporting_documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document IDs supporting the escalation",
                },
            },
            "required": ["issue_id", "escalation_level", "reason", "urgency"],
        },
    },
    {
        "name": "get_workflow_status",
        "description": "Retrieves the current status and progress of a workflow or engagement.",
        "parameters": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow or engagement identifier",
                },
                "include_subtasks": {
                    "type": "boolean",
                    "description": "Whether to include status of all subtasks",
                    "default": False,
                },
                "include_timeline": {
                    "type": "boolean",
                    "description": "Whether to include timeline/milestone information",
                    "default": False,
                },
            },
            "required": ["workflow_id"],
        },
    },
    {
        "name": "log_time_entry",
        "description": "Logs a time entry against a specific engagement or project.",
        "parameters": {
            "type": "object",
            "properties": {
                "engagement_id": {
                    "type": "string",
                    "description": "Engagement or project identifier",
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Date of the work in YYYY-MM-DD format",
                },
                "hours": {"type": "number", "description": "Number of hours worked"},
                "activity_code": {
                    "type": "string",
                    "enum": [
                        "planning",
                        "fieldwork",
                        "review",
                        "reporting",
                        "admin",
                        "travel",
                    ],
                    "description": "Activity category code",
                },
                "description": {
                    "type": "string",
                    "description": "Description of work performed",
                },
            },
            "required": [
                "engagement_id",
                "date",
                "hours",
                "activity_code",
                "description",
            ],
        },
    },
]

ALL_SCHEMAS_STR = json.dumps(TOOL_SCHEMAS, indent=2)
TOOL_NAMES = [tool["name"] for tool in TOOL_SCHEMAS]
TOOL_SCHEMA_MAP = {tool["name"]: tool for tool in TOOL_SCHEMAS}


MAX_SEQ_LENGTH = 2048


def build_system_prompt(tools: list[dict]) -> str:
    """Build the system prompt for a given set of tool schemas.

    Args:
        tools: List of tool schema dicts to include in the prompt.

    Returns:
        The complete system prompt string.
    """
    tools_json = json.dumps(tools, indent=2)
    return f"""You are a helpful enterprise assistant with access to the following tools:

{tools_json}

Each tool is described by its name, description, and the parameters it accepts. Use these tools to fulfill user requests that match their capabilities.

When a user request requires a tool, respond ONLY with a JSON object in this format:
{{"name": "<tool_name>", "arguments": {{"<param1>": <value1>, ...}}}}

If multiple tools are needed, respond with a JSON array of such objects:
[{{"name": "<tool_name_1>", "arguments": {{...}}}}, {{"name": "<tool_name_2>", "arguments": {{...}}}}]

- Only use the tools and parameters exactly as defined above.
- Do not invent new tools or parameters.
- If no tool is appropriate, respond conversationally and do NOT include any JSON.
"""


# Full system prompt with all tools (used at inference time)
SYSTEM_PROMPT = build_system_prompt(TOOL_SCHEMAS)
