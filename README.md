# Mistral-7B Enterprise Function Calling

Fine-tune Mistral-7B-Instruct-v0.3 with QLoRA to reliably perform structured function calling across 16 enterprise tool schemas.

**Adapter**: [alapedriza/mistral-7b-function-calling-adapter](https://huggingface.co/alapedriza/mistral-7b-function-calling-adapter)

## Motivation

Agentic systems depend on reliable function calling, but smaller 7B models are known to be inconsistent: they hallucinate function names, break JSON formatting, use wrong argument types, and omit required fields. This makes function calling one of the clearest cases where fine-tuning demonstrably outperforms prompting alone.

Mistral-7B-Instruct-v0.3 already has basic function calling awareness (special tokens like `[TOOL_CALLS]`), but its reliability on complex or custom schemas is poor. This project fine-tunes it on 16 enterprise-themed tool schemas spanning financial services, compliance, and operations to quantify how much QLoRA can close the gap.

The same pipeline applies to larger models (Mistral Small/Medium) in enterprise settings with GPU budget. The 7B size was chosen for hardware accessibility (Kaggle free-tier T4 GPUs).

## Results

| Metric | Baseline | Fine-Tuned | Delta |
| --- | --- | --- | --- |
| JSON Validity Rate | 87.1% | 97.8% | +10.8 pp |
| Function Name Accuracy | 91.7% | 97.1% | +5.3 pp |
| Required Fields Completeness | 95.5% | 95.6% | +0.1 pp |
| Type Correctness | 100% | 100% | - |
| Enum Compliance | 100% | 99.3% | -0.7 pp |
| Argument Value Match | 76.8% | 83.7% | +6.8 pp |
| Avg Hallucinated Params | 0.008 | 0.000 | -0.008 |
| No-Tool Restraint | 100% | 100% | - |

Evaluated on 154 held-out test examples (139 tool-call + 15 no-tool).

## Tool Schemas

The model is trained on 16 enterprise tools spanning financial services, compliance, and operations:

| Tool | Description |
| --- | --- |
| `get_transaction_history` | Retrieve account transactions within a date range |
| `create_invoice` | Create invoices with nested line items |
| `run_compliance_check` | Run regulatory checks (AML, KYC, MiFID II, GDPR, SOX) |
| `get_portfolio_summary` | Investment portfolio allocation and performance |
| `search_customers` | Search customer database with filters |
| `update_customer_record` | Update customer fields with audit trail |
| `get_customer_risk_profile` | Retrieve customer risk assessment |
| `schedule_client_meeting` | Schedule meetings with attendees |
| `generate_report` | Generate formatted reports (PDF, Excel, CSV) |
| `send_notification` | Send notifications via email/SMS/push |
| `run_database_query` | Execute read-only SQL queries |
| `get_workflow_status` | Check workflow/approval status |
| `create_audit_task` | Create compliance audit tasks |
| `log_time_entry` | Log billable/non-billable time entries |
| `escalate_issue` | Escalate issues with priority levels |
| `extract_document_data` | Extract structured data from documents |

## Repository Structure

```
mistral-7b-enterprise-function-calling/
├── README.md
├── LICENSE
├── data/
│   ├── train.jsonl          # 1228 training examples
│   ├── val.jsonl            # 154 validation examples
│   └── test.jsonl           # 139 test examples
├── src/
│   ├── schemas.py           # Tool schemas and system prompt builder
│   ├── generation.py        # Synthetic data generation
│   ├── validation.py        # Data validation utilities
│   ├── utils.py             # Data loading, saving, stats
│   ├── training.py          # Training pipeline (LoRA, SFTTrainer)
│   ├── inference.py         # Model loading and inference
│   ├── evaluation.py        # Per-example scoring logic
│   └── reporting.py         # Summary tables and qualitative analysis
├── notebooks/
│   ├── 01-dataset-generation # Synthetic dataset creation
│   ├── 02-baseline-evaluation# Baseline Mistral-7B benchmarks
│   ├── 03-fine-tuning        # QLoRA training on Kaggle T4x2
│   └── 04-evaluation         # Fine-tuned vs baseline comparison
└── results/
    ├── baseline_results.jsonl
    ├── baseline_eval_df.csv
    ├── baseline_summary.csv
    ├── baseline_cat_breakdown.csv
    ├── baseline_tool_breakdown.csv
    ├── finetuned_results.jsonl
    ├── finetuned_eval_df.csv
    ├── finetuned_summary.csv
    ├── finetuned_cat_breakdown.csv
    └── finetuned_tool_breakdown.csv
```

## Training Details

### Hardware
- Kaggle T4x2 (2x NVIDIA T4 16GB)
- Pipeline parallelism via `device_map="auto"` (model split across both GPUs)

### Configuration
- **Quantization**: BitsAndBytes 4-bit NF4, double quantization, compute in float16
- **LoRA**: r=16, alpha=32, dropout=0.05, applied to all projection layers (q, k, v, o, gate, up, down)
- **Training**: 1 epoch, batch=1, gradient accumulation=16 (effective batch=16), lr=2e-4, cosine schedule, paged AdamW 8-bit
- **Sequence length**: MAX_SEQ_LENGTH=2048
- **Loss**: Assistant-only (masked system/user tokens)
- **Time**: 511 minutes (77 optimizer steps)

### Key Optimization: Tool Trimming

The system prompt with all 16 tools is ~5500 tokens, making full-context training infeasible on T4 GPUs within a 12h session. The solution: each training example includes only the tools actually used in the assistant response plus 1 random distractor tool. This reduces mean sequence length from ~5500 to ~915 tokens while teaching the model to select from a focused set.

### Training Metrics
- Train loss: 0.1889
- Validation loss: 0.1838 (no overfitting)

## Evaluation Methodology

Each test example is scored on:

| Metric | Description |
| --- | --- |
| `json_valid` | Is the output valid JSON? |
| `func_name_correct` | Does it call the right function? |
| `required_fields` | Fraction of required parameters present |
| `type_correct` | Are parameter types correct? |
| `enum_compliant` | Are enum values valid? |
| `hallucinated_params` | Count of parameters not in schema |
| `argument_match` | Fraction of argument values matching expected |
| `no_tool_restraint` | Does the model abstain when no tool is appropriate? |

Test examples span 5 categories: `simple` (62), `complex` (37), `multi_tool` (24), `ambiguous` (16), `no_tool` (15).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    padding_side="left",
)

# Apply fine-tuned adapter
model = PeftModel.from_pretrained(model, "alapedriza/mistral-7b-function-calling-adapter")

# Build messages with tool schemas in system prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant with access to tools..."},
    {"role": "user", "content": "Get the transaction history for account ACC-123 from 2024-01-01 to 2024-03-31"},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=256, temperature=0.0, do_sample=False)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)
# {"name": "get_transaction_history", "arguments": {"account_id": "ACC-123", "start_date": "2024-01-01", "end_date": "2024-03-31"}}
```

## Notebooks

| Notebook | Purpose | Runtime |
| --- | --- | --- |
| 01-dataset-generation | Generate synthetic training data using GPT-4o | ~30 min |
| 02-baseline-evaluation | Benchmark untuned Mistral-7B | ~45 min |
| 03-fine-tuning | QLoRA training on Kaggle T4x2 | ~8.5 hours |
| 04-evaluation | Compare fine-tuned vs baseline | ~45 min |

All notebooks are designed to run on Kaggle with T4x2 GPU acceleration.

## License

See [LICENSE](LICENSE) for details.
