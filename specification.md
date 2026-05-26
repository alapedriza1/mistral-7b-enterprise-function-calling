# 📋 PROJECT SPECIFICATION
# Fine-Tuning Mistral 7B for Reliable Enterprise Function Calling

---

# 1. PROJECT OVERVIEW

## 1.1 Objective

Fine-tune `mistralai/Mistral-7B-Instruct-v0.3` using QLoRA to significantly improve its ability to reliably perform function calling — selecting the correct function and producing valid, accurate arguments as structured JSON — across a custom set of enterprise-themed tool schemas.

---

## 1.2 Why This Task

The Mistral Applied AI role involves deploying fine-tuned models into enterprise environments and tackling agentic use cases.

Agentic systems depend entirely on reliable function calling. Smaller 7B models are known to be inconsistent at this:

- They hallucinate function names  
- Break JSON formatting  
- Use wrong argument types  
- Omit required fields  

This makes function calling one of the clearest cases where fine-tuning demonstrably outperforms prompting alone — which is the central point you want to prove.

---

## 1.3 Why Mistral 7B Instruct v0.3

- It's a Mistral model — shows direct product familiarity for a Mistral role  
- The Instruct v0.3 variant already has basic function calling awareness in its tokeniser (it uses special tokens like `[TOOL_CALLS]`), but its reliability on complex/custom schemas is poor — giving you a meaningful baseline to improve upon  
- 7B parameter size is fine-tunable with QLoRA on a free-tier Kaggle T4/P100 GPU  

In your README, you'll explicitly note:  

> "In an enterprise setting with GPU budget, this same pipeline applies to Mistral Small/Medium. I chose 7B for hardware accessibility."

---

## 1.4 Constraints

- **Compute**: Kaggle free tier only (T4 16GB or P100 16GB, 30 hours/week)  
- **Cost**: £0 — all tools and libraries are free/open-source  
- **No cloud deployment**: All inference runs locally in a notebook  
- **Time budget**: ~20–25 hours of your time across 2–3 weekends  

---

# 2. DELIVERABLES

| # | Deliverable | Format |
|--|------------|--------|
| 1 | Public GitHub repository | Repo with all code, data, and documentation |
| 2 | Synthetic dataset | `.jsonl` files (train/val/test) |
| 3 | Dataset generation notebook | `01_dataset_generation.ipynb` |
| 4 | Base model benchmarking notebook | `02_baseline_evaluation.ipynb` |
| 5 | Fine-tuning notebook | `03_fine_tuning.ipynb` |
| 6 | Evaluation & results notebook | `04_evaluation_and_results.ipynb` |
| 7 | Trained LoRA adapter weights | Uploaded to HuggingFace Hub |
| 8 | README.md | Technical write-up |

---

# 3. TOOL SCHEMAS — WHAT YOU'RE DEFINING

You'll define **16 tools organised into 4 enterprise domains**.

The domains are flavoured toward financial services and enterprise operations to connect to your professional background, but they're generic enough that anyone can understand them.

---

## 3.1 Domain 1: Financial Operations (4 tools)

- `get_transaction_history`  
- `create_invoice`  
- `run_compliance_check`  
- `get_portfolio_summary`  

---

## 3.2 Domain 2: Customer Management (4 tools)

- `search_customers`  
- `update_customer_record`  
- `get_customer_risk_profile`  
- `schedule_client_meeting`  

---

## 3.3 Domain 3: Document & Data Operations (4 tools)

- `extract_document_data`  
- `run_database_query`  
- `generate_report`  
- `send_notification`  

---

## 3.4 Domain 4: Workflow & Task Management (4 tools)

- `create_audit_task`  
- `escalate_issue`  
- `get_workflow_status`  
- `log_time_entry`  

---

## 3.5 Why 16 Tools?

- **Enough to be non-trivial**: The model must disambiguate between similar tools  
- **Manageable data generation**: 16 tools × ~100 examples = 1,600 examples  
- **Varying complexity**: Flat vs nested vs arrays  
- **Real ambiguity**: Multiple tools could plausibly match a request  

---

# 4. DATASET GENERATION

This is the most important phase. The quality of your training data determines everything.

---

## 4.1 Dataset Structure

Each training example is a conversation containing:

- A system prompt listing all tool schemas  
- A user message containing a natural-language request  
- An assistant message with the correct function call(s) as JSON  

---

## 4.2 Example Format (Single Tool Call)

Each example includes:

- System instructions  
- User request  
- Assistant response in function-call format  

---

## 4.3 Example Format (Multi-Tool Call)

Some examples should require **two tool calls** from a single user request — this is a harder task and more realistic.

---

## 4.4 Example Format (No Tool Needed)

~10% of examples should require **no tool call**, teaching the model:

- Not to force tool usage  
- To respond conversationally when appropriate  

---

## 4.5 Dataset Composition

| Category | # Examples | % |
|----------|-----------|--|
| Simple single-tool | 640 | 40% |
| Complex single-tool | 400 | 25% |
| Multi-tool | 240 | 15% |
| Ambiguous | 160 | 10% |
| No-tool | 160 | 10% |
| **Total** | 1600 | 100% |

---

## 4.6 Dataset Splits

| Split | # | % |
|------|--|--|
| Train | 1280 | 80% |
| Validation | 160 | 10% |
| Test | 160 | 10% |

Important: test set must include all categories proportionally.

---

## 4.7 How to Generate the Dataset

### Step 1: Create generation prompt

Use a frontier model (e.g., Mistral Le Chat):

- Provide all tool schemas  
- Specify category (simple/complex/etc.)  
- Specify target tool(s)  
- Generate both user + assistant outputs  
- Produce ~10 examples per run  

---

### Step 2: Generate in batches

For each tool:

- ~40 simple  
- ~25 complex  
- ~15 multi-tool  
- ~10 ambiguous  
- ~10 no-tool  

---

### Step 3: Validate (critical)

- Parse JSON → must succeed  
- Validate schema compliance  
- Check required fields + types  
- Manually review ~10%  

---

### Step 4: Format & split

- Save `train.jsonl`, `val.jsonl`, `test.jsonl`  

Estimated time: **4–5 hours**

---

# 5. BASELINE EVALUATION (Notebook 2)

Measure **untuned model performance**.

---

## 5.1 Setup

- Load `Mistral-7B-Instruct-v0.3` in 4-bit  
- Run inference on test set  

---

## 5.2 Metrics

- JSON Validity Rate  
- Schema Compliance Rate  
- Function Name Accuracy  
- Argument Accuracy  
- Required Fields Completeness  
- Type Correctness  
- Hallucination Rate  
- No-Tool Restraint Accuracy  

---

## 5.3 Breakdown Dimensions

- By category  
- By tool  
- By schema complexity  

---

# 6. FINE-TUNING (Notebook 3)

## 6.1 Libraries

- transformers  
- peft  
- trl  
- bitsandbytes  
- accelerate  
- datasets  
- wandb  

---

## 6.2 Model Loading

- Load in 4-bit quantisation  
- Use `BitsAndBytesConfig`  

---

## 6.3 LoRA Configuration

- Rank (`r`): 16  
- Alpha: 32  
- Dropout: 0.05  
- Target modules: attention + MLP layers  

---

## 6.4 Data Formatting

Use `SFTTrainer` with chat template.

---

## 6.5 Training Configuration

- Epochs: 3  
- Batch size: 4  
- Gradient accumulation: 4  
- Learning rate: 2e-4  
- Sequence length: 2048  

---

## 6.6 Train

- Train model  
- Save LoRA adapter  

---

## 6.7 Important Notes

- Save outputs frequently (Kaggle timeouts)  
- Adjust batch size if OOM  
- Training time: ~2–3 hours  
- Use W&B for tracking  

---

## 6.8 Hyperparameter Experiments

Run 2–3 experiments:

| Experiment | Change | Purpose |
|-----------|------|--------|
| Baseline | r=16 | Starting point |
| Lower rank | r=8 | Efficiency test |
| More epochs | 5 epochs | Check overfitting |

---

# 7. EVALUATION (Notebook 4)

## 7.1 Load Fine-Tuned Model

- Load base model  
- Apply LoRA adapter  

---

## 7.2 Run Inference

- Run on full test set  
- Compare vs ground truth  

---

## 7.3 Metrics Implementation

Includes functions for:

- JSON validity  
- Function accuracy  
- Field completeness  
- Type correctness  
- Hallucination detection  
- Enum validation  

---

## 7.4 Results Tables

### Table 1: Overall Comparison

- JSON validity  
- Function accuracy  
- Field completeness  
- Type correctness  
- Hallucination  
- No-tool performance  

---

### Table 2: Breakdown by Category

- Simple / complex / multi-tool / ambiguous / no-tool  

---

### Table 3: Breakdown by Tool

- 16 rows  

---

### Table 4: Hyperparameter Comparison

- Compare experiments  

---

## 7.5 Qualitative Examples

Include 10 examples:

- 3 major improvements  
- 3 refinements  
- 2 multi-tool  
- 1 failure  
- 1 no-tool  

---

# 8. REPOSITORY STRUCTURE

```
mistral-7b-enterprise-function-calling/
├── notebooks/
├── data/
├── src/
├── results/
├── README.md
├── requirements.txt
```

---

# 9. TIMELINE

## Weekend 1 (~8–9 hours)

- Setup environment  
- Define schemas  
- Generate dataset  
- Validate  
- Run baseline  

---

## Weekend 2 (~10–12 hours)

- Train models  
- Run experiments  
- Evaluate  

---

## Weekend 3 (~4–5 hours)

- Write README  
- Upload model  
- Clean repo  

---

# Total: ~22–26 hours

---

# 10. README STRUCTURE

- Motivation  
- Approach  
- Tool Schema Design  
- Dataset  
- Training Setup  
- Results  
- Findings  
- Limitations  
- Reproducibility  

---

# 11. CV BULLET POINT

> Fine-tuned Mistral 7B (QLoRA) for enterprise function calling across 16 tools and 1,600 examples, evaluated across multiple metrics and published on GitHub.

---

# 12. INTERVIEW TALKING POINTS

## Why fine-tuning vs prompting

The baseline model already had the full system prompt with all 16 tool schemas, detailed parameter descriptions, and type/enum constraints. Despite this, JSON validity was only 87.1% and argument match 76.8%. The model understood the task but couldn't reliably execute it. Prompting gives the model the information; fine-tuning teaches it the behaviour. The structural improvement from 87.1% to 97.8% JSON validity shows the model needed to internalise the output format at the weight level, not just follow instructions. In production, a 12.9% parse failure rate is unacceptable because you can't retry indefinitely without degrading latency and user experience.

## Function calling as bottleneck

In agentic pipelines, function calling is the interface between reasoning and action. If the model reasons correctly but produces malformed JSON, or picks the right tool but omits a required field, the entire downstream chain breaks. There's no graceful degradation: either the JSON parses and the arguments are valid, or the system fails. This makes function calling a binary reliability problem rather than a quality spectrum. A model that's 90% correct at function calling fails on average once every 10 actions, which in a multi-step agent means most workflows break before completion.

## What I'd improve

- **More data for weak tools**: `create_invoice` (nested arrays) and `run_database_query` (free-form SQL in JSON) still underperform. Targeted data augmentation for these schemas would likely close the gap.
- **Multi-turn conversations**: Current training is single-turn (user asks, model responds). Real agents involve follow-up clarifications, tool results fed back, and sequential tool calls.
- **Full-context training**: The tool trimming strategy was necessary for T4 hardware but means the model never sees all 16 tools simultaneously during training. With more VRAM (A100/H100), training on the full prompt would likely improve disambiguation.
- **Larger base model**: Mistral Small or Medium would start from a higher baseline and likely achieve near-perfect structured output with the same fine-tuning approach.
- **Evaluation on unseen tools**: The current test set uses the same 16 schemas. Testing generalisation to novel tool schemas would measure whether the model learned "how to function call" versus memorising these specific schemas.

## Key learnings

- **Tool trimming as a training optimisation**: Reducing the system prompt from all 16 tools to only the relevant ones (+ 1 distractor) cut mean sequence length from ~5500 to ~915 tokens. This made training feasible within a 12h Kaggle session and is a generally applicable technique for any function-calling fine-tune where the tool registry is large.
- **Pipeline parallelism on consumer GPUs**: Using `device_map="auto"` to split the model across two T4s via pipeline parallelism worked where DataParallel failed (bitsandbytes incompatibility with DP). This is the correct pattern for multi-GPU QLoRA on consumer hardware.
- **Assistant-only loss matters**: Masking loss on system/user tokens focuses the model on learning the output format rather than memorising the prompt. Without this, the model would spend capacity reconstructing the tool schemas.
- **JSON validity vs argument precision are different problems**: Fine-tuning nearly eliminated JSON parse failures (12.9% to 2.2%) but argument match improved more modestly (76.8% to 83.7%). Structural output is learnable quickly; exact value extraction from natural language is a harder generalisation problem that likely needs more diverse training data.
- **Synthetic data quality over quantity**: 1,228 training examples was sufficient for strong results. The key was rigorous validation (schema compliance, type checking, deduplication) rather than volume. Bad examples teach bad habits.

---

# 13. README

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
