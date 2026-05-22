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

- Why fine-tuning vs prompting  
- Function calling as bottleneck  
- What you'd improve  
- Key learnings  

---