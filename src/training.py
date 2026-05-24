"""QLoRA fine-tuning utilities for Mistral 7B function calling."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force single GPU — prevents DataParallel on T4x2

import json
import random
import time
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.inference import MODEL_NAME, BNB_CONFIG
from src.schemas import TOOL_SCHEMA_MAP, TOOL_NAMES, build_system_prompt, MAX_SEQ_LENGTH


# ─── Default Hyperparameters ─────────────────────────────────────────────────

DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": TaskType.CAUSAL_LM,
    "bias": "none",
}

DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "optim": "paged_adamw_8bit",
    "logging_steps": 1,
    "eval_strategy": "epoch",
    "eval_accumulation_steps": 1,
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": False,
    "report_to": "none",
    "max_grad_norm": 1.0,
    "gradient_checkpointing": False,
}

N_DISTRACTOR_TOOLS = 1
HF_REPO_ID = "alapedriza/mistral-7b-function-calling-adapter"

# Training-specific chat template with {% generation %} markers.
# Since merge_system_into_user already folds system into user, this template
# only handles user + assistant roles. Produces the same token format as the
# original Mistral v0.3 template: <s>[INST] {user} [/INST] {assistant}</s>
TRAINING_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "[INST] {{ message['content'] }} [/INST]"
    "{% elif message['role'] == 'assistant' %}"
    "{% generation %} {{ message['content'] | trim }}{{ eos_token }}{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
)


# ─── Logging Callback ────────────────────────────────────────────────────────


class PrintProgressCallback(TrainerCallback):
    """Prints training progress to stdout so it appears in Kaggle logs."""

    def __init__(self):
        self.start_time = None
        self.total_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.total_steps = state.max_steps
        print(f"[TRAIN] Starting | {self.total_steps} total steps | "
              f"batch={args.per_device_train_batch_size} | "
              f"grad_accum={args.gradient_accumulation_steps} | "
              f"epochs={args.num_train_epochs}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        elapsed = time.time() - self.start_time
        elapsed_min = elapsed / 60
        step = state.global_step
        loss = logs.get("loss", logs.get("eval_loss", None))
        lr = logs.get("learning_rate", None)

        parts = [f"[TRAIN] Step {step}/{self.total_steps}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        parts.append(f"elapsed={elapsed_min:.1f}min")

        if step > 0:
            sec_per_step = elapsed / step
            remaining = (self.total_steps - step) * sec_per_step
            parts.append(f"eta={remaining/60:.1f}min")

        print(" | ".join(parts), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = (time.time() - self.start_time) / 60
        print(f"[TRAIN] Complete | {state.global_step} steps | {elapsed:.1f} min total",
              flush=True)


# ─── Message Formatting ──────────────────────────────────────────────────────


def merge_system_into_user(messages: list[dict]) -> list[dict]:
    """Merge the system message into the first user message.

    Mistral v0.3's chat template only injects the system message when the
    user message is the LAST in the conversation (i.e. inference mode with
    [system, user]). For training, where messages are [system, user, assistant],
    the user is not loop.last so the system content is silently dropped.

    This function ensures the system prompt (containing tool schemas) is always
    visible to the model during training.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        A new list with only 'user' and 'assistant' messages, where the
        system content is prepended to the first user message.
    """
    if not messages or messages[0]["role"] != "system":
        return messages

    system_content = messages[0]["content"]
    merged = []
    for msg in messages[1:]:
        if msg["role"] == "user" and not merged:
            merged.append({
                "role": "user",
                "content": f"{system_content}\n\n{msg['content']}",
            })
        else:
            merged.append(msg)
    return merged


def _trim_system_tools(messages: list[dict], n_distractors: int = N_DISTRACTOR_TOOLS) -> list[dict]:
    """Trim the system prompt to only include relevant tools + random distractors.

    Replaces the full 16-tool system prompt with a minimal version containing
    only the tools used in the assistant response plus random distractors.

    Args:
        messages: List of message dicts [system, user, assistant].
        n_distractors: Number of random unused tools to include as distractors.

    Returns:
        Messages with a trimmed system prompt containing only relevant tools.
    """
    if not messages or messages[0]["role"] != "system":
        return messages

    assistant_content = messages[2]["content"] if len(messages) > 2 else ""

    # Identify which tools the assistant actually uses
    used_tools = set()
    try:
        response = json.loads(assistant_content)
        if isinstance(response, list):
            for r in response:
                used_tools.add(r["name"])
        elif isinstance(response, dict):
            used_tools.add(response["name"])
    except (json.JSONDecodeError, KeyError, TypeError):
        pass  # Conversational response — no tools used

    # Select: used tools + random distractors
    available_distractors = [t for t in TOOL_NAMES if t not in used_tools]
    n_dist = min(n_distractors, len(available_distractors))
    distractors = random.sample(available_distractors, n_dist)

    # For conversational examples (no tool used), include a few random tools
    # so the model learns when NOT to use tools
    if not used_tools:
        selected_names = random.sample(TOOL_NAMES, min(3, len(TOOL_NAMES)))
    else:
        selected_names = list(used_tools) + distractors

    # Shuffle so the correct tool isn't always first
    random.shuffle(selected_names)

    # Build trimmed system prompt using the shared builder from schemas
    selected_tools = [TOOL_SCHEMA_MAP[name] for name in selected_names]
    trimmed_system = build_system_prompt(selected_tools)

    return [
        {"role": "system", "content": trimmed_system},
        messages[1],  # user
        messages[2],  # assistant
    ]


# ─── Model Loading ───────────────────────────────────────────────────────────


def load_model_for_training(model_name: str = MODEL_NAME):
    """Load the base model in 4-bit and prepare it for QLoRA training.

    Returns:
        Tuple of (model, tokenizer) ready for LoRA adapter attachment.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Right padding for training
    tokenizer.chat_template = TRAINING_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BNB_CONFIG,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    return model, tokenizer


# ─── LoRA Setup ──────────────────────────────────────────────────────────────


def apply_lora(model, lora_config: dict = None) -> object:
    """Attach a LoRA adapter to the model.

    Args:
        model: Base model prepared for k-bit training.
        lora_config: Dict of LoRA hyperparameters. Uses DEFAULT_LORA_CONFIG
            if not provided.

    Returns:
        PeftModel with the LoRA adapter applied.
    """
    config = lora_config or DEFAULT_LORA_CONFIG
    peft_config = LoraConfig(**config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ─── Dataset Preparation ─────────────────────────────────────────────────────


def _prepare_dataset(examples: list[dict]) -> Dataset:
    """Convert our JSONL examples into a HuggingFace Dataset.

    Trims system prompts to only include relevant tools (+ distractors),
    then merges system into user message for the chat template.

    Args:
        examples: List of dicts with 'messages' key.

    Returns:
        A HuggingFace Dataset with 'messages' column.
    """
    return Dataset.from_list([
        {"messages": merge_system_into_user(_trim_system_tools(ex["messages"]))}
        for ex in examples
    ])


# ─── Truncation Check ───────────────────────────────────────────────────────


def check_truncation(
    examples: list[dict],
    tokenizer,
    max_seq_length: int = MAX_SEQ_LENGTH,
    label: str = "dataset",
) -> pd.DataFrame:
    """Check how many examples would be truncated at the given max_seq_length.

    Applies the same preprocessing as training (trim tools, merge system),
    then tokenizes to count tokens.

    Args:
        examples: List of dicts with 'messages' key.
        tokenizer: The model tokenizer (used to apply chat template).
        max_seq_length: The max sequence length that will be used in training.
        label: A label for the dataset (e.g. "train", "val") used in output.

    Returns:
        A single-row DataFrame with columns: split, total, truncated,
        truncated_pct, mean_tokens, p95_tokens, max_tokens, max_over.
    """
    lengths = []
    for ex in examples:
        trimmed = _trim_system_tools(ex["messages"])
        merged = merge_system_into_user(trimmed)
        formatted = tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=False
        )
        token_ids = tokenizer.encode(formatted)
        lengths.append(len(token_ids))

    truncated_lengths = [l for l in lengths if l > max_seq_length]
    sorted_lengths = sorted(lengths)
    p95_idx = int(len(sorted_lengths) * 0.95)
    n_truncated = len(truncated_lengths)
    max_over = max(truncated_lengths) - max_seq_length if n_truncated > 0 else 0

    return pd.DataFrame([{
        "split": label,
        "total": len(lengths),
        "truncated": n_truncated,
        "truncated_pct": round(100 * n_truncated / len(lengths), 1),
        "mean_tokens": round(sum(lengths) / len(lengths)),
        "p95_tokens": sorted_lengths[p95_idx],
        "max_tokens": max(lengths),
        "max_over": max_over,
    }])


# ─── Training ────────────────────────────────────────────────────────────────


def _create_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> SFTTrainer:
    """Create an SFTTrainer configured for QLoRA chat fine-tuning.

    Uses assistant_only_loss to compute loss only on assistant responses.
    Pushes the adapter to HuggingFace Hub at the end of training.

    Args:
        model: PeftModel with LoRA adapters.
        tokenizer: The model tokenizer.
        train_dataset: HuggingFace Dataset for training.
        val_dataset: HuggingFace Dataset for evaluation.
        output_dir: Where to save checkpoints.
        max_seq_length: Maximum sequence length for tokenization.

    Returns:
        A configured SFTTrainer instance.
    """
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=max_seq_length,
        assistant_only_loss=True,
        packing=False,
        push_to_hub=True,
        hub_model_id=HF_REPO_ID,
        **DEFAULT_TRAINING_ARGS,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[PrintProgressCallback()],
    )

    return trainer


def run_training(
    model,
    tokenizer,
    train_data: list[dict],
    val_data: list[dict],
    output_dir: str,
    max_seq_length: int = MAX_SEQ_LENGTH,
):
    """End-to-end training: prepare data, create trainer, train, push to HF Hub.

    Args:
        model: PeftModel with LoRA adapters.
        tokenizer: The model tokenizer.
        train_data: Training examples (list of dicts with 'messages').
        val_data: Validation examples.
        output_dir: Directory for checkpoints during training.
        max_seq_length: Maximum sequence length.

    Returns:
        Tuple of (trainer, train_result) for inspection.
    """
    print("[STAGE] Preparing train dataset...", flush=True)
    train_dataset = _prepare_dataset(train_data)
    print(f"[STAGE] Train dataset ready: {len(train_dataset)} examples", flush=True)

    print("[STAGE] Preparing val dataset...", flush=True)
    val_dataset = _prepare_dataset(val_data)
    print(f"[STAGE] Val dataset ready: {len(val_dataset)} examples", flush=True)

    print("[STAGE] Creating trainer...", flush=True)
    trainer = _create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
    )
    print("[STAGE] Trainer created, calling trainer.train()...", flush=True)

    train_result = trainer.train()

    print("[STAGE] Training done, pushing to hub...", flush=True)
    trainer.push_to_hub()
    print("[STAGE] Push complete.", flush=True)

    return trainer, train_result
