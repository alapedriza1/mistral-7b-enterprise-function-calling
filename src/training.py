"""QLoRA fine-tuning utilities for Mistral 7B function calling."""

import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.inference import MODEL_NAME, BNB_CONFIG


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
    "logging_steps": 10,
    "eval_strategy": "epoch",
    "eval_accumulation_steps": 1,
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": False,
    "report_to": "none",
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
}

MAX_SEQ_LENGTH = 4096
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


# ─── Model Loading ───────────────────────────────────────────────────────────


def load_model_for_training(model_name: str = MODEL_NAME):
    """Load the base model in 4-bit and prepare it for QLoRA training.

    Returns:
        Tuple of (model, tokenizer) ready for LoRA adapter attachment.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Right padding for training
    tokenizer.truncation_side = "left"  # Truncate beginning of system prompt, keep assistant response
    tokenizer.chat_template = TRAINING_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    model = prepare_model_for_kbit_training(model)

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

    Merges system messages into user messages because the v0.3 chat template
    only injects system content when user is the last message (inference mode),
    not when assistant follows (training mode).

    Args:
        examples: List of dicts with 'messages' key.

    Returns:
        A HuggingFace Dataset with 'messages' column.
    """
    return Dataset.from_list([
        {"messages": merge_system_into_user(ex["messages"])}
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

    Merges system into user before formatting (same as training pipeline),
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
        merged = merge_system_into_user(ex["messages"])
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
    train_dataset = _prepare_dataset(train_data)
    val_dataset = _prepare_dataset(val_data)

    trainer = _create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        max_seq_length=max_seq_length,
    )

    print("Starting training...")
    train_result = trainer.train()
    trainer.push_to_hub()

    return trainer, train_result
