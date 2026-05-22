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
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": TaskType.CAUSAL_LM,
    "bias": "none",
}

DEFAULT_TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "optim": "paged_adamw_8bit",
    "fp16": True,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "none",
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
}

MAX_SEQ_LENGTH = 5248
HF_REPO_ID = "alapedriza/mistral-7b-function-calling-adapter"


# ─── Model Loading ───────────────────────────────────────────────────────────


def load_model_for_training(model_name: str = MODEL_NAME):
    """Load the base model in 4-bit and prepare it for QLoRA training.

    Returns:
        Tuple of (model, tokenizer) ready for LoRA adapter attachment.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Right padding for training

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

    Each example must have a 'messages' key with the chat messages list.

    Args:
        examples: List of dicts with 'messages' key.

    Returns:
        A HuggingFace Dataset with 'messages' column.
    """
    return Dataset.from_list(
        [{"messages": ex["messages"]} for ex in examples]
    )


# ─── Truncation Check ───────────────────────────────────────────────────────


def check_truncation(
    examples: list[dict],
    tokenizer,
    max_seq_length: int = MAX_SEQ_LENGTH,
    label: str = "dataset",
) -> pd.DataFrame:
    """Check how many examples would be truncated at the given max_seq_length.

    Formats each example with the chat template, then tokenizes to count
    tokens. Returns a single-row DataFrame with summary statistics.

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
        formatted = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
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
