"""Model loading and inference utilities for Mistral 7B."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional
from tqdm import tqdm

from src.utils import load_jsonl


# Quantisation config (shared across base + fine-tuned loading) ──
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def load_base_model(model_name: str = MODEL_NAME):
    """Load the base Mistral-7B model quantised with BitsAndBytes and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    return model, tokenizer


def load_finetuned_model(adapter_path: str, model_name: str = MODEL_NAME):
    """Load the base model with a LoRA adapter merged on top."""
    base_model, tokenizer = load_base_model(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Run single-turn inference on the model using a chat message list.

    Applies the chat template, generates a response, and decodes only the
    newly generated tokens (excluding the input prompt).

    Args:
        model: The loaded causal language model.
        tokenizer: The tokenizer matching the model.
        messages: Chat messages in OpenAI format, e.g.
            [{"role": "user", "content": "Hello"}].
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Set to 0 for greedy decoding.

    Returns:
        The model's generated response as a decoded string.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    return response
