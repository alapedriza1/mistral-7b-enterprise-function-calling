"""Model loading and inference utilities for Mistral 7B."""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional
from tqdm import tqdm

from src.utils import load_jsonl


# Quantisation config (shared across inference + training) ──
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


def load_base_model(model_name: str = MODEL_NAME):
    """Load the base Mistral-7B model quantised with BitsAndBytes and its tokenizer.

    Args:
        model_name: HuggingFace model identifier. Defaults to MODEL_NAME.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer


def load_finetuned_model(adapter_path: str, model_name: str = MODEL_NAME):
    """Load the base model with a LoRA adapter merged on top.

    Args:
        adapter_path: Path or HuggingFace repo ID of the LoRA adapter.
        model_name: HuggingFace model identifier. Defaults to MODEL_NAME.

    Returns:
        Tuple of (model, tokenizer).
    """
    base_model, tokenizer = load_base_model(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Run inference on the model using a chat message list.

    Applies the chat template, generates a response, and decodes only the
    newly generated tokens (excluding the input prompt).

    Args:
        model: The loaded base language model.
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


def run_inference_on_test_set(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_data: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> list[dict]:
    """Run inference on the full test set.

    Each test example has 'messages' = [system, user, assistant].
    We feed only [system, user] and capture the model's response.

    Args:
        model: The loaded base language model.
        tokenizer: The tokenizer matching the model.
        test_data: List of example dicts, each with 'messages' and 'category'.
        max_new_tokens: Maximum number of tokens to generate per example.
        temperature: Sampling temperature. Set to 0 for greedy decoding.

    Returns:
        A list of result dicts, each with 'input_messages', 'expected',
        'predicted', and 'category'.
    """
    results = []

    for example in tqdm(test_data, desc="Running inference"):
        messages = example["messages"]

        # Feed only system + user (first 2 messages)
        input_messages = messages[:2]

        # Ground truth is the assistant message
        expected = messages[2]["content"]

        # Run inference
        predicted = run_inference(
            model, tokenizer, input_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        results.append({
            "input_messages": input_messages,
            "expected": expected,
            "predicted": predicted,
            "category": example.get("category", "unknown"),
        })

    print(f"Inference complete: {len(results)} examples")
    return results
