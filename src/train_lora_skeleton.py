"""
LoRA / QLoRA training skeleton.

This file is intentionally a template. Add a real instruction dataset before claiming fine-tuning results.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--output-dir", default="results/lora_adapter")
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    config = LoraConfig(
        r=args.rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    print("\nNext steps:")
    print("1. Load an instruction dataset.")
    print("2. Tokenize prompt/response examples.")
    print("3. Train with transformers.Trainer or TRL SFTTrainer.")
    print("4. Re-run evaluate_mcq.py on the base model and adapter-merged model.")


if __name__ == "__main__":
    main()
