#!/usr/bin/env python3
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, user_text):
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": user_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main():
    p = argparse.ArgumentParser(description="Run base+LoRA adapter for Qwen3.5-4B")
    p.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen3.5-4B")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--prompt", default="Write a Python function to check if a string is palindrome.")
    p.add_argument("--max-new-tokens", type=int, default=192)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    prompt = build_prompt(tokenizer, args.prompt)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
