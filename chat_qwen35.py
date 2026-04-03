#!/usr/bin/env python3
import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, messages):
    """Build chat prompt; disable thinking if tokenizer supports it."""
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


def generate_reply(model, tokenizer, messages, max_new_tokens, temperature, top_p):
    prompt_text = build_prompt(tokenizer, messages)
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

    new_tokens = output_ids[0][model_inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B local chat")
    parser.add_argument(
        "--model-dir",
        default="/root/autodl-tmp/models/Qwen3.5-4B",
        help="Local model directory",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model dtype",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    print(f"Loading tokenizer from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    print("Loading model, this may take a while...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("\nChat ready. Commands: /exit, /quit, /clear")

    messages = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_text = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_text:
            continue

        if user_text.lower() in {"/exit", "/quit"}:
            print("Bye.")
            break

        if user_text.lower() == "/clear":
            messages = [{"role": "system", "content": args.system}]
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": user_text})

        try:
            answer = generate_reply(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except RuntimeError as e:
            print(f"\n[RuntimeError] {e}")
            print("Tips: reduce --max-new-tokens, use --dtype fp16, or clear history with /clear.")
            messages.pop()
            continue

        print(f"Assistant: {answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
