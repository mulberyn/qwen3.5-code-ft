#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from codebleu import calc_codebleu
from datasets import load_dataset
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, messages):
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


def decode_new_tokens(model, tokenizer, prompt_text, max_new_tokens, temperature, top_p):
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    prompt_tokens = int(model_inputs["input_ids"].shape[1])
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **gen_kwargs)
    elapsed_sec = time.perf_counter() - t0
    new_tokens = output_ids[0][model_inputs["input_ids"].shape[1] :]
    generated_tokens = int(new_tokens.shape[0])
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, generated_tokens, elapsed_sec, prompt_tokens


def summarize_speed(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {
            "sample_count": 0,
            "avg_prompt_tokens": 0.0,
            "avg_generated_tokens": 0.0,
            "avg_latency_sec": 0.0,
            "total_generated_tokens": 0,
            "total_generation_time_sec": 0.0,
            "overall_tokens_per_sec": 0.0,
            "avg_tokens_per_sec": 0.0,
        }

    total_prompt = sum(x["prompt_tokens"] for x in items)
    total_generated = sum(x["generated_tokens"] for x in items)
    total_time = sum(x["generation_time_sec"] for x in items)
    avg_tps = sum(x["tokens_per_sec"] for x in items) / len(items)
    overall_tps = (total_generated / total_time) if total_time > 0 else 0.0

    return {
        "sample_count": len(items),
        "avg_prompt_tokens": total_prompt / len(items),
        "avg_generated_tokens": total_generated / len(items),
        "avg_latency_sec": total_time / len(items),
        "total_generated_tokens": int(total_generated),
        "total_generation_time_sec": total_time,
        "overall_tokens_per_sec": overall_tps,
        "avg_tokens_per_sec": avg_tps,
    }


def strip_markdown_fences(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()
    return text.strip()


def trim_humaneval_completion(prompt: str, raw: str) -> str:
    text = strip_markdown_fences(raw)
    if text.startswith(prompt):
        text = text[len(prompt) :]

    stop_markers = ["\nclass ", "\ndef ", "\nif __name__", "\n# Example", "\nprint("]
    cut_idx = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            cut_idx = min(cut_idx, idx)
    text = text[:cut_idx]

    return text.rstrip() + "\n"


def make_model(model_dir: str, dtype: str, adapter_dir: str = ""):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype_map[dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def run_humaneval(model, tokenizer, args, out_dir: Path) -> Dict[str, Any]:
    problems = read_problems()
    task_ids = sorted(problems.keys())[: args.humaneval_n]

    samples: List[Dict[str, str]] = []
    generations: List[Dict[str, str]] = []
    speed_records: List[Dict[str, float]] = []

    for i, task_id in enumerate(task_ids, start=1):
        p = problems[task_id]
        messages = [
            {"role": "system", "content": "You are a careful Python coding assistant."},
            {
                "role": "user",
                "content": (
                    "Complete the following Python function. Return only code continuation, "
                    "no markdown fences.\n\n"
                    + p["prompt"]
                ),
            },
        ]
        prompt_text = build_prompt(tokenizer, messages)
        raw, gen_tokens, gen_time, prompt_tokens = decode_new_tokens(
            model,
            tokenizer,
            prompt_text,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        completion = trim_humaneval_completion(p["prompt"], raw)
        tps = (gen_tokens / gen_time) if gen_time > 0 else 0.0

        samples.append({"task_id": task_id, "completion": completion})
        generations.append(
            {
                "task_id": task_id,
                "prompt": p["prompt"],
                "canonical_solution": p["canonical_solution"],
                "raw_model_output": raw,
                "completion": completion,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "generation_time_sec": gen_time,
                "tokens_per_sec": tps,
            }
        )
        speed_records.append(
            {
                "prompt_tokens": float(prompt_tokens),
                "generated_tokens": float(gen_tokens),
                "generation_time_sec": float(gen_time),
                "tokens_per_sec": float(tps),
            }
        )

        if i % 5 == 0 or i == len(task_ids):
            print(f"[HumanEval] generated {i}/{len(task_ids)}")

    samples_path = out_dir / "humaneval_samples.jsonl"
    generations_path = out_dir / "humaneval_generations.jsonl"
    write_jsonl(str(samples_path), samples)
    write_jsonl(str(generations_path), generations)

    os.environ["HUMAN_EVAL_ALLOW_CODE_EVAL"] = "1"
    he_result = evaluate_functional_correctness(
        str(samples_path),
        k=[1],
        n_workers=args.eval_workers,
        timeout=args.exec_timeout,
        ignore_incomplete=True,
    )

    refs = [g["canonical_solution"] for g in generations]
    preds = [g["completion"] for g in generations]
    codebleu_result = calc_codebleu(
        references=refs,
        predictions=preds,
        lang="python",
    )

    return {
        "task_count": len(task_ids),
        "samples_path": str(samples_path),
        "generations_path": str(generations_path),
        "pass_at_1": float(he_result.get("pass@1", 0.0)),
        "codebleu": {k: float(v) for k, v in codebleu_result.items()},
        "generation_speed": summarize_speed(speed_records),
    }


def normalize_out(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def run_python_program(code: str, stdin_data: str, timeout_sec: float) -> Tuple[bool, str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            input=stdin_data,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        ok = proc.returncode == 0
        return ok, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return False, exc.stdout or "", "TIMEOUT"


def run_livecodebench_lite(model, tokenizer, args, out_dir: Path) -> Dict[str, Any]:
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        trust_remote_code=True,
        streaming=True,
    )

    records: List[Dict[str, Any]] = []
    problem_pass = 0
    compile_ok = 0
    total_tests = 0
    passed_tests = 0
    speed_records: List[Dict[str, float]] = []

    for idx, ex in enumerate(ds, start=1):
        if idx > args.lcb_n:
            break

        prompt = textwrap.dedent(
            f"""
            Solve the following programming problem. Write a complete Python 3 program.
            Requirements:
            - Read input from stdin
            - Write output to stdout
            - Return code only, no markdown fences

            Problem Title: {ex['question_title']}
            Problem Statement:
            {ex['question_content']}
            """
        ).strip()

        messages = [
            {"role": "system", "content": "You are an expert competitive programming assistant."},
            {"role": "user", "content": prompt},
        ]
        raw, gen_tokens, gen_time, prompt_tokens = decode_new_tokens(
            model,
            tokenizer,
            build_prompt(tokenizer, messages),
            args.lcb_max_new_tokens,
            args.temperature,
            args.top_p,
        )
        code = strip_markdown_fences(raw)
        tps = (gen_tokens / gen_time) if gen_time > 0 else 0.0

        try:
            tests = json.loads(ex["public_test_cases"])
        except Exception:
            tests = []

        all_pass = True
        any_compile_ok = False
        per_test = []

        for t in tests:
            if t.get("testtype") != "stdin":
                continue
            inp = t.get("input", "")
            expected = normalize_out(t.get("output", ""))

            ok, stdout, stderr = run_python_program(code, inp, args.exec_timeout)
            got = normalize_out(stdout)
            passed = ok and (got == expected)
            any_compile_ok = any_compile_ok or ok

            total_tests += 1
            if passed:
                passed_tests += 1
            else:
                all_pass = False

            per_test.append(
                {
                    "ok": ok,
                    "passed": passed,
                    "expected": expected,
                    "got": got,
                    "stderr": stderr[:500],
                }
            )

        if any_compile_ok:
            compile_ok += 1
        if all_pass and per_test:
            problem_pass += 1

        records.append(
            {
                "idx": idx,
                "question_id": ex.get("question_id", ""),
                "difficulty": ex.get("difficulty", ""),
                "question_title": ex.get("question_title", ""),
                "raw_model_output": raw,
                "code": code,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": gen_tokens,
                "generation_time_sec": gen_time,
                "tokens_per_sec": tps,
                "tests": per_test,
                "all_public_tests_passed": all_pass and bool(per_test),
            }
        )
        speed_records.append(
            {
                "prompt_tokens": float(prompt_tokens),
                "generated_tokens": float(gen_tokens),
                "generation_time_sec": float(gen_time),
                "tokens_per_sec": float(tps),
            }
        )

        if idx % 5 == 0 or idx == args.lcb_n:
            print(f"[LCB-lite] evaluated {idx}/{args.lcb_n}")

    pred_path = out_dir / "livecodebench_lite_predictions.jsonl"
    write_jsonl(str(pred_path), records)

    return {
        "problem_count": len(records),
        "predictions_path": str(pred_path),
        "public_problem_pass_rate": (problem_pass / len(records)) if records else 0.0,
        "public_test_pass_rate": (passed_tests / total_tests) if total_tests else 0.0,
        "compile_success_rate": (compile_ok / len(records)) if records else 0.0,
        "total_public_tests": total_tests,
        "generation_speed": summarize_speed(speed_records),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate local Qwen model on coding tasks")
    p.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen3.5-4B")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--lcb-max-new-tokens", type=int, default=768)
    p.add_argument("--humaneval-n", type=int, default=20)
    p.add_argument("--lcb-n", type=int, default=20)
    p.add_argument("--eval-workers", type=int, default=4)
    p.add_argument("--exec-timeout", type=float, default=3.0)
    p.add_argument("--output-root", default="eval/results")
    p.add_argument("--adapter-dir", default="", help="Optional LoRA adapter directory")
    return p.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and tokenizer...")
    model, tokenizer = make_model(args.model_dir, args.dtype, args.adapter_dir)

    print("Running HumanEval + CodeBLEU...")
    he = run_humaneval(model, tokenizer, args, out_dir)

    print("Running LiveCodeBench-lite (public tests)...")
    lcb = run_livecodebench_lite(model, tokenizer, args, out_dir)

    metrics = {
        "timestamp": ts,
        "model_dir": args.model_dir,
        "adapter_dir": args.adapter_dir,
        "config": {
            "dtype": args.dtype,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "lcb_max_new_tokens": args.lcb_max_new_tokens,
            "humaneval_n": args.humaneval_n,
            "lcb_n": args.lcb_n,
            "eval_workers": args.eval_workers,
            "exec_timeout": args.exec_timeout,
        },
        "humaneval": he,
        "livecodebench_lite": lcb,
        "notes": [
            "LiveCodeBench-lite metric here uses public tests only, not the official hidden-test leaderboard metric.",
            "HumanEval pass@1 requires executing generated code and can be sensitive to decoding parameters.",
        ],
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
