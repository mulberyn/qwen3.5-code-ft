#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


FIXED_QUESTIONS = [
    {
        "id": "Q1",
        "prompt": "请写 Python 函数 reverse_words(s)，把句子中单词顺序反转（单词内部不反转），并说明时间复杂度和空间复杂度。",
        "func": "reverse_words",
        "tests": [
            (("hello world",), "world hello"),
            (("a b c",), "c b a"),
            (("single",), "single"),
        ],
    },
    {
        "id": "Q2",
        "prompt": "请写 Python 函数 is_palindrome(s)，忽略大小写与非字母数字字符，判断是否回文，并说明复杂度。",
        "func": "is_palindrome",
        "tests": [
            (("A man, a plan, a canal: Panama",), True),
            (("race a car",), False),
            (("",), True),
        ],
    },
    {
        "id": "Q3",
        "prompt": "请写 Python 函数 two_sum(nums, target)，返回和为 target 的两个下标，并说明复杂度。",
        "func": "two_sum",
        "tests": [
            ((([2, 7, 11, 15], 9),), [0, 1]),
            ((([3, 2, 4], 6),), [1, 2]),
            ((([3, 3], 6),), [0, 1]),
        ],
        "normalize": "sorted_list",
    },
    {
        "id": "Q4",
        "prompt": "请写 Python 函数 fibonacci(n)，返回第 n 个斐波那契数（n>=0），并说明复杂度。",
        "func": "fibonacci",
        "tests": [
            ((0,), 0),
            ((1,), 1),
            ((10,), 55),
        ],
    },
    {
        "id": "Q5",
        "prompt": "请写 Python 函数 merge_intervals(intervals)，合并重叠区间，并说明复杂度。",
        "func": "merge_intervals",
        "tests": [
            ((([[1, 3], [2, 6], [8, 10], [15, 18]],),), [[1, 6], [8, 10], [15, 18]]),
            ((([[1, 4], [4, 5]],),), [[1, 5]]),
        ],
    },
    {
        "id": "Q6",
        "prompt": "请写 Python 函数 top_k_frequent(nums, k)，返回出现频率最高的 k 个元素，并说明复杂度。",
        "func": "top_k_frequent",
        "tests": [
            ((([1, 1, 1, 2, 2, 3], 2),), [1, 2]),
            ((([1], 1),), [1]),
        ],
        "normalize": "sorted_list",
    },
    {
        "id": "Q7",
        "prompt": "请写 Python 函数 group_anagrams(strs)，把字母异位词分组，并说明复杂度。",
        "func": "group_anagrams",
        "tests": [
            (((["eat", "tea", "tan", "ate", "nat", "bat"],),), [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]),
        ],
        "normalize": "sorted_groups",
    },
    {
        "id": "Q8",
        "prompt": "请写 Python 函数 valid_parentheses(s)，判断括号字符串是否有效，并说明复杂度。",
        "func": "valid_parentheses",
        "tests": [
            (("()[]{}",), True),
            (("(]",), False),
            (("([{}])",), True),
        ],
    },
    {
        "id": "Q9",
        "prompt": "请写 Python 函数 binary_search(nums, target)，返回目标下标，不存在返回 -1，并说明复杂度。",
        "func": "binary_search",
        "tests": [
            ((([-1, 0, 3, 5, 9, 12], 9),), 4),
            ((([-1, 0, 3, 5, 9, 12], 2),), -1),
        ],
    },
    {
        "id": "Q10",
        "prompt": "请写 Python 函数 longest_common_prefix(strs)，返回字符串数组的最长公共前缀，并说明复杂度。",
        "func": "longest_common_prefix",
        "tests": [
            (((["flower", "flow", "flight"],),), "fl"),
            (((["dog", "racecar", "car"],),), ""),
        ],
    },
]


def build_prompt(tokenizer, user_text: str) -> str:
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


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m[0].strip()
    return text.strip()


def normalize_value(v: Any, mode: str) -> Any:
    if mode == "sorted_list" and isinstance(v, list):
        return sorted(v)
    if mode == "sorted_groups" and isinstance(v, list):
        groups = []
        for g in v:
            if isinstance(g, list):
                groups.append(sorted(g))
            else:
                groups.append(g)
        return sorted(groups)
    return v


def run_tests(code: str, func_name: str, tests: List[Tuple[Any, Any]], normalize: str = ""):
    ns: Dict[str, Any] = {}
    try:
        exec(code, ns)
    except Exception as e:
        return False, 0, len(tests), f"exec_error: {type(e).__name__}: {e}"

    fn = ns.get(func_name)
    if not callable(fn):
        return False, 0, len(tests), f"function_not_found: {func_name}"

    passed = 0
    err = ""
    for t in tests:
        try:
            args, expected = t
            if len(args) == 1 and isinstance(args[0], tuple):
                # packed args compatibility
                got = fn(*args[0])
            else:
                got = fn(*args)
            got_n = normalize_value(got, normalize)
            exp_n = normalize_value(expected, normalize)
            if got_n == exp_n:
                passed += 1
        except Exception as e:
            err = f"runtime_error: {type(e).__name__}: {e}"

    return True, passed, len(tests), err


def score_completeness(answer: str, func_name: str) -> float:
    s = 0.0
    if f"def {func_name}" in answer:
        s += 2.0
    if "O(" in answer or "复杂度" in answer:
        s += 1.0
    if "示例" in answer or "example" in answer.lower() or "print(" in answer:
        s += 1.0
    if '"""' in answer or "参数" in answer or "返回" in answer:
        s += 1.0
    return min(5.0, s)


def score_explanation(answer: str) -> float:
    low = answer.lower()
    s = 0.0
    if "时间复杂度" in answer or "time complexity" in low or "o(" in low:
        s += 2.0
    if "空间复杂度" in answer or "space complexity" in low:
        s += 1.5
    if any(k in answer for k in ["思路", "步骤", "解释", "原理"]):
        s += 1.0
    if any(k in answer for k in ["1.", "2.", "- ", "示例"]):
        s += 0.5
    return min(5.0, s)


def run_model(model, tokenizer, questions, max_new_tokens):
    rows = []
    for q in questions:
        prompt = build_prompt(tokenizer, q["prompt"])
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        latency = time.perf_counter() - t0
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        rows.append(
            {
                "id": q["id"],
                "prompt": q["prompt"],
                "answer": answer,
                "latency_sec": latency,
                "generated_tokens": int(new_tokens.shape[0]),
            }
        )
    return rows


def evaluate(rows: List[Dict[str, Any]], questions: List[Dict[str, Any]]):
    q_map = {q["id"]: q for q in questions}
    out = []
    for r in rows:
        q = q_map[r["id"]]
        code = extract_code(r["answer"])
        runnable, passed, total, err = run_tests(
            code,
            q["func"],
            q["tests"],
            q.get("normalize", ""),
        )
        correctness = (5.0 * passed / total) if total else 0.0
        runnability = 0.0
        if runnable and total > 0 and passed == total:
            runnability = 5.0
        elif runnable:
            runnability = 3.0
        elif code:
            runnability = 1.0
        completeness = score_completeness(r["answer"], q["func"])
        explanation = score_explanation(r["answer"])

        out.append(
            {
                "id": r["id"],
                "correctness": round(correctness, 3),
                "completeness": round(completeness, 3),
                "runnability": round(runnability, 3),
                "explanation": round(explanation, 3),
                "tests_passed": passed,
                "tests_total": total,
                "error": err,
                "latency_sec": round(float(r["latency_sec"]), 3),
                "generated_tokens": int(r["generated_tokens"]),
            }
        )
    return out


def avg_metric(items: List[Dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return round(sum(float(x.get(key, 0.0)) for x in items) / len(items), 3)


def write_markdown_table(path: Path, base_eval, lora_eval):
    bm = {x["id"]: x for x in base_eval}
    lm = {x["id"]: x for x in lora_eval}
    ids = [q["id"] for q in FIXED_QUESTIONS]

    lines = []
    lines.append("# 固定10题前后对比表")
    lines.append("")
    lines.append("评分范围：每项 0-5（越高越好）")
    lines.append("")
    lines.append("| 题号 | Base正确性 | LoRA正确性 | Base完整性 | LoRA完整性 | Base可运行性 | LoRA可运行性 | Base解释质量 | LoRA解释质量 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for qid in ids:
        b = bm[qid]
        l = lm[qid]
        lines.append(
            f"| {qid} | {b['correctness']} | {l['correctness']} | {b['completeness']} | {l['completeness']} | {b['runnability']} | {l['runnability']} | {b['explanation']} | {l['explanation']} |"
        )

    lines.append("")
    lines.append("## 平均分")
    lines.append("")
    lines.append("| 模型 | 正确性 | 完整性 | 可运行性 | 解释质量 | 平均时延(s) | 平均生成token |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| Base | {avg_metric(base_eval, 'correctness')} | {avg_metric(base_eval, 'completeness')} | {avg_metric(base_eval, 'runnability')} | {avg_metric(base_eval, 'explanation')} | {avg_metric(base_eval, 'latency_sec')} | {avg_metric(base_eval, 'generated_tokens')} |"
    )
    lines.append(
        f"| LoRA | {avg_metric(lora_eval, 'correctness')} | {avg_metric(lora_eval, 'completeness')} | {avg_metric(lora_eval, 'runnability')} | {avg_metric(lora_eval, 'explanation')} | {avg_metric(lora_eval, 'latency_sec')} | {avg_metric(lora_eval, 'generated_tokens')} |"
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_analysis(path: Path, base_eval, lora_eval):
    dims = ["correctness", "completeness", "runnability", "explanation"]
    names = {
        "correctness": "正确性",
        "completeness": "完整性",
        "runnability": "可运行性",
        "explanation": "解释质量",
    }
    bm = {x["id"]: x for x in base_eval}
    lm = {x["id"]: x for x in lora_eval}

    lines = []
    lines.append("# fixed10 质量分析")
    lines.append("")
    lines.append("## 维度均分对比")
    lines.append("")
    lines.append("| 维度 | Base | LoRA | 差值(LoRA-Base) |")
    lines.append("|---|---:|---:|---:|")
    for d in dims:
        b = avg_metric(base_eval, d)
        l = avg_metric(lora_eval, d)
        lines.append(f"| {names[d]} | {b} | {l} | {round(l - b, 3)} |")

    lines.append("")
    lines.append("## 分题胜负统计")
    lines.append("")
    lines.append("| 维度 | LoRA更好 | 持平 | Base更好 |")
    lines.append("|---|---:|---:|---:|")
    for d in dims:
        win = 0
        tie = 0
        lose = 0
        for q in FIXED_QUESTIONS:
            qid = q["id"]
            bd = float(bm[qid][d])
            ld = float(lm[qid][d])
            if ld > bd:
                win += 1
            elif ld < bd:
                lose += 1
            else:
                tie += 1
        lines.append(f"| {names[d]} | {win} | {tie} | {lose} |")

    lines.append("")
    lines.append("## 失败用例摘录")
    lines.append("")
    lines.append("| 题号 | Base通过 | LoRA通过 | Base错误 | LoRA错误 |")
    lines.append("|---|---:|---:|---|---|")
    for q in FIXED_QUESTIONS:
        qid = q["id"]
        b = bm[qid]
        l = lm[qid]
        if b["tests_passed"] < b["tests_total"] or l["tests_passed"] < l["tests_total"]:
            be = (b.get("error") or "").replace("|", "/")[:80]
            le = (l.get("error") or "").replace("|", "/")[:80]
            lines.append(
                f"| {qid} | {b['tests_passed']}/{b['tests_total']} | {l['tests_passed']}/{l['tests_total']} | {be} | {le} |"
            )

    lines.append("")
    lines.append("## 结论")
    lines.append("")
    best_dim = max(dims, key=lambda d: avg_metric(lora_eval, d) - avg_metric(base_eval, d))
    worst_dim = min(dims, key=lambda d: avg_metric(lora_eval, d) - avg_metric(base_eval, d))
    lines.append(
        f"- LoRA 相对 Base 提升最大的维度：{names[best_dim]}（差值 {round(avg_metric(lora_eval, best_dim) - avg_metric(base_eval, best_dim), 3)}）。"
    )
    lines.append(
        f"- LoRA 相对 Base 表现最弱的维度：{names[worst_dim]}（差值 {round(avg_metric(lora_eval, worst_dim) - avg_metric(base_eval, worst_dim), 3)}）。"
    )
    lines.append("- 建议结合失败用例继续定向补数据（函数签名约束、边界样例、解释结构化输出）。")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Run fixed 10-question benchmark for base vs LoRA")
    p.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen3.5-4B")
    p.add_argument(
        "--adapter-dir",
        default="lora/sft_outputs/opencode_lora_20260313_184805/adapter",
    )
    p.add_argument("--max-new-tokens", type=int, default=280)
    p.add_argument("--output-dir", default="lora/compare/fixed10")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base.eval()
    base_rows = run_model(base, tokenizer, FIXED_QUESTIONS, args.max_new_tokens)

    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_for_lora = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_model = PeftModel.from_pretrained(base_for_lora, args.adapter_dir)
    lora_model.eval()
    lora_rows = run_model(lora_model, tokenizer, FIXED_QUESTIONS, args.max_new_tokens)

    base_eval = evaluate(base_rows, FIXED_QUESTIONS)
    lora_eval = evaluate(lora_rows, FIXED_QUESTIONS)

    (out_dir / "base_raw.json").write_text(json.dumps(base_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "lora_raw.json").write_text(json.dumps(lora_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "base_scores.json").write_text(json.dumps(base_eval, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "lora_scores.json").write_text(json.dumps(lora_eval, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "base_correctness",
                "lora_correctness",
                "base_completeness",
                "lora_completeness",
                "base_runnability",
                "lora_runnability",
                "base_explanation",
                "lora_explanation",
            ]
        )
        bm = {x["id"]: x for x in base_eval}
        lm = {x["id"]: x for x in lora_eval}
        for q in FIXED_QUESTIONS:
            qid = q["id"]
            b = bm[qid]
            l = lm[qid]
            w.writerow(
                [
                    qid,
                    b["correctness"],
                    l["correctness"],
                    b["completeness"],
                    l["completeness"],
                    b["runnability"],
                    l["runnability"],
                    b["explanation"],
                    l["explanation"],
                ]
            )

    md_path = out_dir / "comparison_table.md"
    write_markdown_table(md_path, base_eval, lora_eval)
    quality_md_path = out_dir / "quality_analysis.md"
    write_quality_analysis(quality_md_path, base_eval, lora_eval)

    summary = {
        "base_avg": {
            "correctness": avg_metric(base_eval, "correctness"),
            "completeness": avg_metric(base_eval, "completeness"),
            "runnability": avg_metric(base_eval, "runnability"),
            "explanation": avg_metric(base_eval, "explanation"),
        },
        "lora_avg": {
            "correctness": avg_metric(lora_eval, "correctness"),
            "completeness": avg_metric(lora_eval, "completeness"),
            "runnability": avg_metric(lora_eval, "runnability"),
            "explanation": avg_metric(lora_eval, "explanation"),
        },
        "artifacts": {
            "base_raw": str(out_dir / "base_raw.json"),
            "lora_raw": str(out_dir / "lora_raw.json"),
            "base_scores": str(out_dir / "base_scores.json"),
            "lora_scores": str(out_dir / "lora_scores.json"),
            "csv": str(csv_path),
            "markdown": str(md_path),
            "quality_analysis": str(quality_md_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
