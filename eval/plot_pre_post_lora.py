#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Plot pre/post LoRA comparison charts")
    parser.add_argument("--pre", default="eval/results/20260313_182948/metrics.json")
    parser.add_argument("--post", default="eval/results/20260313_193331/metrics.json")
    parser.add_argument("--output-dir", default="eval/results/figures")
    args = parser.parse_args()

    pre = load_metrics(Path(args.pre))
    post = load_metrics(Path(args.post))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Pre-LoRA", "Post-LoRA"]

    # 1) Core pass-rate metrics
    core_names = [
        "HumanEval pass@1",
        "LCB problem pass",
        "LCB test pass",
        "LCB compile",
        "CodeBLEU",
    ]
    pre_core = [
        pre["humaneval"]["pass_at_1"],
        pre["livecodebench_lite"]["public_problem_pass_rate"],
        pre["livecodebench_lite"]["public_test_pass_rate"],
        pre["livecodebench_lite"]["compile_success_rate"],
        pre["humaneval"]["codebleu"]["codebleu"],
    ]
    post_core = [
        post["humaneval"]["pass_at_1"],
        post["livecodebench_lite"]["public_problem_pass_rate"],
        post["livecodebench_lite"]["public_test_pass_rate"],
        post["livecodebench_lite"]["compile_success_rate"],
        post["humaneval"]["codebleu"]["codebleu"],
    ]

    x = list(range(len(core_names)))
    width = 0.36
    plt.figure(figsize=(11, 5))
    plt.bar([i - width / 2 for i in x], pre_core, width=width, label=labels[0])
    plt.bar([i + width / 2 for i in x], post_core, width=width, label=labels[1])
    plt.xticks(x, core_names, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Pre vs Post LoRA: Core Metrics")
    plt.legend()
    plt.tight_layout()
    core_png = out_dir / "pre_post_core_metrics.png"
    plt.savefig(core_png, dpi=160)
    plt.close()

    # 2) CodeBLEU breakdown
    cb_names = ["codebleu", "ngram_match_score", "weighted_ngram_match_score", "syntax_match_score", "dataflow_match_score"]
    pre_cb = [pre["humaneval"]["codebleu"][k] for k in cb_names]
    post_cb = [post["humaneval"]["codebleu"][k] for k in cb_names]

    x = list(range(len(cb_names)))
    plt.figure(figsize=(11, 5))
    plt.bar([i - width / 2 for i in x], pre_cb, width=width, label=labels[0])
    plt.bar([i + width / 2 for i in x], post_cb, width=width, label=labels[1])
    plt.xticks(x, ["CodeBLEU", "n-gram", "weighted n-gram", "syntax", "dataflow"], rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Pre vs Post LoRA: CodeBLEU Breakdown")
    plt.legend()
    plt.tight_layout()
    cb_png = out_dir / "pre_post_codebleu_breakdown.png"
    plt.savefig(cb_png, dpi=160)
    plt.close()

    # 3) Speed comparison
    speed_names = [
        "HE tok/s",
        "LCB tok/s",
        "HE latency(s)",
        "LCB latency(s)",
    ]
    pre_speed = [
        pre["humaneval"]["generation_speed"]["overall_tokens_per_sec"],
        pre["livecodebench_lite"]["generation_speed"]["overall_tokens_per_sec"],
        pre["humaneval"]["generation_speed"]["avg_latency_sec"],
        pre["livecodebench_lite"]["generation_speed"]["avg_latency_sec"],
    ]
    post_speed = [
        post["humaneval"]["generation_speed"]["overall_tokens_per_sec"],
        post["livecodebench_lite"]["generation_speed"]["overall_tokens_per_sec"],
        post["humaneval"]["generation_speed"]["avg_latency_sec"],
        post["livecodebench_lite"]["generation_speed"]["avg_latency_sec"],
    ]

    x = list(range(len(speed_names)))
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], pre_speed, width=width, label=labels[0])
    plt.bar([i + width / 2 for i in x], post_speed, width=width, label=labels[1])
    plt.xticks(x, speed_names)
    plt.ylabel("Value")
    plt.title("Pre vs Post LoRA: Speed Metrics")
    plt.legend()
    plt.tight_layout()
    speed_png = out_dir / "pre_post_speed_metrics.png"
    plt.savefig(speed_png, dpi=160)
    plt.close()

    summary = {
        "pre": str(Path(args.pre)),
        "post": str(Path(args.post)),
        "figures": {
            "core_metrics": str(core_png),
            "codebleu_breakdown": str(cb_png),
            "speed_metrics": str(speed_png),
        },
    }
    (out_dir / "pre_post_figures_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
