#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def set_publication_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def safe_rate(x: float) -> float:
    return float(x) if x is not None else 0.0


def extract_metrics(data: dict) -> dict:
    he = data.get("humaneval", {})
    lcb = data.get("livecodebench_lite", {})
    he_speed = he.get("generation_speed", {})
    lcb_speed = lcb.get("generation_speed", {})

    return {
        "HumanEval pass@1": safe_rate(he.get("pass_at_1", 0.0)),
        "CodeBLEU": safe_rate(he.get("codebleu", {}).get("codebleu", 0.0)),
        "LCB problem pass": safe_rate(lcb.get("public_problem_pass_rate", 0.0)),
        "LCB test pass": safe_rate(lcb.get("public_test_pass_rate", 0.0)),
        "LCB compile": safe_rate(lcb.get("compile_success_rate", 0.0)),
        "HE tok/s": float(he_speed.get("overall_tokens_per_sec", 0.0) or 0.0),
        "LCB tok/s": float(lcb_speed.get("overall_tokens_per_sec", 0.0) or 0.0),
        "HE latency(s)": float(he_speed.get("avg_latency_sec", 0.0) or 0.0),
        "LCB latency(s)": float(lcb_speed.get("avg_latency_sec", 0.0) or 0.0),
        "CB ngram": safe_rate(he.get("codebleu", {}).get("ngram_match_score", 0.0)),
        "CB weighted": safe_rate(he.get("codebleu", {}).get("weighted_ngram_match_score", 0.0)),
        "CB syntax": safe_rate(he.get("codebleu", {}).get("syntax_match_score", 0.0)),
        "CB dataflow": safe_rate(he.get("codebleu", {}).get("dataflow_match_score", 0.0)),
    }


def annotate_bars(ax, bars, fmt="{:.3f}") -> None:
    for b in bars:
        h = b.get_height()
        y = h + (0.01 if h <= 1.0 else max(0.02 * h, 0.15))
        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_figure(fig: plt.Figure, out_stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_quality_comparison(pre: dict, post: dict, labels: tuple[str, str], out_dir: Path) -> None:
    metric_names = [
        "HumanEval pass@1",
        "CodeBLEU",
        "LCB problem pass",
        "LCB test pass",
        "LCB compile",
    ]
    pre_vals = [pre[k] for k in metric_names]
    post_vals = [post[k] for k in metric_names]

    x = np.arange(len(metric_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    bars1 = ax.bar(x - width / 2, pre_vals, width, label=labels[0], color="#4E79A7")
    bars2 = ax.bar(x + width / 2, post_vals, width, label=labels[1], color="#E15759")

    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)

    ymax = min(1.0, max(pre_vals + post_vals) * 1.22 + 0.04)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Score", labelpad=4)
    ax.set_title("(a) Core Coding Metrics Comparison", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=15, ha="right")
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)
    ax.legend(frameon=True, borderpad=0.35, handletextpad=0.5, labelspacing=0.3)
    fig.tight_layout(pad=0.45)

    save_figure(fig, out_dir / "topconf_core_metrics")


def plot_codebleu_breakdown(pre: dict, post: dict, labels: tuple[str, str], out_dir: Path) -> None:
    names = ["CB ngram", "CB weighted", "CB syntax", "CB dataflow"]
    nice_names = ["n-gram", "weighted n-gram", "syntax", "dataflow"]

    pre_vals = np.array([pre[k] for k in names])
    post_vals = np.array([post[k] for k in names])

    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    for i, yi in enumerate(y):
        ax.plot([pre_vals[i], post_vals[i]], [yi, yi], color="#9C755F", linewidth=2, alpha=0.8)

    ax.scatter(pre_vals, y, color="#4E79A7", s=70, label=labels[0], zorder=3)
    ax.scatter(post_vals, y, color="#E15759", s=70, label=labels[1], zorder=3)

    for i, yi in enumerate(y):
        delta = post_vals[i] - pre_vals[i]
        ax.text(
            max(pre_vals[i], post_vals[i]) + 0.01,
            yi,
            f"Δ={delta:+.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0.0, min(1.0, max(float(pre_vals.max()), float(post_vals.max())) + 0.25))
    ax.set_xlabel("Score")
    ax.set_yticks(y)
    ax.set_yticklabels(nice_names)
    ax.set_title("(b) CodeBLEU Component Shifts")
    ax.legend(frameon=True)

    save_figure(fig, out_dir / "topconf_codebleu_components")


def plot_efficiency(pre: dict, post: dict, labels: tuple[str, str], out_dir: Path) -> None:
    metric_names = ["HE tok/s", "LCB tok/s", "HE latency(s)", "LCB latency(s)"]
    pre_vals = np.array([pre[k] for k in metric_names])
    post_vals = np.array([post[k] for k in metric_names])

    x = np.arange(len(metric_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    bars1 = ax.bar(x - width / 2, pre_vals, width, label=labels[0], color="#59A14F")
    bars2 = ax.bar(x + width / 2, post_vals, width, label=labels[1], color="#F28E2B")

    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)

    ax.set_ylabel("Value")
    ax.set_title("(c) Throughput and Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(frameon=True)

    save_figure(fig, out_dir / "topconf_efficiency_metrics")


def plot_tradeoff_scatter(pre: dict, post: dict, labels: tuple[str, str], out_dir: Path) -> None:
    # x: latency (lower is better), y: quality (higher is better), size: throughput
    series = [
        ("HumanEval", "HE latency(s)", "HumanEval pass@1", "HE tok/s"),
        ("LiveCodeBench-lite", "LCB latency(s)", "LCB test pass", "LCB tok/s"),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 6.6))

    for bench, lat_k, q_k, tps_k in series:
        x_pre = pre[lat_k]
        y_pre = pre[q_k]
        s_pre = max(pre[tps_k], 1.0) * 8.0

        x_post = post[lat_k]
        y_post = post[q_k]
        s_post = max(post[tps_k], 1.0) * 8.0

        ax.scatter(x_pre, y_pre, s=s_pre, color="#4E79A7", alpha=0.75, edgecolors="black", linewidths=0.6)
        ax.scatter(x_post, y_post, s=s_post, color="#E15759", alpha=0.75, edgecolors="black", linewidths=0.6)
        ax.annotate(f"{bench} {labels[0]}", (x_pre, y_pre), xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.annotate(f"{bench} {labels[1]}", (x_post, y_post), xytext=(6, -12), textcoords="offset points", fontsize=9)

        ax.arrow(
            x_pre,
            y_pre,
            x_post - x_pre,
            y_post - y_pre,
            length_includes_head=True,
            head_width=0.01,
            head_length=0.08,
            linewidth=1.2,
            color="#9C755F",
            alpha=0.85,
        )

    ax.set_xlabel("Avg Latency (s, lower is better)")
    ax.set_ylabel("Quality Score (higher is better)")
    ax.set_title("(d) Quality-Latency Trade-off (bubble size = tok/s)")
    ax.grid(True, alpha=0.3)

    save_figure(fig, out_dir / "topconf_tradeoff_scatter")


def build_markdown_report(
    pre_dir: Path,
    post_dir: Path,
    pre_m: dict,
    post_m: dict,
    labels: tuple[str, str],
    out_dir: Path,
) -> Path:
    rows = [
        "| Metric | {} | {} | Delta (post-pre) |".format(labels[0], labels[1]),
        "|---|---:|---:|---:|",
    ]

    order = [
        "HumanEval pass@1",
        "CodeBLEU",
        "LCB problem pass",
        "LCB test pass",
        "LCB compile",
        "HE tok/s",
        "LCB tok/s",
        "HE latency(s)",
        "LCB latency(s)",
        "CB ngram",
        "CB weighted",
        "CB syntax",
        "CB dataflow",
    ]

    for k in order:
        a = pre_m[k]
        b = post_m[k]
        rows.append(f"| {k} | {a:.6f} | {b:.6f} | {b - a:+.6f} |")

    md = [
        "# Top-Conference Style Comparison: {} vs {}".format(pre_dir.name, post_dir.name),
        "",
        "- Input run A: `{}`".format(pre_dir.as_posix()),
        "- Input run B: `{}`".format(post_dir.as_posix()),
        "- Label A: `{}`".format(labels[0]),
        "- Label B: `{}`".format(labels[1]),
        "",
        "## Figure Files",
        "",
        "- `topconf_core_metrics.png/.pdf`",
        "- `topconf_codebleu_components.png/.pdf`",
        "- `topconf_efficiency_metrics.png/.pdf`",
        "- `topconf_tradeoff_scatter.png/.pdf`",
        "",
        "## Numeric Comparison",
        "",
        *rows,
        "",
    ]

    out_path = out_dir / "topconf_comparison_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate top-conference style comparison figures for two eval runs"
    )
    parser.add_argument(
        "--run-a",
        default="eval/results/20260313_182948",
        help="First run directory containing metrics.json",
    )
    parser.add_argument(
        "--run-b",
        default="eval/results/20260313_193331",
        help="Second run directory containing metrics.json",
    )
    parser.add_argument("--label-a", default="Base")
    parser.add_argument("--label-b", default="LoRA")
    parser.add_argument("--output-dir", default="eval/results/figures/topconf_20260313_182948_vs_193331")
    args = parser.parse_args()

    set_publication_style()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_a = load_metrics(run_a)
    data_b = load_metrics(run_b)

    label_pair = (args.label_a, args.label_b)
    pre_m = extract_metrics(data_a)
    post_m = extract_metrics(data_b)

    plot_quality_comparison(pre_m, post_m, label_pair, out_dir)
    plot_codebleu_breakdown(pre_m, post_m, label_pair, out_dir)
    plot_efficiency(pre_m, post_m, label_pair, out_dir)
    plot_tradeoff_scatter(pre_m, post_m, label_pair, out_dir)

    report_path = build_markdown_report(run_a, run_b, pre_m, post_m, label_pair, out_dir)

    summary = {
        "run_a": run_a.as_posix(),
        "run_b": run_b.as_posix(),
        "labels": {"a": args.label_a, "b": args.label_b},
        "output_dir": out_dir.as_posix(),
        "figures": {
            "core_metrics": (out_dir / "topconf_core_metrics.png").as_posix(),
            "codebleu_components": (out_dir / "topconf_codebleu_components.png").as_posix(),
            "efficiency_metrics": (out_dir / "topconf_efficiency_metrics.png").as_posix(),
            "tradeoff_scatter": (out_dir / "topconf_tradeoff_scatter.png").as_posix(),
        },
        "report": report_path.as_posix(),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
