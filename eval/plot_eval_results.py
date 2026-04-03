#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def collect_runs(results_root: Path):
    runs = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir():
            continue
        m = d / "metrics.json"
        if not m.exists():
            continue
        data = json.loads(m.read_text(encoding="utf-8"))
        data["_run_dir"] = d.name
        runs.append(data)
    return runs


def get_speed(metric_obj, section):
    speed = metric_obj.get(section, {}).get("generation_speed", {})
    return speed.get("overall_tokens_per_sec", 0.0), speed.get("avg_latency_sec", 0.0)


def main():
    p = argparse.ArgumentParser(description="Plot benchmark metrics across eval runs")
    p.add_argument("--results-root", default="eval/results")
    p.add_argument("--output-dir", default="eval/results/figures")
    args = p.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(results_root)
    if not runs:
        raise SystemExit("No metrics.json found under results root.")

    labels = [r["_run_dir"] for r in runs]
    x = list(range(len(runs)))

    pass1 = [r.get("humaneval", {}).get("pass_at_1", 0.0) for r in runs]
    codebleu = [r.get("humaneval", {}).get("codebleu", {}).get("codebleu", 0.0) for r in runs]
    lcb_prob = [r.get("livecodebench_lite", {}).get("public_problem_pass_rate", 0.0) for r in runs]
    lcb_test = [r.get("livecodebench_lite", {}).get("public_test_pass_rate", 0.0) for r in runs]
    lcb_compile = [r.get("livecodebench_lite", {}).get("compile_success_rate", 0.0) for r in runs]

    he_tps = []
    he_lat = []
    lcb_tps = []
    lcb_lat = []
    for r in runs:
        tps, lat = get_speed(r, "humaneval")
        he_tps.append(tps)
        he_lat.append(lat)
        tps, lat = get_speed(r, "livecodebench_lite")
        lcb_tps.append(tps)
        lcb_lat.append(lat)

    plt.figure(figsize=(12, 5))
    width = 0.16
    plt.bar([i - 2 * width for i in x], pass1, width=width, label="HumanEval pass@1")
    plt.bar([i - width for i in x], codebleu, width=width, label="CodeBLEU")
    plt.bar(x, lcb_prob, width=width, label="LCB problem pass")
    plt.bar([i + width for i in x], lcb_test, width=width, label="LCB test pass")
    plt.bar([i + 2 * width for i in x], lcb_compile, width=width, label="LCB compile")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score / Rate")
    plt.title("Coding Benchmark Scores Across Runs")
    plt.legend()
    plt.tight_layout()
    score_png = output_dir / "scores_across_runs.png"
    plt.savefig(score_png, dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    w = 0.35
    plt.bar([i - w / 2 for i in x], he_tps, width=w, label="HumanEval tok/s")
    plt.bar([i + w / 2 for i in x], lcb_tps, width=w, label="LCB tok/s")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Tokens Per Second")
    plt.title("Generation Throughput Across Runs")
    plt.legend()
    plt.tight_layout()
    tps_png = output_dir / "throughput_across_runs.png"
    plt.savefig(tps_png, dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(x, he_lat, marker="o", label="HumanEval avg latency")
    plt.plot(x, lcb_lat, marker="o", label="LCB avg latency")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Seconds / sample")
    plt.title("Generation Latency Across Runs")
    plt.legend()
    plt.tight_layout()
    lat_png = output_dir / "latency_across_runs.png"
    plt.savefig(lat_png, dpi=160)
    plt.close()

    latest = runs[-1]
    summary = {
        "run_count": len(runs),
        "latest_run": latest["_run_dir"],
        "figures": {
            "scores": str(score_png),
            "throughput": str(tps_png),
            "latency": str(lat_png),
        },
    }
    (output_dir / "figures_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
