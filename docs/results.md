# Qwen3.5-4B 评测结果汇总（含速度与图表）

## 1. 最新完整评测结果

- Run 目录: `eval/results/20260313_182948`
- 配置: `bf16, temperature=0.0, top_p=1.0, HumanEval=20, LCB-lite=20`

Temperature： 通过缩放概率分布来全局调整模型的“置信度”。低温 -> 确定性高；高温 -> 随机性高。

Top-p： 通过动态裁剪词汇表来局部限制模型的“视野”。只从概率最高的核心词汇池中选词，避免选到不合适的长尾词。

### 1.1 准确率/通过率

- HumanEval pass@1: `0.10`
- CodeBLEU: `0.2892`
- CodeBLEU ngram: `0.1392`
- CodeBLEU weighted_ngram: `0.1975`
- CodeBLEU syntax: `0.4410`
- CodeBLEU dataflow: `0.3790`
- LCB-lite public_problem_pass_rate: `0.00`
- LCB-lite public_test_pass_rate: `0.00`
- LCB-lite compile_success_rate: `0.20`

### 1.2 生成速度（新增）

HumanEval 生成速度：

- avg_prompt_tokens: `204.35`
- avg_generated_tokens: `78.45`
- avg_latency_sec: `2.4045`
- total_generated_tokens: `1569`
- total_generation_time_sec: `48.0900`
- overall_tokens_per_sec: `32.6264`
- avg_tokens_per_sec: `31.9058`

LiveCodeBench-lite 生成速度：

- avg_prompt_tokens: `538.35`
- avg_generated_tokens: `111.45`
- avg_latency_sec: `3.3567`
- total_generated_tokens: `2229`
- total_generation_time_sec: `67.1342`
- overall_tokens_per_sec: `33.2022`
- avg_tokens_per_sec: `25.0460`

## 2. 已评测数据覆盖范围

`eval/results` 下共有 5 个时间戳目录，其中：

- 有完整 `metrics.json`（用于汇总图表）: `20260313_181747`, `20260313_182948`
- 无完整 `metrics.json`（历史中断运行）: `20260313_174516`, `20260313_181230`, `20260313_181444`

图表是基于“所有可用完整评测结果”自动汇总得到。

## 3. 数据图产物

图表目录：`eval/results/figures`

- 总体分数对比图: `eval/results/figures/scores_across_runs.png`
- 生成吞吐对比图: `eval/results/figures/throughput_across_runs.png`
- 生成时延对比图: `eval/results/figures/latency_across_runs.png`
- 图表摘要: `eval/results/figures/figures_summary.json`

## 4. 使用命令

### 4.1 运行评测（包含速度统计）

```bash
HF_HOME=/root/autodl-tmp/hf_cache \
HF_ENDPOINT=https://hf-mirror.com \
python eval/eval_qwen35_code.py \
	--model-dir /root/autodl-tmp/models/Qwen3.5-4B \
	--dtype bf16 \
	--temperature 0.0 \
	--top-p 1.0 \
	--max-new-tokens 256 \
	--lcb-max-new-tokens 768 \
	--humaneval-n 20 \
	--lcb-n 20 \
	--exec-timeout 4
```

### 4.2 生成图表

```bash
python eval/plot_eval_results.py --results-root eval/results --output-dir eval/results/figures
```

## 5. 微调前后对比（同参数）

- 微调前：`eval/results/20260313_182948`
- 微调后（LoRA adapter）：`eval/results/20260313_193331`
- 详细对比表：`eval/results/pre_post_lora_comparison_20260313.md`

关键结论：

- HumanEval 子集：微调后下降（pass@1: 0.10 -> 0.00）
- LiveCodeBench-lite 子集：微调后上升（public_test_pass_rate: 0.00 -> 0.4444）
- 速度：微调后整体 tokens/s 下降，LCB 平均耗时上升
