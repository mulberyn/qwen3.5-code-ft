# Qwen3.5-4B 编程能力评测报告（HumanEval + CodeBLEU + LiveCodeBench-lite）

- 评测日期: 2026-03-13
- 模型: `/root/autodl-tmp/models/Qwen3.5-4B`
- 目标: 评估当前基础模型在代码生成与问题求解上的基线能力，为后续 OpenCodeInstruct LoRA SFT 提供对照

## 1. 评测范围与说明

本次评测包含三类指标：

1. HumanEval `pass@1`
2. HumanEval 生成结果对应的 CodeBLEU（Python）
3. LiveCodeBench-lite（公开样例）通过率

说明：

- LiveCodeBench 官方排行榜通常基于隐藏测试与官方评测管线；本次使用的是 `livecodebench/code_generation_lite` 的公开样例子集，属于工程可复现近似指标，不等同于官方榜单分数。
- 本次为快速基线，采用子集评测：`HumanEval 20 题 + LiveCodeBench-lite 20 题`。

## 2. 环境信息

- OS: Linux
- Python: 3.10.8
- GPU: NVIDIA GeForce RTX 5090 (32GB)
- 网络: `huggingface.co` 不可达，`hf-mirror.com` 可达

关键依赖（实际安装/使用）：

- `torch`
- `transformers`
- `datasets`
- `human-eval`
- `codebleu`
- `tree-sitter==0.22.3`（由 codebleu 依赖）
- `tree-sitter-python==0.21.0`（与 codebleu 兼容）

## 3. 评测脚本

评测脚本位置：`eval/eval_qwen35_code.py`

功能：

1. 加载本地 Qwen3.5-4B
2. 对 HumanEval 题目生成代码补全
3. 调用 human-eval 执行测试得到 `pass@1`
4. 对 HumanEval 的 `prediction vs canonical_solution` 计算 CodeBLEU
5. 对 LiveCodeBench-lite 流式抽样，执行公开 `stdin` 测试并统计通过率
6. 输出 `metrics.json` 与所有中间产物

## 4. 运行参数（本次实测）

执行命令：

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

参数解释：

- `dtype=bf16`: 减少显存压力并保持较好吞吐
- `temperature=0.0, top_p=1.0`: 贪心解码，保证可重复性
- `max-new-tokens=256`: HumanEval 代码补全长度
- `lcb-max-new-tokens=768`: LCB 题目通常更长，给更大生成预算
- `humaneval-n=20`: HumanEval 子集题数
- `lcb-n=20`: LiveCodeBench-lite 子集题数
- `exec-timeout=4`: 单测试执行超时（秒）

## 5. 执行过程（含问题与修复）

1. 初次运行时，HumanEval 默认要求全量题目，触发断言 `Some problems are not attempted`。

修复：脚本中对 `evaluate_functional_correctness` 增加 `ignore_incomplete=True`，允许子集评测。

2. 二次运行时，CodeBLEU 缺少 python parser，报错 `No module named tree_sitter_python`。

修复：安装 `tree-sitter-python`。

3. 三次运行时，`tree-sitter-python` 新版本与 `codebleu` 组合报错 `TypeError: an integer is required`。

修复：将 `tree-sitter-python` 锁定到 `0.21.0`，与当前 `codebleu` 生态兼容。

4. 四次运行完整成功，生成所有结果文件。

附加工程处理：

- 为避免系统盘写满，将 HF 缓存重定向到数据盘：`HF_HOME=/root/autodl-tmp/hf_cache`。
- LiveCodeBench-lite 使用 `streaming=True`，避免整库下载。

## 6. 最终结果

结果目录（首个完整成功批次）：`eval/results/20260313_181747`

核心指标（来自 `metrics.json`）：

### 6.1 HumanEval

- 题量: `20`
- `pass@1`: `0.10`（即 2/20）

### 6.2 CodeBLEU（基于 HumanEval 20 题）

- `CodeBLEU`: `0.2872`
- `ngram_match_score`: `0.1392`
- `weighted_ngram_match_score`: `0.1975`
- `syntax_match_score`: `0.4410`
- `dataflow_match_score`: `0.3710`

### 6.3 LiveCodeBench-lite（公开样例）

- 题量: `20`
- `public_problem_pass_rate`: `0.00`
- `public_test_pass_rate`: `0.00`
- `compile_success_rate`: `0.20`
- `total_public_tests`: `9`

## 7. 结果产物清单

目录 `eval/results/20260313_181747` 下包含：

- `metrics.json`: 汇总指标与参数
- `humaneval_samples.jsonl`: 供 human-eval 评测的样本
- `humaneval_samples.jsonl_results.jsonl`: HumanEval 执行结果
- `humaneval_generations.jsonl`: 每题 prompt / raw 输出 / completion / canonical
- `livecodebench_lite_predictions.jsonl`: LCB-lite 每题生成代码与测试通过详情

样本数量统计：

- `humaneval_generations.jsonl`: 20 行
- `humaneval_samples.jsonl`: 20 行
- `humaneval_samples.jsonl_results.jsonl`: 20 行
- `livecodebench_lite_predictions.jsonl`: 20 行

## 8. 对 LoRA SFT 的基线解读

在本次参数下，模型表现出以下特征：

1. HumanEval `pass@1=0.10`，说明在基础函数补全任务上已有可用但偏弱的正确率。
2. CodeBLEU 的 `syntax/dataflow` 分项高于 `ngram`，说明模型在结构层面有一定合理性，但与参考答案的实现细节差异较大。
3. LCB-lite 公开样例通过率为 0，且仅 20% 样本可编译运行，表明在竞赛风格端到端程序生成（I/O、边界、完整性）上明显短板。

对你后续 OpenCodeInstruct LoRA SFT 的建议：

- 使用本报告指标作为 pre-SFT baseline。
- SFT 后先复跑同一脚本、同一参数，做严格 A/B 对比。
- 若目标偏竞赛与 Agentic coding，建议提高训练样本中完整程序、I/O 规范、单元测试驱动样本占比。

## 9. 复现实验指南

1. 安装依赖（若缺失）：

```bash
HF_ENDPOINT=https://hf-mirror.com pip install -U \
  evaluate human-eval codebleu tree-sitter-python==0.21.0
```

2. 运行评测：

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

3. 查看结果：

```bash
cat eval/results/<timestamp>/metrics.json

## 11. 速度统计与图表更新（20260313_182948）

最新完整运行目录：`eval/results/20260313_182948`

新增记录的生成速度指标：

- HumanEval overall_tokens_per_sec: `32.6264`
- HumanEval avg_latency_sec: `2.4045`
- LCB-lite overall_tokens_per_sec: `33.2022`
- LCB-lite avg_latency_sec: `3.3567`

所有可用完整评测批次已生成图表：

- `eval/results/figures/scores_across_runs.png`
- `eval/results/figures/throughput_across_runs.png`
- `eval/results/figures/latency_across_runs.png`

说明：`eval/results` 中历史中断目录（无 `metrics.json`）不会纳入图表聚合。
```

## 10. 后续可扩展评测

如果你要作为论文级或长期追踪基线，可进一步扩展：

1. HumanEval 全量 164 题（`--humaneval-n 164`）
2. 多采样评测（`temperature>0`，每题 `n` 次）并统计 `pass@k`
3. 增加 MBPP、EvalPlus、MultiPL-E 等补充基准
4. 接入 LiveCodeBench 官方完整管线（隐藏测试）做更严谨对比
