# qwen3.5-code-ft

一个围绕 Qwen3.5-4B 的代码能力实验项目，包含：

- 本地对话推理（chat）
- 基线评测（HumanEval / CodeBLEU / LiveCodeBench-lite）
- LoRA SFT 训练与前后效果对比

适合做“微调前后 A/B 对比”的小型研究与复现实验。

## 项目结构

```text
.
├── chat_qwen35.py                 # 本地聊天脚本
├── eval/
│   ├── eval_qwen35_code.py        # 代码能力评测主脚本
│   ├── benchmark_report_20260313.md
│   └── results/                   # 历次评测产物与图表
├── lora/
│   ├── sft_opencode_lora.py       # LoRA SFT 训练脚本
│   ├── run_lora_model.py          # LoRA 推理脚本
│   ├── compare/                   # 微调前后对比结果
│   └── sft_outputs/               # LoRA 训练产物
└── docs/
    ├── sft.md
    ├── tutorial.md
    └── results.md
```

## 环境依赖

推荐 Python 3.10+，并安装常用依赖：

```bash
pip install -U torch transformers datasets peft human-eval codebleu tree-sitter-python==0.21.0
```

如果需要镜像源，可参考：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始

### 1) 本地聊天

```bash
python chat_qwen35.py \
  --model-dir /path/to/Qwen3.5-4B \
  --dtype bf16
```

### 2) 运行基线评测

```bash
python eval/eval_qwen35_code.py \
  --model-dir /path/to/Qwen3.5-4B \
  --dtype bf16 \
  --temperature 0.0 \
  --top-p 1.0 \
  --humaneval-n 20 \
  --lcb-n 20
```

评测结果会输出到 `eval/results/<timestamp>/`，包含 `metrics.json` 与各类中间文件。

### 3) 运行 LoRA SFT

```bash
python lora/sft_opencode_lora.py \
  --model-dir /path/to/Qwen3.5-4B \
  --train-samples 2000 \
  --max-steps 30
```

训练产物默认保存在 `lora/sft_outputs/`。

## 当前基线（示例）

基于 2026-03-13 的子集评测（20 题 HumanEval + 20 题 LCB-lite）：

- HumanEval pass@1: 0.10
- CodeBLEU: 0.2872
- LCB-lite public problem pass rate: 0.00

详细报告见：`eval/benchmark_report_20260313.md`

## 文档入口

- SFT 说明：`docs/sft.md`（LoRA 细节也可看 `lora/sft.md`）
- 结果汇总：`docs/results.md`
- 使用教程：`docs/tutorial.md`
