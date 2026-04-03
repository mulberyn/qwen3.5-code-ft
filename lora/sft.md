# Qwen3.5-4B 使用 nvidia/OpenCodeInstruct 进行 LoRA SFT 微调实战记录

## 2面问题

三抽一：

1. 梯度下降是什么？

英文问题：

1. 介绍一下你的项目
2. 介绍一下一本最喜欢的书？

综合问答：

1. LoRA 微调了哪几层？
2. LoRA 的原理？LoRA 的初始化方式？LoRA 的训练方式？LoRA 的推理方式？
3. LoRA 中的参数（Rank）是怎么选择的？为什么选择这个值？
4. 我这个 Loss 是怎么计算的，微调后的 Loss 是怎么计算的？（注意一下细节）

## 1. 目标与产出

本次目标：

1. 使用 `nvidia/OpenCodeInstruct` 对本地 `Qwen3.5-4B` 做 LoRA SFT。
2. 完整记录数据格式、数据处理、训练方法、训练过程、问题与排查。
3. 成功加载微调后的模型并完成一次推理验证。

最终产出：

1. 训练脚本：`lora/sft_opencode_lora.py`
2. 微调模型启动脚本：`lora/run_lora_model.py`
3. 微调输出目录：`lora/sft_outputs/opencode_lora_20260313_184805`
4. 运行摘要：`lora/sft_outputs/opencode_lora_20260313_184805/sft_run_summary.json`

## 2. 环境信息

- OS: Linux
- Python: 3.10.8
- GPU: NVIDIA GeForce RTX 5090 (32GB)
- 模型目录: `/root/autodl-tmp/models/Qwen3.5-4B`
- 网络：`huggingface.co` 不稳定，使用 `HF_ENDPOINT=https://hf-mirror.com`
- 缓存目录：`HF_HOME=/root/autodl-tmp/hf_cache`（避免系统盘压力）

## 3. 数据集与数据格式

数据集：`nvidia/OpenCodeInstruct`

通过流式读取看到的关键字段：

- `id`: 样本唯一 ID
- `input`: 用户指令（题目描述）
- `output`: 目标输出（通常是代码）
- `domain`: 领域标签
- `generation_algorithm`: 数据生成方式
- `llm_judgement`: 评估信息
- `unit_tests`: 单测文本
- `tests_execution_status`: 测试通过状态
- `average_test_score`: 平均测试分数

本次 SFT 使用的核心监督字段：

- 输入：`input`
- 标签：`output`

## 4. 数据处理方式

在 `lora/sft_opencode_lora.py` 中采用以下处理流程：

1. 使用 `streaming=True` 读取训练集，避免一次性下载和加载全部数据。
2. 对每条样本进行清洗：
   - `input` 或 `output` 为空时跳过。
   - 去除前后空白。
3. 构造 chat 格式训练文本：
   - system: `You are a helpful coding assistant.`
   - user: `input`
   - assistant: `output`
4. 调用 tokenizer 的 `apply_chat_template` 生成完整训练文本。
5. 进行 tokenization：
   - `max_length=1024`
   - 超长截断
6. 构建 `input_ids / attention_mask / labels`：
   - `labels` 与 `input_ids` 相同（标准 Causal LM SFT）
7. 取前 `train_samples=800` 条有效样本组成本地训练集。

说明：

- 这是快速可复现的 SFT 流程，适合先跑通与做第一版效果验证。
- 若后续追求更高质量，建议加入更精细的 loss mask（仅对 assistant 部分计算 loss）。

## 5. 微调方式（LoRA 配置）

LoRA 核心参数：

- `r=16`
- `lora_alpha=32` alpha 越大，越强调 LoRA 模块的更新，相当于放大了 LoRA 的学习率，越大可能出现过拟合的情况。
- `lora_dropout=0.05`
- `bias=none`
- `task_type=CAUSAL_LM`
- `target_modules`:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

训练参数：

- `max_steps=20`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- `learning_rate=2e-4`
- `weight_decay=0.0`
- `max_length=1024`
- `bf16=True`

实际运行命令：

```bash
HF_HOME=/root/autodl-tmp/hf_cache \
HF_ENDPOINT=https://hf-mirror.com \
python lora/sft_opencode_lora.py \
  --model-dir /root/autodl-tmp/models/Qwen3.5-4B \
   --output-root lora/sft_outputs \
  --train-samples 800 \
  --max-length 1024 \
  --max-steps 20 \
  --per-device-batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-4
```

## 6. 微调过程记录

运行结果：成功完成 20 steps，保存 checkpoint 与 adapter。

训练日志（节选）：

- step 1: loss 0.6767
- step 10: loss 0.2720
- step 20: loss 0.2845
- train_runtime: 186.7s
- train_steps_per_second: 0.107
- train_loss: 0.3135

完整运行摘要：

- 文件：`lora/sft_outputs/opencode_lora_20260313_184805/sft_run_summary.json`
- 关键值：
  - `train_samples=800`
  - `max_steps=20`
  - `elapsed_sec=262.2810`

输出目录结构：

- `lora/sft_outputs/opencode_lora_20260313_184805/adapter`
- `lora/sft_outputs/opencode_lora_20260313_184805/checkpoint-20`
- `lora/sft_outputs/opencode_lora_20260313_184805/sft_run_summary.json`

## 7. 微调后的模型启动与验证

启动命令：

```bash
python lora/run_lora_model.py \
  --model-dir /root/autodl-tmp/models/Qwen3.5-4B \
   --adapter-dir lora/sft_outputs/opencode_lora_20260313_184805/adapter \
  --prompt "Write Python function two_sum(nums, target) and return indices." \
  --max-new-tokens 180
```

结果：成功加载 base model + LoRA adapter，并生成可执行的 Python `two_sum` 函数实现。

这说明“微调后模型可正常启动并推理”。

## 8. 微调中遇到的问题与处理

### 8.1 镜像与网络问题

问题：环境对 `huggingface.co` 访问不稳定。

处理：统一设置

- `HF_ENDPOINT=https://hf-mirror.com`
- `HF_HOME=/root/autodl-tmp/hf_cache`

效果：数据和缓存拉取稳定，避免系统盘被占满。

### 8.2 `bitsandbytes` 未安装

问题：环境里 `bitsandbytes` 缺失。

处理：本次采用标准 LoRA（bf16）流程，不依赖 QLoRA 的 4bit/8bit 加载，仍可在 32GB 显存完成小规模 SFT。

### 8.3 加速算子缺失提示

问题：日志有提示缺少 flash-linear-attention / causal-conv1d，退回 torch 路径。

处理：本次未阻塞训练，仅影响速度，不影响正确性。

### 8.4 `torch_dtype` 弃用提示

问题：运行时提示 `torch_dtype is deprecated`。

处理：属于告警，不影响训练；后续可改为 `dtype` 参数以消除告警。

## 9. 你后续可直接复用的流程模板

1. 先用较小 `train_samples` 与 `max_steps` 跑通（例如本次配置）。
2. 确认 loss 曲线和推理可用后，逐步放大：
   - `train_samples` -> 5k / 20k
   - `max_steps` -> 200 / 1000+
3. 训练后固定用 `lora/run_lora_model.py` 做回归验证。
4. 同时复跑 `eval/eval_qwen35_code.py`，对比微调前后 HumanEval / CodeBLEU / LCB-lite。

## 10. 结论

本次已经完成以下闭环：

1. 使用 `nvidia/OpenCodeInstruct` 对 `Qwen3.5-4B` 进行 LoRA SFT。
2. 成功产出 LoRA adapter。
3. 成功加载并运行微调后的模型。
4. 全流程、参数、问题与处理已完整记录在本文档。

## 备忘

训练数据：nvidia/OpenCodeInstruct（流式读取，取 800 条有效样本）
微调方式：LoRA（r=16, alpha=32, dropout=0.05，Q/K/V/O + MLP 投影层）
训练配置：max_steps=20, batch_size=1, grad_accum=8, lr=2e-4, max_length=1024, bf16
训练成功结束，摘要文件：sft_run_summary.json
微调后模型成功启动并生成代码（two_sum 示例），验证命令已在文档中记录
