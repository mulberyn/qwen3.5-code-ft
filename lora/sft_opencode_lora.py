#!/usr/bin/env python3
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def parse_args():
    """
    用途: 定义并解析命令行参数，方便在终端动态调整超参数。
    输入: 无 (从 sys.argv 读取)
    输出: argparse.Namespace 对象 (包含所有参数名和值)
    """
    p = argparse.ArgumentParser(description="LoRA SFT on nvidia/OpenCodeInstruct for Qwen3.5-4B")
    p.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen3.5-4B") # 本地模型路径
    p.add_argument("--output-root", default="lora/sft_outputs")                  # 输出根目录
    p.add_argument("--dataset-name", default="nvidia/OpenCodeInstruct")         # 数据集 HuggingFace 路径
    p.add_argument("--train-samples", type=int, default=2000)                   # 训练样本数
    p.add_argument("--max-length", type=int, default=1024)                      # 最大序列长度
    p.add_argument("--lora-r", type=int, default=16)                            # LoRA 秩 (Rank)
    p.add_argument("--lora-alpha", type=int, default=32)                        # LoRA 缩放系数
    p.add_argument("--lora-dropout", type=float, default=0.05)                  # LoRA 层丢弃率
    p.add_argument("--learning-rate", type=float, default=2e-4)                 # 学习率
    p.add_argument("--weight-decay", type=float, default=0.0)                   # 权重衰减
    p.add_argument("--per-device-batch-size", type=int, default=1)              # 每个 GPU 的 Batch Size
    p.add_argument("--grad-accum", type=int, default=8)                         # 梯度累加步数
    p.add_argument("--max-steps", type=int, default=30)                         # 最大训练步数
    p.add_argument("--logging-steps", type=int, default=1)                      # 日志记录间隔
    p.add_argument("--save-steps", type=int, default=30)                         # 模型保存间隔
    p.add_argument("--seed", type=int, default=42)                              # 随机种子
    return p.parse_args()


def format_sample(tokenizer, user_text: str, assistant_text: str) -> str:
    """
    用途: 将原始的 User/Assistant 文本转换为模型特定的 Chat Template 格式。
    输入: tokenizer (分词器), user_text (用户指令), assistant_text (期望回答)
    输出: str (拼接好的 Prompt 字符串)
    """
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    try:
        # 尝试应用聊天模板（针对支持 Qwen/Llama3 等新版模板的分词器）
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,            # 返回字符串而非 token 序列
            add_generation_prompt=False, # 训练模式下不添加生成前缀
            enable_thinking=False,      # 针对推理模型(如 R1)的特殊处理
        )
    except TypeError:
        # 兼容旧版本 Transformers 的调用方式
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

def build_local_train_dataset(args, tokenizer):
    """
    用途: 流式加载原始数据集，格式化并进行 Tokenization。
    输入: args (配置参数), tokenizer (分词器)
    输出: datasets.Dataset (预处理后的数据集对象)
    """
    # streaming=True 可以在不完整下载大数据集的情况下读取数据
    ds = load_dataset(args.dataset_name, split="train", streaming=True)

    rows = []
    for ex in ds:
        inp = (ex.get("input") or "").strip()   # 获取输入字段
        out = (ex.get("output") or "").strip()  # 获取输出字段
        if not inp or not out: continue

        # 1. 转换为对话格式
        text = format_sample(tokenizer, inp, out)
        # 2. 转换为 token ID
        tok = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )
        # 过滤过短的样本
        if len(tok["input_ids"]) < 16: continue

        rows.append({
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": tok["input_ids"], # SFT 中 labels 通常与输入一致（内部会自动 shift）
        })
        if len(rows) >= args.train_samples: break

    if not rows: raise RuntimeError("No valid training samples collected from dataset.")
    return Dataset.from_list(rows)


def collate_fn(features, pad_token_id):
    """
    用途: 将长度不一的样本填充成相同长度的 Batch。
    输入: features (List[dict]), pad_token_id (填充符ID)
    输出: Dict[str, torch.Tensor] (张量字典)
    """
    # 提取序列并转换为 LongTensor
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    # padding 操作
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    # labels 的 pad 值为 -100，PyTorch CrossEntropy 会自动忽略这个值的 Loss 计算
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed) # 设置随机种子保证结果可复现

    # 创建带时间戳的输出文件夹
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"opencode_lora_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Qwen 系列通常需要手动指定 pad

    # 2. 构建数据
    train_ds = build_local_train_dataset(args, tokenizer)

    # 3. 加载预训练模型 (使用 bf16 精度以节省内存并提高速度)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False           # 训练时关闭 KV Cache 以支持梯度回传
    model.gradient_checkpointing_enable()    # 开启梯度检查点技术以节省显存

    # 4. LoRA 配置：目标指向所有的 Linear 层，提升调优效果
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg) # 包装模型，只允许 LoRA 参数更新
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 5. 配置训练参数
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(), # 优先使用 BF16
        fp16=False,
        report_to="none",               # 不上传日志到 wandb 等
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=args.seed,
    )

    # 6. 初始化训练器并开始训练
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=lambda feats: collate_fn(feats, tokenizer.pad_token_id),
    )
    trainer.train()

    # 7. 保存 LoRA Adapter 和 Tokenizer
    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # 8. 生成并保存运行总结 (JSON)
    summary = { ... } # 省略具体内容，包含所有超参数和运行耗时
    (out_dir / "sft_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
