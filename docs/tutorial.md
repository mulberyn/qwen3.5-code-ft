# Qwen3.5-4B 从零下载、部署与对话测试教程

本文基于模型仓库：<https://huggingface.co/Qwen/Qwen3.5-4B>

目标：

- 从零下载 Qwen3.5-4B
- 本地部署并加载模型
- 完成一次真实对话测试

## 1. 环境准备

### 1.1 机器配置（本次实测）

- OS: Linux
- Python: 3.10.8
- GPU: NVIDIA GeForce RTX 5090 (32GB)
- 磁盘：
  - 系统盘 `/` 约 30GB（剩余较少）
  - 数据盘 `/root/autodl-tmp` 约 50GB（建议下载到这里）

### 1.2 检查命令

```bash
python --version
pip --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /
df -h /root/autodl-tmp
```

## 2. 安装依赖

```bash
pip install -U torch transformers huggingface_hub accelerate safetensors
```

说明：

- 若你已经安装了 `transformers`，也建议升级到最新版本，避免 `qwen3_5` 架构不识别。

## 3. 网络与下载策略

实测环境中，`huggingface.co` 无法直连（超时）。
因此采用镜像端点：`https://hf-mirror.com`。

先验证连通性：

```bash
curl -I --max-time 15 https://huggingface.co
curl -I --max-time 15 https://hf-mirror.com
```

若官方站可连通，可直接用官方；若不可连通，按下面镜像方式下载。

## 4. 从零下载模型

将模型下载到数据盘，避免系统盘不足。

```bash
mkdir -p /root/autodl-tmp/models/Qwen3.5-4B
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  Qwen/Qwen3.5-4B \
  --local-dir /root/autodl-tmp/models/Qwen3.5-4B
```

下载完成后，目录中应包含：

- `model.safetensors-00001-of-00002.safetensors`
- `model.safetensors-00002-of-00002.safetensors`
- `model.safetensors.index.json`
- tokenizer/config 等文件

可检查大小：

```bash
du -sh /root/autodl-tmp/models/Qwen3.5-4B
ls -lh /root/autodl-tmp/models/Qwen3.5-4B | head -n 20
```

## 5. 本地部署与对话测试

### 5.1 编写测试脚本

创建 `chat_test_qwen35.py`：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "/root/autodl-tmp/models/Qwen3.5-4B"

# 加载 tokenizer 和模型
# 使用 bfloat16 + device_map=auto，在 32GB 显存上可直接运行

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are a concise and helpful assistant."},
    {
        "role": "user",
        "content": "请直接回答，不要输出思考过程。用两句话介绍 Qwen3.5-4B 的特点。",
    },
]

# 新版 Qwen Chat Template 支持 enable_thinking 参数；
# 若当前 tokenizer 不支持，则回退到普通模板
try:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
except TypeError:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=96,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response)
```

### 5.2 运行测试

```bash
python chat_test_qwen35.py
```

### 5.3 实测成功输出（示例）

```text
Qwen3.5-4B 是通义千问系列中参数量为 4B 的增强版模型，在保持轻量级的同时显著提升了语言理解与生成能力。它支持多语言交互、复杂逻辑推理及代码生成，适用于移动端、嵌入式设备等多种资源受限场景。
```

到此，说明“下载 -> 部署 -> 对话”链路已完成。

## 6. 常见问题

### 6.1 报错：`model type qwen3_5 not recognized`

原因：`transformers` 版本过旧。

解决：

```bash
pip install -U transformers
```

### 6.2 官方 Hugging Face 无法访问

可切换镜像端点下载：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...
```

### 6.3 显存不足

可尝试：

- 减少 `max_new_tokens`
- 使用量化（如 4bit/8bit）
- 切换更小模型

## 7. 一条命令快速复现（镜像环境）

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3.5-4B --local-dir /root/autodl-tmp/models/Qwen3.5-4B && \
python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
m='/root/autodl-tmp/models/Qwen3.5-4B'
tok=AutoTokenizer.from_pretrained(m, trust_remote_code=True)
mod=AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
msgs=[{'role':'system','content':'You are helpful.'},{'role':'user','content':'你好，请做一个自我介绍。'}]
try:
    text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp=tok([text], return_tensors='pt').to(mod.device)
out=mod.generate(**inp, max_new_tokens=80)
print(tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True))
PY
```

---

如果你希望下一步做 API 服务化部署（FastAPI/vLLM/OpenAI-Compatible API），可以在本地模型目录基础上直接继续。
