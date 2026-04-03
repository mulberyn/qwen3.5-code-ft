# 微调前后同题回答对比

## 测试问题

请用 Python 写一个函数：输入一个字符串列表，按字符串长度分组并返回字典；同时给出时间复杂度和空间复杂度。

## 运行命令

### 微调前（Base）

```bash
python - <<'PY' > lora/compare/base_answer.txt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_dir='/root/autodl-tmp/models/Qwen3.5-4B'
prompt='请用 Python 写一个函数：输入一个字符串列表，按字符串长度分组并返回字典；同时给出时间复杂度和空间复杂度。'
tok=AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
mod=AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
msgs=[{'role':'system','content':'You are a helpful coding assistant.'},{'role':'user','content':prompt}]
try:
    text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp=tok([text], return_tensors='pt').to(mod.device)
with torch.no_grad():
    out=mod.generate(**inp, max_new_tokens=260, do_sample=False)
print(tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True))
PY
```

### 微调后（LoRA）

```bash
python lora/run_lora_model.py \
  --model-dir /root/autodl-tmp/models/Qwen3.5-4B \
  --adapter-dir lora/sft_outputs/opencode_lora_20260313_184805/adapter \
  --prompt "请用 Python 写一个函数：输入一个字符串列表，按字符串长度分组并返回字典；同时给出时间复杂度和空间复杂度。" \
  --max-new-tokens 260 > lora/compare/lora_answer.txt
```

## 原始输出文件

- 微调前：`lora/compare/base_answer.txt`
- 微调后：`lora/compare/lora_answer.txt`

## 现象对比

1. 两者都能给出正确的分组函数实现，并且复杂度均给出 `O(n)` 时间与 `O(n)` 空间。
2. 微调前回答使用了 `defaultdict` + 类型注解，结构较规范，但输出末尾出现截断（结尾停在“总”）。
3. 微调后回答更偏“教学模板”风格，函数 + 示例 + 复杂度解释完整收尾，整体可读性更完整。
4. 在这个问题上，微调后的主要变化是“回答收束更完整、表述更稳定”，而不是算法本质变化。

## 结论

在该同题测试上，LoRA 微调后模型的回答完整性和可直接复用性略优于微调前；两者核心算法正确性差异不大。
