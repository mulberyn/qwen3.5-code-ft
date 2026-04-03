# fixed10 质量分析

## 维度均分对比

| 维度 | Base | LoRA | 差值(LoRA-Base) |
|---|---:|---:|---:|
| 正确性 | 3.0 | 3.5 | 0.5 |
| 完整性 | 4.3 | 4.7 | 0.4 |
| 可运行性 | 3.6 | 4.0 | 0.4 |
| 解释质量 | 3.4 | 3.3 | -0.1 |

## 分题胜负统计

| 维度 | LoRA更好 | 持平 | Base更好 |
|---|---:|---:|---:|
| 正确性 | 1 | 9 | 0 |
| 完整性 | 3 | 7 | 0 |
| 可运行性 | 1 | 9 | 0 |
| 解释质量 | 1 | 7 | 2 |

## 失败用例摘录

| 题号 | Base通过 | LoRA通过 | Base错误 | LoRA错误 |
|---|---:|---:|---|---|
| Q1 | 0/3 | 3/3 | exec_error: SyntaxError: unterminated string literal (detected at line 33) (<str |  |
| Q5 | 0/2 | 0/2 | exec_error: SyntaxError: invalid syntax (<string>, line 1) | exec_error: SyntaxError: invalid syntax (<string>, line 1) |
| Q6 | 0/2 | 0/2 | runtime_error: TypeError: cannot unpack non-iterable int object |  |
| Q8 | 0/3 | 0/3 | exec_error: SyntaxError: invalid syntax (<string>, line 1) | exec_error: SyntaxError: invalid syntax (<string>, line 1) |

## 结论

- LoRA 相对 Base 提升最大的维度：正确性（差值 0.5）。
- LoRA 相对 Base 表现最弱的维度：解释质量（差值 -0.1）。
- 建议结合失败用例继续定向补数据（函数签名约束、边界样例、解释结构化输出）。
