# Top-Conference Style Comparison: 20260313_182948 vs 20260313_193331

- Input run A: `eval/results/20260313_182948`
- Input run B: `eval/results/20260313_193331`
- Label A: `Base`
- Label B: `LoRA`

## Figure Files

- `topconf_core_metrics.png/.pdf`
- `topconf_codebleu_components.png/.pdf`
- `topconf_efficiency_metrics.png/.pdf`
- `topconf_tradeoff_scatter.png/.pdf`

## Numeric Comparison

| Metric | Base | LoRA | Delta (post-pre) |
|---|---:|---:|---:|
| HumanEval pass@1 | 0.100000 | 0.000000 | -0.100000 |
| CodeBLEU | 0.289197 | 0.277531 | -0.011666 |
| LCB problem pass | 0.000000 | 0.200000 | +0.200000 |
| LCB test pass | 0.000000 | 0.444444 | +0.444444 |
| LCB compile | 0.200000 | 0.350000 | +0.150000 |
| HE tok/s | 32.626353 | 23.562100 | -9.064253 |
| LCB tok/s | 33.202156 | 25.003205 | -8.198951 |
| HE latency(s) | 2.404498 | 2.535852 | +0.131354 |
| LCB latency(s) | 3.356710 | 7.455044 | +4.098335 |
| CB ngram | 0.139242 | 0.157730 | +0.018488 |
| CB weighted | 0.197479 | 0.171061 | -0.026418 |
| CB syntax | 0.441034 | 0.426494 | -0.014540 |
| CB dataflow | 0.379032 | 0.354839 | -0.024194 |
