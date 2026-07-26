# Commands

以下命令均从仓库根目录 `2026OpticsMoE` 执行，命令中没有续行反斜杠。

## 100-sample smoke

```bash
python -m experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16 --config experiments/qwen3_vl_2b_cc3m_general_distillation_pca224_moe16/configs/cc3m_smoke100.json --phase all
```

## Formal CC3M run

```bash
python -m experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16 --config experiments/qwen3_vl_2b_cc3m_general_distillation_pca224_moe16/configs/cc3m.json --phase all
```

## Separate phases

```bash
python -m experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16 --config experiments/qwen3_vl_2b_cc3m_general_distillation_pca224_moe16/configs/cc3m.json --phase fit_pca
```

```bash
python -m experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16 --config experiments/qwen3_vl_2b_cc3m_general_distillation_pca224_moe16/configs/cc3m.json --phase pca_oracle_check
```

```bash
python -m experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16 --config experiments/qwen3_vl_2b_cc3m_general_distillation_pca224_moe16/configs/cc3m.json --phase precompute_teacher
```

```bash
python -m experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16 --config experiments/qwen3_vl_2b_cc3m_general_distillation_pca224_moe16/configs/cc3m.json --phase train_vision
```

```bash
python -m experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16 --config experiments/qwen3_vl_2b_cc3m_general_distillation_pca224_moe16/configs/cc3m.json --phase train_language
```

```bash
python -m experiments.qwen3_vl_2b_cc3m_general_distillation_pca224_moe16 --config experiments/qwen3_vl_2b_cc3m_general_distillation_pca224_moe16/configs/cc3m.json --phase train_joint
```

## Tests

```bash
pytest experiments/qwen3_vl_2b_cc3m_general_distillation_pca224_moe16/tests -q
```
