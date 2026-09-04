# LGVQ 实验台入口

实验人员只编辑 `LAB_CONFIG.yaml`，然后运行：

```powershell
python -m experiments.lab_lgvq.prepare_lab
```

初始配置使用包内真实存在的原厂 LUT；在当前光路完成 128 点标定且验证报告为
`recommended_for_use=true` 后，再按主部署文档切换到新生成的线性化 LUT。

硬件标定和六次光路/四阶段微调的唯一完整顺序见：

```text
experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/LAB_DEPLOYMENT.md
```

不要沿用旧 Qwen/MNIST 包中的阶段目录或配置；本目录所有输出独立写入
`experiments/lab_lgvq`。
