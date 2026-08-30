# Qwen 光学平台交接入口

这里不是新的单一任务，而是把现有工程整理成两份可移交项目的来源：

- `qwen_optical_simulation_project`：服务器仿真、训练、评估与 mask 导出。
- `qwen_optical_hardware_project`：实验室标定、播放、采集、一致性评价与逐层微调。

两份项目之间通过明确的 stage manifest、checkpoint SHA、光学几何和 task contract
交接，不能通过手工复制几个 BMP 或 checkpoint 猜测对应关系。

先阅读：

1. `SIMULATION_PROJECT.md`
2. `HARDWARE_PROJECT.md`
3. `NEW_TASK_WORKFLOW.md`
4. `AI_MODIFICATION_RULES.md`
5. `COMMANDS.md`

验证任务合同：

```powershell
python -m experiments.qwen_optical_platform_handoff `
  experiments\qwen_optical_platform_handoff\templates\simulation_retrieval.contract.json
```

生成两份代码包：

```powershell
python -m experiments.qwen_optical_platform_handoff.build_packages `
  --output-dir handoff_exports
```

默认不复制 Qwen 4 GB 权重、数据集、训练 checkpoint、CCD、运行日志和历史标定结果。
如需让硬件包在断网实验室直接加载 Qwen，可显式传入
`--hardware-offline-model-dir`；权重只进入硬件包，不会在两个包中重复。

当前 `qwen_mnist4_early_robust_full_data_lab` 是已经配好 Caltech101 的“实验实例包”，
可以继续用于本次四层采集和微调，但它不是新数据集的通用模板。新任务必须先在仿真工程中
新建自己的任务目录、split、head/loss/metric 和导出合同，再把 task payload 交给硬件工程。

两份包的边界和需要额外交接的运行资产见 `PROJECT_BOUNDARIES.md`。
