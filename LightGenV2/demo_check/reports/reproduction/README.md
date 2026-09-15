# 师弟实验复现检查

保留 `adrenal_softsign_code_export_20260915_145336/code/` 原始字节；独立入口
`../../reproduction/run_adrenal.py` 调用其中的模型、数据处理、训练、选模和评估函数。

本轮先检查 2/4/6 层、MoE/D2NN、有/无 OEO 的 12 个配置前向和反向，再复跑
2 层、seed17 的四组，各完整 50 轮。固定已报告的 Softsign，不重新筛选激活；
结果不能称为原始 68 次搜索及五种子统计的完整复现。

数据文件 SHA256：`22bc193a85c43be85f093bf12f3ffdc5ed79b5efda63925bd9e063a70594bf93`。
GPU 默认只用一张。运行目录分别放 `demo_check/runs/smoke/` 与
`demo_check/runs/simulation/`，不可覆盖历史归档。

```bash
CUDA_VISIBLE_DEVICES=0 python LightGenV2/demo_check/reproduction/run_adrenal.py --phase smoke --data /absolute/path/data.npz --out /absolute/path/demo_check/runs/smoke/adrenal_20260915
CUDA_VISIBLE_DEVICES=0 python LightGenV2/demo_check/reproduction/run_adrenal.py --phase train --depth 2 --seed 17 --data /absolute/path/data.npz --out /absolute/path/demo_check/runs/simulation/adrenal_L2_s17_20260915
```

每个 run 保存实际配置、命令、源码 SHA256、Git commit、环境和运行状态；训练保存
逐轮验证、best/last 权重，全部四组完成并封存后才加载测试集，保存逐样本预测。
原始代码和配置不修改，迁移只在独立入口中指定数据与输出位置。

EuroSAT 的本地归档不含数据图片或训练权重；该项目前尚未完成运行复现。
