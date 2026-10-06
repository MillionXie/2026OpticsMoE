# 已保存PNG的T12指标工具

这是2026-09-28 TEST2304五模型比较的指标源码入口，不是训练或重新生成图片。
原始源码、全部CSV及六张对照图仍保留在仓库根
`outputs/t12_perceptual_comparison_20260928`；不修改原结果。

使用模块入口 `python -m LightGenV2.tasks.t12_text_to_image.perceptual.eval_lpips`
（或 `eval_dists`、`eval_regions`），参数沿用原脚本；必须给新 `--output`。
这些工具需要已有数据、PNG、官方LPIPS/DISTS模型及原环境依赖；没有自动下载或重新训练。
`--repo` 指向统一主工程，其他路径显式指定。各脚本拒绝已存在输出目录。
采用版仅修复两个跨脚本导入名，其余计算逻辑原样保留；特别是不改变ROI、量化、
掩码、固定样本及指标口径。评估掩码不进入模型推理。

原区域脚本引用 `t12_eval_perceptual_20260928`；原件与交付包改名后的
`eval_lpips.py` 逐字节相同。现使用任务内完整模块路径，不依赖私人临时目录。
证据：`maintenance/storage/T12_PERCEPTUAL_DEPENDENCY_IDENTITY_20261006.json`。

源码合同测试不读取数据或模型，不证明已在当前环境复算历史分数。
汇总入口现为：
`python -m LightGenV2.tasks.t12_text_to_image.perceptual.assemble_three_task_scores --input <旧CSV目录> --output <新CSV路径>`。
它只拼接已有摘要，不计算新指标；拒绝重复model/mode键及非768样本，输出采用独占创建，
不会覆盖历史CSV。15行各字段已与原表逐项核对一致，源CSV的SHA在测试前后不变。
旧汇总器及作图器仍作为原交付包保留；作图器会在导入时打开ZIP，尚不作为新模块执行入口。
