# T12 历史训练入口：实际服务器闭包补齐

目的不是追加实验或改变最终模型，而是避免已保存的训练命令在 main 上缺模块。
实际运行来源为 `.worktrees/t12_physical_robust_v2_20260927`，HEAD
`7093ec46082eed2fae127ec5028d3e2e8548b592`；逐文件核验了服务器实际内容，而非只看 HEAD。

## 补入范围

- `product_global_redesign_training.py`：整幅商品编辑潜变量缓存与历史训练流程。
- `qwen_mini_small_run.py`：小型 Qwen 编辑器缓存/训练 CLI。
- `compact_turbo.py`、`product_scene_training.py`：上述入口在 main 缺失的必要训练依赖。

四项原源码逐字节迁入，没有重写训练函数、改变模型或替换任何现存的 main 模型模块。
来源/大小/SHA 分别在 `T12_PRESERVED_ENTRY_ADDITIONS_20261004.json` 和
`T12_PRESERVED_ENTRY_DEPENDENCIES_20261004.json` 中登记。前两项也与受保护的本地原有修改一致。

发布核心提交：`b3ae1710d10ebf20231af719e8d0d726072e530f`。

## 验证与明确边界

初次只补两个入口，CPU 导入实际报缺 `compact_turbo`；未发布该失败候选。
随后沿原服务器源码静态检查 57 项依赖，确认只缺另两项；实际服务器 SHA 与来源 Git 一致。
完整候选从 Git 树直接在服务器 CPU 导入，27 项实际导入源码留存 SHA；四项新增来源全一致。
缓存和训练两个 CLI 的全部传参通过现存真实函数签名绑定，训练函数由不写文件的桩替代。

这是源码依赖及调用合同验证，不是重新训练或性能复现。没有读取科学数据、加载权重、
下载模型、分配 CUDA 或碰硬件；原运行目录、best/last、缓存、结果和全部测速均未动。
工作目录旧版本与 main 仍不完全相同，14 项原有真实本地差异继续审计。
大权重/教师依赖、Windows 设备闭包及旧工作树治理未因此宣布全部完成。
