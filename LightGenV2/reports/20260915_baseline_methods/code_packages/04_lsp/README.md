# 04 关键点检测 · LSP baseline 代码

对应指标：PCK@0.2 0.7226。源码版本：`bcec3a5977ccf00c71c6b5e8c00bf436cc3e305e`。

`source/` 保存该版本的原始源码和依赖配置，`METHODS.md` 为技术说明，`SOURCE_MANIFEST.json` 为逐文件校验清单。公共模块保留原目录结构，训练时仅运行下方 baseline 入口。

## 准备

使用 Python 3.11、Git 和 CUDA 环境，安装 `requirements.txt`（PyTorch 2.6.0 对应 CUDA 12.4；其他依赖未作为完整锁文件固定）。
在当前目录执行：

```bash
python run_baseline.py check
```

本页是9月15日固定历史baseline的复现记录，不是最新模型入口；日常开发仍使用统一main。
原ZIP、738份历史源码、权重/数据身份与全部测速不改。当前包装启动器增加创建嵌套仓库的
权限检查及Windows长路径只读支持，默认不能再建Git工程；不修改原模型或原指标。确需仓库外独立历史复现并另获
用户许可时才可执行 `python run_baseline.py --allow-new-repository init-source`；
该开关本身不是授权。历史来源仍以 `SOURCE_ORIGIN.json` 为准。
配置路径均相对于 `source/`。数据、Qwen 预训练权重、训练后的任务头及特征缓存需另外提供，包内不含这些文件。
准备 LSP 2,000 张图与 joints.mat；从头训练还需 HR-LSPET 9,428 张及标注。模型为 Qwen3-VL-Embedding-2B，snapshot 与 ABO 相同。固定权重复评需要原姿态头 `teacher_best_train_loss.pt`。

## 运行

在 `source/experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt2.yaml` 中补充当前机器的 `dataset.data_root`、`qwen.model_id`（本地模型目录）、`qwen.local_files_only: true` 和新的 `output_dir`，保留原 base_config。

```bash
python run_baseline.py train -- --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt2.yaml
python run_baseline.py evaluate -- --model /path/to/Qwen3-VL-Embedding-2B --data-root /path/to/lsp_pose --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt2.yaml --checkpoint /path/to/teacher_best_train_loss.pt --run-dir LightGenV2/tasks/t02_keypoint_detection/runs/simulation/handoff_qwen
```
仅 `teacher_train` 属于此基线，训练 40 轮并按训练损失选头。所固定 commit 是指标复评版本；原历史训练启动 commit 尚未从记录中确定，因此不声称从头训练必定精确返回该小数。

原入口的完整参数可通过 `python run_baseline.py <入口名> -- --help` 查看。首次修改配置前可先执行 `check`；修改后的配置和训练记录归属于新复现运行。
代码包已进行源码字节、依赖收集和语法校验；本次整理未重新训练或运行完整数据集评价。固定权重复评需使用下列权重身份，重新训练所得指标另行报告。

## 权重与数据身份

原头 SHA256：`0a4569f288f6de424b1b412452fa804a96e58d8f7e8655efaa23f461ab8cc735`。
