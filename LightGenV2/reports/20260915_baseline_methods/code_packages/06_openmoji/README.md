# 06 语义交互 · OpenMoji baseline 代码

对应指标：修改格准确率 0.8420。源码版本：`966f407000c5ecef4c7c6415d8995065aeed7484`。

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
准备 OpenMoji 17.0.0 彩色 72px 图标素材和 Qwen3-VL-2B-Instruct snapshot `89644892e4d85e24eaac8bacfd4f463576704203`。可由 prepare 生成 seed 73 的 v2 数据（5,000/1,000），也可提供原完整 v2 数据和清单。固定权重复评需要 qwen_shared_s73 的 best_checkpoint.pt。

## 运行

在 `source/LightGenV2/tasks/t04_semantic_interaction/configs/qwen_shared.yaml` 中补充 `dataset.data_dir`（v2 数据目录）、`dataset.asset_dir`（图标目录）、`qwen.checkpoint`（本地模型目录），使用绝对路径并保留原 base_config。

```bash
python run_baseline.py prepare -- --device cuda
python run_baseline.py train -- --device cuda --run-dir LightGenV2/tasks/t04_semantic_interaction/runs/simulation/handoff_qwen_shared
python run_baseline.py evaluate -- --device cuda --checkpoint /path/to/best_checkpoint.pt --run-dir LightGenV2/tasks/t04_semantic_interaction/runs/simulation/handoff_recheck
```
完整冻结图文主干，训练同结构网格头和适配器 100 轮。固定权重入口不重新训练；必须配套 v2 场景及真实 source_grid。

原入口的完整参数可通过 `python run_baseline.py <入口名> -- --help` 查看。首次修改配置前可先执行 `check`；修改后的配置和训练记录归属于新复现运行。
代码包已进行源码字节、依赖收集和语法校验；本次整理未重新训练或运行完整数据集评价。固定权重复评需使用下列权重身份，重新训练所得指标另行报告。

## 权重与数据身份

原 run：`qwen_shared_s73`，选中 epoch 25 EMA，训练 manifest 记录的恢复版本为上述 commit。权重 SHA256 见 `REFERENCE.json`（若原记录未提供则保留 null，不推测）。
