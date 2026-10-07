# 05 显著性分析 · SALICON baseline 代码

对应指标：CC 0.8748。源码版本：`d8653fa6057f96b52d53259999343ff2117c1526`。

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
准备 SALICON 2015r1 train/val 图像与 fixations_train2014.json、fixations_val2014.json。模型为 Qwen3-VL-Embedding-2B。固定权重复评另需 50 轮基线选中的 best_checkpoint.pt。

## 运行

在 `source/LightGenV2/tasks/t03_saliency/configs/qwen_aligned_head_50.yaml` 中补充 `dataset.data_root`、`qwen.model_id`（本地模型目录）、`qwen.local_files_only: true`，保留原配置其余内容。

```bash
python run_baseline.py train -- --run-dir LightGenV2/tasks/t03_saliency/runs/simulation/handoff_head50
python run_baseline.py evaluate -- --checkpoint /path/to/best_checkpoint.pt --run-dir LightGenV2/tasks/t03_saliency/runs/simulation/handoff_recheck
```
冻结主干，仅训练 282,596 参数的适配器及密度头，预算 50 轮、无增强；固定权重评价保持 batch 96。训练和复评使用不同的新输出目录。

原入口的完整参数可通过 `python run_baseline.py <入口名> -- --help` 查看。首次修改配置前可先执行 `check`；修改后的配置和训练记录归属于新复现运行。
代码包已进行源码字节、依赖收集和语法校验；本次整理未重新训练或运行完整数据集评价。固定权重复评需使用下列权重身份，重新训练所得指标另行报告。

## 权重与数据身份

原 best SHA256：`689d086b5a390c399df109d07b73185575cd2245998b7bd427f6ccea5b0fe019`；选中 epoch 45。
