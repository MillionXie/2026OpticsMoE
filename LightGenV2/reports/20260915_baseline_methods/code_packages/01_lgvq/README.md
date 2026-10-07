# 01 视频质量评价 · LGVQ baseline 代码

对应指标：Temporal SRCC 0.7663；Spatial SRCC 0.6908。源码版本：`4b24e9ad4df027db3bac92a7721ae8cf052a6f50`。

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
准备 LGVQ 原视频、MOS.txt、prompt_cls.json，以及 Qwen3-VL-2B-Instruct snapshot `89644892e4d85e24eaac8bacfd4f463576704203`。清单中的视频路径需要指向当前机器。

## 运行

```bash
python run_baseline.py prepare -- --dataset-root /path/to/LGVQ --output /path/to/lgvq_split.csv --seed 42
python run_baseline.py temporal -- --phase all --model /path/to/Qwen3-VL-2B-Instruct --manifest /path/to/lgvq_split.csv --run-dir LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/handoff_temporal
python run_baseline.py spatial -- --phase all --model /path/to/Qwen3-VL-2B-Instruct --manifest /path/to/lgvq_split.csv --run-dir LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/handoff_spatial
```
`all` 依次提取特征、训练评分头、完整测试；也可分别指定 `--phase extract/train/benchmark`。两个入口固定四帧。仅固定权重复评时，将对应 best_checkpoint.pt 放入新 run 的 `checkpoints/frames4/` 后执行 `--phase benchmark`，不执行训练。A100 配置的 GPU 名称检查位于 `timing.gpu`，更换硬件须明确修改该项并重新报告。

原入口的完整参数可通过 `python run_baseline.py <入口名> -- --help` 查看。首次修改配置前可先执行 `check`；修改后的配置和训练记录归属于新复现运行。
代码包已进行源码字节、依赖收集和语法校验；本次整理未重新训练或运行完整数据集评价。固定权重复评需使用下列权重身份，重新训练所得指标另行报告。

## 权重与数据身份

时间头 SHA256：`e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2`；空间头 SHA256：`9155ec7a45a19e14ac87661021b44599317d5efb16475df77707d70e866ad913`。
原 LGVQ manifest SHA256：`607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc`（改写绝对路径后文件哈希会改变，需另存新哈希并保留样本和划分）。
