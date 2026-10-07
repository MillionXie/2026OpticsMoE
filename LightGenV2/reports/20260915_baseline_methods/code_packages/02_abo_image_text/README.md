# 02 商品检索 · ABO 图搜文 baseline 代码

对应指标：R@1 0.7358。源码版本：`b9a4be689f198f55ee0caaf4e5bcddd4dfc80200`。

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
准备 ABO easy100 数据目录：`manifest.csv`、`train.csv`、`test.csv`、`titles.csv` 和清单对应图像；100 商品、4,800/2,400 图像。准备 Qwen3-VL-Embedding-2B snapshot `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`。该基线无需任务头权重。

## 运行

```bash
python run_baseline.py evaluate -- --model /path/to/Qwen3-VL-Embedding-2B --data-root /path/to/abo_easy100 --run-dir LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/handoff_qwen
```
该入口直接编码完整 2,400 张测试图和 100 条标题，无训练阶段。保留动态长宽、2,048 维和原始 prompt。不同 GPU 可显式传入 `--expected-gpu`；计时附加结果属于该次运行。

原入口的完整参数可通过 `python run_baseline.py <入口名> -- --help` 查看。首次修改配置前可先执行 `check`；修改后的配置和训练记录归属于新复现运行。
代码包已进行源码字节、依赖收集和语法校验；本次整理未重新训练或运行完整数据集评价。固定权重复评需使用下列权重身份，重新训练所得指标另行报告。

## 权重与数据身份

模型 safetensors SHA256：`c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`。
测试清单 SHA256：`4ca5834abf4bb4d614b41f05b5b2481b3d41720dd78c1c37b05923c97657dfb8`；标题表 SHA256：`c97e8da3c6c608a706f2cc0d284aeafb873fc6cc17d04a7c9f9f66b2135846c1`。
