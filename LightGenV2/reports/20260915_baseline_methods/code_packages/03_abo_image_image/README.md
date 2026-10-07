# 03 商品检索 · ABO 图搜图 baseline 代码

对应指标：R@1 0.85125（表中 0.8513）。源码版本：`5f731fae979ef81a09770c1dd721b5cecd8b3392`。

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
准备 ABO similarity10 原始 200 商品、2,400 张图及 `data/abo_similarity10_manifest.csv`，清单路径相对于数据目录；随后生成每 SKU 八图库/四查询的新划分。准备与图搜文相同的 Qwen3-VL-Embedding-2B snapshot。无任务头权重。

## 运行

```bash
python run_baseline.py prepare -- --data /path/to/abo_similarity10_data --output /path/to/enrolled_protocol
python run_baseline.py evaluate -- --data /path/to/abo_similarity10_data --manifest /path/to/enrolled_protocol/protocol.json --model /path/to/Qwen3-VL-Embedding-2B --output LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/handoff_qwen64
```
仅运行 `qwen64`：完整冻结模型、前 64 维、native aspect、1,600 图库/800 查询。原结果在 RTX 4090、batch 4 上完整评价，A100 汇总沿用该性能；这里不将计时复测称为重新得到该完整测试指标。

原入口的完整参数可通过 `python run_baseline.py <入口名> -- --help` 查看。首次修改配置前可先执行 `check`；修改后的配置和训练记录归属于新复现运行。
代码包已进行源码字节、依赖收集和语法校验；本次整理未重新训练或运行完整数据集评价。固定权重复评需使用下列权重身份，重新训练所得指标另行报告。

## 权重与数据身份

协议 manifest SHA256：`f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde`。重新生成协议时 provenance 字段可能改变；样本顺序、每 SKU 8/4 划分和图片身份应一致。模型 safetensors SHA256：`c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`。
