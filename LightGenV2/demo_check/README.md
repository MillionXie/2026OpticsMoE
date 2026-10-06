# 历史 demo / 必要对照入口

这里是历史实验与必要 baseline，不是另一个日常主工程。正式任务仍从仓库根 `START_HERE.md` 和 `LightGenV2/tasks/` 进入。原数据、权重、逐样本预测及全部测速保留。

## EuroSAT 历史协议请分开看

| 协议 | 源码入口 | 原结果范围 |
| --- | --- | --- |
| 无电子分类旁路、相位训练 | `pure_optical/run.py`、`pure_optical/evaluate_holdout.py` | 动态MoE/整孔径D2NN，旧2000验证39.85%/29.20%；不是当前任务成绩 |
| 同一冻结CNN、输出概率0.5融合 | `frozen_electronic/run.py`、`frozen_electronic/config.json` | 电子76.05%、MoE融合75.85%、D2NN融合75.90%，旧2000验证；融合没有带来净收益 |
| 同一冻结CNN先编码光场，无输出电子分类旁路 | [shared_frontend](shared_frontend/README.md)；单run独立测试 `shared_frontend/evaluate_single_holdout.py` | 原20轮验证77.20%/73.65%，后续独立空间测试与续训另记，不混成同一个数值 |

原实验说明、限制与完整报告见各协议 README 和 `reports/reproduction/`。本轮没有重新训练、测试或采集，也没有把历史验证精度改称独立测试精度。

## 本轮收录范围

三份此前遗漏的入口已与 `/DATA/DATA1/guest3/demo_reproduction_20260915` 的实际文件逐字节核对。源码身份见 [清单](HISTORICAL_RUNTIME_IDENTITY_20261006.json)。融合配置来自既有提交 `81fb4e79332b4d07601e7218e7e69fa59dc480c2`。CPU测试只校验来源、固定配置、小型指标计算，不执行历史训练或访问正式数据。

历史训练入口仍按原脚本方式运行，部分使用同目录导入与动态导入，不保证任意 `python -m` 调用方式。环境、全部模型/数据闭包及历史打包入口还没有完成整体验收，不能把本页当作新机器部署完成证明。

独立发布包内五份改写源码保留为交付版本，不能用包内源码SHA替换训练源SHA。其余历史分析/绘图/数据准备工具仍待用途核对，本轮未删除任何文件。

EuroSAT 53,784张图像包的校验使用主线只读入口，不再依赖分析目录里的
`.codex_plot_deps`、机器私有 `IMAGE_EXPORT_LOCAL.json` 或自动重命名/解压：

```bash
python LightGenV2/demo_check/verify_eurosat_image_archive.py --archive /path/images.zip --export-report /path/SERVER_EXPORT_RESULT.json --split /path/SPLIT.json --image-manifest /path/IMAGE_MANIFEST.json
```

依赖 Pillow；校验 ZIP/每图文件及像素SHA、56×56 RGB、固定划分与CRC。
只向终端输出结果，不写报告、不解压、不替换 `.partial`。旧验证脚本和原报告保留；
本轮仅用合成小样本测试安全行为，没有重新读取整个正式图片包或复评模型。

旧纯光学 ZIP 的 SHA、13份文件清单、CRC与解压目录逐字节核验通过，见 [发布包身份](HISTORICAL_RELEASE_IDENTITY_20261006.json)。原训练提交为 `8c48e5caacb5cd2a8f82d18eaad41d5cb7a80d5f`，打包提交为 `558cfcebbbcc768345cf0a6dc3f909ae51495675`。当前main的模型已增加可选OEO/共用振幅入口，配置增加默认 `oeo_activation=none`，准备源码抽出 `decode_pair`；因此旧包不是当前源码逐字节副本。

原打包入口按原metadata的来源SHA校验；当前来源不匹配时应失败，不应改摘要绕过。此次只核验旧发布物完整性，没有重新构建、运行或上传包，没有重新证明当前扩展模型与旧版数值等价。

原包的冗余解压目录现已可恢复归档至本机 `archive/legacy_code_snapshots/eurosat_phase_only_delivery_20261006`，14份原文件逐SHA不变；原ZIP保留在原releases位置。见 [搬迁与恢复收据](../../maintenance/storage/EUROSAT_RELEASE_EXTRACTION_ARCHIVE_20261006.json)。不删除数据，不计新增磁盘释放；其他主机不必持有这份本地归档。

Kather源码包的 `- Copy.zip` 也已确认与正式ZIP逐字节重复并可恢复归档。原包70份manifest成员及CRC核验通过，正式ZIP与全部验证/传输记录原位保留；见 [重复包收据](../../maintenance/storage/KATHER_DUPLICATE_RELEASE_ARCHIVE_20261006.json)。没有移动训练工程、权重、数据或测速。
