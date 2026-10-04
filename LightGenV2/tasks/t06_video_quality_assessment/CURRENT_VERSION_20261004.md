# T06当前结果、实际源码与待收敛边界

此页整理已有版本身份，不重训、不重复数据集评估、不改变设备合同。以下两项已有
实拍，是彼此独立的模型，不能把Spatial和Temporal的PT或数据复用为同一项结果。

## 已验证的实拍版本

| 版本 | 固定原模型仿真SRCC | 未适配实拍SRCC | PT SHA256 |
| --- | ---: | ---: | --- |
| Spatial，单视频4帧，原1M读出 | 0.6710968960 | 0.5786364901 | `95e12397ccf8c960fa30ba9dfb400b69d2c2ebd02288ac6a4879870ab592828b` |
| Temporal，16视频×4帧，同场并行 | 0.8043868643 | 0.7977138739 | `5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c` |

两项源数据均为原558条test；空间3348张、时间210张六层有效CCD。原实验报告在
[空间实测](reports/reproduction/SPATIAL_SHS_20260914.md)和
[时间实测](reports/reproduction/TEMPORAL_SHS_20260914.md)。空间末端适配的不同划分、
混合test适配、重代入指标另见原读出报告，不能用不同分母的最高分替代上表。

## 代码及交付包已核验，但不冒称main已经完全兼容

两项实际离线推理包的源码commit为
`8e869473787f4ffceb2a6a77f4430b94c206f459`。各包15份运行Python与其内部SHA清单
及该固定Git源码全部一致；PT也与包声明及原服务器权重完全一致。2026-10-04在服务器
CPU上直接从既有ZIP读取源码和PT做strict reload，两模型均通过；不重新评估558条数据，
未写入/解压新的独立工程。

实拍包在训练服务器主工程 `LightGenV2/tasks/t06_video_quality_assessment/releases/`：

- `20260914_shs_spatial06710.zip`，SHA `fe90b0ac2cdedbe9b71f1d21c1f6cc5dcd6f2e237cefdd9bed1c9e94cde61d2c`。
- `20260914_shs_temporal08044.zip`，SHA `8421e7418ade4c9ec59f924cdad8e274db4100600c001cf802a8291ab7a67aff`。

源码从Git迁移、数据/PT从manifest+SHA管理；ZIP留作已验证旧部署与恢复证据，不把其代码
直接覆盖main或当前设备工程。七项运行文件仍与当前main不同，需连同训练/适配依赖逐项
兼容检查后才迁入，不能直接换PT冒称新入口已能复现。

## 历史默认profile的实际问题

Temporal-36是“单视频36帧”独立历史baseline，不是上述16视频×4帧实拍版本。
当前默认profile及部分Spatial旧profile锁定的源码SHA与多个实际工作目录不匹配。
部分权重只是移到别处，并未丢失：paired-flip Spatial best/last在服务器
`/DATA/DATA1/guest3/lightgen_spatial_067/promote_s1201/`，两项SHA核验通过。
Temporal-36旧默认路径的PT在本次限定名称搜索中未找到；不能据此判定被删除。

因此目前不得把默认 `--phase evaluate` 视为已验证的最终实拍复现入口，也不修改SHA
或取消检查来强行启动。正式实拍版本按已固定包及原报告读取；默认入口兼容性仍在收敛。

## 数据、baseline与测速保留

原数据/划分、缓存、best/last、全部原CCD/收据、逐视频结果、部署更新和诊断均不移动。
Temporal-36/9×4/16×4、Spatial不同电子容量、自研卷积及被否决的预训练上界各自保留
身份；上界仍标为不合规，不改为正式方案。全部历史A100/5090D测速和功率证据保留，
且只绑定当时的源码/PT/工作负载，不套给本页模型。

机器可读盘点在仓库 `maintenance/storage/T06_RUNTIME_IDENTITY_20261004.json`。
它是限定范围审计，不是全盘数据备份，也不是批准删除旧目录的清单。
