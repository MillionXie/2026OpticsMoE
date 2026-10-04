# Spatial 0.6710固定权重：SHS六层实测

这是固定光电学生的硬件前向评估，不是训练或微调。原电子残差/读出头保留；全部六次传播都用实测CCD替换，没有缺层仿真兜底。

## 唯一结果与身份

- 本地run：`runs/hardware/spatial_full_20260914`，入口`00_查看这里.md`。
- 最终师弟电脑会话：`E:/code/guest/2026OpticsMoE/LGVQ_Spatial_Lab_SHS_8um/sessions/spatial_lang1600_20260914`。
- 模型`spatial_readout_1m_srcc067`，checkpoint SHA256 `95e12397ccf8c960fa30ba9dfb400b69d2c2ebd02288ac6a4879870ab592828b`。
- 原离线推理包commit `8e869473787f4ffceb2a6a77f4430b94c206f459`；分层曝光增量包commit `3ef6ee7cc33218999119d8c5b5b34ec7e221c0ad`，ZIP SHA256 `deb1e88fe39d0b67cd5f301820d95b00cfc63a2f5a4ce2fa639b6c4eea2d61b2`。更新前验证原文件SHA，保留基础文件，记录`runtime_updates/installed.json`。
- 单视频4帧、每幅场输出一个Spatial MOS，完整原test划分558视频，3348张有效CCD。未挑选或剔除视频，未重新选模。
- 缓存保持训练时固定Spatial prompt与冻结Qwen前端；未删除文本、未添加Transformer或Attention。

| 指标 | 同权重仿真复评 | 六层实测 |
| --- | --- | --- |
| SRCC | 0.6710968960031272 | 0.5786364900822758 |
| KRCC | 0.48864788239622425 | 0.41208281220401133 |
| PLCC | 0.6883636086894079 | 0.6154004575783413 |
| RMSE | 8.243528366088867 | 8.9367036819458 |
| MAE | 6.587212085723877 | 7.207368850708008 |

历史训练记录SRCC为0.6710079009479295；表中是完整打包复评值，不混报。实测与复评相差0.0924604059个SRCC单位，未复现仿真性能。
本地`results.json`保存558个原始预测和标签，`independent_metric_check.json`用SciPy独立复算SRCC/KRCC/PLCC，与远端一致。

## 硬件与曝光调整

SHS-202-M相机1920×1080 Mono8、100fps、Gain_X4；Holoeye振幅SLM1920×1080、8μm；Meadowlark HDMI相位SLM1920×1200、8μm。
沿用20260914四角ROI及既定左右镜像、相位BMP方向与255-g反向。传播10cm、532nm，17μm模型的478有效孔径保持物理尺寸重采样到约1016硬件像素。
相位LUT仍为19x12_8bit_linearVoltage；不是重新标定的线性相位LUT。相位由本地SDK保持，师弟电脑RTX4060完成逐层电子处理和最终评估。

| 层 | 曝光μs | 输入网络的CCD倍率 | 实采p99范围 |
| --- | ---: | --- | --- |
| vision_router | 400 | 1/255 | 29–41 |
| vision_expert | 400 | 1/255 | 27–68 |
| vision_global | 400 | 1/255 | 17–27 |
| language_router | 1600 | 1/(255×4) | 19–19 |
| language_expert | 1600 | 1/(255×4) | 29–30 |
| language_global | 400 | 1/255 | 11–12 |

所有层240ms等待、持续排空旧帧；正式CCD仅478×478透视ROI PNG及记录，无TIFF、无逐张min-max拉伸。
第四层400μs两次被近暗底拦截（p99=7），输入有效区平均灰度0.395/255、非零约1.79%；1600μs诊断p99=19且无饱和，4000μs出现少量饱和。
第五层400μs部分采集p99=9、标准差约1.24，主动暂停；1600μs诊断p99=29、最大值231且无饱和后整层重采。
诊断和400μs第五层部分数据保留，不混入最终评估。倍率补偿基于曝光比例，不根据test标签或理论图拟合；它不是暗场扣除或完整光度标定。

## 继承与复现

原`spatial_full_20260914`前三层保持不动；新`spatial_lr1600_20260914`只改变第四层曝光。
随后`spatial_lang1600_20260914`只改变第五层曝光，继承前四层。继承必须验证每个继承层的有效硬件设置与读出尺度完全相同，并核对全部原图、输入、相位和上游SHA。
新记录封装附`inherited_acquisition`，保存原记录SHA、原硬件身份、来源会话；不能声称这些层重新采过。
暂停第五层遗留的锁在确认原PID23936不存在后清理，仅删除该锁文件，未删除图片或其他进程。

操作命令、环境与分层曝光命令见[COMMAND_SHS.md](../../hardware/COMMAND_SHS.md)。本轮每层实际命令保存在本地`*_launch.json`，实际配置为`LAB.lang1600.json`。
复算：在师弟电脑项目目录，使用同级ABO工程`.venv_gpu/Scripts/python.exe`执行`run.py evaluate --session spatial_lang1600_20260914 --device cuda --config LAB.lang1600.json`，不要重新capture或覆盖会话。

全量CCD、身份清单和结果ZIP位于最终会话同级：`spatial_lang1600_20260914_results.zip`，505,968,348字节，SHA256 `43e0c1670253c807c11a504d76f13c169739680cd051703d73bffac83bad3dd0`。
本地仅下载完整结果/身份清单和每层示例，不声称所有CCD已在本地。`measurement_audit.json`和`measurement_SHA256.json`记录全量校验。

## 限制

第六层信号仍偏弱；曝光补偿没有消除底噪、杂散光、相位响应或方向误差。没有本轮去光消融，也没有重新认证相位SLM实际响应/三者方向，因此不能由最终SRCC推出光学贡献或光场完全匹配。
没有测整机能耗；每张采集调用耗时之和也不等于包含生成BMP、诊断、换层和评估的总耗时。本轮未优化权重来追逐test指标。
