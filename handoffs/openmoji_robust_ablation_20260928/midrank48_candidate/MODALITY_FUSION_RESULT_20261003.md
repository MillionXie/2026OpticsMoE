# OpenMoji 本轮固定模态融合结果（2026-10-03）

本轮已完成，但没有满足用户要求的完整组合。所有数值为同一原始 TEST1000 的
Changed-cell Accuracy，不是整幅完全正确率。TEST 每5epoch选最高PT经用户授权，
属于开发结果，不是独立泛化验证；TEST没有进入梯度，未用VAL选模。

| 阶段 | Changed-cell | 保留格准确率 | 整幅完全正确 |
| --- | ---: | ---: | ---: |
| 原冻结权重正常CPU仿真 | .8935 | .986923 | .635 |
| 同权重六层真实CCD直接部署 | .1965 | .981463 | .104 |
| 原末端decoder微调后，严格CPU复载 | .7580 | .956587 | .362 |

原仿真在88–90%窗口内；直接部署下降69.70个百分点，不是期望的30–40个百分点。
微调后距原仿真仍低13.55个百分点（相对15.17%），未达到原仿真97%的.866695门槛。
改动格提升同时保留格下降，不能只报告.7580掩盖这一取舍。

## 精确模型及训练范围

- 初训候选 language55_vision80，editor48/originaldecoder64，共享头240664参数、decoder30162。
- 语言两块光占比alpha=.55，视觉两块=.80；在初训前固定，不是实拍微调时补加的层。
- 无新增CCD/DC/grid鲁棒训练trick，无适配时新增层/支路。
- 原PT SHA256 `d94dd0450edd415eee613ee28e424242239bf3dd1dd559cae57fe27d634b8c86`，初训选e40。
- 源码已发布Git `439018d2a9955cd4629993951742fbcbf3617752`，实际既有工作树/保护overlay及精确后端SHA见MODALITY_FUSION_20261003.md。
- 视觉router集中（最大pair=.994、审计不通过），语言通过；不能称路由完全均衡。
- 新TEST及独立TRAIN分别各1000场景×六层=6000CCD；两套每层1000PNG+1000收据，来源及ID与TEST零交叉。
- 末端微调TRAIN1000梯度，原TEST1000每5epoch开发选择，160epoch已完成，选e155；无VAL选择。
- 严格CPU缓存基线=.1965，best/last严格加载，保护上游哈希始终一致：
  `0a6552d348f5f1f5cccb999e807a2342df8bf9d4ef9b4339555b214cb1ff28b7`。
- 最后e160 TRAIN Changed-cell=.928、loss=.55645，不据本轮有限160epoch宣布绝对优化上限；也不自动继续无限TEST优化。

## 光学合同与有效性

2000us、GainX4、wait240，原ROI、phase hv/inverse、camera flip_v、保零tanh振幅和
round255BMP保持。理想六层接口桥接通过。TEST及TRAIN全部相位/收据/信号审计通过。
初次严格1%饱和pilot失败记录保留；按用户允许饱和，新sat15身份TEST接受15%饱和、
TRAIN采用既有95%几乎全白守卫，逐帧保存裁切比例，暗p99<15仍停止。实际TEST最大
饱和约1.451%、TRAIN约1.389%；没有修改曝光或故意制造低baseline。
有亮帧/收据通过不等于真实传播与理想传播形状一致，直接精度失配仍是失败结果。

## 已完成任务与证据

实验室既有工程：
`E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Rank64_SHS_20261002`

- TEST：`runs/modality_l55_v80_sat15_test1000`
- TRAIN：`runs/modality_l55_v80_sat15_train1000`
- 适配：`runs/modality_l55_v80_sat15_decoder_train1000_testselected`
- 顺序日志/状态：`runs/modality_l55_v80_sat15_pipeline.log` / `_pipeline_status.json`
- 一次性任务 `OpenMoji_Modality55_Sat15_Pipeline_1003` Ready、最后返回0；pipeline complete。
- 无本任务Python/SDK进程残留；bench4060回到319MiB桌面、0%利用率，两个初训3090已先前释放。
- execution.json保留开始时的running字段是历史入口记录，不覆盖为新事实；完整report/progress/strict_reload/任务返回0证明结束，不据旧字段重启。

本地已保留正常仿真初训、TEST/TRAIN实拍报告及审计；适配交付目录为
`modality_l55_v80_sat15_decoder_20261003/`，包括best、last、report、strict_reload、
execution、160epoch history、1000逐样本结果和progress，逐文件SHA见manifest.json。
原CCD/收据/缓存留实验室，不删除；旧ABO/G5/G2正式权重不覆盖。外部上传仍暂停。

本轮完整交付后停止对应监督，不重新采集、复活旧任务或启动新仿真候选。
