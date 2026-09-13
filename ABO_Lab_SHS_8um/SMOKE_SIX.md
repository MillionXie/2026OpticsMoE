# 六层小样本实测排障

用途：验证真实振幅下发 → 相位换层 → SHS取图 → CUDA电子处理，逐层走完六层。
不借用理论CCD；不把小样本分数作为正式准确率；不代替 `dual_run.py` 的正式参考库验收。

1. 两台电脑的相机/SLM GUI关闭。本地运行相位 SDK，师弟电脑运行振幅、相机和RTX4060。
2. 必须先用实测振幅标记得到坐标映射，并用未参与拟合的不对称图确认；把证据报告SHA写入诊断配置。
3. 诊断配置另存师弟工程 `results/smoke_configs/smoke_<日期>.json`，不改 `LAB.local.json`。
   `diagnostic_only=true`，`diagnostic_session=smoke_<日期>`，`geometry_evidence.method=measured_markers_with_asymmetric_check`。
   `geometry_confirmed=true` 在此仅表示实测映射已通过**诊断**检查，必须在证据里声明尚未正式标定。
   固定曝光、显式增益、[0,255]存储尺度；相位翻转与反灰度必须基于当前设备实测，不能照搬。
4. 本地独立link配置中设置 `phase_startup_cycles=0`、`phase_retry_cycles=2`、
   `phase_display_align_top=true`、`phase_settle_s=1`。临时对齐只改变相位屏Y位置，结束恢复。
   曝光须按真实网络输入检查，不能只按满亮棋盘格决定。如果较高曝光让探针饱和，
   可在配置指定经实测的 `diagnostic_probe_bmp`（results下的原尺寸振幅BMP）及
   `diagnostic_probe_sha256`，例如减小棋盘格亮块灰度。只改变排障探针，绝不改变网络输入或CCD存储尺度。
5. 本地运行（Python需numpy/Pillow/paramiko/pywin32；相位SDK和LUT路径沿用本机link配置）：

```powershell
python smoke_six.py --link-config results/smoke_configs/link.json `
  --remote-config results/smoke_configs/smoke_<日期>.json `
  --limit 4 --out results/smoke_runs/smoke_<日期>
```

密码从交互输入或 `SHS_SSH_PASSWORD` 环境变量读取，勿写入配置/文档。
一次执行生成、采集六层并评估；只允许新会话，已有数据不覆盖。

六层依次为 vision_router、vision_expert、vision_global、language_router、language_expert、language_global。
可在新诊断配置声明 `diagnostic_query_indices: [0,600,1200,1800]`，按固定间隔取4张，
避免默认前4张恰好同一类别；索引必须在采集前确定，不能依据检索结果挑样本。
4张查询的采集量依次为4/4/4/104/104/104，共324张，后3层包含全部100个标题。
每层另拍平相位挑战、目标相位两次重复、采后复查；PCC和亮度漂移不通过即停止，不继续推理。
这些检查证明响应改变和重复稳定，**并不独立证明完整灰度—相位曲线正确**。

仅为了排查六层数据流，可显式加 `--brightness-warning-only`：PCC仍需≥0.97，
不足信号/饱和/相机设置改变仍停止，但>20%的整体亮度漂移单独标为警告，
结果 `photometric_stability_passed=false`，**不能据此批准正式实验**。
若只因这项亮度漂移停下，且该层所有图已采完并保存采后PCC，可加 `--resume`
从下一层继续；保留原停止报告和协议变更记录，不修改已采像素。其他不完整层不允许这样跳过。
正式 `dual_run.py` 的参考库、亮度门限和审批规则不受此诊断选项影响。

`generated/smoke_<日期>/` 是独立相位/标定BMP目录，`sessions/smoke_<日期>/` 保留
真实canonical CCD、身份/文件SHA、路由与最终结果。默认只保留478×478 PNG及必要JSON，
24张全幅排障探针另外保存在results，不为每张正式输入保存大幅原图。
结果中 `production_qualified=false`；若光路尚未精调，低分应保留如实报告，不能筛选样本提高分数。
