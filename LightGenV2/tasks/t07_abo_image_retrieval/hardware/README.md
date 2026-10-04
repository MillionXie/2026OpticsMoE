# 封存 rank72：六层离线回放核心

本目录的 `replay.replay_batch` 是从已验证 Windows 最终流程提取的计算核心，
不导入相机、DVP、SLM 或旧独立工程，不启动设备，也不默认评估800查询。
封存权重仍是 SHA256
`25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22`。
它保留视觉和语言的 router→expert→global 六次注入、原块内alpha和
rank72 RGB patch旁路的末端图像token .5/.5混合。

调用方必须提供严格加载并置为CPU/eval的模型、原processor产生的batch、
对应唯一sample IDs，以及 `capture_stage(stage, bounded_active_amplitude, ids)`。
回调返回 `(CPU CCD tensor, receipt, amplitude_scale)`，CCD必须保持原线性强度，
不得按帧归一化；实际振幅的保零有界编码使用原 `standalone.bounded_export`。
回调不由本模块创建，设备独占、收据/相位/曝光/增益/ROI/身份校验仍由上层负责。

## 已核验与未完成边界

- 原Windows计算主体逐句AST核对：只移除路径、processor及设备全局包装，
  capture调用改为显式回调；六层计算和旁路混合逻辑不变。
- 封存PT在训练服务器从候选Git树严格CPU加载；一个合成输入的六层理想CCD桥接
  描述子最大误差为0，两次回放逐位相同。没有重评原图或查询集。
- 原 `abo_full_query_flow_snapshot_20260928.py` 的旧 `main()` 未采用：其CUDA、
  可选模拟图库和默认曝光不能被当作最终实拍入口。
- **尚未完整迁移逐层采集/断点恢复runner。**不能仅有此模块就删旧工程或宣称
  main已经可替换实验室正式入口。原14400有效CCD、收据、权重和结果仍原样保留。

源码身份审计（不需要私有恢复对象，不打开设备）：

```bash
python maintenance/git_safety/check_t07_replay_source.py --commit main
```

维护机持有原恢复记录时加 `--audit-archive`，另核对原计算图。
合成桥接需已核验的私有封存PT和兼容PyTorch环境；不需要原图/CCD：

```bash
python maintenance/git_safety/check_t07_hardware_replay.py --repository /path/to/2026OpticsMoE --commit main --checkpoint /path/to/rank72_epoch13_best_snapshot.pt
```

来源及边界见 `maintenance/storage/T07_OFFLINE_REPLAY_SOURCE_20261004.json`。
这不是训练或正式检索命令，也不改变最终版指标。
