# Hardware common

通用硬件能力当前由 `experiments/hardware_sdk` 提供，包括：

- Meadowlark 1024×1024、17 µm 振幅 SLM；
- 1920×1200、8 µm 相位 SLM；
- TUCam CCD；
- LUT、曝光、双 SLM 对齐、菲涅尔和 CCD 单应性标定；
- 文件夹播放、采集和 478×478 canonical warp。

本目录先定义共享边界，不复制 SDK。任务的播放顺序、相位权重和逐层微调属于任务
自己的 `hardware/`。迁移时必须用设备回归测试证明与旧 SDK 行为一致后，才移动
driver 源码。
