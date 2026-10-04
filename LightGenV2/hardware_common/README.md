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

## SHS 高速相机：已归并的控制依赖

`shs/` 收录实际实验室 SHS 路径使用的四项控制依赖：`sdk.py`、
`capture.py`、`phase_hdmi.py`、`slm_camera.py`。来源和恢复身份见
`maintenance/storage/SHS_SHARED_CONTROLLER_SOURCE_20261004.json`。
相机解码与相位显示源码不改数值逻辑；相对导入改为包内导入，振幅驱动统一调用
已核验的 `experiments.hardware_sdk.devices`，不再动态寻找邻近的 `vendor_driver.py`。

`Controller(config, config_base=...)` 必须显式提供机器配置所在目录；SDK、LUT、
ROI 和原始机器配置保持在实验室，不进 Git，也不能将相对路径默认解释成本包目录。
导入这些模块不会打开设备。其连续排队丢帧、原位相机配置恢复、原生相位尺寸与
SHA 检查保留；这不是触发同步保证，也不更改任何任务的曝光或方向合同。

纯 CPU 合同测试：

```bash
python -m pytest LightGenV2/hardware_common/tests/test_shs_contract.py experiments/hardware_sdk/tests/test_hardware_sdk.py -q
```

本次归并仅完成控制依赖，**不表示 main 已可完整接管 ABO/OpenMoji 实拍**。
任务的六层 runner、几何、数据前端、精确权重和断点收据仍须按各任务核验。
实验室原入口及其有效采集保持原样；不要为整理而启动设备或重拍已有数据。

发布源码身份检查不要求旧工作树或私有恢复引用，可在普通 main clone 中执行：

```bash
python maintenance/git_safety/check_shs_source_identity.py --commit main
```

持有原始恢复记录的维护机器可额外加 `--audit-archive`，核验归并前源码。
这两种检查均不打开设备；普通检查只证明发布文件与已登记 SHA 一致，
不声称重新核验了实验室原文件、机器资产或整套采集流程。
