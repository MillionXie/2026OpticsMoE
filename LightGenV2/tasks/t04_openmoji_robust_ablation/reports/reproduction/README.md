# OpenMoji 当前复现入口

当前版本身份、指标和物理合同只维护在[任务 README](../../README.md)。
操作源码尚未完整合入本目录；不把“已有入口”说成“已完成源码统一”。

原已验证复现证据：

- [rank64 G5 最终权重、strict reload、冻结参数与原数据位置](../../../../../handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/FINAL_RANK64_VERSION_20261002.md)。
- [rank64 G2 TRAIN2000、最佳/last及校准权重SHA](../../../../../handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/G2_RANK64_TRAIN2000_RESULT_20261002.md)。
- [rank48 历史两路线记录](../../../../../handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/TWO_ROUTES_20261002.md)，仅作重要容量对照，不覆盖 rank64 结果。

复现所需：精确代码/overlay、配置、原G2/G5 PT与适配 PT、OpenMoji素材许可、
源身份互斥 TRAIN/TEST manifests、词嵌入/视觉前端、各层真实CCD与收据及设备合同。
代码通过 Git 同步，数据和 PT 通过 manifest+SHA；不上传凭据，不把暗帧或旧mask数据混入。
