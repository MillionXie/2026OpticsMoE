# 历史测速源码：依赖闭合与保护收据

本轮补齐主线缺失的两份历史 A100 测速脚本，以及其中 PositionReadout 的原始模块。
来源为服务器实际源码保全提交 `43bf2f1d214c252b48e8104ef5816643ce6a7f4f`；
逐文件 SHA 见 [纳入清单](TIMING_PROFILER_ADDITIONS_20261004.json)。
主线候选 `353fb63cab87bcd7eae50642ba320862ade27390` 在服务器已有环境中，
直接从 Git 树内存加载，未创建工作树、未覆盖源码。

## 实际检查

- 三份新增源文件逐字节匹配实际服务器身份并编译通过。
- 两份 profiler 及所需模块从候选 Git 树成功导入；没有运行 profiler main。
- CPU 合成数据检查 Top-2 四能量路由、原 OpenMoji 神经头、位置线性读出。
- 原头输出形状为 `[1,17,6,6]`、`[1,6,6]`、`[1,4]`，数值有限。
- 测速 RAW 数组始终为空；没有数据评估、训练、CUDA 使用或 SLM/CCD 操作。
- 并未替换 T04 modeling/settings，未改变现用 OpenMoji 工作树或任何 PT。

初步历史 Git 树静态扫描列出五份 T04 缺失模块，但其中四份仅为旧训练分支依赖。
按实际主线 CPU 导入验证，本轮测速只需其中 embedding_model.py 的 PositionReadout。
未把旧整套训练实现覆盖为新 OpenMoji 最终版；该任务的正式源码迁移仍单独审计。

## 原报告与测速边界

`LightGenV2/reports/20260914_latest_optical_electronics_a100_NARROW_CLEAN_FINAL`
原 SHA 清单中 10 项全部通过本地逐字节核验，包括约 24.8MB 原始逐次计时、
约 2.7MB 功率采样、CSV 汇总、报告、源码和合并脚本。原文件未修改或删除。
其中 profiler SHA `71ebc1e…dce100` 与本次服务器源码一致。

该 narrow 报告量的是 GPU 已驻留、预成形输入的电子算子，使用 pooled CUDA-event
median；并行残差与 bridge 单列，不含搬运、文件 I/O、SLM 排布或 RMS/alpha 融合。
功率来自 10ms nvidia-smi board power 采样。不能称为整套光路端到端实測。
另一个 latest profiler 口径更宽，包含 post-CCD/fusion/reload；两口径不能混为一个数字。

服务器 profiler 的旧 ABO 图搜图 head 是 linear64、77 language tokens；本地独有版本
改为 spatial2x2_64/71 tokens/V5 并增加测试选项。双方源码均保留，未用服务器旧测速
代替 rank72 最终模型的速度。最终模型若没有独立测速证据，指标继续留空。
目前 main 的部分任务模块已迁移，因此本轮 CPU 兼容检查不等于在当前 main 重现旧
GPU 测速；复现旧数值必须使用原报告源码身份、环境、配置与对应模型。

私有原始 CPU 检查收据保存在 `.codex_tmp/t11_source_20261004/timing_profiler_candidate_cpu_20261004.json`。
源码恢复包 SHA 与恢复引用见 [原保全记录](TIMING_SOURCE_RECOVERY_20261004.json)，其
`pending`/`published=false` 是保全当时快照，以本页后续依赖检查为覆盖说明。
本轮不删除任何测速文件及重复副本，不重新启动实验，不恢复外部上传。
