# 主工程已切换至统一主线（2026-10-05）

本次只处理开发入口和历史保全，不改变模型、实验协议或运行中的实验。

- 主目录 `C:/Users/Xml12/OneDrive/2026OpticsMoE` 当前实际分支为 `main`，切换目标 commit 为 `4a5a8bd73f57309934af14c650f8a39599755cbd`。
- 238 个统一主线路径已落实到工作目录。18 份原有修改及 7 处不同内容未覆盖；1295 个旧历史路径仍在原位。未删除实验数据、权重或测速。
- 原主目录历史保存于 `refs/archive/root-before-main-20261005`，完整历史 bundle SHA256 为 `e413e7681063f6d4de49461876e68a6a8a10cae4445289954baa9988b6e870ac`。
- 切换前 1385 个文件的恢复包已逐文件核验，SHA256 为 `5fb0f8acf3d6e355c15fa6204573c5de5991d2b78a8f0b49571a0837db4bba37`。私有收据及恢复包位于 `.codex_tmp/main_checkout_transition_preservation_20261005/`，不进入 Git。
- 旧 T12 根目录分支名已退出；旧 `t12_cross_modal` 工作树退出开发分支，5411 个文件及其原修改保持不变，历史保存在 `refs/archive/frozen-local-20261005/t12-cross-modal-lifelong-overlay`。目录尚未删除。
- 旧 `.worktrees/t04-qwen` 对照工作树也已退出开发分支：核验 4680 个文件及原修改保持不变，未检测到本机进程引用该目录；历史保存于 `refs/archive/frozen-local-20261005/t04-qwen-baseline-overlay`，完整历史 bundle SHA256 为 `706942a3ea46cc708284569de1c11a703ea05667af8bfa104ad92e29747cf645`，独有文件恢复包 SHA256 为 `eb4b9d9cd482980ff341dc59845eedb0968ff4663973eafe7d912581894ae23a`。必要对照与全部测速仍在原位，不因退出分支而失效。
- 旧 `.codex_tmp/demo_reproduction_worktree` 已退出开发分支，5392 个文件及原修改前后身份一致；完整历史保存于 `refs/archive/frozen-local-20261005/demo-reproduction-overlay`，bundle SHA256 为 `bf41bf489dc1802494e14b1e7d9646a767ec4b4f99a5cc2f906d188c604c9121`。独有内容恢复包 SHA256 为 `f96c2eaf64f16288478237785699644e7001f6ab2af89b0fcfd810a54c0ac375`。未删除目录或文件，独有 OEO / 共享前端差异仍待语义审计。
- 本地开发分支现为 3 个：`main`、受保护的 OpenMoji robust 分支、历史 Qwen5090 报告分支；这不代表工作树、服务器运行目录或所有未提交修改已收敛。

## 尚未完成

后续已将共享 optical_contract 依赖及 T12 配置接口测试后纳入主线，并将 T11 model/test_contract 的旧文件与已审计主线补齐；原文件恢复包仍保留。T11 两份源码逐字节 LF 比较与已发布树一致；本机 PyTorch DLL 导入失败，但服务器从精确 `dab476e80e23b72082f1408bb4ae4b353fade964` Git 树编译 31 份源码、18 项 CPU 合同测试全部通过，无 GPU 或新 checkout。

T12 的本地 lazy import 和窄 Qwen-style 层说明已审计：配置导入不加载 Torch，公开模型导出保持原身份，2 项隔离测试通过；共享合同的 3 项测试也通过。说明明确自定义窄层不是原预训练 Qwen decoder 层，不修改计算图或正式 PT。随本次提交归主线后，主目录剩 12 个实际源码/文档差异待审计。

后续 T12 的 audited_unified、CCD bridge、训练数据标注、sealed loader 与合同测试五份旧文件，以及两份旧 README 已补齐为精确已发布主线。原文件仍有恢复包；主线中的 17M 与 9.96M、不同 alpha 下限和 baseline/测速历史分别保留，不用一个版本口径覆盖其他版本。服务器从 `a5521baf933bbccb467276c38c1d30fba3adc27b` Git blob 运行 12 项 CPU 合同测试全部通过，未读取科学数据、加载正式 PT、使用 GPU 或改运行 checkout。主目录现在剩 5 处真实差异：pure_optical/config、旧 profiler、ABO retrieval_adapt/retrieval_refine/robust_training。以上不是全工程验收完成。

Git 索引的换行/文件状态噪声已按逐文件 HEAD/index/归一化内容三者相同的条件刷新：96 条 tracked 修改提示降至 16 条真实差异，工作文件未改写、无 staged 内容变化、未设置忽略修改标记。随后 T11 两处补齐，剩 14 条。这不是删除文件或压低真实差异计数。

服务器及实验室运行目录尚未统一切换；已发布 Git 引用同步不等于运行目录切换。其余历史目录必须在确认无运行占用、独有代码和下游依赖，并保有可恢复备份后再退出。全部测速、必要 baseline、最终模型及有效数据继续保留。

因此本次是主目录统一入口的实际落地，不是整个治理任务验收完成。

旧 narrow profiler 的独有单任务、均值统计与短协议选项已保留；ABO 的 77-token/linear64 与 20260927 的 71-token/spatial2x2 两种历史身份改为显式选择，默认维持原主线口径。环境报告记录所选身份并明确没有测试封存 rank72；默认任务集合不新增反向检索。无 Torch/GPU 的合成接口测试检查两种 token/head/kernel 身份及未知身份拒绝，不重跑任何历史测速。此项源码提交不意味着服务器运行目录已切换。

ABO 的 retrieval_adapt、retrieval_refine、robust_training 三份旧源码也已逐文件补齐为已发布 main 的内容。原旧版缺少新版 TRAIN 留出、rank72 与训练辅助接口，不能与新版其他文件拼接使用；旧源码仍在切换前恢复包及历史引用中，不删除历史实验语义。此次仅修复主目录源码一致性，未修改封存权重、数据、相位或实验室运行目录，也未训练或评估查询集。主目录剩余两处实际源码差异为旧测速 profiler 和 pure_optical/config，仍需分别审计；全部测速记录保持原位。
