# 剩余本机历史目录：先保护独有内容，再逐用途收敛

## 已核验的范围

2026-10-05 对四套尚有改动的历史工作目录作只读盘点，逐文件与已发布
`main` 提交 `070618689f2f2b2260fdfd69311bc17cdd43e067` 比较。
随后将这些修改／未跟踪／忽略内容另存私有恢复包：所有成员 CRC、SHA256 和原文件
再次核验通过，四目录 Git 状态不变。没有切换 checkout、取消分支或删除文件。

| 原目录 | 固定 HEAD | 已保护文件数 | 用途与后续边界 |
| --- | --- | ---: | --- |
| `.worktrees/t12_cross_modal` | `11ca2e7e388a7dd07be5614d07027afd769c9315` | 165 | T14 冻结共享读出、T15 CCD 窗口读出及 T16 路由图等历史协议／对照。不是图文编辑 T12；保留必要对照，不把失败几何候选当最终版。 |
| `.worktrees/t04-qwen` | `ccf6630ba5c4540dbb966617f179a99a30743c39` | 177 | OpenMoji 旧语义场景／Qwen baseline 与论文选图报告；有本机模型、配置和测试差异，须先核对应服务器版本。 |
| `.codex_tmp/demo_reproduction_worktree` | `1c7222a1119385475e1b464ca7efdb67cdc21a4c` | 373 | 复现 demo 的纯光学模型、配置和共享前端差异。两份 T09 源码与 main 完全一致，其余内容不能据此一并删除。 |
| `.worktrees/t04_openmoji_robust_20260928` | `9d6b24f01b877d4f4594adbedde65be3502ab44d` | 34 | 本机历史 robust 文档及缓存；两份受扰评估源码与 main 一致，README 仍不同。实际实验台 rank64 工程不在此处，也未改变。 |

合计保护 749 份现存文件，包含源码、报告、结果图及字节码缓存；这**不是 749 份独有
源码**。忽略文件和未跟踪文件没有被当作垃圾，包内没有混入任何其他 checkout 内容。
已提交历史仍由原 Git 对象／分支保留，本次恢复包保护的是额外工作内容，不替代完整
Git 历史，更不证明其他未纳入范围的数据已全部审计。

## 私有恢复证据

逐文件身份：`.codex_tmp/remaining_local_overlays_20261005.json`。
四包及复核收据：`.codex_tmp/remaining_local_overlay_preservation_20261005/receipt.json`。
这些私有产物不提交 Git、不传外部服务器，不覆盖原文件。

| 包 | SHA256 |
| --- | --- |
| overlay_0.zip | `e5166c2480fdb1d6de4c21c436502ac7a8ae0293fd5bb4147ac38ad53ef41c69` |
| overlay_1.zip | `eb4b9d9cd482980ff341dc59845eedb0968ff4663973eafe7d912581894ae23a` |
| overlay_2.zip | `f96c2eaf64f16288478237785699644e7001f6ab2af89b0fcfd810a54c0ac375` |
| overlay_3.zip | `ad5d69855b2f3e61c20cf882f5c45f5b7c01863d2afeee76ad0f1e772dd1844e` |

## 尚未完成

### 服务器来源补核（本日后续）

实际冻结运行目录 `/DATA/DATA1/guest3/t12_cross_modal_20260920` 仍为
`2f2f6946f38c7d06320eb29314654a6260f3d618`。本机 T14／T15 的22份 Python／Markdown
与该目录当前字节比较，只有2份忽略CRLF后相同。进一步逐个检查服务器83个已登记
checkout 的同路径17份 Python，只有1份找到完全相同的LF字节；其余16份未匹配。
这是限定路径、限定已登记目录的核查，不证明服务器整个文件系统不存在其他副本。
因此不能将这批本机 overlay 冒称服务器最终版，也不能用它覆盖已发布最终T16。

main 中 T14 目前明确只保留最终T16实际导入的 `data.py`／包入口与导航说明，
不是将旧T14整套训练协议标为已经迁入主线。T15 README 明确为暂停重设计、被否决
几何候选的历史诊断；其中 checkpoint 不得填最终矩阵。原始对照、未匹配本机代码
和图表仍在原路径、恢复包与Git历史保留。私有只读收据为
`.codex_tmp/lifelong_history_runtime_compare_20261005.json` 和
`.codex_tmp/lifelong_source_matching_checkouts_20261005.json`。

这些结论收紧了版本选择边界，不把文件名相似或时间较新当作可合并依据；此处未进行
训练、测试集重评或模型替换。

1. 按协议核对 T14／T15 与服务器的源码差异，确定必要对照的正式导航和依赖；备份
   并不等于已经进入主线或可以清退原目录。
2. Qwen 旧 baseline 模型／配置差异与现有报告逐项绑定，不覆盖当前 OpenMoji。
3. demo 的三份差异继续按模型合同核验，不能仅凭文件较新选用。
4. 三端已同步与实际 checkout 已统一分开核验：本批主线 070618 已发布到 GitHub、
   训练服务器引用相同且原运行 HEAD／索引／修改不变；实验台 SSH 连接超时，尚未
   导入本批 Git bundle。主目录没有切 main，整体迁移未完成。

全部测速仍原位保留；本批释放空间为零，不增加历史清理释放量。

## Qwen 分层场景来源进一步定位

核对83个服务器已登记目录后，找到已有
`/DATA/DATA1/guest3/LightGenV2_worktrees/t04_layered_1bc120428`，HEAD为
`a61a3746d39765991048d95e40812d1028d70c93`。8份本机目标中5份与该目录忽略CRLF后
逐字节相同；modeling、settings及论文选定报告仍未匹配，不能整体覆盖main。

实际run不在该源码目录，而在服务器主目录的T04 `runs/simulation` 中。
已有完整报告的Qwen baseline为epoch30／修改格0.8120；电子expansion0.5的100轮
方案best为epoch70／0.9365。它们都是历史仿真开发指标，不是当前rank64实拍。
原论文选定0.8715又是另一种固定选定口径，不能用100轮best冒充。
另一个30轮paper目录虽有best/last，但缺标准训练完成报告、选中PT测试报告及逐样本
文件，暂不能将其中PT与论文选定0.8715绑定。所有现存文件保留，未重新跑TEST。

三套run的原提交、best/last与报告／逐样本SHA已登记在
[分层场景baseline身份](T04_LAYERED_BASELINE_IDENTITY_20261005.json)。
当前源码目录HEAD不能倒填为旧run的源码身份；同类名称不等于同一模型、PT或测速。
