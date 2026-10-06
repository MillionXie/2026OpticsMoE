# T09旧图表与验证报告：保留位置核对

后续服务器定位结果见 [资产位置清单](T09_REPORT_ARTIFACT_LOCATIONS_20261005.json)：
八个引用目标位于原 `demo_reproduction_20260915` 工程的报告树，不在当前主工程
同名目录。六份具体文件已登记大小/SHA，两个目录仅核对存在，不冒称递归内容身份
全部验证。下文关于服务器位置待核对的文字保留为初次盘点时的历史状态。

主线129份任务Markdown的298个内联链接只读检查发现，T09复现说明中的五个历史
图表目录不在main提交树内。这不是源码丢失证明，也不是清理许可。

当前本地保留位置（均相对于仓库根）：

- `.worktrees/t12_cross_modal/LightGenV2/tasks/t09_multimodal_matching/reports/figures`
- `.codex_tmp/demo_reproduction_worktree/LightGenV2/tasks/t09_multimodal_matching/reports/figures`

核对目录为 `text_encoding_s17_20260917_v2`、`test_s17_20260917`、
`generalization_s17_20260917`、`phase_budget_s17_20260917`、
`audio_matching_s17_20260917`。每份副本各59文件，路径集合相同；20份逐字节相同，
其余39份JSON/SVG经CRLF→LF归一后相同。原字节仍不同，不冒称SHA相同。
未计算新预测、未修改图表/报告、未移动或删除副本。审计工具为
`maintenance/git_safety/audit_t09_report_assets.py`，输出当前每文件大小及SHA，
不是当时运行已有的身份凭据。

服务器主工程同名reports/figures位置当前不存在，不能把本地身份替代服务器资产
完整性证明。下一步仍需核对服务器实际保留位置和报告到原run的绑定；本轮不传输
资产、不重评TEST。上述本地副本涉及科学报告，解除历史工作树前必须继续保护。

Git链接检查只检查提交树中的内联目标，不检查锚点、外部URL、私有资产或运行导入。
OpenMoji私有handoff及T11/T16报告图片亦需单独资产核验；不能用链接检查通过宣称
全部复现完成。
