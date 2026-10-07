# Baseline 代码交接包

本页只保存2026-09-15的六项历史baseline，不是六个日常开发工程。正式任务从仓库
`START_HERE.md`进入，按任务页区分最新模型与历史对照；尤其OpenMoji本包.8420不是
后续layered应用baseline或rank64 robust baseline，不能互套指标。

原ZIP及其中源码保持SHA不变；主线保存方法、动作、数据/PT身份和逐文件清单。
主线中的六个包装启动器已增加嵌套Git创建权限检查，默认只读核验不会创建工程。
这不是重新训练、评估或生成交付包。源码payload和原ZIP仍是独立保留的历史资产，
fresh clone需要原SHA校验资产或历史Git恢复对象，不以当前main冒充原提交。

1. [01 视频质量评价 · LGVQ](01_lgvq/README.md) · [ZIP](01_lgvq.zip)
2. [02 商品检索 · ABO 图搜文](02_abo_image_text/README.md) · [ZIP](02_abo_image_text.zip)
3. [03 商品检索 · ABO 图搜图](03_abo_image_image/README.md) · [ZIP](03_abo_image_image.zip)
4. [04 关键点检测 · LSP](04_lsp/README.md) · [ZIP](04_lsp.zip)
5. [05 显著性分析 · SALICON](05_salicon/README.md) · [ZIP](05_salicon.zip)
6. [06 语义交互 · OpenMoji](06_openmoji/README.md) · [ZIP](06_openmoji.zip)
