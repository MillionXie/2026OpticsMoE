# T12 正式版本与必要对照（安全收敛记录）

本页是2026-10-03治理新增的版本入口，不改变模型或历史指标。服务器实际运行目录
为 `.worktrees/t12_physical_robust_v2_20260927`，源码 commit
`7093ec46082eed2fae127ec5028d3e2e8548b592`。任务是RGB图像加文字指令的商品场景编辑，
不是T08的图文检索，也不等于开放域文生图。

## 采用哪个版本

17.026642M鲁棒小版及其仅末端电子decoder微调版均保留。149.755866M大版、
Qwen28匹配baseline与pix2pix-Turbo baseline是必要对照，不能作为中间垃圾删除。
旧9.96M小版及其测速仍保留，但旧速度不得套用于17M版本。

统一TEST2304、原生256×256，逐图PSNR再平均；下表引用已完成结果，本次没有重新推理。

| 权重/用途 | 干净仿真 PSNR / SSIM | 实拍 PSNR / SSIM |
| --- | --- | --- |
| 原17M鲁棒权重，5b4f9a37…27cac | 34.277751 / .927140 | 27.551289 / .876325 |
| 仅decoder微调权重，eeeca764…97a14 | 28.890612 / .903006 | 31.552886 / .906013 |
| 大版，2a91d812…0c5b5 | 31.428599 / .882211 | 本页无实拍证据 |
| Qwen28匹配baseline，599805ad…b52 | 27.260453 / .837653 | 本页无实拍证据 |
| pix2pix-Turbo匹配baseline，3a347c58…4c7c | 20.746391 / .740299 | 本页无实拍证据 |

原权重与适配权重不能拼成同权重对照。微调权重的干净仿真下降，且原报告保留
validation warm-start限制；语言专家平衡守卫未通过，不宣称已实现路由均衡。

## 源码、权重和数据在哪里

main只接入上述固定Git版本的最小源码依赖闭包、5个已测试文件与历史README，
不是整分支合并。闭包及逐文件SHA见
[T12源码清单](../../../maintenance/storage/T12_SOURCE_IMPORT_20261003.json)。
本链接应从仓库根读取：`maintenance/storage/T12_SOURCE_IMPORT_20261003.json`。
原README中的旧工作树/分支操作建议属于历史记录，现行根AGENTS禁止自动新增分支/工作树。

实际服务器任务目录：
`/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t12_physical_robust_v2_20260927/LightGenV2/tasks/t12_text_to_image`。
原17M采用 `runs/simulation/20260927_language_balance_input/last_checkpoint.pt`，
而非该轮训练器best；Qwen28与pix2pix分别在
`runs/simulation/20260927_matched_qwen28_baseline`、
`runs/simulation/20260927_pix2pix_turbo_matched`，best与last都留存。
上述原17M、两项baseline的best/last共5份PT已在本次按SHA重新核验。
微调PT及大版PT的SHA本次仅引用原交付记录，未宣称再次完整验证。

本地证据：`handoffs/t12_lab_robust17m_20260927`、
`handoffs/t12_four_group_summary_20260927`及原发布包。原始CCD/收据、数据、
PT、逐样本指标和所有测速原位保留，不进Git。Windows原实验工程保留不切换。

## 验证边界与尚未完成

固定服务器源码的CPU合同32项通过（6.08秒），无GPU、采集或重训。
发布前核验共享源码/YAML与main完全一致；现有不同本地文件未覆盖。
这不是全套资产已经从零复现：外部Qwen/VAE/PCA资产、官方pix2pix vendor、
完整数据/CCD身份审计、Windows硬件工程和全部旧测速依赖仍需逐项梳理。
现有本地目录也尚未切到main；请勿直接以旧分支里的同名文件冒充main发布内容。
