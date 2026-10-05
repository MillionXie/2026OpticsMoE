# EuroSAT共享冻结前端对照：按run绑定原源码

2026-10-05只读实际服务器 `/DATA/DATA1/guest3/demo_reproduction_20260915`。
18份当前shared_frontend源码/配置/说明已核SHA；三个原run仍存在，各有三份best PT，
共9份PT完整读取核SHA，没有载入模型、执行训练/评估或删除产物。
完整私有收据：`.codex_tmp/shared_frontend_run_binding_20261005.json`。

| 原run（LightGenV2/demo_check/runs/simulation下） | 运行记录源码commit | 来源绑定数 | 当前文件与记录相同 |
| --- | --- | ---: | ---: |
| eurosat_shared_frontend_20260916 | 8cb125cfaa4829e7714b5b23389f0c96e4844a72 | 8 | 4 |
| eurosat_shared_frontend_continuation_20260916 | ef70e50f802302a900e321be5f5188a258631990 | 8 | 4 |
| eurosat_frozen_cnn_20260916 | 81fb4e79332b4d07601e7218e7e69fa59dc480c2 | 7 | 4 |

三组共23项记录源码均能在各自原commit中找到，内容SHA与原metadata一致。
角谱后端 `adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py`
在Git中保留CRLF；必须按原字节核SHA
`b250627b14357a254eb12c785caf20f3aff2bf3fe30121a05321094803c73a7d`。
将输出转LF会得到另一SHA，不能据此误判源码丢失或修改。

## 用途与迁移边界

共享前端组复用单独预训练的冻结CNN，移除分类头后将128维特征编码进光场，
MoE/D2NN输出探测能量分类；没有额外输出电子残差。frozen_cnn组则是原电子/光学
输出0.5/0.5融合对照，不是同一个计算图。低学习率续训也是独立run，不覆盖原20轮。
源CNN best SHA `281523a2a62525ddf491b5937c1eaf4b2e13bbfbf76e8c4652e2bc79e28ad4fc`
当前仍与共享前端配置声明一致。

不能把当前工作文件直接配旧PT称历史精度复现。正式主线迁入应保留配置对应计算图，
并核验当前模型对原PT的兼容性；必要时使用原commit而非当前HEAD。此处23项只是
run记录的源码范围，不代表所有动态依赖、原图、环境及测速闭包均已证明。
数据、9份best、last、报告、旧测速以及本地独有OEO代码全部原位保留；不替换现用实验。
