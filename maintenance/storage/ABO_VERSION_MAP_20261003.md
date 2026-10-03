# ABO 三种检索任务的保留与迁移边界

2026-10-03 用户确认：rank72只是图搜图；文搜图、图搜文同样必须整理。
本页只核对现有证据，不训练、重评、删除或指定未经核验的新模型。
“文搜图”是检索已有图片，不是T12的生成/编辑图像。

| 方向 | 现有工程及数据协议 | 已有结果与身份 | 本轮处理 |
| --- | --- | --- | --- |
| 图搜图 | T07；200个已登记SKU，1600图库+800查询；六层14400有效CCD | 用户封存rank72，第13轮；仿真R@1 .83375、去光 .8000、未微调实拍 .81125；PT SHA `25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22` | main采用简洁最终入口的方向已获批准；旧历史/对照完整保留，训练与Windows部署源码仍需依赖闭包审计 |
| 图搜文 | T08；easy100，4800 TRAIN/2400 TEST图像，100个官方标题候选 | 2026-09-07性能优先仿真R@1 .79875；强均衡 .7983333333，二者不能按较新日期相互覆盖 | 已有main源码与报告保留；核验服务器实际版本、PT及数据SHA后再声明完整交付 |
| 文搜图 | T08反向独立工程；100个官方标题查询，在2400张TEST图片中检索 | 10cm/alpha约.40主体PT `cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae`；仿真 .86、去光 .73、未微调实拍 .79；用户2026-09-26采用10epoch读出适配，实拍 .85，读出PT `89e25360c8f12c9d13dbceba49e8d81073d4dd75c6acf667ab12a38b8fd293c5` | 主体+末端读出须成对保存；不能直接使用旧图搜文入口，不能只换PT |

## 文搜图的另一版本不能混入上述实拍链

2026-09-28九任务A100计时报告另锁定10cm文搜图PT
`8a96132d8ba68c671426d4e0b6064e87b5b3943505c0ad8b2aa694c879c20883`，
仿真R@1 .8800，窄范围电子计时1.4004ms+公式物理6.2682ms=7.6686ms。
它不是上述cc977b83主体，不将其.8800与另一主体实拍.85相减，不将公式计时
冒称端到端实验台实测。部署与完整源代码身份待核验；此版本也保护，不擅自取消。

原文搜图主体实际训练源码来自`t08_text_to_image_20260920`工作树；
已保存run manifest记录Git `94856741263742c14fa1851ca0bdac7fc5efc7a1`。
独立Windows部署在`E:/code/guest/2026OpticsMoE/ABO_T2I_10cm_alpha040_20260925`。
其six-stage TEST、TRAIN5100 CCD、best/last、预测与配置不属于可清理的试错垃圾。
原始版本记录来自既有交接文件；2026-10-03进一步完成下述有界身份核验。

## 本轮实际核验（2026-10-03）

- 服务器两组图搜文 best/last 及 final_report/run_manifest 已逐文件重算 SHA，
  与报告记录一致；原运行提交分别为 `b44bfe3d...`、`46d35fc2...`，保留两组。
- 服务器文搜图主体 best 的 SHA 为 `cc977b83...`，与本机交接主体相同；其
  last `1c60ff8e...` 也保留。easy100 的 train/test/titles/manifest 四份 CSV
  与原报告 SHA 全部一致；没有重算全体原图，也未重评科学精度。
- 本机文搜图10轮读出 best `89e25360...`、last `ec338134...`、report/history
  已核验；Windows 实拍目录本轮未重哈希，不能称三端全部资产已验证。
- 原训练 manifest 记录 dirty。当前服务器反向工作树 HEAD `d0662a7d...`
  的入口 Git blob、工作文件与本机交接源码都为 SHA `ebf026e5...`，任务已跟踪
  源码无修改；这补足入口身份，不表示46个静态依赖和动态入口已完整迁移。
- 静态依赖审计发现3处现有文件不同：反向 `optical_moe.py`、T01架构名称生成、
  robust后端的显式传播距离覆盖。共享文件未覆盖，反向工程尚不能直接用main部署。
- 旧 config 所指工作树下数据目录已不存在，但主目录的数据清单SHA匹配。
  后续应核验可移植路径解析，不能仅复制旧绝对路径声称可运行。
- 服务器既有图搜文CPU合同6项通过（2.43s）；本机PyTorch DLL加载失败，
  不算通过。本轮没有训练、光路操作、精度重评、数据删除或外部上传。

完整精确身份见[T08核验记录](../../LightGenV2/tasks/t08_abo_image_text_retrieval/reports/reproduction/FINAL_IDENTITIES_20261003.json)。

## 核对用的原证据

- 图搜图：[T07封存入口](../../LightGenV2/tasks/t07_abo_image_retrieval/README.md)。
- 图搜文：[T08任务页](../../LightGenV2/tasks/t08_abo_image_text_retrieval/README.md)、
  [原始汇总](../../LightGenV2/tasks/t08_abo_image_text_retrieval/reports/optical_router_moe_20260907/summary.json)。
- 文搜图实拍：本机私有`handoffs/abo_text_to_image_10cm_alpha040_20260925/README.md`、
  `SELECTED_PHYSICAL_RESULT.md`、`FINETUNE_TRAIN800.md`及`finetune10/`原报告。
  它们不进入公开Git，发布源码不能要求新clone默认持有私有文件。
- 文搜图计时：[九任务计时说明](../../LightGenV2/reports/20260928_nine_task_a100_table/README.md)，
  该目录尚未完成主线来源迁移；不能用链接存在代替同步完成。

## 后续收敛顺序

1. T07简洁最终入口不等于“整个ABO只有一个版本”；分别完善以上三个方向。
2. 先核对实际运行源码、未提交overlay、数据划分、主体/适配PT的逐文件SHA。
   同一T08名称下正反方向的实现不同，不能整目录覆盖或整分支合并。
3. 源码审计、依赖测试通过后进入唯一main；数据/PT/CCD保留原目录，以manifest+SHA登记。
4. 确认被替代中间代码无运行、独有内容和下游依赖，并备份可恢复之后才清理。
5. 各方向TEST选择规则按原批准记录，开发指标不称独立泛化；未测性能/时间留空。
