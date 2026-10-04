# T02 多版本身份：LSP结果与个人照片必须分开

2026-10-03只读核验，不训练、不操作设备、不删除。证据：
[资产SHA、划分与原报告](../../../../../maintenance/storage/T01_T02_ASSET_AUDIT_20261003.json)。

## 官方LSP：保留四类不同候选

服务器run根为 `/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t02_keypoint_detection/runs/simulation/`。

| run | 角色/协议 | 原报告仿真PCK@0.2 | 同权重去光 | best epoch |
| --- | --- | ---: | ---: | ---: |
| moe_router_scale_dc20_seed42 | 原始DC20公平对照主方法 | .5772857 | 本页不新增 | 100 |
| d2nn_matched_dc20_seed42 | 原始DC20普通D2NN对照 | .6751429 | 不适用 | 100 |
| refinement_20260909/staged_heatmap | 历史交付的低alpha最佳光学候选 | .7347857 | .7330 | 50 |
| alpha40_distill_seed42_20260911 | 后续高光占比热图蒸馏候选 | .7282857 | .7043571 | 5 |
| qwen_deconv40_seed42_20260909 | 完整冻结Qwen视觉、较小Deconv40头 | .6381429 | 不适用 | 37 |

低alpha候选的73.48%不能替换DC20公平对照表中的57.73%；Deconv40的63.81%
也不能冒充旧Deconv128完整头的72.17%。全部必要baseline和历史结果保留。
这些是原1000张LSP周期TEST上的仿真指标，TEST参与选模，不是独立泛化或实拍结果。

历史最佳best SHA：`495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518`。
alpha40蒸馏best SHA：`dbc059e2a7eddefac73d3b9bb158bf0140d440dfb396e0aa2956ad41670fc96a`；
两处实际alpha约.41806/.41805。两份PT均重新计算SHA，与各自最终报告一致，last同样保留。
报告声明的部署图不因训练教师热图而新增教师推理支路；未在本轮新机器严格加载整套模型。

官方划分CSV SHA：`2cb33989b3a5436408edad5e9e7452245814ec0724bc5ae20637cc40b4ba1d79`。
已核对DC20两组、alpha40蒸馏及Deconv40共享相同CSV：10428 TRAIN、1000 periodic_test，
11428个ID无重复，原图路径均存在；历史最佳CSV SHA也相同。原标注、缓存和数据全部保护。

2026-10-04补齐官方11428张原图逐文件SHA，总2,876,095,861字节，内容清单SHA
`f143ff286cd43000ad086f3ff0ca268a1c8a0ca3d19155d6289dac6797a59fc2`。
两份原始 `joints.mat` 匹配历史protocol中的SHA；五类官方run的10份best/last、
共享随机初始化PT、蒸馏教师及1.83GB教师热图缓存已核验。
[完整内容资产收据](../../../../../maintenance/storage/T02_CONTENT_ASSETS_20261004.json)
同时保留原配置/报告SHA，没有改写旧run或重新计算性能。

## 个人照片：不是正式ground-truth准确率

后来的 `personal_few10_*_pilot_s42_20260928` 是独立预标注小样本域迁移。
原报告明确 `formal_ground_truth_evaluation=false`，数值是伪标签一致性：
ours在94个人体样本上的PCK .74710→.76139，baseline .85165→.89276。
不能把94人体样本当94张官方LSP图，不能把这些pilot替代上面的1000图指标。
人工标注审核及各自分组划分仍需核验；pilot代码、原图、标签和best/last先全部保留。

后续用户筛选版本也不能遗漏：`personal_curated20_head_testselect100_pilot_s42_20260928`
是20张拟合、54张测试照片（55人体、654有效四肢点）的末端头续训；最高仍为起点
PCK12=.83333333，best_epoch=0，PT SHA
`d8287ddb8dc4ae8e06a502cc4037b1c459a9e5f4452f952dd1ef543f2b7e5c7a`。
原60轮头适配PT `0c57938c…1c2fe13` 和未适配Deconv128 baseline同批.87461774也保留。
用户是在看过预测后筛选照片，续训TEST又参与选模，故均为开发期预标注一致性，
不是官方LSP真值或独立泛化。不能用.8333替换上表官方.7283或.7348。
原101张和筛选74张工作图均匹配各自标注中的图片SHA，标注和筛选/映射JSON身份已记录；
未把 `reviewed` 标记改成人工确认，未公开照片、坐标或拍摄元数据。

## 实际代码在哪里，为什么不能立即删旧工程

后续alpha/蒸馏/个人图片的完整工程为服务器
`/DATA/DATA1/guest3/t02_personal_source_20260928`，HEAD
`2fa901dc7800d8798658b29e242e38bebac97989`，本任务已跟踪源码无未提交修改。
alpha40蒸馏原run记录训练commit `8de890f316e3c58d50eff2e3a900a676c2d6ad68`。
既有工程CPU测试39项通过（包含T01合同）；没有新训练、完整评估或GPU占用。

其refine/personal/figure入口静态闭包66项。后续源码按上述固定commit逐项SHA核验，
缺失入口、配置、测试及历史协议以缺项追加方式收敛到main；`modeling.py` 的alpha架构标识
和 `run.py` 的四个训练入口按实际运行版补齐。清单见
[`T02_PINNED_SOURCE_IMPORT_20261003.json`](../../../../../maintenance/storage/T02_PINNED_SOURCE_IMPORT_20261003.json)。
这不代表原运行目录已切换，也没有重训/重新测指标。原工程继续保留；PT、缓存、图片等外部资产
仍需各自清单管理，不能删除个人工程或直接把新PT塞进旧入口。
旧110.3万头baseline、73.48%交付、alpha50/global_noise必要对照及所有测速继续原位保护。

前端路径注意：旧DC20配置写的 `cache/qwen` 现在不存在，但已发布解析器会在工程祖先目录
回退到 `/DATA/DATA1/guest3/.cache/huggingface/hub`；已只读验证实际解析到T01收据所核验的
`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` snapshot。不是模型丢失，也不修改旧配置补造历史。
新复现应显式使用已校验的不可变snapshot；历史训练revision尚不能倒推证明。
本轮只补内容身份，动态运行与完整测速绑定仍未全部核定。
资产只读复核：`python maintenance/storage/check_t02_assets.py`；冻结前端另用
`check_t01_assets.py` 核对同一snapshot。两者不运行任务入口、不加载模型、不写原run。
