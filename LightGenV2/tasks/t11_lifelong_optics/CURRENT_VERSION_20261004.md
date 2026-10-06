# T11 当前源码、架构与资产入口

2026-10-04治理记录。T11是病理分类/终身学习与必要对照，不是T16多模态终身学习。
此次不重训、不重评TEST、不修改光学几何或选模含义。

## 架构不能混用

| 实现 | 目的和合同 | 实际服务器来源 |
| --- | --- | --- |
| `model.py:OpticalMoE` | 纯光、固定槽位、相干叠加；旧专家冻结，router/global可塑；无电子分类头/OEO | `t11_optical_20260919`，HEAD `604cd89e3e4a88ed0b580eaa41f88af7de2da86c` |
| `model.py:OpticalD2NN` 与joint/continual入口 | 固定容量两相位面；联合训练、顺序replay、冻结迁移分别是不同baseline | `t11_joint_d2nn_20260920`，HEAD `1430e8331739b0fd640c23d3adba244991408abc` |
| `crc9_model.py:CRC9Optics` | CRC九分类，两个OEO及相同小MLP；MoE后续冻结旧专家和global，D2NN适配只更新读出 | `t11_crc9_distill_20260921`，HEAD `72180033c6edd605c7151b2faa8a599dc15910ee` |

最新服务器源码包含上述旧协议入口与新增对照实现，所以采用精确CRC9 HEAD作完整源码
基线，不是把CRC9架构当作所有历史任务的模型。每次仍须按原run配置选对应入口和PT。
源码经Git恢复包取回、SHA审核；root原本地模型/测试文件不覆盖。

## 已有结果口径

纯光病理四域协议历史结果（验证集BA、seed17，不是实拍或独立测试）如下；完整阶段、
训练预算、BWT及权重身份见 [原复现报告](reports/reproduction/README.md)：

| 协议 | 最终四域平均BA |
| --- | ---: |
| MoE少量replay | 77.43% |
| D2NN少量replay | 77.88% |
| MoE全量replay | 80.02% |
| D2NN全量replay | 77.42% |
| MoE离线联合 | 79.43% |
| D2NN离线联合 | 77.98% |

不同预算/数据访问权限不能作为同预算架构比较。CRC9结果与这些二分类BA分开记录，
不把其九分类或合成域结果混入本表，也不因某试验分数高便改称正式最终版。

2026-10-06补核CRC9实际四份NPZ：SHA与原运行记录相同；每域5026 TRAIN／718 VAL／
1436 TEST，九类均存在，各域内部三划分图像身份互斥。四域对应划分的ID顺序及标签
逐位相同，是原始／染色／扫描变换的同一批图像，不是四个独立病理数据集，也不能把
四域TEST相加称5744张独立测试图像。该口径与上表早期四数据集二分类协议分开。
只解码ID／标签并读图像数组头，未重建变换、解码图片或重评模型；证据见
[`CRC9划分核验`](../../../maintenance/storage/T11_CRC9_SPLIT_IDENTITY_20261006.json)。

## 数据与权重保持原位

- 纯光run：`/DATA/DATA1/guest3/t11_optical_20260919/LightGenV2/tasks/t11_lifelong_optics/runs/`。
- 必要baseline：`/DATA/DATA1/guest3/t11_joint_d2nn_20260920/LightGenV2/tasks/t11_lifelong_optics/runs/`。
- CRC9所有正式/试验结果：`/DATA/DATA1/guest3/t11_crc9_runs/`。
- 数据、来源/划分manifest、best/last、预测、原日志和所有测速继续保留。
  stale `status=training`不等于仍在跑；未完成run不冒充最终结果。
- 只读资产身份清单：仓库 `maintenance/storage/T11_ASSET_IDENTITY_20261004.json`。
  清单仅记录选定正式记录及best/last的SHA，不代表已复制备份全部原数据。

## 验证与复现前检查

55项源码/配置/协议从精确服务器Git树纳入main，PNG报告图仍保留原位置，不为整理重复
提交数据。31份Python可编译；服务器原源码18项CPU合同测试通过。候选main同源码的
CPU测试工具支持从Git blob直接加载，避免创建新的工程副本或覆盖用户工作目录。
本地现有Python的torch DLL初始化失败，未修改该环境。服务器既有CPU环境已直接从
发布树 `ba0effba9ab867206d95c1c0292246cc8d347d70` 读取31份Python，18项合同测试通过。
原任务目录没有切换或覆盖。第一次Git加载工具的namespace错误已修正并复测；不把
工具错误归为模型失败。发布收据见仓库 `maintenance/storage/T11_PUBLICATION_20261004.md`。

CRC9测试历史使用 `tasks.*` 导入，纯光测试使用 `LightGenV2.tasks.*`，因此正常磁盘运行
要从仓库根执行并设置 `PYTHONPATH=<仓库>/LightGenV2`，不能把少设路径误判模型错误。

运行前核对配置中的架构、几何、类别数、checkpoint、数据manifest和原run source commit。
整理时仅检查代码合同，不再次评价科学数据；重训必须另获授权并使用新run ID。

源码发布与资产收尾是两件事：三套旧运行目录仍原样保留，未做删除；历史时间不套给
CRC9或新的main入口。将来解除目录依赖前仍需确认实际进程及本地独有内容。
