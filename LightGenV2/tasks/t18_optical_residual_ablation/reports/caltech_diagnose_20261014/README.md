## 最新部分阶段：前三组40轮已完成并CPU审核

基线/关闭量化/整面LN三组完整40，均val NLL选E40，train/val分别33.75/28.98%、
34.69/30.39%、33.82/32.16%；关闭量化仅小幅改善且仍7类零召回，整面LN减少零召回至1类，
但macro和NLL更差、小gap不代表成功正则。独立CPU保存审核和真实PNG/SVG/CSV已完成、图已实际视检。
[三组共同预算结果与逐类图](first3_stage/README.md)。
原三子实际退出；GPU2 RGB、GPU6 dense、GPU5 preserve_global正常接续，剩MixUp.2/.6与旧9专家待派。
未完成全九屏障、无test/部分正式锁、121源冻结不改。

## 最新执行：仅GPU2/5/6、完整状态接续

人工明确指定物理2/5/6，覆盖下方初始6/2/1名单；本项目GPU1子已退出，其他项目不动。
旧CPU父3054100和三子已实际退出，base/unquantized完整E17、fullfieldLN完整E12状态保留。
新CPU父3077108已实际在新run `caltech_selected20_s17_L4rho03_regression_diagnose40_gpu256_20261014`
接续：GPU6 base3077640、GPU2 unquantized3077832、GPU5 fullfieldLN3078020；PID是启动历史，后续实时核对。

[原进程退出及保存检查点身份](execution_handoff_gpu256.json)、[CPU完整恢复检查](gpu256_cpu_restore_checks.json)、
[九项新实际GPU预检](gpu256_numerical_preflight.json)、[新启动及三GPU真实恢复门槛](gpu256_continuation_start.json)。
三组raw/EMA/Adam逐项完全相同，原best保留参与选模，完成轮次不重播；fixed LR无scheduler。
三张实际GPU恢复门槛repeatprob0、所有相位梯度有限非零；121运行来源冻结e4cb1400fdaed5d4a1867770474e91b0ddc19617。
旧prefix来源582c90a1c保留，模型/光学/科学配置完全不改，仅执行入口/audit/GPUplan变动。
诊断总40轮/7200s含prefix，失败保留，无test或正式训练启动。

持续监控已更新只允许2/5/6；另设每日北京时间09:00实际结果报告，报告进度/失败也不编造满意结果或保证75%。
下面的旧GPU1与source582初始运行描述仅作原阶段证据。

# 2026-10-14 回归消融与新类别准备

人工要求先诊断精度退步，再开始正式训练；四层残差单独检查过拟合，逐类监控。
旧同数据100轮残差：2层train57.80%/val51.59%，4层train76.90%/val61.84%；当前部署版150轮2层36.22/36.04%、4层57.22/46.64%。不是同一架构或正则的受控比较。
[旧真实保存指标](previous_fit_reference.json)。旧4层80%以上印象不套成这次已核实的76.90%。

九项main4/rho.3共同40轮消融固定旧20类，分别检查量化、行归一化、RGB保留、Top2、global场重编码、MixUp.2/.6；另含当前基线和旧九专家整套参考。已真实启动CPU父3054100，初GPU6 A100基线子3054519、GPU2不量化子3054710、GPU1整面LN子3054898；随后以current_processes与实际proc/UUID为准，不能重复启动。
只train/val，逐类support/recall/CM；无test数组和推理，不自动正式训练、不无限候选。量化关闭仅定位，不可直接作为部署候选。
[明确预算、差异与正式门槛](../../CALTECH_DIAGNOSE_PROTOCOL_20261014.md)。

新正式20类遵照人工排除Faces/Motorbikes/airplanes，保留其余17类，只用train/val固定特征选择3个替补。
类别改变与消融分开；新结果仍为开发实验，不冒称独立泛化。不会因未来test失败继续删类别。
A100本轮人工允许；最多三项目子、每子2GiB+800MiB，GPU6/2优先，第三卡从允许0/1/4的余量选择。其他项目原样保留；最终真实退出后核对显存释放。
[只读资源盘点](resource_inventory.json)。不连接实验室，不改封存ABO/旧训练源/结果。

## 实际启动与数值门槛

九项门槛全部真实passed，20类样本均检查，重复micro1概率误差0，各相位梯度有限非零；八个当前图变体完全同initialSHA，旧九专家889696参数另有身份。基线与封存部署版908552参数前向/状态精确相同。
[真实数值检查](numerical_preflight.json)、[早期消融接口失败留存](preflight_interface_failures.json)。RGB通道与旧模型class配置接口错误已修正并Git发布；旧失败目录不覆盖，不把适配错误当成旧低精度原因。
[实际CPU父/三GPU子命令、来源和早期live证明](production_start.json)。121源码当前/原训练582c90a1c08b1e24a84df32b9d0f7ab04b66c210 Gitblob一致。实际初期base/unquantized E2、fullfieldLN E1，所有生产smoke通过，test0。
实测初期普通轮base54.4s/unquantized53.3s，GPU1共享负载首轮含全train107.4s；预计九组共同40轮约2-3小时，估计受负载影响，不是结果或结束承诺。

新类已冻结：car_side/joshua_tree/stop_sign替补，保留其他17；1937图/train1246/val261/test430。旧保留类1295个train/val像素及角色逐项相同；以前fit/val进入新test为0。
[新数据身份、支持、固定筛选和独立CPU审核](revised_data_cpu_receipt.json)。固定缩略图proxy宏召回52.66%仅筛选指标，不是光学精度。缓存SHA21436ad0b269d5319176e3e4d40fd382b21e1d34d41345ec979176231fb0e3c0，manifestSHA017dff61d36bb5161ac9442cd22d8436b15cf68a963273f797229d624868f30b。
正式训练仍未启动；先根据消融检查信息损失/优化，并独立评估4层残差正则与逐类失效，再冻结正式方案。旧test不复测，不拿test选候选。

## 接续后的独立CPU核验与结构线索

接续真实base/unquantized E20、fullfieldLN E15保存记录已独立CPU核验；
原17/17/12轮history/gradients/order/transforms前缀逐项完全相同，Adam1720/1720/1290步，
全部相位梯度有限非零；独立重生全部截至当前order/affine/MixUp hash，包含宏尾12。
[实际保存审核](gpu256_partial_cpu_audit.json)，没有模型前向或test数组。

旧3层rho.3获选E150模型仅router组件，用每类首5个TRAIN样本（共100）检查：
100个均选底部专家2/3，未随输入改变。不是全train统计，更没有test重推。
[真实训练样本router组件探针及原PT身份](trained_L3_router_probe.json)。
另用零raw相位、均匀振幅的末端expert合成组件检查六种Top2组合：
选择底部2/3时，上半分类窗能量极低，最大/最小分类窗能量比约5.23e5。
[无数据、CPU合成几何探针](synthetic_router_detector_geometry.json)。

这两项说明路由固定化及末端expert/分类窗能量偏置值得优先排查，
不能直接证明完整已训练模型的因果失败、不能把合成能量当准确率。
现有dense_router共同40轮消融会验证去除硬Top2后的训练/验证变化；
3层正式方案还必须单独确认末端布局，不允许仅换类别掩盖此结构问题。
后续正式验收增加路由按样本变化/各槽位占用及逐类持续失效检查，
既要诊断欠拟合，也要4层残差独立控制泛化差距，不能只看train提高。

七个物理量化变体的fresh synthetic真实native BMP写盘/读取/裁剪/逆映射检查已全部通过：
模拟与native BMP回放概率误差均0、黑输入各层振幅严格0、五次捕获正确，无额外峰值拉亮。
[七种真实设备尺寸BMP合同检查](variant_native_bmp_checks.json)。这不是已训练候选的最终部署验收或实拍。
若preserve_global被选中，其global输入478，需要独立符合此合同的导出入口，
旧224-only exporter及旧rowLN/Top2合同不能直接套用其他变体。

## 固定旧权重的路由干预：不能直接切推理开关

对旧3层rho.3获选E150相同参数，只在每类首5个TRAIN样本（100个、每类等支持）
对照原hardTop2与dense4，参数SHA严格相同，没有重训、选模或test数据。
原路径subset accuracy/macro60%/60%、bNLL1.76751；直接切dense4为9%/9%、bNLL3.06231，
原Faces/Leopards/Motorbikes/airplanes/bonsai五类仍为0，不能称直接推理切换已修复。
[同权重组件路径干预完整收据](sameweight_L3_routing_intervention.json)。
这说明旧权重依赖其已训练的传播路径；不能据此否定fresh dense训练，也不能将路由固定化
单独认定为完整精度退步的唯一原因。当前匹配fresh40轮dense消融仍必需，
最终选择需要完整train/val、逐类健康及物理一致性，不能拿此100训练样本结果替代。
