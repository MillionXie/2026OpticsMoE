# Caltech筛选20类当前四层：完整100轮与独立科学最终结果

两组完整100/maximum_epochs、无early；各生产子原始exit_code0并实际退出，queue_exit原始rc0、CPU campaign/queue实际退出。rootstate caltech_current4_complete_science_passed，active0/pending0/fail0/test0，GPU2/5项目0/A1000。自己的独占审核第一次真实passed，audit_completed SHA实际匹配，无任何failure。

| 指标 | 有光学残差rho.3 | 无光学残差rho0 | 残差差值 |
|---|---:|---:|---:|
| 训练准确率 | 78.0612% | 61.3703% | +16.6910pp |
| 验证准确率 | 62.1908% | 55.1237% | +7.0671pp |
| 验证宏平均召回 | 59.7451% | 51.8116% | +7.9335pp |
| 验证balanced NLL（越低越好） | 1.6403502369 | 1.9969947965 | -0.3566445596 |
| 训练-验证准确率差 | 15.8704pp | 6.2466pp | — |
| 按首次最低验证EMA balanced NLL选中轮次 | 100 | 100 | 同100预算 |

验证准确数176/283对156/283，净多20正确，领先7.0671个百分点；macro领先7.9335个百分点。本次已有明确正向配对差距，但残差验证62.1908%仍未80；两个原严格开发精度门槛都未通过，不把科学真实性审核passed等同精度验收成功。单seed17，没有跨种子稳定性保证。

## 收敛和薄弱类别

末十轮验证NLL均值对前十轮仍改善：残差3.2333%、对照2.0789%，两组按NLL均选E100，说明100轮末仍学习，不能宣称充分收敛。残差训练78.0612/验证62.1908，gap15.8704pp；对照61.3703/55.1237，gap6.2466pp，残差拟合更强同时泛化差更大，不能只归为未收敛或认为过拟合已解决。

残差螃蟹10%、雨伞18.18%，对照雨伞0%并有持续零召回，不能只挑高差距类别掩盖弱类。验证每类仅10–20张，单类一张正确就变化5–10个百分点；逐类差异是当前保存结果，不等于新类别组合训练成绩。

## 全部逐类验证召回

| 类别 | 验证支持 | 有残差 | 无残差 | 差值pp |
|---|---:|---:|---:|---:|
| Faces | 20 | 85.00% | 80.00% | +5.00 |
| Leopards | 20 | 65.00% | 65.00% | +0.00 |
| Motorbikes | 20 | 100.00% | 95.00% | +5.00 |
| airplanes | 20 | 70.00% | 65.00% | +5.00 |
| bonsai | 19 | 63.16% | 47.37% | +15.79 |
| crab | 10 | 10.00% | 20.00% | -10.00 |
| crayfish | 10 | 50.00% | 30.00% | +20.00 |
| grand_piano | 14 | 85.71% | 85.71% | +0.00 |
| helicopter | 13 | 53.85% | 30.77% | +23.08 |
| kangaroo | 12 | 33.33% | 8.33% | +25.00 |
| ketch | 17 | 52.94% | 76.47% | -23.53 |
| laptop | 12 | 58.33% | 50.00% | +8.33 |
| lotus | 10 | 40.00% | 40.00% | +0.00 |
| minaret | 11 | 72.73% | 90.91% | -18.18 |
| revolver | 12 | 66.67% | 50.00% | +16.67 |
| schooner | 10 | 60.00% | 30.00% | +30.00 |
| trilobite | 12 | 75.00% | 41.67% | +33.33 |
| umbrella | 11 | 18.18% | 0.00% | +18.18 |
| watch | 20 | 45.00% | 40.00% | +5.00 |
| yin_yang | 10 | 90.00% | 90.00% | +0.00 |

[逐类训练/验证支持和召回CSV](class_recall.csv)。

## 来源、预算和审核身份

run `caltech20_screened_current4_s17_pair100_input100_mixup02_gpu25_20261015`，实际训练Git `763e6a242a3fe27a513044dc6a6282a0ef574b7d`。155登记来源current/original逐项核一致；独立审核额外引用audit_resisc20_input_pair.py不在155里，完成后已另核当前字节及训练Git763 blob均SHA9bee138135b1cfb5f616208629f541aebf4021a96325e32e561aa7d4e5a4013b。不篡改训练登记/claims，不冒称155包含它。

原固定Caltech20/cache/划分不变，train1372/val283/test2064仅metadata；本次未读test图。旧Caltech测试已观察，不声称独立未见任务泛化。当前四224专家、四主层E/G/E/G+router、908552相位、478global/inputRGB100/8bit/OEO/固定20窗口、无电子可训练分类支路。两组同seed17/init/order/affineMix，共同MixUp.2/LS.08/TV.05/capture.2/EMA.95/lr.002，只rho不同，macro16/micro8/每E86Adam/尾12，每臂8600更新。

拟合秒残差826.86363、对照834.40766；含必要端点评估总秒829.56753/837.16959，两卡并行约14分钟。均在10800秒安全限内，没有预算延长或重试。

自己的独占保存科学audit SHA `588b5a7046f4f7e4ca6349459d9515874c153a6c31e2a943e48203bed54508a0`，actual_audit_returncode0/completed真实一致。独立核source/config/rawEMA/PT/shapeinit/all11grad/8600Adam/orderaffineMix/LR/预算/保存IDlabelargmax20CMsupportmacroNLL/firstminimum，两CPUmodel只shape/seed、0forward/test。

残差selected PT `ffa25f8597aa400aaf65ec0d2357ec2803b6e059dfe2ac9c639558546e86699e`，history `cdc7fa257667cc555afb60c57f1ed56d095f22924454bfbb94508154b36c5de5`。
对照selected PT `28a1ba0f3170751c072ccc5b936e57a6d852283964c9fa657f5188b4dfa380c5`，history `d579adef87fd4b72db1038076e221980878e7bdd0244febc9ebaea4e98821a8b`。

## 本轮结束边界

本次授权的一对拟合及独占科学已完成。差距是验证+7.0671pp，不是测试差距，旧九专家的测试+7.95不能套成本次成绩。没有自动test/额外优化/深度/改类/电子支路/A100/native/formal6/lab或图作者。所有旧source/PT/data/history/失败/claims/markers保留，rawJSON/完整日志/逐样本ID/PT私有。保留t18与每日09晨报ACTIVE，本次完成通知一次，之后正常完成状态静默。
