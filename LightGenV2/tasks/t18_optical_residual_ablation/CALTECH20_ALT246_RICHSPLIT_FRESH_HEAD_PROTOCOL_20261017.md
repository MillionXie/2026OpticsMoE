# Fresh六层同一线性头：自然样本比例目标

人工持续针对性优化授权范围内的一条有据策略，不改类别/划分/光学推理图、冻结无残差100及现有候选。

已完成freshKD完整100轮，按预登记VAL总体accuracy选E81，自己的科学和一次TEST/保存数学核验通过、生产和CPU均退出。新方案父固定该E81，不按本次测试重选epoch。TRAIN/VAL保存特征诊断显示均衡TRAIN目标与自然VAL梯度cos=-.53745，自然TRAIN目标cos=+.82965；仅TRAIN梯度虚拟探针VAL accuracy78.8512→79.8956，而原旧表示同诊断没有accuracy提高。因此不重复旧温度/旧头候选，针对新表示只训练同一5140参数线性头。

使用自然TRAIN均值CE、LS.05和相对父头anchor L2=.1；完整CPU LBFGS每外迭代max_iter1/strong-Wolfe，最多20迭代或参数完全不变停止、1800秒故障保护。raw float64头和优化器只存last，小历史只含头与预测；原光学与中间电子/alpha逐张量冻结，0光学forward/更新。每步按float32实际头保存TRAIN/VAL概率。选whole-prefix最高VAL总体accuracy、同分最低自然NLL、再最早头更新；含step0父模型，不按TEST挑step。新头若不胜父则原模型和TEST真实复用。

新来源为fresh240闭包加4叶文件=244。CPU独立证明真实父保存概率匹配<=3e-6、20类解析CE+anchor梯度及micro1vs20<=1e-10，0optimizer/TEST。新独占科学NumPy重算每步目标、softmax/CM/accuracy/macro/NLL、冻结全部非头状态、5140参数/同合同/选步，原始producer rc0实际退出后才一次新digest TEST，之后保存ID/label/prob/features/argmax/CM/support/NLL独立核验。旧digest用已核验结果，不能重推理。

测试GPU仅空闲非A1002/5/4/0一张、一卡一子、最多项目4卡，Torch2GiB+800MiB；当前248两训练不打断。源码仅Git，未同步到服务器前不执行；发布到公共GitHub的批准与私有GPU服务器执行范围区分，绝不绕过已拒绝的报告公共上传。所有旧source/PT/claims/失败保护，新输出唯一，不重试失败。不加新电子网络，不声称新同预算因果或独立泛化，教师/多次TEST开发影响保留。新科学通过也不等于6>4或10点目标成功。
