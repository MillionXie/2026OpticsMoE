# 四层82.48%：只开放最后global

原四层同一头继续128步已独立审核通过，但仍选父step0、339/411；新错误池201个不同TRAIN近邻。当前72张TEST错图56张更近别类TRAIN，提示仅调整线性边界可能有限。新候选继续从4aef3d12...82.48%的精确状态开始，只开放第四物理层global的phase_sets.9、electronic.3.local.weight、fusion_logits.3及同一linear头五张量，前三层、router、中间电子与其他alpha字节冻结。

复用已发布六层末global训练与独立核验的纯函数内核；没有复制或新增模型实现。独立CPU明确核四层stage_slices[3]=(9,10)、EG×2和72错例/201近邻，不能套六层phase14/prefix5；实际GPU4自己核微批梯度和冻结范围。更新64步、每8步TEST、batch8近邻+8回放、TRAIN余弦正负例约束及系数校准均同尾层协议，学习率phase.0002/末电.00008/alpha.00001、EMA.9。仅best/last，保留父step0，若未严格胜339不替换。

TEST知情开发、原数据角色不变、额外预算、非独立泛化/等预算因果；二层、无残差和实验室均不改。所有准入/来源/producer原始退出/独占科学/实际保存ID预测/最后Adam/EMA边界必须真passed，失败保留不自动重试。六层正在运行的源字节保持不变。
