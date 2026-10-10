# T18光残差的算子表述

router没有残差：`model.py`仅选择expert_bank/global_fcs的PhaseLayer。
六层主干是三组专家变换与global变换，额外router独立；六层共30张调制面有残差。

为避免将系统写成单张相位乘光场，以U表示一段的输入，T_theta表示该段
可学习光学调制/模式变换，P表示后续传播与几何裁剪，B=P∘I表示同位置
未调制但同样传播的支路，N表示后续非线性。对专家和global分别有：

    R_(alpha,theta)(U)=(1-alpha) P(T_theta(U))+alpha B(U)
    H_(l+1)=N_l(R_(alpha_l,theta_l)(H_l))

此式仅当混合发生在非线性之前且该段P为共同线性传播时与当前代码等价。
未调制不等于未传播，P不得默认为I。alpha是相干振幅混合系数，不是功率占比。
专家fanout/路由权重/复光场拼接算子记为S_(g)：

    V_l=N_l^E(S_(g)({R_(alpha_l^E,theta_(lk))(F_k(H_l))}_k))
    H_(l+1)=N_l^G(R_(alpha_l^G,theta_l^G)(V_l))

F_k是含权重振幅的relay输入装载，S是空间场拼接/必要叠加，不能写成
分类结果或光强的任意加权平均。当前g由router一次计算，并在三个周期复用。
这些式子是相位面局部残差，不声称整个MoE块有一条不受影响的身份通道。

若讨论真实Kerr器件，可以在未来协议将N替换为经验证的N_Kerr；当前T18使用
强度探测→非仿射LayerNorm→ReLU→Softsign→零相位振幅重编码的OEO。
Kerr与OEO非线性不能在文稿中混称；未调制泄漏更不能在经过Kerr之后假定仍保持
输入，因为Kerr响应本身依赖光强，合束位置会影响计算图。

传播影响大小必须由合同和证据确认；本协议5cm角谱传播参与空间混合与探测窗
收光，不能仅因它是线性算子就忽略。外部LLM映射与抗噪收益尚未由T18验证。
