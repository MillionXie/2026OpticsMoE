# FFA 实验记录

## 2026-10-11：初始化与审查

- 用户授权创建唯一顶级 `FFA` 目录，讨论后直接尝试，不建分支或工作树。
- 完整对话含五组问答；保留用户对预训练意义、当前 BP/PAT 公平对照和物理反馈来源的要求。
- 本地、服务器主目录与 GitHub 起始提交为 `1b771a0a718c4b02ea7348f9dd33add6141e6a16`，分支为 main。
- 本地既存 OpenMoji/handoffs/输出未跟踪文件保留；服务器主目录起始已跟踪修改为零。
- GPU 1 的 FishNet 已有计算进程，不使用；GPU 6 A100 初检无计算进程，按 UUID 指定。
- 根 AGENTS.md、FixedFeedbackSFT README/MIGRATION、P11/P12 实现和任务协议均已核对。
- 三位子代理负责架构/训练审计、数学及文献审查、强度接口实现；路径互不重叠，Git 事务由主代理串行执行。
- 旧 LSP phase-only common start 未保留，LSP 已适配 best 不作为共同起点。P11 source 与 frozen stem 仍可使用。
- 本轮采用新共同头起点和独立 run 身份。配置为 `pilot.json`，只做验证集初筛。
- 本机 PyTorch c10.dll 加载失败；源码静态语法检查通过，数值测试在服务器已有 xml 环境执行。
- 实现保留物理相位映射的当前控制链式因子；冻结的是完整光学强度响应，不是全部电子计算。

## 审查结论与争论

数学代理验证 `gφ=+2a Im(Ha)`；任意非酉传播也必须用共轭伴随、不能用逆算子。
实现代理将自定义梯度边界移动到真实强度量，保留 token 物理排列、填充行、归一化和 residual。
训练代理指出旧 common 缺失、电子补偿、缓存计算精度、buffer 冻结、反馈身份及匹配起点的风险，均纳入新协议。

复制源响应仍不是教授所说的动态反馈形成；只有初筛通过后才开始拟合形成规则。
未扰动的实数振幅与强度无法一般识别完整相位更新响应；未来形成阶段需要受控相位扰动或等价额外信息。
单层漂移界不能保证八层收敛。每独立轴线的整体相位是不可观测 gauge，不能把所有相位位移直接解释为反馈失配。

## 文献核对

教授本地稿：[On the Surprising Effectiveness of Fixed Feedback Weights for Language Model Supervised Fine-Tuning](../FixedFeedbackSFT/literature/paper.pdf)，第1–2页只固定连接器、第3页预训练语言模型 SFT、第4页说明尚未建立计算优势。PDF metadata 不能证明公开发表状态。

PAT 原文：[Deep physical neural networks trained with backpropagation](https://arxiv.org/html/2104.13386)，物理前向与可微数字模型提供梯度。DAT 原文：[Dual adaptive training of photonic neural networks](https://arxiv.org/html/2212.06141)，实测强度与模拟相位融合。因此“不测真实相位”本身不是 FFA 新颖性。

其他需正面对照的工作：[NeurIPS 2024 随机噪声预训练](https://proceedings.neurips.cc/paper_files/paper/2024/hash/18d3a2f3068d6c669dcae19ceca1bc24-Abstract-Conference.html)、[Bio-inspired backpropagation-free training for optical neural networks](https://www.nature.com/articles/s41377-026-02394-3)、[Hughes 等原位光学梯度](https://arxiv.org/html/1805.09943)。不能把“同光路不能反向”作为本方案理由，比较的是完整误差注入/测量/计算代价。

## 运行状态

初始源码提交 `0de3b0da0` 已经同步本地、GitHub 与服务器 main。
服务器 xml / PyTorch 2.6.0+cu124 环境下 12 项数值测试全部通过（1.963 s），包括
CUDA float32、双精度 gradcheck、有限差分、token/feature 两层电子链、源响应冻结与恢复。
测试结束后 GPU 上未保留测试进程。训练程序的 buffer 哈希增加标量张量处理后再发布运行。
实际训练结果将追加在此；测试通过不代表 FFA 适配有效。
