# FFA 实验记录

## 2026-10-11：初始化与审查

- 用户授权创建唯一顶级 `FFA` 目录，讨论后直接尝试，不建分支或工作树。
- 完整对话含五组问答；保留用户对预训练意义、当前 BP/PAT 公平对照和物理反馈来源的要求。
- 本地、服务器主目录与 GitHub 起始提交为 `1b771a0a718c4b02ea7348f9dd33add6141e6a16`，分支为 main。
- 本地既存 OpenMoji/handoffs/输出未跟踪文件保留；服务器主目录起始已跟踪修改为零。
- GPU 1 的 FishNet 已有计算进程，不使用；GPU 6 A100 初检无计算进程，按 UUID 指定。
- 根 AGENTS.md、FixedFeedbackSFT README/MIGRATION、P11/P12 实现和任务协议均已核对。
- 三位子代理负责架构/训练审计、数学及文献审查、强度接口实现；路径互不重叠，Git 事务由主代理串行执行。
- 旧 LSP phase-only common start 未保留，LSP 已适配 FA-source best 不作为共同起点。P11 source 与 frozen stem 仍可使用。
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

## 第一轮终态：完成，GPU 已释放

- run：服务器 `/DATA/DATA1/guest3/2026OpticsMoE/FFA/runs/lsp_intensity_pilot_20261011`。
- 执行代码 main `3ac18278502be4e634de32a75f6676e140e132e9`，PID 339952，终态 `result.json=status:complete`。
- 30 epoch 新头暖启 + 每组 8 epoch；每组恰好 8 行适配历史、五组同一个共同起点 SHA。
- 共 273.5672 秒；测试集未评估，未进行真实光路实验或动态反馈形成。
- 骨干电子参数与全部持久 buffer 完整哈希在训练前后相同；冻结反馈 tensor SHA 在每组前后相同。
- 八层相位在全部更新组都有有限非零梯度；无 AMP 跳过的优化步。
- 32 份 JSON/CSV、477,326 字节记录已下载到本地同名 `FFA/runs`，逐文件 SHA 核验；权重保留在服务器，best/last SHA 进入各组结果。传输清单保存在被忽略的 `FFA/runs/transfer_manifest_20261011.json`。
- 进程退出后查询 GPU compute-app，A100 UUID 已无进程；GPU 1 FishNet 未干预。

| 方法 | val best PCK | val last PCK | 末轮梯度余弦（同状态对 BP） |
|---|---:|---:|---:|
| head_only | 26.171875% | 25.920759% | 无光学更新 |
| bp_current | 26.953125% | 26.953125% | 1.0000 |
| fa_source_connector | 26.925223% | 26.869420% | 0.9996 |
| source_full_response | 26.897321% | 26.869420% | 0.9471 |
| random_full_response | 26.199777% | 26.060268% | 0.0676 |

BP 相对同预算 head-only 的增益 0.78125 pp，源完整响应恢复 92.8571%；这是开发描述性比值，未检验统计显著性。头暖启仍在明显学习，小数据结果不代表旧完整任务成绩。原 FA 保留原二维 FFT，起点 val 相较一维实现差 0.0558 pp（两关节），全网 legacy/current 梯度 cosine 0.99999991、relative L2 0.000427、输出 max abs 0.000193；不解读小数点级排名。主比较 BP/source-full/random-full 共用一维前向。

源完整响应末轮平均绝对相位位移 0.00408061 rad、RMS 0.0128294 rad，仍是很局部的范围。因此补做旧 50-epoch FA-source 已保留终点的只读梯度诊断；旧 best 本身选在 epoch45，是否完成50轮由原 result 收据确认，不能由 `run_epochs=50` 推断。

## 旧终点审计：严格来源检查先行

首个审计目录 `FFA/runs/lsp_retained_endpoint_audit_20261011` 在源相位哈希检查处退出，failure.json 保留。
随后只读重建实验确认是 CPU 线程设置下的 float32 舍入差异：默认 40 线程时源相位 SHA
与旧 checkpoint/声明完全一致 `dec25a27...`；4 线程时为 pilot 使用的 `e1237e98...`。
总 1,204,224 个元素仅44个不同，max abs `9.536743e-7 rad`、mean abs `8.666205e-12 rad`。
同一 P11 文件 SHA 保持 `c3ad0b78...`；电子 parameter+buffer 哈希在 source/旧 endpoint 严格一致。

修复仅将线程设置移到默认设置下完成原样 source 重建与严格加载之后，不放宽来源校验，
保留旧 source tensor 字节。新试验本身的源相位、代码和数据身份在运行时已独立记录，
不将两种舍入后的物理相位 tensor SHA 混为一个值。审计重试使用新目录，未覆盖失败记录。

第二次审计通过所有权重身份检查并完成第一批梯度计算，但输出记录将裁剪坐标的
`list[Tensor]` 误作为样本身份写入 JSON，序列化失败。记录和日志保留在 `_r2`。
修复只限定样本身份为有精确数量的 sample_id/source/image_path 字符串列表，
未改梯度计算。第三次在新 `_r3` 目录运行，最终结果必须由 complete 收据判定。

第三次审计完成首批后，第二批 legacy2D/1D 前向最大绝对差 `0.000500828`，刚超过
原 `0.0005` 的绝对阈值而退出；phase gradient cosine `0.9999999846`、relative L2
`0.000177953`，head relative L2 `0.000104204`。原光学算子一维等价测试已通过，
该绝对输出阈值不随训练后的头数值尺度变化。
因此改为记录全部 absolute 差异，同时用 forward **relative L2≤0.001** 验收；
phase/head 梯度门禁保持不变。改动不影响前向、梯度或旧初筛结果。第四次使用 `_r4`。
