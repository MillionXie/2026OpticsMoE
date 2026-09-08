# 语言Router坍缩修复与公平Qwen对照

## Linux baseline 启动前的文件句柄限制

冻结 Qwen 缓存包含每条样本的变长文本张量。多进程 DataLoader 在默认 1024 文件句柄限制下可能报 `Too many open files`。
在同一个 shell 先执行 `ulimit -n 65536`，再执行 README 的训练命令；这只提高当前进程资源上限，不改变模型、batch、样本或训练协议。
训练器从 `966f4070` 起跳过权重为零的相位正则，避免纯 Qwen 没有 PhaseLayer 时误报错。失败日志保留，完整冻结特征缓存可复用，不必重新提取。

## 原因与修复范围

旧 `embedding_alpha40_lean_s73` best第80轮，修改格89.40%、整场景82.10%，语言Router所有1000个样本都选0/1。
64条test诊断：语言在224×224振幅中仅最上方4–14行非零；四区平均能量比例47.50%、48.16%、1.49%、2.86%。
这证明存在强烈的输入照明位置偏置；并不声称这是唯一优化问题。

新 `FullApertureLanguageRouter` 在光Router这一次曝光前，把有效token行通过固定双线性展开铺到现有224方形中，
再恢复同一总功率。该方形仍放在478逻辑ROI中心，ROI、专家位置、SLM像素间距、10cm距离、检测区域全部不变。
不增加电子分类器，不从任务标签选择专家，不强制轮流选专家，不按batch配额分配。
只是Router输入BMP改变，专家/global输入编码不改；硬件导出需使用Router实际记录的 `last_input_amplitude`，
不能沿用旧语言Router输入BMP。现有硬件包尚未针对新合同重新导出验证，不宣称可直接替换旧包。

新模型从头训练，光Router/专家/global及电子残差均可训练，alpha仍在[0.4001,0.95]，两组末端条件卷积。
保留原20%–30%特征相位未调制项和CCD噪声，不使用像素平移；均衡配置两个候选，其余结构/训练预算一致。
现有Router传播另有原配置的相位dropout；不能把特征相位DC配置误称成所有Router传播也含相同DC。

每次周期test记录两个模态的专家槽位占比、Top-2组合直方图。合格线：每专家占比5%–45%，
且任一固定专家组合不能覆盖超过80%样本；按合格优先、再按修改格准确率选best。
如果没有合格轮次，仍保留最佳诊断权重，但 `selected_router_accepted=false`，不能宣称修复完成。

## 两个方法完全相同的末端读出

实际调用同一 `SharedGridReadout` 类，不是另写一个“差不多”的头。共享指架构/初始值相同，不共享训练后的权重。
输入是14×14×192视觉特征和L×192语言特征（L≤64）。语言用64个固定位置的可学习线性系数汇总，再LN+Linear+GELU；
视觉经0.1尺度文本FiLM、坐标映射、两组深度卷积条件残差（dilation1/2），再原SemanticGridDecoder产生6×6类别和编辑logits。
两者都有相同4类操作辅助头和同一任务损失。后端381976参数，初始SHA写入 `student_architecture.json`，并有逐位输出一致测试。

主方法：冻结词表/patch前端 → 双模态光Router/expert/global+电子残差 → 同一末端头。
Qwen baseline：**完整冻结原生Vision和Language**，不加LoRA、不微调主干。视觉取最后原生Vision层、合并前的196×1024特征，
保持14×14空间分辨率，不先缩到7×7又放大。文本取完整多模态Language最后一层中原指令token对应的L×2048特征，
不取平均/最大拼接、不用额外提示描述。为进入同一头，仅增加1024→192和2048→192两个Linear+LN。
两个方法的主干功能与参数量本来不同，不能声称整个网络参数量相同。

baseline的原生vision merger和deepstack仍正常用于Qwen内部完整多模态推理；图文融合是模型原生机制，
不是为了baseline另添网络。视觉最后层hook与全部原生层计数在特征提取中逐张校验。
只缓存完整冻结前向输出以节约训练开销；推理结构仍包含完整Qwen，不能据缓存训练速度报告大模型推理加速。
cache位于任务dataset/openmoji_grid_v2/qwen_shared_head_v1，包含split SHA、样本顺序、模型snapshot和实际层计数。

## 公平性与复现

- 相同v2源场景/指令去重清单，5000train、1000test，原始224RGB、同一条指令内容，无validation。
- baseline使用Qwen原生chat包装，这是模型输入格式差异；不添加任务标签、答案、额外解释或教师输出。
- seed73、100epoch、batch32、AdamW、相同基础学习率3e-4、5% warmup+cosine、EMA0.995。
- 两者共享任务损失：类别CE权重、编辑BCE/Dice、保留项、操作分类权重都一样。
  光方法独有光学正则/Router均衡，baseline没有这些不适用项。
- epoch1/每5/末轮test选模，有选择偏倚；光方法额外要求Router不过度集中，报告该限制。
- 相同 `MetricAccumulator`，同时输出修改格、前景类别、编辑IoU、对象F1、整场景及每操作指标。
- 最终合成都使用数据给定 `source_grid` 保留未修改区域；双方一致披露此额外结构化源信息，不冒充纯RGB端到端。
- 保留best与last，不写每5轮PT；主方法自动做同best去光，不另训纯电子模型。
- 旧D2NN51.75%与旧Qwen53.9%不是这一共享读出协议的结果；需要相同协议时D2NN也应重训，不混填。

运行入口在任务README；三个新结果目录分别是 `runs/simulation/routerfill_shared_s73`、
`routerfill_shared_balance_s73`、`qwen_shared_s73`。检查各目录的 `run_manifest.json` 获取实际Git commit和完整命令。
本轮不测5090D速度/能耗，仅测性能，不人为限制baseline成绩。
