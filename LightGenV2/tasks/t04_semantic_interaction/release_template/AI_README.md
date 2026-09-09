# 给接手同学 / AI 的技术交接

## 锁定的版本

只用 `weights/best_checkpoint.pt`，epoch40、standard 原头，architecture 字符串必须与 MANIFEST 一致。以 `reference/student_architecture.json` 和实际 forward 为准，不能套用历史 contextual-text / slim-head 模型。禁止把 reference 中的服务器路径当作当前运行路径；入口已逐一重定位。

原训练 5000 / 1000、seed73、100epochs、周期 test 选 best。源数据清单随包固定。保留 train/test 分离，不用测试目标制作输入；不得用目标网格、操作标签代替推理结果。

## 数据流

1. 文字：固定 Qwen tokenizer 的 token 对应原始冻结 embed_tokens，缓存 `[S,2048]`。这是词嵌入查表结果，不是完整 Qwen 的 contextual hidden states，不经过原生 Transformer。
2. 图像：`[B,3,224,224]` 经冻结 Qwen patch embedding + position，得到 14×14，即196个空间位置的视觉特征。不是把196幅图与文字拼在一起。
3. Language 光电模块先处理文字：光 Router 从CCD探测区能量选 Top2 / 4 experts，然后专家传播和 global；有同尺度电子残差融合，alpha 约束区间 `[0.4001,0.95]`。
4. 语言结果压缩成条件向量，经轻量尺度/偏置调制视觉特征；Vision 同样为光 Router → experts → global。文字条件不是视觉 Transformer 的 attention。
5. 原 standard 共享读出头包含条件卷积与编辑解码器，输出每格类别和编辑概率。通过 source_grid 保留未编辑格，再渲染图标；不是用大模型生成句子。

实际细节请阅读 `LightGenV2/tasks/t04_semantic_interaction/{embedding_model,router_repair,shared_readout}.py`。`modeling.build_model` 根据配置选择本包 OURS；依赖文件里的其他历史类 / 延迟 import 不表示它们进入本模型 forward。不得把这些公共代码删到 import 失败，也不能据目录名断言模型里含 Transformer。

## 权重怎么读

确认文件来自可信交付并先运行 verify；PT 使用 torch.load，不能加载未知来源恶意 PT。

```python
import torch
checkpoint = torch.load('weights/best_checkpoint.pt', map_location='cpu', weights_only=False)
print(checkpoint['architecture'], checkpoint['epoch'])
state = checkpoint['model']
for name, tensor in state.items():
    if 'raw_phase' in name or 'raw_router_phase' in name:
        print(name, tuple(tensor.shape))

phases = torch.load('weights/phases.pt', map_location='cpu', weights_only=False)
for name, phase in phases['phases'].items():
    print(name, tuple(phase.shape), 'radians')
```

state 中 raw 相位是训练参数，物理相位为 `2*pi*sigmoid(raw)`。`phases.pt` 已转换，不能再次 sigmoid。矩阵 `[y,x]`；语言和视觉有各自 router、4个专家和 global 参数，按 key 明确选择所研究的那一个，不能把不同模块当同一专家的时间序列。这里只有最优一次，不包含逐epoch权重。

物理相位PT不是已经过SLM LUT标定的8bit BMP；实机使用还需按既有光学几何、波长、像素间距、ROI和LUT导出并标定，不可任意拉伸矩阵。

## 复现与变更约束

- 先 verify → demo → evaluate，通过后再开发；完整结果与单图的 scene_exact 不混淆。
- 原始参考文件只读。开发产生的数据放新 outputs 子目录，不覆盖交付 best。
- 用新指令 / 新数据必须重做匹配的原始 token 缓存；不要调用完整 Qwen 生成 contextual states，否则已改架构。
- 原始参数与训练损失见 settings.json、training.py；训练包含光路由均衡、相位优化及未调制分量设置，不能默默关掉来追分。
- baseline不在本包；reference去光是同一权重消融，不是单独训练的另一个baseline。
- 记录依赖版本、随机种子、参数、数据清单 hash、选模口径。不要宣称 GPU/CPU 逐位一致或新训练一定复现到相同 best。

本包不存 SSH 密码、硬件配置和其他任务的数据集；无网络情况下可以执行随包数据复现。完整大模型与硬件控制需另行准备，不能将本包宣传为任意新文本推理或实机部署的一键包。
