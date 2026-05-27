# Visual Prompting 项目技术导读

这份文档面向刚开始接触大模型/视觉大模型的读者，解释这个项目在做什么、代码怎么组织、`Train/Test for CLIP` 和 `Train/Test for Vision Models` 分别是什么意思，以及命令行参数为什么要这样写。

你可以先读这份文档，再看代码：

- `main_clip.py`：CLIP 视觉提示训练和测试入口。
- `main_vision.py`：传统视觉预训练模型的视觉提示训练和测试入口。
- `models/prompters.py`：visual prompt 的具体实现。
- `utils.py`：accuracy、checkpoint、学习率调度等通用工具。

## 1. 这个项目到底在研究什么？

项目来自论文 **Exploring Visual Prompts for Adapting Large-Scale Models**。

核心问题是：

> 如果我们有一个已经训练好的大模型，不想改它的参数，能不能只在输入图像上加一小块可训练的“视觉提示”，让模型适配新的任务？

这里的 **visual prompt** 可以理解为“加在输入图片上的一组可训练像素”。

普通图像分类是：

```text
image -> model -> class logits
```

Visual Prompting 是：

```text
image
  |
加上可训练 prompt
  |
prompted image
  |
冻结的大模型
  |
class logits
```

训练时：

- 冻结大模型参数。
- 只更新 visual prompt 参数。
- 用分类交叉熵训练。

这和 prompt tuning / prefix tuning 的思想类似，只不过这里的 prompt 不在文本 token 空间，而是在图像像素空间。

## 2. 为什么这对你的光学 MoE 有启发？

你的想法里有一个关键点：

> 低速可切换的光学 prompt 控制高速图像光场在不同专家路径中的传播。

这个项目提供了一个很好的软件原型：

```text
任务变化
  |
切换/训练不同 prompt
  |
冻结主模型仍然复用
  |
完成不同任务适配
```

迁移到光学 MoE 时，可以类比为：

```text
Visual Prompting 项目:
数字图像 + 数字 prompt -> 冻结视觉模型

光学 MoE 想法:
高速图像光场 + 低速光学 prompt -> 固定或半固定光学专家系统
```

所以复现这个项目的目的不是最终使用 CLIP，而是先搞清楚：

- prompt 如何作为任务控制信号；
- 主模型如何冻结；
- 只训练 prompt 时 loss 如何回传；
- 训练和评估路径如何组织。

## 3. 项目结构

当前主要文件：

```text
visual_prompting/
  main_clip.py
  main_vision.py
  models/
    prompters.py
    download_models.sh
  utils.py
  requirements.txt
  REPRODUCE_CLIP.md
  PROJECT_TECHNICAL_GUIDE_CN.md
```

### `main_clip.py`

用于 CLIP 实验。

CLIP 是图文对齐模型，它有两个编码器：

```text
image encoder: image -> image feature
text encoder : text  -> text feature
```

分类时会把类别名称变成文本，例如 CIFAR10 类别 `cat` 会变成：

```text
This is a photo of a cat
```

然后 CLIP 比较图像特征和各类别文本特征的相似度，得到分类 logits。

### `main_vision.py`

用于普通视觉模型实验，例如：

- `rn50`
- `instagram_resnext101_32x8d`
- `bit_m_rn50`

这些模型本身就是图像分类模型，输入图片后直接输出分类 logits。

### `models/prompters.py`

定义 visual prompt 长什么样。

目前有三种方法：

```text
padding
fixed_patch
random_patch
```

### `utils.py`

放通用函数：

- `accuracy`：计算 top-k 准确率。
- `AverageMeter`：记录 loss/accuracy 平均值。
- `ProgressMeter`：打印训练进度。
- `save_checkpoint`：保存 prompt 参数。
- `cosine_lr`：余弦学习率调度。
- `refine_classname`：把类别名格式变成自然语言形式。

## 4. `models/prompters.py` 是干嘛的？

这个文件是项目最核心的部分之一。

它定义的是“输入图像上的可训练 prompt”，不是大模型本身。

输入图像张量一般是：

```text
x shape = [batch_size, 3, image_size, image_size]
```

例如：

```text
[8, 3, 224, 224]
```

含义是：

- 8 张图片；
- 3 个颜色通道 RGB；
- 每张图片大小是 224 x 224。

### 4.1 `PadPrompter`

对应命令参数：

```bash
--method padding
```

它在图像四周加一圈可训练像素：

```text
+----------------------+
|      pad_up          |
+---+--------------+---+
| L | original img | R |
+---+--------------+---+
|      pad_down        |
+----------------------+
```

代码里可训练参数是：

```python
self.pad_up
self.pad_down
self.pad_left
self.pad_right
```

这些都是 `nn.Parameter`，所以会被优化器更新。

主模型参数不更新，只有这些 prompt 参数更新。

### 4.2 `FixedPatchPrompter`

对应命令参数：

```bash
--method fixed_patch
```

它在图像固定位置放一个可训练 patch，默认是左上角：

```text
+------+----------------+
|patch |                |
+------+                |
|                       |
|    original image     |
|                       |
+-----------------------+
```

可训练参数是：

```python
self.patch
```

### 4.3 `RandomPatchPrompter`

对应命令参数：

```bash
--method random_patch
```

它也是一个可训练 patch，但每次 forward 时随机放到图像中的某个位置。

这种方法可以理解为一种带随机扰动的数据增强。

## 5. Train/Test for CLIP 是什么意思？

README 里写的：

```text
Train/Test for CLIP
```

意思是：

- Train：在冻结 CLIP 的情况下训练 visual prompt。
- Test：加载训练好的 visual prompt，在验证/测试集上评估。

这里的 Test 不是训练 CLIP，也不是测试 CLIP 仓库本身，而是测试保存下来的 prompt 是否有效。

### 5.1 CLIP 训练流程

以 CIFAR10 为例：

```text
CIFAR10 image
  |
preprocess 到 224x224，并按 CLIP 需要的 mean/std 归一化
  |
prompter(image)
  |
prompted image
  |
冻结的 CLIP image encoder
  |
image feature

类别名称
  |
"This is a photo of a {class_name}"
  |
CLIP text encoder
  |
text features

image feature 和 text features 做相似度
  |
输出每个类别的 logits
  |
CrossEntropyLoss(logits, target)
  |
只更新 prompter 参数
```

### 5.2 CLIP 路径里的输入输出

输入：

```text
images: [batch_size, 3, 224, 224]
target: [batch_size]
texts : 类别文本列表
```

输出：

```text
output: [batch_size, num_classes]
```

例如 CIFAR10：

```text
output shape = [8, 10]
```

每一行表示一张图片属于 10 个类别的分数。

### 5.3 CLIP 实验为什么重要？

CLIP 是视觉语言大模型，具备强大的 zero-shot 能力。

Visual Prompting 想验证：

> 不微调 CLIP，只改输入图像上的 prompt，能不能让 CLIP 更适合某个下游数据集？

这对应大模型适配中的一个重要范式：

```text
冻结大模型 + 小规模可训练参数
```

## 6. Train/Test for Vision Models 是什么意思？

`main_vision.py` 不是用 CLIP，而是用普通视觉预训练模型。

支持的模型包括：

```text
rn50
instagram_resnext101_32x8d
bit_m_rn50
```

训练目标仍然是：

```text
冻结视觉模型，只训练 visual prompt
```

流程是：

```text
image
  |
prompter(image)
  |
prompted image
  |
冻结的 ResNet/BiT/ResNeXt
  |
class logits
  |
CrossEntropyLoss(logits, target)
```

### 6.1 为什么还要做 Vision Models 实验？

因为 CLIP 和普通视觉分类模型的能力来源不一样。

CLIP：

- 通过图文对比学习训练；
- 分类时依赖文本 prompt；
- 更接近视觉语言大模型。

Vision Models：

- 通过图像分类或大规模视觉监督训练；
- 直接输出分类 logits；
- 更接近传统视觉 backbone。

同时做这两个实验，可以验证 visual prompting 不是只对 CLIP 有效，而是一种更通用的输入端适配方法。

这对你的光学 MoE 也有意义：

> 如果 prompt 控制机制在不同冻结模型上都有效，那么未来用光学 prompt 控制不同专家路径也更有说服力。

## 7. 训练 loss 是什么？

两个入口的主 loss 都是分类交叉熵：

```python
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, target)
```

其中：

```text
output: [batch_size, num_classes]
target: [batch_size]
```

举例：

```text
output[0] = [1.2, -0.3, 4.1, ...]
target[0] = 2
```

表示第 0 张图片真实类别是第 2 类，loss 会鼓励模型把第 2 类的分数变高。

关键点：

```text
loss 会反向传播
但被冻结的大模型参数 requires_grad=False
所以只有 visual prompt 的 nn.Parameter 会被更新
```

在 CLIP smoke test 中，你会看到：

```text
CLIP trainable parameters: 0
Visual prompt trainable parameters: 69840
```

这说明训练目标确实只作用在 prompt 上。

## 8. 为什么命令行要加那么多参数？

命令行参数的作用是让同一个脚本能跑不同实验。

例如：

```bash
python main_clip.py --dataset cifar10 --root ./data --smoke_test
```

可以拆开理解：

```text
python main_clip.py
```

运行 CLIP visual prompting 入口。

```text
--dataset cifar10
```

选择数据集。当前 CLIP 路径支持：

```text
cifar10
cifar100
eurosat
```

```text
--root ./data
```

告诉 torchvision 数据集放在哪里。

如果本地没有数据，`download=True` 会尝试下载到这个目录。

```text
--smoke_test
```

运行极小规模冒烟测试。它不是正式训练，而是检查代码路径能不能跑通。

## 9. `--smoke_test` 是什么？

`--smoke_test` 是我为复现路径新增的工程测试模式。

它做几件事：

```text
1. 只使用很小的数据子集
2. 只训练 1 个 epoch
3. batch size 限制到最多 8
4. num_workers 设置为 0，减少 Windows/conda 多进程问题
5. 检查 CLIP 没有可训练参数
6. 检查 visual prompt 有可训练参数
7. 保存 checkpoint
8. 打印 train/eval accuracy
```

它的目的不是训练出高精度模型，而是确认：

```text
环境 OK
数据 OK
CLIP OK
prompt OK
反向传播 OK
checkpoint OK
```

你可以把它理解为：

```text
正式实验前的最小健康检查
```

## 10. 常用命令解释

### 10.1 安装依赖

如果你的目录结构是：

```text
2026OpticsMoE/
  CLIP/
  visual_prompting/
```

在 `visual_prompting` 目录下执行：

```powershell
conda activate RFL
pip install -r requirements.txt
pip install -e ../CLIP
```

`pip install -e ../CLIP` 的意思是：

```text
把本地 ../CLIP 仓库安装到当前 Python 环境
```

否则 `main_clip.py` 里的：

```python
import clip
```

会找不到包。

### 10.2 CLIP smoke test

```powershell
python main_clip.py --dataset cifar10 --root ./data --smoke_test
```

适合第一次验证。

你应该看到类似输出：

```text
CLIP trainable parameters: 0
Visual prompt trainable parameters: 69840
Smoke test checkpoint saved: ...
Smoke test train/eval Acc@1: train=..., eval=...
```

### 10.3 正式训练 CLIP visual prompt

```powershell
python main_clip.py --dataset cifar10 --root ./data
```

如果显卡显存不大，不建议直接用默认 `--batch_size 256`。可以先用：

```powershell
python main_clip.py --dataset cifar10 --root ./data --batch_size 64 --num_workers 0
```

如果仍然 OOM，再改成：

```powershell
python main_clip.py --dataset cifar10 --root ./data --batch_size 32 --num_workers 0
```

常用可调参数：

```text
--method padding
--prompt_size 30
--batch_size 256
--learning_rate 40
--epochs 1000
```

当前代码已经把 CLIP 的文本特征提前算好并缓存。这样训练时每个 batch 只需要跑 image encoder，不会反复运行 text encoder，显存和时间都会更稳。

### 10.4 评估保存的 CLIP prompt

```powershell
python main_clip.py `
  --dataset cifar10 `
  --root ./data `
  --evaluate `
  --resume ./save/models/<run_name>/checkpoint.pth.tar
```

`--evaluate` 表示只测试，不训练。

`--resume` 指向保存的 prompt checkpoint。

### 10.5 Vision Models 训练

```powershell
python main_vision.py --model rn50 --dataset cifar100 --root ./data
```

注意：当前 `main_vision.py` 原始代码主要面向 CIFAR100，数据加载仍然硬编码为 `CIFAR100`。

### 10.6 Vision Models 评估

```powershell
python main_vision.py `
  --model rn50 `
  --dataset cifar100 `
  --root ./data `
  --evaluate `
  --resume ./save/models/<run_name>/checkpoint.pth.tar
```

## 11. 输出文件是什么？

训练会保存 checkpoint：

```text
save/models/<run_name>/
  checkpoint.pth.tar
  model_best.pth.tar
```

checkpoint 里主要有：

```python
{
    "epoch": epoch + 1,
    "state_dict": prompter.state_dict(),
    "best_acc1": best_acc1,
    "optimizer": optimizer.state_dict(),
}
```

注意：

```text
保存的是 visual prompt 参数，不是 CLIP 或视觉大模型参数。
```

因为主模型是冻结的，可以从预训练权重重新加载。

## 12. 代码执行流程：以 `main_clip.py` 为例

整体顺序：

```text
parse_option()
  |
读取命令行参数
  |
clip.load(...)
  |
加载冻结 CLIP
  |
创建 prompter
  |
加载 CIFAR10/CIFAR100/EuroSAT
  |
构造类别文本 prompt
  |
创建 optimizer，只优化 prompter.parameters()
  |
for epoch:
    train(...)
    validate(...)
    save_checkpoint(...)
```

训练内部：

```text
images, target
  |
images.to(device)
target.to(device)
  |
prompted_images = prompter(images)
  |
output, _ = model(prompted_images, text_tokens)
  |
loss = CrossEntropyLoss(output, target)
  |
loss.backward()
  |
optimizer.step()
```

其中 optimizer 是：

```python
optimizer = torch.optim.SGD(prompter.parameters(), ...)
```

所以即使 loss 从 CLIP 输出算出来，真正更新的也只有 prompt。

## 13. 代码执行流程：以 `main_vision.py` 为例

整体顺序类似：

```text
parse_option()
  |
加载 rn50 / instagram_resnext101_32x8d / bit_m_rn50
  |
model.eval()
  |
创建 prompter
  |
加载 CIFAR100
  |
optimizer 只优化 prompter.parameters()
  |
train / validate / save_checkpoint
```

训练内部：

```text
images
  |
prompted_images = prompter(images)
  |
output = model(prompted_images)
  |
如果是 ImageNet 预训练模型，取 CIFAR100 对应类别 indices
  |
loss = CrossEntropyLoss(output, target)
  |
只更新 prompter
```

## 14. CLIP 路径和 Vision 路径的核心区别

| 对比项 | CLIP 路径 | Vision Models 路径 |
|---|---|---|
| 入口文件 | `main_clip.py` | `main_vision.py` |
| 主模型 | CLIP ViT-B/32 | ResNet/BiT/ResNeXt |
| 输出方式 | 图像-文本相似度 logits | 分类器 logits |
| 类别信息 | 用文本模板生成 | 用模型分类头输出 |
| 训练参数 | 只训练 prompt | 只训练 prompt |
| 主要意义 | 验证视觉语言大模型适配 | 验证普通视觉模型适配 |

## 15. 你现在应该怎么读代码？

建议顺序：

1. 先读 `models/prompters.py`
   - 看 prompt 是怎么加到图像上的。

2. 再读 `main_clip.py` 的 `main()`
   - 看 CLIP 如何加载、冻结、创建 prompt、加载数据。

3. 再读 `train(...)`
   - 看 loss 如何计算，optimizer 如何更新 prompt。

4. 再读 `validate(...)`
   - 看 `Original Acc@1` 和 `Prompt Acc@1` 的区别。

5. 最后读 `main_vision.py`
   - 对比不用 CLIP 时，普通视觉模型路径有什么不同。

## 16. `Original Acc@1` 和 `Prompt Acc@1` 是什么？

验证时会同时算两个准确率：

```text
Original Acc@1
```

不加 prompt，直接把原图送入冻结模型。

```text
Prompt Acc@1
```

先加 visual prompt，再送入冻结模型。

如果训练有效，通常希望：

```text
Prompt Acc@1 > Original Acc@1
```

不过在 smoke test 里不一定，因为 smoke test 数据太少，只是为了检查流程。

## 17. 和光学 Prompt / 光学 MoE 的对应关系

这个项目可以作为你的软件基线：

```text
数字 visual prompt
  -> 改变输入图像
  -> 控制冻结模型输出
```

你之后可以把它映射到光学系统：

```text
光学 prompt 相位/振幅调制
  -> 改变输入光场传播
  -> 控制专家路径或专家权重
```

进一步做光学 MoE 时，可以把几个概念对应起来：

| Visual Prompting | 光学 MoE 想法 |
|---|---|
| 可训练像素 prompt | 可切换光学 prompt |
| 冻结 CLIP / Vision model | 固定光学专家硬件 |
| prompt 控制输入分布 | prompt 控制传播路径/权重 |
| 下游任务适配 | 多任务光学推理 |
| checkpoint 保存 prompt | 保存不同任务的光学 prompt 配置 |

## 18. 当前复现状态

当前已经验证过：

```powershell
conda activate RFL
cd visual_prompting
python main_clip.py --dataset cifar10 --root ./data --smoke_test
```

关键输出：

```text
CLIP trainable parameters: 0
Visual prompt trainable parameters: 69840
Smoke test checkpoint saved: ...
Smoke test train/eval Acc@1: train=87.500, eval=75.000
```

这说明：

- CLIP 确实冻结；
- prompt 确实可训练；
- CIFAR10 数据可以加载；
- 训练和验证路径可以跑通；
- checkpoint 可以保存；
- `--evaluate --resume` 评估路径也已经跑通。

## 19. 进一步解释：CLIP 路径和 Vision 路径到底是不是你理解的那样？

你的理解大方向是对的：

```text
main_clip:
基于 CLIP 这种图文对比学习预训练模型，训练 visual prompt。

main_vision:
基于传统视觉分类预训练模型，训练 visual prompt。
```

但要补一个关键细节：

```text
main_clip 的下游 prompt 训练仍然使用 label。
```

区别不在于“有没有 label 训练 prompt”，而在于冻结主模型如何输出分类分数。

### 19.1 `main_clip.py`

CLIP 原始预训练方式是图文对比学习：

```text
image encoder(image) -> image feature
text encoder(text)   -> text feature
```

在 CIFAR10 上做分类时，会把类别名变成文本：

```text
This is a photo of a airplane
This is a photo of a car
This is a photo of a bird
...
```

然后 CLIP 比较：

```text
prompted image feature
和
每个类别文本 feature
```

得到相似度 logits。

但是 loss 仍然是：

```python
loss = CrossEntropyLoss(output, target)
```

其中 `target` 还是 CIFAR10 的 label。

所以更准确的说法是：

```text
main_clip:
冻结 CLIP，用图像-文本相似度作为分类 logits，再用下游数据集 label 训练 visual prompt。
```

### 19.2 `main_vision.py`

`main_vision.py` 使用的是普通视觉预训练模型，例如 ResNet50、BiT、ResNeXt。

这些模型没有 text encoder，不需要类别文本 prompt。

流程是：

```text
image -> visual prompt -> frozen vision model -> class logits
```

然后同样用：

```python
loss = CrossEntropyLoss(output, target)
```

所以更准确的说法是：

```text
main_vision:
冻结传统视觉分类模型，直接用分类 logits 和 label 训练 visual prompt。
```

## 20. `padding` prompt 的尺寸为什么这么设计？

假设：

```text
image_size = 224
prompt_size = 30
```

输入图像 shape 是：

```text
[batch_size, 3, 224, 224]
```

`padding` 方法的意思是在图像四周放一圈可训练像素，中间原图区域不加 prompt。

边框宽度是 `prompt_size = 30`。

### 20.1 为什么 `pad_up` 不需要 `image_size - pad_size * 2`？

上边框覆盖整张图的宽度。

所以：

```python
self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
```

对应：

```text
[1, 3, 30, 224]
```

含义是：

```text
1 张 prompt 模板
3 个 RGB 通道
高度 30
宽度 224
```

下边框 `pad_down` 同理。

### 20.2 为什么 `pad_left` 需要 `image_size - pad_size * 2`？

左边框不应该覆盖上边框和下边框已经占据的区域。

完整高度是 224。

上边框占 30。

下边框占 30。

所以左边框只负责中间高度：

```text
224 - 30 - 30 = 164
```

因此：

```python
self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
```

对应：

```text
[1, 3, 164, 30]
```

右边框 `pad_right` 同理。

### 20.3 `PadPrompter.forward()` 一步步在做什么？

代码：

```python
def forward(self, x):
    base = torch.zeros(1, 3, self.base_size, self.base_size, device=x.device)
    prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
    prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
    prompt = torch.cat(x.size(0) * [prompt])

    return x + prompt
```

第一步：

```python
base = torch.zeros(1, 3, self.base_size, self.base_size, device=x.device)
```

`base` 是中间不加 prompt 的区域。

如果 `image_size=224`、`prompt_size=30`：

```text
self.base_size = 224 - 30 * 2 = 164
base shape = [1, 3, 164, 164]
```

第二步：

```python
prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
```

`dim=3` 是宽度方向。

横向拼接：

```text
pad_left  [1, 3, 164, 30]
base      [1, 3, 164, 164]
pad_right [1, 3, 164, 30]
```

得到：

```text
[1, 3, 164, 224]
```

第三步：

```python
prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
```

`dim=2` 是高度方向。

纵向拼接：

```text
pad_up   [1, 3, 30, 224]
middle   [1, 3, 164, 224]
pad_down [1, 3, 30, 224]
```

得到完整 prompt：

```text
[1, 3, 224, 224]
```

第四步：

```python
prompt = torch.cat(x.size(0) * [prompt])
```

如果 batch size 是 8，`x.size(0)=8`。

这一步把同一个 prompt 复制给 batch 里的每张图：

```text
[1, 3, 224, 224] -> [8, 3, 224, 224]
```

最后：

```python
return x + prompt
```

把 prompt 加到输入图像上。

这里训练的是一个“全任务共享 prompt”，不是每张图一个 prompt，也不是每个类别一个 prompt。

## 21. `random_patch` 是不是每次都随机叠加在一个位置？

是的。

`RandomPatchPrompter.forward()` 中：

```python
x_ = np.random.choice(self.isize - self.psize)
y_ = np.random.choice(self.isize - self.psize)
```

每次 forward 会随机采样 patch 的左上角位置。

然后：

```python
prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch
```

把同一个可训练 patch 放到这个随机位置。

注意当前代码里，一次 forward 只采样一个 `(x_, y_)`，所以同一个 batch 里的所有图片使用同一个随机位置。

### 21.1 不是每个类别一个 prompt 吗？

不是。

这个项目默认是：

```text
一个任务/数据集训练一个 prompt。
```

例如：

```text
CIFAR10 -> 一个 prompt
CIFAR100 -> 一个 prompt
EuroSAT -> 一个 prompt
```

而不是：

```text
airplane -> 一个 prompt
cat -> 一个 prompt
dog -> 一个 prompt
```

所以 `random_patch` 的含义不是“类别专属 prompt 随机出现”，而是“同一个任务 prompt 在随机位置出现”。

这更像一种随机增强，迫使 patch 不要只依赖固定位置。

你的直觉也没错：如果未来你要做光学 prompt，随机位置可能不太适合，因为光学系统里的 prompt 通常是固定相位板、固定 SLM 区域或可控调制区域。对光学 MoE 来说，`padding` 或 `fixed_patch` 更容易解释和迁移。

## 22. `utils.py` 里的函数为什么要有？

### 22.1 `accuracy(output, target, topk=(1,))`

它计算 top-k 分类准确率。

模型输出：

```text
output shape = [batch_size, num_classes]
```

比如 CIFAR10：

```text
output shape = [8, 10]
```

每一行是模型对 10 个类别的分数。

`top-k` 的意思是：

```text
看分数最高的前 k 个类别里，是否包含真实类别。
```

`top-1 accuracy`：

```text
分数最高的那个类别必须等于真实类别。
```

`top-5 accuracy`：

```text
分数最高的前 5 个类别里包含真实类别，就算对。
```

这个项目里主要调用：

```python
accuracy(output, target, topk=(1,))
```

所以实际报告的是 top-1。

为什么函数写成 top-k 通用形式？

因为 ImageNet 等视觉分类任务常报告 top-1 和 top-5，这是视觉分类代码里的常见工具函数。

### 22.2 `AverageMeter`

训练时每个 batch 都有一个 loss 和 accuracy。

但是我们通常要看平均值，而不是只看最后一个 batch。

`AverageMeter` 做的是：

```text
sum += 当前值 * batch_size
count += batch_size
avg = sum / count
```

在代码里：

```python
losses = AverageMeter('Loss', ':.4e')
top1 = AverageMeter('Acc@1', ':6.2f')
```

分别记录平均 loss 和平均 accuracy。

### 22.3 `ProgressMeter`

用于打印训练进度。

你看到的这种输出：

```text
Epoch: [0][0/2] Time ... Loss ... Acc@1 ...
```

就是它组织出来的。

### 22.4 `save_checkpoint`

用于保存训练结果。

这里保存的是：

```python
prompter.state_dict()
```

也就是 visual prompt 参数。

不是保存 CLIP 或 ResNet 参数，因为主模型是冻结的。

### 22.5 `cosine_lr`

用于学习率调度。

训练初期 warmup，后面按余弦曲线下降。

训练循环里每个 batch 都会调用：

```python
scheduler(step)
```

它会更新 optimizer 里的学习率。

### 22.6 `convert_models_to_fp32`

CLIP 在 GPU 上可能会用 fp16。

这个函数把模型参数转成 fp32，减少混合精度训练时的数值问题。

### 22.7 `refine_classname`

CLIP 需要把类别名填进文本模板：

```python
template = 'This is a photo of a {}'
```

如果类别名是：

```text
maple_tree
```

`refine_classname` 会转成：

```text
maple tree
```

这样得到的文本更自然，CLIP text encoder 更容易理解。

## 23. `--prompt_size 30` 怎么理解？

`prompt_size` 是 visual prompt 的空间尺寸。

不同方法下含义略有不同。

### 23.1 对 `padding`

```bash
--method padding --prompt_size 30
```

表示边框宽度是 30 像素。

如果输入给 CLIP 的图像是：

```text
224 x 224
```

那么中间不加 prompt 的区域是：

```text
224 - 30 * 2 = 164
```

prompt 占据四周边框。

### 23.2 对 `fixed_patch` / `random_patch`

```bash
--prompt_size 30
```

表示 patch 是：

```text
30 x 30
```

### 23.3 原图大小是多少？

CIFAR10/CIFAR100 原始图片是：

```text
32 x 32
```

但是 CLIP 和这些预训练视觉模型通常需要：

```text
224 x 224
```

所以数据加载时会经过 preprocess，被 resize/crop 到 224。

代码里的默认值：

```bash
--image_size 224
```

对应的是“送入模型的图像尺寸”，不是 CIFAR 原始尺寸。

## 24. 为什么 `--learning_rate 40` 这么大？

这个学习率看起来很大，是因为训练对象不是普通网络权重，而是输入空间里的 prompt 像素参数。

几个原因：

1. 只训练 prompt，参数量很小。
2. 主模型冻结，梯度只更新输入端 prompt。
3. prompt 初始化是随机像素，可能需要较大步长快速改变输入分布。
4. 原论文/官方代码采用了这种设定。

但是这不代表它永远合理。

如果你发现训练不稳定，可以尝试：

```bash
--learning_rate 1
--learning_rate 5
--learning_rate 10
--learning_rate 20
```

对于你未来的光学神经网络，学习率不一定能沿用 40。光学网络里的相位、振幅或衍射层参数通常有物理约束，学习率需要重新调。

## 25. CIFAR10 和 CIFAR100 有什么区别？

两者都是小图像分类数据集，原始图像大小都是：

```text
32 x 32
```

区别是类别数：

```text
CIFAR10:  10 类
CIFAR100: 100 类
```

CIFAR10 类别较少，例如：

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

CIFAR100 更细粒度，类别更多，任务更难。

对于你未来做光学神经网络，我建议先用 CIFAR10：

```text
类别少，训练快，debug 容易。
```

等流程稳定后再上 CIFAR100。

## 26. 如果未来光学神经网络要在 CIFAR10 上训练与测试，该怎么改？

如果你的光学神经网络是“有 label 的监督训练模式”，它更接近 `main_vision.py` 的范式，而不是 `main_clip.py`。

你需要的最小训练流程是：

```text
CIFAR10 image
  |
预处理/编码成光学输入
  |
光学神经网络
  |
输出平面光强
  |
读出成 10 类 logits
  |
CrossEntropyLoss(logits, label)
```

### 26.1 数据集替换

在代码中从 CIFAR100 改成 CIFAR10：

```python
from torchvision.datasets import CIFAR10

train_dataset = CIFAR10(args.root, transform=preprocess, download=True, train=True)
val_dataset = CIFAR10(args.root, transform=preprocess, download=True, train=False)
```

### 26.2 输出类别数

你的模型最后输出应该是：

```text
[batch_size, 10]
```

因为 CIFAR10 有 10 类。

如果光学网络输出的是二维光强图，你需要一个读出方式，例如：

```text
把输出平面划分成 10 个探测区域
每个区域积分光强
得到 10 个类别分数
```

伪代码：

```python
intensity = optical_model(images)      # [B, H, W]
logits = detector_readout(intensity)   # [B, 10]
loss = F.cross_entropy(logits, labels)
```

### 26.3 是否还需要 visual prompt？

取决于你的实验设计。

如果你只是训练一个普通光学分类网络：

```text
image -> optical net -> logits
```

那不一定需要 visual prompt。

如果你要研究光学 prompt / 光学 MoE：

```text
image + optical prompt -> optical experts -> logits
```

那 prompt 就是一个额外可训练或可切换的控制输入。

### 26.4 推荐路线

建议分三步：

1. 先做普通 CIFAR10 光学分类：

```text
image -> optical network -> 10 类 logits
```

2. 再加固定 optical prompt：

```text
image + prompt -> optical network -> 10 类 logits
```

3. 最后做光学 MoE：

```text
image + task prompt -> router/光学调制 -> 多专家路径 -> logits
```

这样每一步都有可验证 baseline，不会一开始就把问题变得太复杂。
