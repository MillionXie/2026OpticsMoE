# 给师姐 / 接手 AI：架构与实验部署合同

## 先固定版本，不要重新猜模型

Best SHA256：`c9926cbaaa1ef066d9657a8028dffc130a33915aa9f392551573dc79894192d0`。
Protocol SHA256：`f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde`。
源推理/训练验证 commit：`67566b888d56a1b1913e956012ac3567d7dd54f7`；交付新增文档/入口的 commit 见 MANIFEST。
严禁把旧 70% 类别检索的 assets、teacher cache、split 搭到这份权重上。
原 protocol 的 `initialization_required` 指从头训练，当前是相同协议下已训练权重的 continuation，不能因此误重置 best。

读代码顺序：`frontend.py` → `model.py` → `optics.py` → `retrieval_screen.py`；继续训练看 `retrieval_adapt.py`、`retrieval_refine.py`。
所有都是 `standalone/` 内的相对导入，无需旧 experiments 代码。历史 `export.py` 是旧权重转换器，本版不运行。
加载：`torch.load('assets/best.pt', map_location='cpu', weights_only=True)`；
`OpticalRetrieval(payload['metadata'])` 后 `load_state_dict(payload['state_dict'], strict=True)`，必须使用 metadata 而非手写默认结构。

## 实際推理链（一个图片输入，一个 64D 描述符）

```text
RGB → 保长宽比白边补齐224² → 冻结 Qwen patch Conv3D + 位置项
    → [B,196,1024]
    → V 输入投影192
       → V光router → V光expert + E1 同尺度融合
       → V光global + E2 同尺度融合
    → 投影回1024 → 原patch恒等残差 + sigmoid(gate)×更新量
    → 冻结Qwen 2×2 merger → [B,49,2048]
固定prompt tokenizer + 冻结选定词嵌入 → 插入49个图像token
    → [B,77,2048]（包含28个文本/特殊token）
    → L输入投影192
       → L光router → L光expert + E1 同尺度融合
       → L光global + E2 同尺度融合
    → token mean/max拼接384 → LayerNorm → Linear384→64 → L2
    → 与1600张gallery的64D向量做余弦排序 → exact-SKU Hit@1
```

无 native Qwen Transformer/attention/大型隐含视觉网络。冻结 frontend 仍包括 patch、位置、merger、词嵌入。
prompt 固定；不是用商品标题来检索。L 输入包含图像，因此 L router 随图片变化。
原 patch skip 是无额外网络的恒等残差，不能删除或解释为又添加一个训练分支。
电子残差为 Vision depthwise3×3 / Language causal1D5，pointwise 与 MLP192→384→192。
每次融合把 E/O 调到同 RMS 再做 `(1-alpha)E + alpha O`，随后保持尺度；以 `model.py` 为准。
当前 readout 是原有 linear64，83%→83.125%只把 TRAIN 同SKU内协方差收缩线性变换折叠进该 Linear 的 W/b。
推理无协方差拟合、无 SKU 标签、无新网络。拟合强度.02由 QUERY比较选中，注意披露选模口径。

## 光路不可更改的部分

- 波长532nm，传播10cm，逻辑pitch17μm；有效478×478，计算padding到518×518；padding不显示在SLM。
- 每侧3次捕获：router → expert → global；两侧共6次，不是6块透镜串联的全光网络。
- router振幅224×224居中（有效面127:351），相位同位置；CCD四个59×59探测窗左上角为 `(y,x)` 的164/255组合。
- expert 4个224×224，2×2，间距30；有效面起点 `(0,0),(0,254),(254,0),(254,254)`，依次TL/TR/BL/BR。
- router电子softmax温度2、Top2；权重以L2归一化施加在**振幅**，不是强度；全场相干传播，不分专家独立传播后拼图。
- global为478×478相位，输入重新编码前次融合后的特征，沿用该侧router权重。
- 原相位参数是raw，物理相位 `2*pi*sigmoid(raw)`；raw=0意味着pi；不能将raw直接线性缩放当相位。
- FP32相位，无k滤波、无训练8bit直通量化。训练有未调制/CCD扰动；正式参考 `eval()` 无随机训练噪声，不声称83.125是20%固定漏光实测值。
- 当前readout `prefix_rows`：CCD均值归一化→clip12→log1p→pool224²→rowLN→ReLU→取V196/L77行→Linear224→192。
  **不能擅自改成全场行重采样或取消log后仍声称同一模型**。可做新版本对照，但必须复评。

## 硬件改造入口与逐阶段采集

代码已有注入点（只取非负线性强度，不取对数显示截图）：

```python
model.vision.optics.router.measured_ccd = ccd_v_router  # [B,478,478]
model.vision.optics.measured['expert'] = ccd_v_expert
model.vision.optics.measured['global'] = ccd_v_global
# language同理；model.eval()，模型与CCD使用同一设备。
# 结束后清空，避免下一批误用上一批CCD：
model.vision.optics.router.measured_ccd = None
model.vision.optics.measured.clear()
```

这些只是替换光传播结果的接口，不会替你触发相机、生成本次振幅或按样本寻址。
接手AI需要实现 hardware runner：在 `encode`/router/fanout/propagate 边界导出当前振幅，捕获对应 CCD，注入后继续。
务必按下列链实时或逐批推进，不能拿仿真路由结果生成所有后续BMP后假称完整光路由实测：

1. V router振幅/相位 → 实测router CCD → 原探测窗计算Top2。
2. 按实测权重fanout V expert振幅/相位 → 实测CCD → decode与E1融合。
3. 根据融合特征重新encode V global振幅 → 实测CCD → decode与E2融合。
4. Qwen merger与固定prompt拼接后，重复L router/expert/global。
5. 原读出头输出64D；gallery与query均按相同硬件合同处理，不混用不同域图库。

每个缓存至少记录sample_id、stage、checkpoint/phase/amplitude SHA、LUT SHA、ROI/homography、曝光、等待、帧序号。
断点续采不能仅按文件名排序；已存在输出拒绝覆盖。先单图6阶段、再多图循环检查时序、最后全量。
实测注入将截断相位梯度；不能宣称将静态CCD喂进autograd就能直接训练真实相位。
可先用实测CCD微调下游电子，再以仿真代理/校准梯度优化相位、重新导出采集，并报告每轮差异。

### 物理尺寸和标定，最容易出错

已知目标设备为振幅1024²/17μm，相位1920×1200/8μm；实际型号仍须现场核对。
有效物理宽度478×17=8126μm，在8μm相位面约1015.75像素；每专家224×17=3808μm，对应476像素。
不要把478逻辑像素直接显示成478相位面板像素，也不要把上述非整数边界逐块round造成累计漂移。
按统一物理坐标、现场放大倍率、中心和像素面积映射重采样整张相位，包中不武断提供免标定物理BMP。
相位2π→灰度映射依赖该面板/波长LUT，线性灰度只是理想假设。振幅灰度需要已标定的field-amplitude LUT。
`encode`振幅可能>1；若硬件0..1范围，保存全帧公共缩放并记录，不可直接clip或对每专家独立拉伸，否则改变门控权重。
SLM外围须按光阑/最暗透过率处理；相位0不是遮光。不要复制其他电脑的LUT/曝光/ROI当作已正确。
CCD保存线性原始强度（优先16bit），矫正到逻辑478²；方向由非对称/单角标定确定，homography已处理镜像后不得再翻。
暗场、曝光不饱和、幅度响应、相位响应、双SLM配准、CCD映射、写入完成等待及丢旧帧都要独立验证。
不允许用逐图min-max/CLAHE/gamma增强后的PNG喂模型，再混称原始CCD；显示图与模型输入分开。

## 结果检查

正常/去光必须同一best、同一照片、同一gallery，不重训去光版本；`set_remove_optical(True)`跳过router/expert/global并使用E1/E2。
QUERY的专家选择份额（Top2事件数1600为分母，四项总和100%）：
V `[26.75,26.3125,22.5,24.4375]%`，L `[13.875,30.5625,26.875,28.6875]%`。
这不是每样本激活率或光能比例，不是全四专家同时激活。完整分布见reference报告。
复现先比较逐图结果，再查processor版本/白边预处理/浮点路径。别直接改权重或读取缓存伪装成原图推理。
训练、评估输出写新 `runs/<清楚的实验名>`；保留best/last、命令、环境、指标与SHA，勿把失败run不断打包给实验同学。
