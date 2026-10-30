# 自然两句F9：冻结500请求与光路交接

## 固定评价结果

E662电子30步与F9单组LG12步各生成500张512×512图；输入白底ABO RGB256与两句自然提示。
相同source-only种子、同一请求，Qwen仅前向一次，无生成图回输、参考VAE输入旁路或GT推理输入。
冻结清单先于首次预测创建，模型／请求／采样／合并LoRA／评分不变；没有根据TEST结果调参或选模。

| 类别 | 模型 | CLIP-T | DINOv2-I | CLIP-I | ImageReward compact |
|---|---|---:|---:|---:|---:|
| 总体500 | 电子30步 | .3011685 | .3485514 | .7802747 | -1.8456397 |
| 总体500 | LG12步 | .2959249 | .3309013 | .7685478 | -1.8414377 |
| 台灯250 | 电子30步 | .2976862 | .3916494 | .7933228 | -1.7402514 |
| 台灯250 | LG12步 | .2953103 | .3738508 | .7829847 | -1.7549496 |
| 靠垫250 | 电子30步 | .3046508 | .3054534 | .7672267 | -1.9510281 |
| 靠垫250 | LG12步 | .2965396 | .2879518 | .7541110 | -1.9279258 |

总体与两类别12项代理均值均通过用户最新10%容限，最差下降5.7297%为靠垫DINO-I。
负值IR下降仍按`(reference-actual)/abs(reference)`，不是直接取负分比值；归一化改变不是概率。
CLIP-T为完整提示的ViT-B/32余弦，DINOv2-small/CLIP-I为整幅参考与生成图的外观代理；
IR使用固定≤35 token简写，仅为辅助偏好，不能评估每个完整提示子句。
没有合格匹配场景真实分布，因此没有将白底输入冒充FID真实图池。

500请求实际来自10款、10原始视图、80图文上下文的多seed抽样，并非500独立商品。
这些是工程接受均值，不是已做显著性／统计等效检验或独立逐属性准确率。
全1000原始PNG、元数据和逐图分数已SHA核验下载；全量图页不按分数筛选。
人工系统检查的是各80上下文固定第一seed的160张输出，不是全1000图盲评。
仍见两端遗漏背景元素／改变参考造型、侧视靠垫变椅背，个别LG把侧视靠垫生成灯。
这些失败保留在图页，数字可比不等于全部指令已满足或绝对生成质量已合格。

## 模型和计时身份

- 电子E662：`scene_two_sentence_qwen_paired_20261029`，SHA
  `662ff90f9a46670b28bf86d61629c59e4fb619ed69a62e4ae52ed8527ea76f76`。
- 单组光前端F6：`qwen3_scene_LG_frontend_20261029`，SHA
  `6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e`。
- 生成F9：`scene_natural_single_EG_joint12_training_20261030`，SHA
  `f9fbdb7fd36d1c862686616a624e7c18f4e9149cef61b0704bbf97767de6eaec`。

每模态一router＋一E/G，4专家top2；保留视觉6／语言4原TF，移除42/52，光宏后无TF，
电残差与光残差保留，alpha约.649且≥.6，共6光周期。478 active/518 canvas/224 expert，
有界tanh只做一次，保复／符号相位，BMP round255*a，不逐帧峰值归一。
硬top2在DEV128固定V[0,2]／L[0,1]，不宣称均衡动态路由。
LG的既有image QueryResampler使用visual＋hidden（joint_image_context=true），E662为false；
没有增加头参数，但这不是只替换TF的单因素消融。尾部从30到12步且额外训练原头／UNet LoRA。

匹配F9／v4的独立DEV8请求×3进程／模型软件完整wall：电子2873.149673ms，
LG1240.683641ms，2.315779倍。B1、保留首请求、无额外预热、模型加载窗外，
PIL RGB／文字就绪至CPU RGB；包含CPU等待、FFT仿真和电残差。
这不是本次TEST500全量时延，不是物理光学整机实测，也不是光学模块单独的加速贡献。

## 冻结与可追溯证据

固定执行源码`d4ab78d5326cb90651c97fee1fa3b0e3f6c835ae`。UTC10:42:30.495691建立的
frozen manifest SHA `fa241dafc73f4d39aac3dfde2afc9c50c5045dc3afaa84ef8106070275bdd50c`；
请求SHA `27b9ae4866937f5c92569ac68d260587da4dfaeec64eb0c290dd8f619ff7038e`。

任务`runs/simulation/`中的证据：

- `scene_natural_frozen500_workflow_20261030/report.json` SHA
  `6afd21a170eb909a3b1963b14ebc9fee8e56c8d5c29060fceb6e0d46552163cb`。
- 同目录`quality_gates.json` SHA
  `51dbd86172fe0c4f94fe46ea92c004af039f64320d7e93be2493f3ff5eaaa27e`。
- 同目录`gpu_release_receipt.json` SHA
  `54b68d1b13d4e25f08d14997c95b32fdaba0c92b9d21fb0f0415495c703bced1`。
- `scene_natural_electronic30_fused_TEST500_20261030`及
  `scene_natural_single_EG12_fused_TEST500_20261030`保存全部原生PNG／逐输入身份。
- `scene_natural_frozen_metrics1000_20261030`保存全部逐图分数和固定评分器资产SHA。
- `scene_single_EG_frozen500_delivery_20261030/index.html`为本地全量同题图页，
  `context_review/selection.json`记录不按效果筛选的80上下文人工检查索引。

原workflow及三个子PID均已完成退出，GPU释放true；历史权重、失败及原始结果保留。

## 光路交接边界

用户授权的“保住光路性能-实验室残差优化”对话接手光路专用入口、部署与现场校准。
准备报告见[HARDWARE_READINESS_20261030](HARDWARE_READINESS_20261030.md)，它是历史只读快照。
本次正式交接位于任务`runs/hardware/scene_F9_deployment_handoff_20261030`；handoff SHA
`e5ea8ba5dd115d453014b8a3c4e7d50565a550fb02892ce55095b5a18097e0d8`。

持久资产闭包在`runs/hardware/scene_F9_persistent_asset_closure_20261030`：34份基础模型
14,583,745,230字节逐SHA匹配原资产，无需/tmp；manifest SHA
`de7655ad9ffc0fea0eed12dbceb9a8e4d01d0f7b7b11c396ff3751bcacf4dd2f`。
三PT／必要JSON及白底参考已下载本地并逐SHA核验，不传源代码，不把完整原资产总量冒充部署参数。

**EXP–SIM≤5%是下一阶段目标，尚无实测验收结果。** 先完成离线接口、真实器件会话与标定核验、
固定TRAIN／校准DEV pilot，再按层全量采集。后续输入必须由前层实测CCD重算；不得拿理想后段
输入冒充实测传播，不得静默FFT回退、复用其他任务会话或用TEST500适配刷门槛。
新适配权重须另存SHA和匹配SIM参照，不能混权重／样本凑5%。
此报告不放行未经核验的设备会话，不宣称离线参考包已具备现场SDK或物理整机2倍加速。
