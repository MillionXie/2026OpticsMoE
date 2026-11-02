# LG2图文条件场景生成：仿真推理包

固定原F6光学前端、70块UNet和八步采样。输入是白底商品RGB图与完整两句描述，
Qwen图像信息为256，输出512。运行包含GPU FFT光学仿真，不需要相机或SLM SDK，
也不使用训练特征缓存代替RGB／文字的真实前向。

## 目录与运行

目标根目录为`/autodl-tmp/lg2`；本机该路径指向已有数据盘`/root/autodl-tmp/lg2`。
本包位于`t12/`，不改原有bloodmnist／kather等任务的数据。

```bash
cd /autodl-tmp/lg2/t12

# 先检查全部数据与权重SHA；不会初始化GPU
bash run.sh --verify-only

# 单个固定TEST请求的烟雾测试；输出目录必须为新目录
bash run.sh --device 0 --limit 1 --output "$PWD/outputs/smoke_$(date +%Y%m%d_%H%M%S)"

# 全部500个固定TEST请求
bash run.sh --device 0 --limit 500 --output "$PWD/outputs/TEST500_$(date +%Y%m%d_%H%M%S)"
```

`run.sh`与本说明链接到Git同步的源码；`SOURCE_VERSION.json`列出发布commit和逐源码SHA。
不能把目标主目录HEAD当成T12的实际执行版本。推理使用独立`.venv`，不会升级服务器共享Python。
如需重建环境：`python -m venv .venv`后用`.venv/bin/pip install -r requirements.txt`。
需要CUDA GPU；当前目标3080 Ti为12GB，`--offload auto`自动采用逐模块CPU offload，
保留BF16、原VAE、网格和权重，仅改变驻留位置，生成会比24GB常驻执行慢。
选中的GPU有其他任务时入口拒绝启动，不会停止其他进程。

## 身份与内容

- LG2所选PT：`aa2d277bd743d4dd831ec33852629dbf3001e772b55a8ac436a0d2a557a06fd8`。
- F6前端PT：`6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e`。
- 70块原UNet、deep_keep10；6V／4L、单router／E／G每模态、MoE4 top2、六光周期。
- 八步`958,859,727,595,463,364,232,100`；CFG7.5、IP0.25、conditioning1024、joint_image_context=false。
- `data/`为原624TRAIN目标／参考／文字及500TEST请求／参考，保持原划分和逐文件身份。
- `models/`包含Qwen、SDXL、IP-Adapter的原资产、配置、分词器及相应许可／来源说明。
- `weights/`是所选LG2与F6；`provenance/`只有必要的身份报告，没有教师特征PT。
- `reference_simulation/`保留同aa2权重在原4090上的完整500图及元数据；
  `reference_metrics/`是原报告。新GPU或执行后端的PNG可能不同，不能把这些参考图当成本机新生成结果。
- `DATA_manifest.json`列明1313项左右的资产身份；实际数量以收据为准。
  `DATA_transfer_receipt.json`记录目标端逐文件SHA核验结果。

TEST为18款商品、26视图的500请求，没有真实场景GT；原TEST已用于选PT，不能称独立泛化。
TRAIN为44款76视图的624条审核保留样本（365完整／259弱监督）；原703候选中79条被排除。
ImageReward不参与当前验收，不为本包新计算。原始失败图和历史报告保留。
本包不包含ResNet或三个外部baseline，也不表示现场光路已经验收。
单张烟雾测试只能证明入口能运行，不能替代完整500评分或物理实验。
