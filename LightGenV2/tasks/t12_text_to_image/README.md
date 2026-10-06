# T12 图文条件商品编辑：main 统一入口

当前实拍主表使用17.03M小版：原权重 `5b4f9a37…` 与末端decoder适配权重 `eeec…`
分别保留。原权重仿真34.277751dB、未适配实拍27.551289dB；适配权重实拍31.552886dB、
其自身仿真28.890612dB，不能跨权重拼成sim-exp差距。精确指标口径和原产物见下面复现入口。

大版149.76M、小版9.96M及其他17M候选是分别保留的历史版本/必要对照，
不是几条新的长期开发分支。每套支持换背景、换目标、同时修改；高级材质及失败的
开放形态实验不作为最终版本。此页在main维护，旧分支名只记录原实验出处。

唯一入口：[复现与最终产物](reports/reproduction/README.md)。旧报告保留作历史记录，不代表当前部署入口。

正式 17M 实验室捕获与电子适配的 17 份源码已逐 SHA 收录；来源、旧入口恢复点、CPU 检查和尚未迁移的环境边界见 [硬件源码登记](lab_shs8um/SOURCE_STATUS_20261006.md)。这是源码身份收敛，不是新实验或硬件运行环境验收。

已保存PNG的逐图感知/区域指标工具现在由[任务内指标入口](perceptual/README.md)管理。
仅修复旧导出包的模块改名依赖，不改变已有图片、结果或模型；旧作图及汇总源码保留。

旧 `handoffs/t12_small_baseline_share_20260928` 是17M模型对外交付快照，不是第二个开发工程。
其中231份源码/报告/配置载荷已逐字节或显式CRLF核对主线/原历史提交，并保留三端Git恢复身份；
原文件、数据、权重及全部测速不删，仅退出Git待同步列表。两个交付辅助脚本及两份不同的
历史说明继续可见，未被整目录忽略。精确范围见
[旧交付快照身份](../../../maintenance/storage/T12_SHARE_SOURCE_PAYLOAD_VISIBILITY_20261006.json)。
保留本地交付文件的机器可用 `python maintenance/git_safety/check_source_archives.py --t12-share-payloads`
只读检查这231份原字节是否变化；源码新克隆未附私有载荷时会失败，不自动重建或覆盖。

配对数据导出工具已归任务：`python -m LightGenV2.tasks.t12_text_to_image.export_baseline_pairs
--dataset-root <原数据目录> --instruction-cache <原指令缓存> --split test --output <新目录>`。
它不再寻找交付包里的source/assets，保留原round量化、配对字段及20736/2304/2304数量守卫，
拒绝覆盖已有输出；GT只用于导出目标，不能交给baseline推理。三项合成CPU测试不读取正式数据。
该入口使用当前main的数据实现；未做历史渲染逐像素等价核验，不能宣称重建原交付PNG。
原交付export_pairs.py及其冻结source、数据、结果继续保留，不替换历史引用。
旧stage/materialize_pairs.py所用的单split `input/target/pairs.csv`布局，可在同一主线
导出入口加`--manifest-format csv`得到；保留prompt和原配对列，不再另依赖交付source路径。
CSV不支持all，避免旧目录布局混用。原物化脚本和已导出数据仍保留；这里只验证格式合同，
未重新导出正式数据或声称历史像素逐一复核。

原17M单样本干净仿真另有明确入口：`python -m LightGenV2.tasks.t12_text_to_image.infer_formal_sample --help`。
显式提供原5b4f PT、数据、指令缓存和embedding缓存，固定CPU、不调用设备，拒绝其他权重及覆盖输出；
保留旧交付工具的1042+index随机种子和floor PNG量化。推理仅接收reference、prompt，不传GT。
合成测试验证种子与量化，不代表正式图像性能重放或全部资产闭包。旧infer_ours.py仍原样保留。

## 2026-09-27/28 同任务 baseline 补训与五组汇总

新增外部baseline：官方pix2pix-Turbo，upstream锁定 `86f54146590ffb4543c8cf85b5a36657da670924`。沿用当前20736/2304/2304配对数据及256输出；SD-Turbo预训练骨干，不从零训练主干。CLIP文本编码器冻结，微调UNet/VAE LoRA、输入卷积及官方VAE skip卷积。完整推理参数1,299,445,747，按约定排除词嵌入50,593,792后1,248,851,955；实际微调9,505,160。没有Qwen或PCA条件接口，不把它标作Qwen baseline。3轮训练在完整VAL选中step31104，固定权重TEST2304取得PSNR20.746391dB/SSIM0.740299；权重SHA256 `3a347c58affb53d8e7efc583bb5aecdaa2ac33bd316d792c12c805fa837b4c7c`。这一外部baseline明显弱于当前模型，不能宣称凭参数规模质量必然更好。来源、实现差异及命令见复现入口。

完整 Qwen28 + 原电子 UNet/adapter + 冻结 VAE baseline 已完成当前 TRAIN20736 的3轮补训，VAL2304 按逐图平均PSNR选中step15000，固定后评估TEST2304。没有缩减Qwen层数，没有用窄头缓存代替Qwen28。训练源码 `b813628ea`，导出源码 `c657f959d`，checkpoint SHA256 `599805ad3062bebf67acf3b515f0a812fb843506643e18251004f180131b4b52`。

统一256×256、浮点RGB[0,1]、逐图PSNR后平均：原17M小版仿真34.277751dB，小版电子decoder微调后EXP31.552886dB，大版仿真31.428599dB，同任务补训baseline27.260453dB。参数预算依次17.026642M、17.026642M、149.755866M、1834.345963M；含固定条件缓冲值，按约定另列冻结词嵌入311.164928M。

**小版主表仿真与EXP不是同一权重**：原5b4f权重EXP27.551289dB；微调eeec权重仿真28.890612dB。不可用34.28与31.55之差代表同权重仿真—实测差距。当前表比较已完成的模型，不等于排除不同训练历史后光学模块的因果优势。旧跨任务baseline保留为历史诊断，不纳入当前主表。

本地交付 `handoffs/t12_four_group_summary_20260927`，四模型每个包含2304张原生生成PNG、对应输入/GT、逐图CSV/JSON，代表图按类别/模式固定索引选取而非质量筛选。表格位于 `outputs/t12_four_group_summary_20260927/T12_four_group_performance.xlsx`。完整复现命令与局限见复现入口。本任务不采集光路，不使用GT贴回、超分或锐化。

新增pix2pix-Turbo原生TEST全量图片、报告、同样本拼图及五组逐图汇总位于 `handoffs/t12_pix2pix_turbo_20260927` 和 `handoffs/t12_five_group_summary_20260928`；五组表格 `outputs/t12_five_group_summary_20260928/T12_five_group_performance.xlsx`。保留旧四组文件原样，避免覆盖已引用的结果。未新增跨架构统一延迟实测，不把既有小版旧计时挪用于17M或pix2pix。

## 历史记录：2026-09-27 小版物理鲁棒性候选

本分支追加语言均衡/极强扰动试验，**均衡尚未解决**：所有试验实际语言top2仍为专家1/2。仅作为图像鲁棒性改善候选交付 `5b4f9a37…`（17,026,642参数，与前版预算相同），完整VAL2304：clean34.4315/.928509、stress29.3008/.892036、severe28.4256/.885481、extreme27.7628/.879783。候选保留原standardized_region_energy读出，不启用实验性log读出。来源为 `20260927_language_balance_input/last_checkpoint.pt`，不是该run未通过均衡守卫的best_checkpoint.pt；在完整VAL上仅按图像质量/干净保护人工选为备选，不能称为均衡选模成功。新增极强扰动已进入TRAIN，不再称为留出未见扰动；TEST仅固定权重后评估。交付目录 `handoffs/t12_language_balance_20260927`。

最新候选为 `5496204d…`、17,026,642参数、VAL选择step800。训练源码 `4065cbfa2`，新增文字条件空间先验，不改变光路或幅度/BMP编码。完整2304条VAL clean=34.1925/.924963、stress=27.7766/.877212、severe=27.0017/.870978、extreme=26.3708/.864833（PSNR/SSIM）；全部未实测。alpha下限.35，实际四融合值.4730/.4618/.4615/.4668；语言专家仍偏斜，残影/纹理不足仍存在。交付入口 `handoffs/t12_channel_severe_20260927`，完整配置与局限见复现报告；下面13cf版为历史候选，不是最新权重。

分支 `codex/t12-physical-robust-v2-20260927`。从 bounded-tanh 权重 SHA256 `3901e4fb…` 继续训练，不改变网络、tokenizer、Qwen-style 两层文字头或 478×478 相位区域；预算仍为 9,958,098。新权重 `13cf9a201bce42f58c19a0ff85fd11fe9940e18be036a86adf163320303edc3e`。

TRAIN 微调、VAL 选模，保留干净教师约束和干净/扰动双前向。训练通道明确将 30% 定义为干涉前的名义泄漏支路**功率比例**，使用 sqrt(0.7)/sqrt(0.3) 的复场叠加；不直接给图像加常数。CCD、错位、k-space、phase bypass dropout 仅训练/扰动评估时开启。部署幅度严格保持保零 `tanh(abs(E)/0.5)`，保留复场相位；BMP 仅 `round(255*a)`。

完整 2304 条 VAL：干净 PSNR 33.6471→33.5685，SSIM 0.920208→0.919405；组合扰动 PSNR 27.3636→27.7775，SSIM 0.867222→0.871593。选模清晰度保护阈值为 PSNR 下降≤0.2 dB、SSIM 下降≤0.002。新权重**尚未做光路实测**，不能将合成扰动改善等同于实测改善；用户另行安排实测。

## 架构与边界

输入 RGB 图像及文字，输出单次前向 256×256 RGB。原 Qwen tokenizer 和冻结词嵌入保留；文字主干是训练过的两层缩窄 Qwen-style Transformer，不是原预训练 Qwen 层。语言和视觉均调用审计版 DC20 光电模块：电子/光学同输入并行、router/top-2 experts、global、RMS 融合，历史alpha下限.4，本轮候选.35。478×478 有效光学区域，四个 224×224 专家分区；两路纯相位参数合计 958,728。

大版为 D640 文字头、条件适配器、128/256/512 窄 UNet、冻结 VAE encoder/decoder；小版为 D512 文字头与 48/96/160/224 CNN encoder/decoder，无额外 decoder refinement。冻结词嵌入 311,164,928 参数按约定单独列出，不计预算。大版神经网络 144,630,618 参数，另固定条件缓冲值 5,125,248，合计 149,755,866；小版 9,958,098。

推理不做 GT 掩码贴回或商品检索；模型使用可学习软融合。训练 GT 背景是程序合成的有限场景，目标商品来自封闭目录，因此不可宣称开放域生成或未见商品设计。种子多样性弱、专家负载偏斜仍是局限。

数据为 ABO 灯/桌/靠垫，12 款指定目标。基础 train/val/test 为 1728/192/192 图像，商品身份 144/24/24；扩展指令对 20736/2304/2304。训练包含像素、边缘、感知与局部细节损失和弱 PatchGAN；小版从大版蒸馏。判别器和感知网络仅训练使用，不计部署参数。

原实验整合分支为 `codex/t12-audited-editors-20260926`，这是历史出处，不是当前开发入口。
当前源码归入main；已有运行目录仍保护，不因此切换或覆盖。权重自带结构配置，严格加载，
历史兼容源码及必要checkpoint保留用于baseline/旧结果复查。

原17M实拍权重可从主线只读核验：
`python -m LightGenV2.tasks.t12_text_to_image.verify_formal_checkpoint --checkpoint <原5b4f权重路径>`。
该入口仅CPU检查SHA、原构建器严格加载及参数数目，不读取数据/调用SDK；不接受eeec适配权重，
也不重新测精度。配置与光学实现仍复用T01及历史后端，相关依赖未解除前不能删除。
