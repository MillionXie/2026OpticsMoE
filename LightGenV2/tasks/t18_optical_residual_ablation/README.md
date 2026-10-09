# T18 Kather六层MoE：相干未调制分量消融

用户2026-10-09批准振幅混合：`U'=[(1-rho) exp(i phi)+rho] U`，固定rho=0或0.3。
不是30%功率；相干干涉使透射功率依赖相位。理想透射强度范围为
`(1-2rho)^2`至1，rho0.3时为0.16至1。不额外做逐层能量恢复，OEO沿用原协议。
未调制支路绕过相位调制，但仍经相同传播与OEO，不是绕过整个块的恒等捷径。

## 完整模型合同

沿用 `demo_check/reproduction/kather2016_experiment.py` 的历史六层MoE＋逐层OEO。
RGB150×150 → 固定抗混叠缩放/训练仿射 → 四块50×50 `[R,G;B,meanRGB]`
→ 振幅100×100。没有CNN、Qwen或可训练电子读出头。
路由相位100×100、5cm传播、九个32×32探测框，温度100 softmax得p，入口振幅
`p/sqrt(sum(p^2))`；3×3九专家均激活，理想relay复制同一图像，孔径146×146、
间距176。router不加残差，保持输入条件路由不变。

三个周期，每周期九张独立专家相位（局部padding30、206×206角谱传播裁回146）
→ 整面专家OEO → 全局498×498相位 → 全场传播 → 全局OEO。
九并行相位面算一层，每周期两层，合计六层＋一个额外router。
总参数1,329,544，仅相位参数；27专家相位＋3全局相位采用固定rho，router原样。
六次OEO：有效498×498整面强度LayerNorm（无仿射，eps1e-6）、ReLU、Softsign、
零相位振幅重编码。波长532nm、像素8um、画布500×500、主干及末端距离5cm。
最后八个互不重叠32×32 CCD框（两行四列、起点126/198、步距40），能量归一化
后argmax分类，无电子Linear。初相位沿用此历史协议raw uniform[-1,1]、2pi sigmoid。

任务模型仅替换30张主干PhaseLayer的前向；保留同一Parameter与state_dict键。
为确保专家批量FFT捷径不会绕过残差，**两组都关闭专家vectorize**，使用同一串行
专家实现。smoke须核验rho0与原计算一致、初始参数逐元素相同、全部主干都有梯度。

## 数据、配对与选模

Kather2016原5000张八分类，固定3496/752/752 train/val/test、每类437/94/94。
原数据CC BY4.0；固定缓存及manifest哈希核验，图像级划分，非患者独立验证。
服务器缓存 `/DATA/DATA1/guest3/demo_reproduction_data/kather2016/kather2016_fixed_split.npz`。
两组从头训练：seed17、batch16、AdamW lr0.002余弦至0.0002、30轮上限、至少15轮、
patience8、EMA0.95、梯度裁剪1、label smoothing0.02；分类NLL＋0.2捕获损失＋
0.02圆周相位平滑；训练旋转±10°、平移±3、尺度±5%。所有设置共用，验证NLL
选EMA checkpoint，不以测试选比例/超参。相同seed保证相同初始化、样本顺序和增强。
历史Kather测试已见过，本实验属探索性单种子，不给标准差或预设残差一定更好。
rho0重新训练作配对对照，不直接拿旧MoE表混作同次实验。

每组run写入本任务 `runs/simulation/`，只存best/last，保存配置、命令、Git、
源SHA、数据SHA、环境、功率诊断、完整曲线、逐样本结果。训练不读test；两组完成
后核验配对并锁定，再分别测试一次。GPU上限两张，按UUID选择空闲卡，进程退出释放。
不修改历史源码或旧run，不新建分支/工作树，源码按main发布后服务器Git同步。

入口：`run.py --phase smoke|train|evaluate --rho 0|0.3 --data ... --out ...`。
当前尚未产生训练结果。
