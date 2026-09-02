# ImageNet-21K/22K 光学骨干训练契约

本工程的中心目的，是将已固化的 8 层 P11 光学特征提取器扩展到大词表监督预训练，
而不是把 ImageNet-1K 图片换一个 21,841 类分类头就宣称完成了 22K 实验。

当前服务器没有 ImageNet-21K/22K，也没有可用的下载凭证。因此目前能做的只有代码、
索引和 100-batch 管线验证；正式启动脚本会在任何 GPU/NCCL 初始化和输出目录创建之前，
检查数据 manifest、源目录、文件哈希、P11 资产和磁盘空间，缺一项就终止。

版本必须由数据方 manifest 精确声明：

- 原始 Fall11：21,841 类、14,197,122 张；模板训练 90 epoch；无官方验证集，
  只导出 last raw/EMA，不使用 “best” 一词。
- MIIL ImageNet-21K-P Fall11：11,221 类、训练 11,797,632 张、验证 561,052 张；
  模板训练 80 epoch；训练/验证索引分别固化。
- MIIL Winter21-P 是 10,450 类的另一版本，不能仅凭文件夹名称当成 11,221 类版本。
  其他 P 版本样本数必须由所持有的确切发布版 manifest 给出，代码不猜数。

初始化只读取 `_assets/8stage/checkpoints/backbone.pt`。新建大词表 readout 后，严格断言
缺失键恰好是 `readout.*`，unexpected 为空，且 stem/backbone SHA 完全匹配；原 1000 类头
不会被复制。

训练采用 layer-wise AdamW、独立相位学习率、Mixup/CutMix soft-target CE、AMP、EMA 和
DDP 精确恢复。临时分类头不计入可部署 backbone 的光学参数占比。完成 21K/22K 预训练后，
下一阶段是在独立工程中重置 1000 类头，以较低 backbone LR 进行 30 epoch ImageNet-1K
微调，再开展冻结线性探测、下游迁移和鲁棒性验证。
