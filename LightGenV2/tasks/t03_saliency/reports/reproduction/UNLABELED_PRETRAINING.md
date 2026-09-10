# 额外无标签图像：预训练数据准备

状态：只准备、审计图像池，**尚未生成教师预测、尚未训练、没有新CC结果**。
当前两张GPU仍用于教师预热/联合续训对照，不能因本方案占用第三张卡。

## 为什么检查这条路线

在同一10000张SALICON图像上反复续训的已完成best为0.86204960，目标0.88尚未达到。
可以尝试将现有教师对额外未标注图像的输出用于预训练，再回原SALICON真值微调，
让原光电网络接触更多视觉变化，而不增加推理网络/参数。
依据是[Noisy Student，CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.html)
利用教师伪标签与额外无标签数据、学生训练扰动的思路。该论文是ImageNet分类，
这里不是其完整复现，也不能据此承诺显著性CC会提升。具体预训练损失/预算待当前对照后确定。

**公平性必须披露：若采用本路线，学生多用了无标签预训练图像，不能再声称与原baseline
使用完全相同的训练数据预算。**原10000图训练结果与新方案分行保存，不覆盖baseline。
教师只在离线生成目标时运行，不能加入学生推理；所有既有光路、Top2、alpha及轻量结构约束不变。

## 已有服务器数据与重叠风险

服务器已有 `data/COCO2017/train2017` 118287张，未下载或移动任何数据。
只按COCO数值ID核查发现：与SALICON train重叠10000张，**与SALICON public-test重叠4376张**。
因此不能直接拿整个COCO train2017预训练。排除SALICON train与val全部15000个ID后，
剩103911个候选。初步seed17042选择20000张的现有文件共3248018579字节（约3.25GB），
这些是已有磁盘内容，不会复制成另一份数据集。

正式准备器进一步对完整SALICON train/val文件及选中图像计算SHA256：

1. 要求排除目录完整含10000/5000张，拒绝错误/缺失目录和重复数值ID。
2. 排除全部训练、测试数值ID，固定seed从排序候选中抽样，再按ID排序。
3. 排除与训练/测试文件字节完全相同、或选中集合内重复的文件；不会为补齐数量偷偷改变抽样。
4. 保存实际数量、每个原图的SHA与文件名、ID哈希、排除原因和Git commit；不保存额外图像副本。

仅ID初选的20000个ID哈希（还不是内容去重后的正式manifest）：
`beb2014d0029c7cda21543b8964ea8fbb5e864bf2cb5189cd975985f1a3e4624`。
哈希口径是数值排序、十进制ID逐行、末尾换行。不能与旧流程无末尾换行的sample_id哈希直接比较。
上述排除不保证发现不同ID且重新编码的近重复图片；该边界须保留。教师导出时还必须验证图像
可解码、文件SHA与manifest一致，且不能把伪标签当成原始眼动真值。

## CPU准备命令

在服务器仓库已同步、已测试的源码版本中运行；输出必须不存在。该命令不导入torch、不使用GPU，
只顺序读取文件计算哈希；它不是预训练入口。

```bash
python -m LightGenV2.tasks.t03_saliency.prepare_unlabeled_pool \
  --coco-root /DATA/DATA1/guest3/2026OpticsMoE/data/COCO2017/train2017 \
  --salicon-root /DATA/DATA1/guest3/2026OpticsMoE/data/SALICON \
  --count 20000 --seed 17042 \
  --output /DATA/DATA1/guest3/2026OpticsMoE/cache/qwen3_vl_embedding_2b_salicon_lightgen/coco20k_pretrain_20260910/image_manifest.json
```

后续教师预测、光电预训练和真值微调应分别有身份清单；只在空出当前两张卡之一后执行。
预训练仅使用上述排除后的额外图像；正式微调仍只使用10000张SALICON train，5000张public-test
仅按既有已披露的公开测试选模口径评估。完成标准独立复评之前不得更新正式成绩。
