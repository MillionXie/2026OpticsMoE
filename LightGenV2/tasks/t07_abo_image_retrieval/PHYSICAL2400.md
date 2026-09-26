# 2026-09-26全实拍图搜图

用户授权重新采集1600图库与800查询，共2400输入、六层14400张CCD，不微调。原20260925目录保留作混域对照，不覆盖。

师弟电脑输出：E:/code/guest/2026OpticsMoE/ABO_I2I_Lab_DVP_8um/runs/abo_i2i_20260926/full_physical2400。

日志为同级full_physical2400.log；progress.json记录已完成输入数（不是查询数）；batch_results每4张保存一次，ccd按阶段存ROI透视/方向处理后的线性PNG。无逐帧亮度归一化。最终report.json仅使用本轮800实拍查询与1600实拍图库；features.pt含全部2400实拍特征及对应仿真对照。

命令入口：run_full_physical2400_20260926.cmd，先加--max-batches 1做4张六层短测，成功并核查光场后去掉该参数续跑。默认20ms曝光、240ms等待、每次取6帧用最后一帧；使用20260925已记录的ROI和每层方向配置。本轮不重新以PCC筛方向。

先图库后查询。总600个batch；旧的200个查询batch不会混入。断点续跑严格核验run_contract.json。最终800查询与图库sample_id互斥；同SKU不同图为正确匹配。实拍域一致不意味着光路误差消失，结果须实际评估。
