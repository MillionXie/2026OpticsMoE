# 方法、公式与实现边界

## 1. 单层前向

第 `l` 个 OEO stage 的输入是非负实振幅 `a_l`，相位参数是
`raw_phi_l`：

```text
phi_l = 2*pi*sigmoid(raw_phi_l)
E_l = a_l * exp(i*phi_l)
U_l = P_z(E_l)
I_l = |U_l|^2
N_l = LayerNorm_full_plane(I_l)
R_l = ReLU(N_l)
[alpha_l, beta_l] = softmax(residual_logits_l)
a_(l+1) = alpha_l*R_l + beta_l*a_l
```

`P_z` 是 5 cm、无 padding 的角谱传播。每个 stage 都执行一次平方律探测、
归一化、非线性和振幅重新加载，因此 20 stage 包含 20 次 OEO。

## 2. 三种 backward connector

设光学线性算子为：

```text
A_l(phi) = P_z diag(exp(i*phi))
```

前向对三种方法完全一致：

```text
U_l = A_l(phi_l_current) a_l
```

返回前层的误差信号分别为：

```text
BP:             A_l(phi_l_current)^H * delta_l
FA-pretrained:  A_l(phi_l_pretrained)^H * delta_l
FA-random:      A_l(phi_l_random)^H * delta_l
```

其中 `delta_l` 来自当前 batch、当前模型和当前 loss。`phi_pretrained` 在微调
开始时复制一次并注册为 buffer；`phi_random` 每层采样一次后固定。

## 3. 为什么 random feedback 仍使用物理形状算子

400 x 400 光场展平后有 160,000 维。显式构造一个
`160000 x 160000` 的随机矩阵不可行，也与物理传播的尺度不匹配。因此
FA-random 使用：

```text
固定随机 phase-only screen + 同一个自由空间传播算子
```

它与 FA-pretrained 具有相同输入输出 shape、单位模长相位和传播尺度，但不包含
预训练结构，是当前实现中合理的随机固定反馈 baseline。

## 4. 局部相位梯度

当前代码没有把相位参数的梯度替换成直接随机投影。对于当前层相位：

```text
dL/d(raw_phi_l)
```

仍由当前 `a_l`、当前 `phi_l` 和当前 `grad_output` 计算。只有返回给上一层的
`dL/da_l` 使用冻结反馈相位。代码在自定义 autograd function 中分别计算这两
个 VJP。

这个定义对应“固定层间反馈连接器 + 当前局部权重更新”，而不是冻结完整
backward graph。

## 5. 自适应残差

V1 的 residual 初始化为 optical/skip = 0.10/0.90。它能让 20 层网络稳定
开始训练，但也容易让网络绕开光学路径。V2 改为 0.35/0.65，并记录每层与
全网的 optical/skip 权重。

两项权重：

- 都是可训练参数；
- 始终为正；
- 总和为 1；
- 三种反馈方法中都使用精确 BP；
- 没有硬性 optical weight 下限。

因此必须观察训练后 `residual_optical_weight_min/mean/max`，而不能只看任务
精度。

## 6. V2 对比学习读出

```text
final amplitude
-> adaptive average pool 20 x 20
-> flatten 400
-> affine LayerNorm(400)
-> Dropout(0.1), train only
-> Linear(400,128)
-> L2 normalization
```

最终 embedding 允许正负值。Dropout 只在电子读出中启用，phase dropout
关闭，避免把硬件鲁棒性噪声与反馈方法比较混在一起。

预训练：

```text
L_pre = SupCon(z, y)
```

微调：

```text
L_ft = SupCon(z, y) + 0.5 * CE(cos(z, leave-one-out prototype)/tau, y)
```

评估时使用固定 CIFAR-10 support split 构建每类 prototype，对 validation/test
embedding 做 cosine nearest-prototype 分类。

## 7. 几何指标

设共享预训练参数为 `theta_pre`，方法 `m` 在匹配 epoch `T` 的参数为
`theta_m,T`：

```text
Delta_m,T = theta_m,T - theta_pre

relative_parameter_drift
  = ||Delta_m,T||_2 / ||theta_pre||_2

drift_ratio_to_BP
  = ||Delta_m,T||_2 / ||Delta_BP,T||_2

endpoint_cosine_to_BP
  = cosine(Delta_m,T, Delta_BP,T)
```

`endpoint_cosine_to_BP` 比较的是从预训练点出发的**累计更新向量**，不是单个
batch 的瞬时梯度。瞬时梯度 cosine 也可作为诊断，但必须与 endpoint cosine
分开命名。

NoFT 的更新向量为零，因此 cosine 未定义，应报告 N/A，不能写成 0。

## 8. 公平性约束

主比较必须保持：

- 同一个 pretrained checkpoint；
- 相同模型初始化状态；
- 相同数据和 split；
- 相同 epoch/sample budget；
- 相同 batch order hash；
- 相同 sample-level augmentation seed；
- 相同 optimizer、LR 和 weight decay；
- 相同 dropout 概率与随机种子；
- 相同 checkpoint policy。

测试集不得用于 checkpoint 选择。任务表现可以额外报告 validation-selected
结果；参数几何必须在相同 epoch 比较。
