# Ali-CCP ESMM 特征域 attention 实验

> 历史实验记录：下文结果、路径和运行状态来自此前实验；2026-10-20 归档，尚未与远端最终文件核对。数据、图表与模型产物不随 Git 分发。

在完整 Ali-CCP 数据上增加特征域之间的 self-attention。代码、启动脚本、模型和结果独立存放。原 `esmm_aliccp_full` 的学习率实验不修改。

## 结构

1. 原始 23 个特征域；用户/物品 ID embedding 各 64 维，其余 21 域各 18 维。
2. 每个域内仍保留全部 token 和原始浮点权重，做加权求和。拼接共 506 维。
3. 每个域用独立的无偏置线性层投影为 64 维，得到 `[batch, 23, 64]`。
4. 1 层 multi-head self-attention，4 头，每头 16 维；pre-LayerNorm、attention dropout=0.1、残差 dropout=0.1、输出 LayerNorm。没有额外的 Transformer FFN，也没有时间位置编码。
5. 按固定字段顺序展平为 1472 维，进入 CTR/CVR 两个独立塔：`1472 → 360 → 200 → 80 → 2`，ReLU、dropout=0.1。两个输出之差作为二分类 logit。
6. `pCTCVR = pCTR × pCVR`，损失仍为全部有效曝光上的 `BCE(click, pCTR) + BCE(conversion, pCTCVR)`。

字段投影各自独立，用来区分不同字段语义。已知且原始权重非零的 token 决定字段是否存在；缺失/未知字段不能作为 attention key，输出也置零。全部字段缺失时临时放行一个哑 key，最终字段输出仍全零，防止全掩码 softmax 出现 NaN。

这是参考 [AutoInt](https://arxiv.org/abs/1810.11921) 思路的自定义 ESMM 扩展，不是该论文的完整复现。attention 发生在 23 个聚合后的特征域之间；`109_14/110_14/127_14/150_14` 中的历史兴趣仍先按原权重聚合，没有加入兴趣集合内部的 attention。数据没有逐次行为时间戳，因此不解释为时间序列 Transformer。

## 对照和评估

固定对照：`esmm-id64-lr1e-4-20261007-185351-13286`。

- Adam，固定学习率 **0.0001**，无 weight decay、无学习率调度。
- batch=4096，seed=42，梯度裁剪范数 5。
- 最多 10 轮，验证 CTCVR AUC 连续 3 轮未超过改善阈值 0.0001 时早停；保存实际最高验证 AUC 的 checkpoint。
- 从头训练，不加载之前训练好的权重。embedding 初始化在新 attention 层创建前，保持原初始化顺序。
- 训练 38,069,915 条，验证 4,229,990 条，测试 43,016,614 条。
- `base_model.py` 是原模型源码的逐字副本，复用其加权 embedding 和 ESMM 乘积损失。`data.py` 和 `metrics.py` 也逐字保留。
- 启动时核对三份基础源码 SHA256、训练/验证行号、词表 SHA256、所有已有训练超参数与对照一致，记录在 `reference_comparison.json`。
- checkpoint 只根据验证集选择，选定后对完整原始测试集评估一次。记录 CTR AUC、点击子集 CVR AUC、全曝光 CTCVR AUC、AP、log loss。

这轮比较的是整套特征交互结构变化，包含投影、归一化、attention 和更宽的塔输入。若指标提升，还需要同样投影/归一化/塔输入但移除 attention 的消融，才能分离注意力本身的收益。单随机种子也不能证明稳定优势。后续调参依据验证集，重复使用同一测试集会削弱其最终独立评估作用。

## 启动和查看

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation/esmm_aliccp_attention
bash start.sh
```

脚本先冻结源码到独立的 `runs/<run>.launch`，再启动后台任务。等待指定的 0.0001 基线进程退出、`TRAINING_COMPLETE`、最终指标和 checkpoint 记录齐备、GPU 占用低于 2 GiB 后自动开始训练。等待上限 16 小时，基线失败则停止并记录原因；不会终止已有进程。文件锁避免本 attention 队列重复运行。

启动时返回 `RUN`、`LOG`、`QUEUE`。查看 `tail -f <LOG>`；队列状态保存于 `<RUN>.queue.json`，包括等待、开始、训练、完成或失败。训练后配置、源码快照、数据划分、梯度检查、每轮指标、最佳 checkpoint 和最终结果保存在 `<RUN>`。

2026-10-07 19:05 已启动后台队列，运行名 `esmm-field-attn64-h4-lr1e-4-20261007-190518-13969`，初始状态为 `waiting_for_baseline`，等待上述基线完成。完整模型参数量为 296,400,752；本轮架构比纯 ESMM 增加 744,800 个参数。冻结源码已通过对照配置和三份基础源码 SHA256 检查，记录在同名 `.preflight.json`。

```bash
tail -f /root/autodl-tmp/deeplearning-lab/recommendation/runs/esmm-field-attn64-h4-lr1e-4-20261007-190518-13969.log
```

## 检查

```bash
/root/miniconda3/bin/python test_attention.py -v
/root/miniconda3/bin/python test_queue.py -v
/root/miniconda3/bin/python smoke_batch.py
```

覆盖独立 NumPy attention 公式、CPU/CUDA mask、字段间梯度、批次隔离、原始加权聚合/损失一致性、embedding/attention/双塔梯度、未知和全空输入、checkpoint 恢复、超参数漂移检查、队列成功启动及失败阻断。GPU 冒烟验证使用真实 batch=4096 和 `[4096,23,64]` attention 形状、合成多值字段与小词表；它不等价于完整词表的显存测试。
