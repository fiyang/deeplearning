# Ali-CCP：AITM 作者参数对照实验

> 历史实验记录：下文结果、路径和运行状态来自此前实验；2026-10-20 归档，尚未与远端最终文件核对。数据、图表与模型产物不随 Git 分发。

使用 [AITM 作者仓库](https://github.com/xidongbo/AITM) 的两任务 `AITM.py` 和 `process_public_dataset.py`，移植到当前 PyTorch/CUDA 环境。原始源码、MIT 许可证和 SHA256 保存于 `official/`，固定 commit 为 `b695cfafdeaf82101aa7bd457d3bd82fd87f38e5`。论文：https://arxiv.org/pdf/2105.08489。

这是有明确差异记录的官方方法移植实验，不是作者 TensorFlow 1.10 环境的逐位复现。未使用 EasyRec：它发布的 Ali-CCP 基准配置依赖 4 个加工分值字段，但没有附带可核对的加工逻辑，配置中的输出类别数与现版代码也不兼容。

## 模型和训练参数

| 项目 | 配置 |
|---|---|
| 输入 | 作者选取的 18 个特征域，各取一个 ID |
| Embedding | 每域 5 维，拼接为 90 维；两任务共享 |
| 点击塔、购买塔 | 各 `90 → 128 → 64 → 32`，ReLU |
| Dropout | 0.1、0.3、0.3；信息传递层 0.3 |
| 信息传递 | 点击塔 → Linear(32,32) + ReLU + Dropout → 与购买塔进行两路注意力聚合 |
| 输出 | pCTR 与 pCTCVR，各一个 sigmoid 输出 |
| 优化器 | 按 TF1 Adam 公式移植，lr=0.001，betas=(0.9,0.999)，epsilon=1e-8 |
| Batch | 2,000 |
| 正则化 | embedding 和注意力三个矩阵的 L2，系数 1e-6；作者代码没有对 Dense 权重加 L2 |
| 顺序约束 | `0.6 × mean(max(pCTCVR − pCTR, 0))` |
| 最大轮数 | 100，使用作者 early_stop=1 |
| 最佳模型 | 验证 CTR AUC + CTCVR AUC 的和最高；连续一轮未提高就停止 |

损失为全曝光上的点击 BCE + 购买 BCE + 顺序约束 + L2。保留作者 `tf.losses.log_loss` 的对数内 epsilon=1e-7，L2 使用 `0.5 × coefficient × sum(weight²)`。注意力按照作者逐路 `sum(Q*K)/sqrt(32)` 加 softmax 计算，并不是完整 Transformer 的跨 token `QKᵀ`。

本模型没有独立的条件 CVR 输出。原文 purchase AUC 对应全曝光 CTCVR AUC，不能称为点击子集 CVR AUC，也不能把 purchase 与 click 再相乘。

## 特征处理与实验差异

18 个域：`101,121,122,124,125,126,127,128,129,205,206,207,216,508,509,702,853,301`。

按作者脚本的字典处理：同一域出现多次时取最后一个 ID，公共特征覆盖样本中的同名域；忽略原始浮点权重。未采用原 23 域实验的 4 个用户历史聚合域和物品 intent 多值域。因此这次是完整方法对照，并非只改变网络架构的消融实验。

作者脚本首次出现时将频次计为 0，后续每次加 1，保留条件是 `counter > 10`，实际对应至少 **12 次出现**；本实验保留这个实际行为。低频、缺失或训练时未见 ID 映射到每域可训练的 0 号向量，没有哈希分桶。

为与我们的 ESMM 保持可比较性，保留 seed=42 的同一训练/验证行号划分和原始完整测试文件；只从实际训练分区统计频次、建立词表。作者是在划分验证集之前统计词表，使用 seed=2020、约 10% 随机验证集和局部窗口乱序。本实验记录这些差异，不把验证/测试 ID 加入训练词表，使用确定的排序编码和每轮全训练行乱序。词表容量根据实际结果生成并包含 OOV 行，不直接抄作者硬编码的容量。

训练 38,069,915 条，验证 4,229,990 条，测试 43,016,614 条。所有分区继续排除 click=0/conversion=1，与此前实验一致。选择 checkpoint 后才评估测试集；报告 AUC、AP 和 log loss，原始 0.5 阈值准确率不是主要比较指标。

## 运行

服务器目录：`/root/autodl-tmp/deeplearning-lab/recommendation/aitm_aliccp_official`。

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation/aitm_aliccp_official
bash start.sh
```

首次运行自动把已验证的原始 CSR 缓存转换成作者的 18 域输入，并核对独立字典重建结果。输出缓存位于 `recommendation/data/aliccp-aitm-author-seed42-v1`，每次训练生成独立 `recommendation/runs/aitm-author-*` 目录和同名 `.log`。请在 GPU 空闲时启动，避免重复训练。

2026-10-07 已完成运行 `aitm-author-20261007-170658-9941`。全量输入转换完成；训练/验证行号 SHA256 与 ESMM 基线完全匹配；训练词表下模型总参数 6,133,702。所有分支梯度检查通过。第 3 轮按作者规则早停，选择第 2 轮，最终测试已完成。

```bash
tail -f /root/autodl-tmp/deeplearning-lab/recommendation/runs/aitm-author-20261007-170658-9941.log
```

对照对象是 `esmm-aliccp-id64-20261007-153915-7377`：测试 CTR AUC=0.610467、CTCVR AUC=0.625594。AITM 不产生直接可比的条件 CVR 输出，因此不把其 purchase AUC 与原 ESMM 的 CVR AUC 混为一谈。

| 完整测试集指标 | ESMM ID 64 维 | AITM 作者参数移植 |
|---|---:|---:|
| CTR AUC | 0.610467 | 0.613476 |
| CTCVR AUC | 0.625594 | 0.635926 |
| CTR AP | 0.05522567 | 0.05792528 |
| CTCVR AP | 0.000349451 | 0.000360830 |
| CTR log loss | 0.17025300 | 0.16275810 |
| CTCVR log loss | 0.002078715 | 0.002121832 |
| 参数量 | 295,655,952 | 6,133,702 |
| 峰值 GPU 分配内存（GiB） | 16.77 | 3.30 |
| 训练及评估总耗时（分钟，不含本次特征转换） | 53.95 | 23.19 |

本次同一完整测试集上，AITM 的 CTR/CTCVR AUC 分别增加 0.003009/0.010333，模型约小 48 倍。CTCVR log loss 略变差；平均预测转化概率为 0.000534381，真实转化率为 0.000213755，仍有高估。该单种子结果支持“本次排序指标改善”，不能单独归因于注意力或认定为稳定显著收益，概率估计也并非全面改善。

第 3 轮验证 CTCVR AUC 从第 2 轮的 0.696450 升到 0.698233，但 CTR AUC 从 0.678546 降至 0.676625，两者之和从 1.374996 略降到 1.374859。因此按作者的联合选择指标保留第 2 轮，不是 CTCVR 单指标触发早停。未根据测试结果重新选择 checkpoint。

CPU/CUDA 的 6 个核对测试均已通过：源码哈希与字段次序；字典覆盖和多值处理；训练分区频次门槛及 OOV；独立 NumPy 前向公式与 checkpoint；损失/正则项/各分支梯度；独立 NumPy TF1 Adam 更新公式。

```bash
/root/miniconda3/bin/python -m unittest -v test_author_port
```

实验目录保留源代码快照、原作者源码、数据清单、配置、差异说明、梯度检查、历史指标、最佳模型和测试预测列定义。原 ESMM 的 18 维、64 维运行及 checkpoint 均保留。
