# MovieLens：ID embedding 与浅层网络联合训练

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

新增实验：[Amazon Beauty 生成式推荐](generative_beauty/README.md)：商品内容 → RQ-VAE 语义 ID → T5 生成下个商品，并与热门商品、SASRec 风格模型进行全库 Top-K 对照。

后续实验：[属性与历史特征对照](FEATURES.md)、[AUC 选模与交互模型对照](AUC_EXPERIMENT.md)、[Transformer 用户行为特征](SEQUENCE_EXPERIMENT.md)。下文保留第一轮仅使用 ID 的实验记录。

使用 PyTorch，从用户 ID、电影 ID 学习两张向量表，并和一个隐藏层一起训练。目标是对已发生的评分预测「是否达到 4 星」，再按预测分数排序。这是推荐排序的入门实验，不是点击率模型。

## 网络结构

```text
用户 ID → 用户 Embedding（64 维） ┐
                                 ├→ 拼接（128 维）→ Linear(128,64) → ReLU → Linear(64,1) → Sigmoid
电影 ID → 电影 Embedding（64 维） ┘
```

只有一个隐藏层；最后的线性层负责输出分数。训练时直接把输出 logit 交给 `BCEWithLogitsLoss`，该损失包含 Sigmoid 所需的数学处理；推理时才显式调用 Sigmoid。两张 embedding 表从随机值开始，与 MLP 同时反向传播更新。输入不包含电影文本、用户人口属性或额外画像特征。

第一轮共有 **552,641 个参数**，包含保留给未知 ID 的两行零向量。未知 ID 未参与评估，示例推理脚本会明确拒绝没有学到向量的 ID。

## 数据与标签

来源：[GroupLens MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)。官方压缩包包含 6,040 名用户、3,706 部有评分的电影、1,000,209 条评分。数据自身没有现成的用户/电影 embedding。

- 4～5 星：标签 1；1～3 星：标签 0。
- 只使用实际评分记录，没有把未评分电影当作负样本。
- 分数反映已评分样本中的高评分倾向；未经真实曝光/点击数据训练，也没有做线上概率校准。
- `prepare_data.py` 下载官方压缩包并核对 SHA-256；解压时保留原始 README。官方数据说明允许非商业研究用途，其他用途应查看随数据附带的条款。

归档 SHA-256：`a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`。

## 时间划分与评估范围

全体评分按时间约 70% / 15% / 15% 划分；相同时间戳留在同一集合。词表只用训练数据创建，验证与测试仅保留用户和电影都在训练期出现过的样本。

| 集合 | 时间划分后 | 排除未知用户/电影后用于评估 |
|---|---:|---:|
| 训练 | 700,144 | 700,144 |
| 验证 | 150,033 | 35,124 |
| 测试 | 150,032 | 75,930 |

训练期覆盖 **4,870 名用户、3,633 部电影**，因此实际导出的向量数量小于全量数据的 ID 数量。验证覆盖率约 23.4%，测试覆盖率约 50.6%；下方结果只适用于这些已知用户与电影，不能代表冷启动效果。

模型按验证集 AUC 选最佳 checkpoint，测试集在选定模型后统一评估一次。NDCG@10 对每位用户在对应集合中已经评分的电影排序，只纳入同时有高分、低分电影的用户，再按用户平均。测试共有 859 名符合条件的用户，每人的候选数中位数为 45。这不是全量电影库上的 Top-10 推荐效果。

## 第一轮结果

运行目录：`runs/mlp64-20261007-085012`。CUDA 训练，随机种子 42，总流程约 13.5 秒。最多 30 轮，第 12 轮早停；最高验证 AUC 为 0.77986，出现在第 9 轮。

| 测试指标 | 电影评分基线 | 联合训练 Embedding + MLP |
|---|---:|---:|
| AUC ↑ | 0.73226 | **0.76182** |
| NDCG@10 ↑ | 0.81741 | **0.81968** |
| Log loss ↓ | 0.60673 | **0.59210** |
| 分类准确率（阈值 0.5）↑ | 67.40% | **69.42%** |

基线对每部电影只计算训练集的平滑高评分率：`(高评分数 + 10 × 训练集总体高评分率) / (评分数 + 10)`。同一电影对所有用户的基线分数相同。

这一轮的 AUC 有改善，但 NDCG@10 只小幅提高；单个种子、单次划分的结果不证明稳定的排序优势，也不能与 CIFAR-10 的图像分类准确率直接比较。

训练参数：Embedding 64 维，隐藏层 64 维，AdamW，学习率 0.001，weight decay 0.0001，batch size 4,096，梯度裁剪 5。早停 patience=4，超过 0.0001 的验证 AUC 改善才重置计数；checkpoint 始终保留最高的实际验证 AUC，即使改善小于该阈值。

已验证：AUC 与独立两两比较公式一致；两张 embedding 表都有梯度且所有已知 ID 的向量均有更新；CPU 重新加载 checkpoint 后分数与 GPU 结果一致；单独加载导出向量的打分结果与 ID 查表方式一致；本机与远端模型、向量文件的 SHA-256 相同。

## 在 AutoDL 上重新训练

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation
bash start_train.sh
```

脚本自动准备数据，并为每次运行生成独立目录及同名 `.log`。复用已安装的 `/root/miniconda3/bin/python`（PyTorch 2.8.0+cu128、NumPy）；可用 `PYTHON_BIN` 指定其他 Python。Mac 上可指定已安装 PyTorch 和 NumPy 的解释器，程序会自动使用 CPU。

调整下一次实验的参数：

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation
bash start_train.sh --embedding-dim 64 --hidden-dim 64 --epochs 30 --patience 4
```

查看本轮完整日志：

```bash
tail -f /root/autodl-tmp/deeplearning-lab/recommendation/runs/mlp64-20261007-085012.log
```

本轮已经结束，最后一行为 `TRAINING_COMPLETE`。

## 复用 embedding 并排序

每次运行保存：

| 文件 | 内容 |
|---|---|
| `best.pt` | 最佳模型权重、结构参数、最佳轮数和验证指标 |
| `user_embeddings.npz` | `user_ids` 原始 ID；`embeddings` 形状为 `[4870, 64]` |
| `item_embeddings.npz` | `item_ids` 原始 ID；`embeddings` 形状为 `[3633, 64]` |
| `id_maps.npz` | 训练词表的原始用户/电影 ID |
| `config.json` | 完整超参数、数据哈希、划分时间、覆盖范围 |
| `metrics.csv` | 每轮训练损失、验证指标、耗时 |
| `summary.json` | 最终测试结果、基线结果和校验记录 |
| `split_indices.npz` | 各集合在原始 `ratings.dat` 中的零基行号 |
| `test_predictions.npz` | 测试样本 ID、原始评分、标签及分数 |
| `source/` | 本轮实际使用的模型和训练代码快照 |

向量行号不是原始 ID。读取时应通过同文件中的 ID 数组建立对应关系：

```python
from pathlib import Path
import numpy as np

run = Path('/root/autodl-tmp/deeplearning-lab/recommendation/runs/mlp64-20261007-085012')
users = np.load(run / 'user_embeddings.npz')
items = np.load(run / 'item_embeddings.npz')
user_row = np.flatnonzero(users['user_ids'] == 1164)[0]
item_row = np.flatnonzero(items['item_ids'] == 1)[0]
user_vector = users['embeddings'][user_row]   # shape: (64,)
item_vector = items['embeddings'][item_row]   # shape: (64,)
features = np.concatenate([user_vector, item_vector])  # shape: (128,)
```

实际评分示例：从文件读取用户 1164 的向量和下面十部电影的向量，再运行浅层网络，按分数从高到低输出。用户按测试集中「至少十条评分且同时有高分与低分」的最小 ID 选出；这十部电影是其测试记录中的前十部，仅用于验证调用，不用于挑选好案例。

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation
/root/miniconda3/bin/python score.py \
  --run-dir /root/autodl-tmp/deeplearning-lab/recommendation/runs/mlp64-20261007-085012 \
  --user-id 1164 \
  --item-ids 2987 715 1249 2996 728 587 589 1 6 3006
```

已保存输出到运行目录的 `example_ranking.json`。例如电影 ID 1（Toy Story）的模型分数为 0.90674；该数字不是已经校准的真实点击概率。该示例只排序给定候选，不负责召回，也不自动过滤用户已经看过的电影。

实现文件：`model.py` 定义模型，`train.py` 负责训练与评估，`score.py` 演示加载导出向量后的排序，`start_train.sh` 提供启动入口。
