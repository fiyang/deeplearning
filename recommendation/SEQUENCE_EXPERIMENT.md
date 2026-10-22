# 用 Transformer 从用户评分历史提取兴趣特征

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

本轮运行：`runs/sequence-20261007-103741`。代码为 PyTorch；基于上一轮 seed=42 的 DeepFM 最佳验证 AUC checkpoint 继续训练。两个候选都增加相同的行为字段，一个用平均池化，一个用 Transformer，验证注意力编码是否有增益。

## 特征如何进入打分模型

```text
用户最近最多 50 条历史评分
  每条：电影 ID、评分 1～5、距预测时刻的时间桶、时间组位置、电影类别/年份
      ↓
  电影 embedding + 评分 embedding + 时间 embedding + 位置 embedding + 电影属性投影
      ↓
  [batch, 50, 64] → LayerNorm → Dropout(0.1)
      ↓
  Transformer Encoder × 2
  每层：4 头自注意力（每头 16 维）+ 前馈网络 64→128→64，残差、LayerNorm
      ↓
  对有效位置求平均 → LayerNorm → [batch, 64] 用户兴趣向量
      ↓
  与原 DeepFM 深层分支的 3008 维特征拼接 → [batch, 3072]
      ↓
  Linear(3072,64) → ReLU → Linear(64,1)
      ↓
  加上原线性分支和 FM 二阶交互分支 → Sigmoid → 高评分预测分数
```

序列中的每条电影表示会先通过自注意力吸收其他历史记录的信息，再汇总成用户向量。平均池化对照保留同样的电影、评分、时间、位置和属性编码，但跳过两层 Transformer，直接汇总。它也是有时间特征的模型，不能把两者差异理解为「有时间信息」对「无时间信息」。

兴趣向量可以从用户历史独立提取并缓存；候选电影改变时不必重复编码相同历史。候选电影仍在后续打分网络中与兴趣向量交互。本轮没有做以候选电影为 query 的 DIN 注意力，也不是逐位置预测下一部电影的 SASRec 复现。

原 DeepFM 仍使用最近 20 条高评分（≥4 星）的历史池化及统计特征；新增序列使用最近 50 条**所有评分**，包括低评分。两条历史分支共享电影 ID embedding。

## 时间边界与缺失历史

- 训练样本只读取时间戳严格早于当前评分的记录，当前电影评分和同时间戳评分都不能进入它自己的输入。
- MovieLens 时间戳是提交评分的时间，不是完整观看、点击或曝光轨迹。许多评分时间相同，不能推断其真实内部先后。
- 同一时间戳的事件使用相同的位置编码；位置 1 表示最新时间组，2 表示再前一个时间组。相同时间组内的存储和截断使用确定的电影 ID 顺序，不当作真实行为顺序。
- 时间桶为 `min(floor(log2(1 + 间隔秒数/3600)) + 1, 31)`，0 留给 padding。
- 训练时以当前样本时间计算间隔。验证、测试的历史及计算间隔的时钟均固定在训练截止时刻 `974862386`，不注入任何验证或测试评分。这样缓存向量与离线预测一致，但不是在线实时更新行为的评估。
- 不足 50 条补零并 mask；空历史最终输出全零兴趣向量，避免全 mask 注意力产生 NaN。
- 所有 token 都是已经发生的过去行为，因此历史内部允许双向注意力；这里没有逐 token 的下一项预测任务，不需要三角形自回归 mask。

沿用同一份全局时间划分：训练 700,144 条、验证 35,124 条、测试 75,930 条。训练词表覆盖 4,870 位用户和 3,633 部电影；验证和测试只评估已知 ID。训练样本的有效历史平均 41.36 条，中位数 50 条；12,069 个样本在严格时间边界下没有历史。

## 训练与公平比较

共同配置：seed=42、AdamW、学习率 0.0003、weight decay=0.0001、batch size=1024、最多 30 轮、早停 patience=6、min_delta=0.0001、梯度裁剪 5、BCEWithLogitsLoss。

两者加载同一份原 DeepFM 参数，新增兴趣特征对应的隐藏层权重从零开始，因此开始训练前两者预测均与原模型一致。所有参数共同微调，包括原来的用户/电影 embedding。共同新增的 embedding 和投影使用相同初始化，样本 shuffle 使用独立的相同种子生成器；Transformer 内部 dropout 会引入不同的随机运算过程。参数数量不同，因此不是参数量匹配实验。

checkpoint 按验证集 AUC 选择；同时保留按验证 NDCG@10 选择的 checkpoint 供查阅，但报告的测试结果统一使用 `best_auc.pt`。两轮训练均完成、候选与原 DeepFM 的选择写入 `final_selection.json` 后，才统一评估测试集。本轮为同一已看过测试集上的探索性实验，只有一个种子，不能据此断言稳定增益。

标签为「已评分电影是否 ≥4 星」，不是 CTR；准确率阈值为 0.5。NDCG@10 只在用户实际评分过的候选中计算，测试涉及 859 位同时有正负标签的用户，不是全库召回指标。

## 运行和查看日志

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation
bash start_sequence.sh
```

每次运行生成独立目录，依次训练平均池化和 Transformer 两个版本。已有实验文件保留。

```bash
tail -f /root/autodl-tmp/deeplearning-lab/recommendation/runs/sequence-20261007-103741.log
```

文件说明：

- `sequence_data.py`：按时间构造严格历史序列。
- `sequence_model.py`：编码器、特征提取、DeepFM 打分和加载。
- `train_sequence.py`：对照训练、验证选模、测试和向量导出。
- `score_sequence.py`：加载独立 checkpoint，计算用户兴趣向量并为电影打分。
- `test_sequence.py`：时间边界、同时间事件、空历史、padding、梯度、checkpoint 和缓存特征一致性检查。

## 提取已训练的用户兴趣向量

每个版本的运行子目录都会保存 `user_interest_features.npz`：

```python
import numpy as np

data = np.load("runs/sequence-20261007-103741/transformer/user_interest_features.npz")
user_ids = data["user_ids"]       # [4870]，原始 MovieLens 用户 ID
features = data["features"]     # [4870, 64]，该 checkpoint 在固定历史下提取的兴趣特征
history_count = data["history_count"]  # [4870]，有效历史长度
cutoff = int(data["cutoff"])     # 历史截止时间（不含边界）
```

此文件的 `features` 来自 Transformer 编码历史，与另存的 `user_embeddings.npz` 中按用户 ID 查表的向量不同。与候选电影 embedding 拼接后可以另训练一个浅层网络；本轮已经把它加入原打分模型，并联合训练了编码器。

`score_sequence.py --include-interest` 可以同时输出 64 维向量；它使用 checkpoint 内保存的训练期历史，并不会自动收集新行为。生产式更新需要向序列加入新记录，重新计算时间桶并编码。

可运行的示例（用户和候选均为本次已知 ID）：

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation
/root/miniconda3/bin/python score_sequence.py \
  --checkpoint runs/sequence-20261007-103741/transformer/best_auc.pt \
  --user-id 1164 --item-ids 2987 715 1249 2996 \
  --include-interest
```

## 实测结果

总流程约 256 秒。平均池化训练约 83 秒，Transformer 约 141 秒；两者都在第 7 轮早停，最高验证 AUC 都在第 1 轮。这里是从已有 DeepFM 继续微调，不是从零仅训练 1 轮。

| 版本 | 参数量 | 验证 AUC | 测试 AUC | 测试 NDCG@10 | 测试准确率 | 测试 log loss |
|---|---:|---:|---:|---:|---:|---:|
| 原 DeepFM，seed=42 | 750,277 | **0.78861** | 0.77142 | 0.83490 | 70.31% | 0.58244 |
| 新序列 + 平均池化 | 761,541 | 0.78720 | **0.77255** | **0.83647** | **70.51%** | **0.58003** |
| 新序列 + Transformer | 828,485 | 0.78846 | 0.77218 | 0.83643 | 70.49% | 0.58133 |

Transformer 相对平均池化的测试 AUC 差为 **-0.000375**，NDCG@10 差为 -0.000046，本轮没有明确的注意力增益。Transformer 相对原 DeepFM 测试指标略高，但同时改变了输入和训练过程，不能把差异单独归因于注意力。按预先约定的验证 AUC 选择规则，仍保留原 DeepFM 为首选；不根据这次测试集表现改选平均池化模型。

两种序列模型都已经训练、保存并导出有效兴趣特征。训练曲线显示损失持续下降而验证 AUC 下降，增加微调轮数没有改善本轮泛化表现。注意力、评分、时间、位置及电影属性投影均有非零梯度；新增兴趣向量对应的打分权重已更新。

![训练曲线](runs/sequence-20261007-103741/sequence_training.png)

结果文件：根目录 `summary.json`、`comparison.csv`、`final_selection.json`；各版本 `metrics.csv`、`best_auc.pt`、`best_ndcg.pt`、`test_predictions.npz`、`user_interest_features.npz`、用户和电影 ID 向量。完整核验记录为 `verification.json`。

核验覆盖源码与数据哈希、时间划分、从原始训练记录独立重建所有 4,870 位用户的最终历史、验证选模、独立打分与导出向量。CPU/GPU 的 Transformer 加速内核存在轻微浮点差异：抽查向量最大绝对差约 0.000125，向量 L2 相对差约 0.005%，对应最终分数差约 0.0000012。关闭 GPU 融合加速路径后，CPU/GPU 向量最大差约 0.000001；同一 GPU 路径下缓存向量与在线编码打分误差约 0.00000012。核验分别约束向量数值误差和更严格的最终打分误差，不要求跨设备逐位相等。
