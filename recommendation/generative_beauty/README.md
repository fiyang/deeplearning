# 第一轮生成式推荐：Amazon Beauty + TIGER 风格模型

> 历史实验记录：下文结果、路径和运行状态来自此前实验；2026-10-20 归档，尚未与远端最终文件核对。数据、图表与模型产物不随 Git 分发。

用户历史 → 语义 ID 序列 → Transformer Encoder–Decoder → 逐 token 生成目标语义 ID → 真实商品。

这是独立编写的 PyTorch 教学实验，不是 Google 官方实现，也不是严格复现论文成绩。第一轮不加 MoE；先建立可信的完整链路和相同评估口径的对照。

## 可选公开数据与方法

| 数据 | 适合的用途 | 本轮选择 |
|---|---|---|
| [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) | 已有电影评分序列，便于 ID-based 序列推荐；内容属性较少 | 留作另一领域对照 |
| [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) 的 Beauty/Sports/Toys | 有时间戳、商品标题/品牌/类别；是 TIGER 的经典数据 | **Beauty 5-core 全子集** |
| [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | 新一些、更大的商品与行为数据 | 后续扩展；不能直接与 2014 成绩对比 |
| [OpenOneRec / RecIF-Bench](https://github.com/Kuaishou-OneRec/OpenOneRec) | 约 1 亿次交互、20 万用户，短视频/广告/商品、指令推荐 | 后续大规模实验 |

方法上，先对比训练期热门商品、SASRec 风格序列模型与 TIGER 风格生成式模型；后续可加随机 ID 消融、多个随机种子，再考虑 MoE、会话级生成和偏好优化。

## 真实数据规模与 schema

官方原始文件：`reviews_Beauty_5.json.gz` 44,819,276 bytes；`meta_Beauty.json.gz` 99,156,148 bytes。

- 22,363 个用户、12,101 个商品、198,502 条评论交互；没有用户/商品重复对。
- 训练期交互 153,776 条；展开为 next-item 样本 **131,413 条**。每个用户训练前缀的第一条没有历史，不单独作为预测样本。
- 验证 **22,363 条**，测试 **22,363 条**；每个用户各留一个目标。
- 交互使用 `reviewerID, asin, unixReviewTime`。保留所有评分等级；目标是下一条评论对应商品，不是点击率、购买率或高评分分类。
- 内容向量只使用商品 `title, brand, categories, price`。不输入评论内容、用户姓名、销量排名、also-bought 图等。
- 商品信息覆盖全部 12,101 个商品。下载 manifest 保存来源、大小、SHA-256；公开模型从 ModelScope 的同名 Sentence-T5 镜像获取并核对该镜像文件清单中的 SHA-256。

## 划分及评估约束

1. 按每个用户的时间排序，最后一个交互为测试、倒数第二个为验证，其余训练；最多输入 20 个历史商品。
2. 同时间戳用原始文件行号稳定排序。共有 77,402 对相邻同时间戳记录；数据不能恢复这些记录之间的真实先后。
3. 这是论文风格的 per-user leave-one-out，**不是全局时间切分**。结果用于离线学习，不能当成真实上线时序效果。
4. 候选范围是全商品库；删除该用户当前时刻以前所有已交互商品，不能只删除截断后的 20 个。所有模型使用同一候选规则，不采样 100 个负例来评估。
5. 固定目录下所有商品内容可用于编码器/量化器。这是已知商品目录的 transductive 设置；不使用验证/测试用户交互去训练。元数据也是静态快照，不模拟历史上架时间。
6. 测试集只在两个模型都根据完整验证集选好 checkpoint 后各评估一次。主选模指标 NDCG@10；同时记录 Recall/NDCG@5/10/20。
7. 生成式模型在全库语义 ID 前缀树上做 beam search，宽度 20。它是近似检索，不保证找到全库联合概率的精确 Top-20。评分保留原始全词表 log probability，过滤非法路径后不重新归一化。
8. 每个完整语义 ID 必须唯一对应商品。命中用真实商品行号判定，不能以前三个语义码相同就算命中。

## 模型与本轮参数

**内容编码和量化（训练推荐模型前冻结）**

- `sentence-transformers/sentence-t5-base`，768 维；最大文本长度 128，batch 128，L2 归一化。
- 内容向量整体去均值并乘 `sqrt(768)`，保留欧氏距离相对关系，改善量化器的优化尺度。
- RQ-VAE 编码器 `768→512→256→128→32`；对称解码器；3 级码本，每级 256 个 32 维向量。
- Autoencoder 重建预训练 500 steps、Adam lr=0.001；逐级 MiniBatchKMeans 初始化码本。
- RQ-VAE 再训练 3,000 steps、Adam lr=0.001、batch=1024、commitment beta=0.25、clip=5。
- 第 4 个 token 区分前三码相同的商品，最多 256 个冲突槽；超限直接失败，不静默截断或错误合并。
- 初次直接使用 Adagrad 0.4 出现码本坍缩，已被冲突校验拦截；失败日志保存在 `runs/prepare-adagrad04-failed.log`。该失败编码未用于推荐模型。

**TIGER 风格生成式模型**

- 从随机权重训练的 T5 Encoder–Decoder；4 层 Encoder + 4 层 Decoder。
- d_model=128，6 heads，d_kv=64（注意力内部拼接宽度 384），d_ff=1024，dropout=0.1。
- 3 个特殊 token + 4×256 个分层语义 token + 2,000 个用户 hash token；总词表 3,027。
- 用户通过稳定 SHA-256 映射到 2,000 桶。输入为用户 token + 最多 20×4 个历史 token；目标为下个商品的 4 个 token。
- teacher forcing + 全词表交叉熵；固定生成 4 个位置，合法商品 ID 通过前缀树约束。

**SASRec 风格对照**

- 2 层因果 Transformer，宽度 128，4 heads，FFN=512，dropout=0.1；商品向量同时作为输出权重。
- 使用相同的训练前缀样本和 20 个历史商品，直接对全商品库做 softmax CE。这是 SASRec 风格对照，不是原论文的负采样 BCE 实现；不加用户 embedding。

**两个推荐模型的共有训练设置**

- seed=42，最多 30 epochs，batch=512。
- AdamW，峰值 lr=0.001，weight decay=0.01，clip=1。
- warmup=min(500,total_steps/10)，之后 cosine 降到峰值的 10%；BF16 autocast，FP32 参数。
- 第 1 轮以及每 2 轮评估完整验证集；连续 5 次验证 NDCG@10 没提高则早停。
- 保留验证集 NDCG@10 最高的 checkpoint；固定热门商品对照仅统计训练期频数。

### 与 TIGER 论文的差异

参考 [TIGER 论文](https://arxiv.org/html/2305.05065v3) 的数据、Sentence-T5、3×256 RQ-VAE、4-token ID、20-item 历史和 4+4 T5 框架。主要差异为 PyTorch 实现、RQ-VAE 的中心化/重建预训练/优化器/训练预算、推荐模型的 AdamW 与 warmup/cosine、30 epochs 的短预算、约束解码和明确过滤历史。只跑一个 seed。上述成绩不能直接作为论文复现或架构优劣的因果证据。

## 运行与日志

远端目录：`/root/autodl-tmp/deeplearning-lab/recommendation/generative_beauty`。

独立 `.venv` 使用现有 PyTorch 2.8 CUDA 环境；额外安装 sentence-transformers、transformers、scikit-learn、sentencepiece。每次运行保存完整 `environment.txt`。

```bash
ssh deeplearning-dl-1
cd /root/autodl-tmp/deeplearning-lab/recommendation/generative_beauty

# 已完成数据准备后运行校验与短流程。短流程不评估测试集。
.venv/bin/python test_pipeline.py
bash start_train.sh --smoke

# 正式运行；建议在 screen 中运行以便 SSH 断开后继续。
bash start_train.sh
```

每次生成 `runs/beauty-tiger-日期-时间-PID/` 和同名 `.log`，不覆盖旧实验。日志包含 `MODEL_START`、`STEP`、`EPOCH`、`TEST`、`FINAL`；只有出现 `TRAINING_COMPLETE` 和 `status.json: complete` 才代表正式实验完成。

结果目录保存 source 快照、配置、数据校验信息、每模型 history.csv、best.pt、best_validation.json，以及最终 summary.json、测试 Top-20 结果、固定前五个用户的推荐示例。

加载已选模型并生成推荐：

```bash
.venv/bin/python recommend.py --run-dir runs/对应正式运行目录 --user-index 0
```

测试覆盖：指标手算、历史过滤不看未来、完整 ID 区分冲突、beam 与小目录穷举联合概率一致、padding 不改变结果、模型和量化器梯度。另需 GPU 短流程验证真实数据训练/评估与存取 checkpoint。
