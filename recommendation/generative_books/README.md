# Amazon Books：两种生成式推荐结构对照

> 原始实验代码于 2026-10-22 与远端核对归档；2026-10-28 的续训、扩容及召回诊断见下方入口。后面的原始实验章节保留各自的历史配置。数据、权重及完整预测文件不随 Git 分发。

## 2026-10-28 实验入口

| 实验 | 代码与说明 | 已确认结果 |
|---|---|---|
| 小版 TIGER 低学习率续训 | [tiger_refinement](tiger_refinement/README.md) | 5,043,712 参数，第 18 轮最佳权重续训至第 21 轮；完整测试 Recall@10 从 3.333786% 至 3.371555% |
| 保留原权重的模型扩容 | [tiger_growth](tiger_growth/README.md) | 扩至 14,281,344 参数，选中第 23 轮；完整测试 Recall@10 为 3.529423% |
| 1,428 万参数模型继续训练 | [tiger_continuation](tiger_continuation/README.md) | 从第 23 轮最佳权重追加最多 5 轮，提交时仍在训练；尚无本轮最终测试结果 |
| 验证集召回诊断 | [diagnostics](diagnostics/README.md) | 固定小版模型，分别考察 beam 宽度与 Recall@10/50/100；不混用不同 checkpoint 的成绩 |

各阶段使用独立运行目录，配置、数据与源码哈希、启动检查及已完成结果保存在 [`results/`](results/)。本次新增代码的 17 项 CPU 回归检查和提交时的训练快照见 [commit_verification.json](results/books-tiger-wide-continue5-lr3e-5-20261028-142351-11623/commit_verification.json)。检查涵盖扩容前后预测一致性、断点恢复、学习率边界、非有限梯度保护及召回计数。

续训脚本复用 AutoDL 上已有的数据、Python 环境和父运行快照；仅克隆仓库还不能直接续训。入口默认父目录对应这次已验证实验，具体路径、停止上限及日志命令见各自 README。脚本共用训练锁，按用户要求完成后保持机器开机。

这轮扩大到 Amazon Reviews 2014 Books **完整 5-core 文件**，已完整处理 8,898,041 条评论交互，约为 Beauty 198,502 条的 44.8 倍。实际 603,668 名用户、367,982 件商品，训练期交互 7,690,705 条，展开训练样本 7,087,037 条，验证/测试各 603,668 条。本轮没有重复用户-商品被剔除。原始规模和哈希见 `data/dataset_info.json`。

官方数据入口：<https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html>。两份文件来自该页面的 Stanford 官方链接：

| 文件 | 字节数 | 说明 |
|---|---:|---|
| reviews_Books_5.json.gz | 3,223,678,899 | 完整 Books 5-core 评论文件，只使用用户、商品和时间构建序列 |
| meta_Books.json.gz | 825,767,823 | Books 商品元数据，流式提取当前商品库对应记录 |

合计约 4.05 GB（3.77 GiB）。2026-10-09 启动前检查数据盘有约 67 GiB 空闲，暂未删除 ESMM/MMoE 的任何数据或结果。8 路 HTTP Range 下载校验每段 Content-Range、长度和合并后长度，并记录 SHA-256；哈希用于记录本次文件身份，不能冒充数据作者发布的校验和。gzip CRC 在完整数据解析时校验。

## 实验设计

| 项目 | TIGER 风格 | Decoder-only SID |
|---|---|---|
| 主体 | T5 Encoder–Decoder | GPT 风格因果 Transformer |
| 层数 | Encoder 4 + Decoder 4 | Decoder 5 |
| 主宽度 | 128 | 256 |
| Attention | 6 heads，d_kv=64 | 4 heads，每头 64 维 |
| FFN 宽度 | 1024 | 1024 |
| 可训练参数 | 5,043,712 | 5,182,976 |
| 输入 | 用户 hash token + 最近 20 个商品的 SID | 同样输入 + BOS，然后自回归续写 |
| 输出 | 目标商品 4 个 SID token | 目标商品 4 个 SID token |
| 生成 | 稀疏前缀合法性约束，beam=20 | 相同约束与 beam；使用 KV cache |

共同条件：相同数据、商品语义编码、样本顺序、训练轮数、优化器、历史长度和评测用户。两者从随机参数开始训练；Sentence-T5 内容编码器独立冻结。主模型参数相差约 2.8%，训练样本预算一致，**计算量不是严格相等**。结构比较包含位置编码、激活函数等实现差异，不把结果归因于单个组件。

Decoder-only 使用 GPT-2 模块从头训练，并不加载 GPT-2 预训练语言权重。这是可在当前单卡上运行的约 5M 参数结构对照，**不是 OpenOneRec、OneRec-V2 或其基础模型的官方复现**。TIGER 也属于教育实现，非官方代码和训练预算复现。

参考：

- TIGER：<https://arxiv.org/abs/2305.05065>
- Decoder-only 与语义 token 推荐的大规模研究背景 OpenOneRec：<https://arxiv.org/abs/2512.24762>
- HF GPT-2 接口：<https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/gpt2>

## 商品编码

标题、类目、品牌、价格、描述 → 冻结 Sentence-T5 768 维 → RQ-VAE `768→512→256→128→64` → 三个 512 项码本 → 第四位区分同前三码的商品。

- 内容最多 128 个文本 token，L2 归一化后全商品库共同中心化、乘 sqrt(768)。
- AE 预训练 1,000 steps，逐级 MiniBatchKMeans 初始化（最多 200,000 件商品样本）。
- RQ-VAE 训练 10,000 steps，batch 1024、Adam lr=0.001、commitment beta=0.25、梯度裁剪 5。
- 码本和商品编码在推荐训练之前固定；此次先比较生成结构，暂不混入 LETTER 协同编码改动。
- 第四位使用 1,024 个区分槽，前三层仍各为 512。生成器用分层不同的取值范围和混合进制前缀索引；用户 token 与商品 token 不重叠。
- 若碰撞组超过第四位容量，流程失败停止，不合并不同商品，也不拿前缀命中冒充商品命中。
- 词表：3 个特殊 token + 3×512 个语义 token + 1,024 个区分 token + 2,000 个用户 hash token = 4,563。
- 用户 hash 仅作特征桶，不用它决定数据划分；不同用户碰撞不会合并其行为序列。
- 商品目录和内容元数据视为预先已知，包含验证/测试商品，是 transductive 设置；商品评论文本、用户未来交互、also-bought 图和销售排名不进入编码训练。

与 Beauty 首轮的编码差异：3×256 升为 3×512，latent 32 升为 64，增加书籍描述，量化训练预算增加。Books 内部两个模型共用这一套编码，不能将跨数据集变化解读为某个组件的独立收益。

2026-10-09 11:02，首次编码因最大碰撞组 569 超过最初设定的 512 个区分槽而停止，尚未开始正式推荐训练。修复扩大第四位到 1,024，保留前三层码本配置与全部商品。修复前代码保存在 `runs/pre-collision-fix-20261009/`，原失败日志保留在总日志中。6 项回归检查包含 569 项碰撞、混合码长范围的逐商品概率对照与 KV cache 校验。

数据质量：商品元数据均能找到，但 81,679 件商品没有标题，其中 61,838 件同时缺少描述；有 346,619 种不同的实际内容文本，最大完全相同文本组 356 件。缺失内容可能影响语义编码质量，扩大区分槽只解决唯一标识问题，不补足内容语义。

## 数据与评测口径

1. 每个用户按时间排序，相同时间用原文件行号打破并列。重复用户-商品保留时间最早的一次；不足 5 条的用户排除并计数。
2. 最后一次交互为测试，倒数第二次为验证，其余是训练期。**不是全局时间切分**；评论时间常以天为粒度，不能恢复当天真实顺序。
3. 展开训练期所有可用历史前缀预测下一个商品，最多保留最近 20 件作为模型输入。用完整训练样本，不抽取小训练子集。
4. 为控制每轮成本，从全体验证用户中以 seed=42 均匀固定抽取 20,000 人，每轮用同一批用户选择模型；索引和哈希写入运行目录。**验证用户有抽样，候选商品不做负采样**。
5. 全部模型选定后，各自对全部测试用户进行最终评测；测试不用于选超参数、早停或选 checkpoint。短流程不评测测试集。
6. 候选为完整商品库，过滤用户在目标发生之前的所有历史商品，包括超出输入最近 20 件的部分。
7. 记录 Recall/NDCG@5、10、20，实际填满候选槽比例、重复结果数、评测耗时。比较的是完整真实商品 ID；beam 是近似搜索，过滤后不足 K 的位置按未命中处理。
8. 目标是“下一次评论对应的商品”，所有评分都保留，不代表 CTR、购买或用户一定喜欢。1 个 seed，只能作为本轮实验观察。

训练配置：最多 **5 epochs**，batch=512，AdamW lr=0.001、weight decay=0.01，warmup=min(1,000, total_steps/10)，cosine 到峰值的 10%；dropout=0.1、clip=1、BF16。每轮验证，连续 3 次 NDCG@10 不改善早停。保存 best/last 权重、优化器和随机状态；正式 test 在两种模型都完成选择后开始。

## 运行与日志

远端目录：`/root/autodl-tmp/deeplearning-lab/recommendation/generative_books`。
复用已安装且验证过的 `../generative_beauty/.venv` 和冻结 Sentence-T5 权重，不改旧实验代码与数据。

本轮后台流程：下载完成 → 正确性检查 → 全量数据预处理 → 生成商品编码 → Books 真实 GPU 短流程 → 两种模型正式训练 → 最终测试。

```bash
ssh deeplearning-dl-1
cd /root/autodl-tmp/deeplearning-lab/recommendation/generative_books

# 总流程日志和阶段
tail -f runs/pipeline-20261009.log
cat runs/pipeline_stage.txt

# 下载进度
cat data/download_status.json

# 最近一次短流程或正式训练目录
cat runs/latest_run.txt
```

总流程日志明确标记 `books_gpu_smoke`、`formal_training`。只有正式目录的 `status.json` 为 complete、生成 summary.json 且有 TRAINING_COMPLETE 才表示完成。`pipeline_stage.txt` 为 complete 和 PIPELINE_COMPLETE 表示整套流程结束。请不要把 fixture 或 smoke 的指标当成 Books 正式成绩。

后台流程正在运行时，不要再启动第二套。数据准备完成后可单独启动新训练：

```bash
bash start_train.sh --epochs 5 --batch-size 512 --val-users 20000
```

中断后从已完成 epoch 恢复，需要传入原来的参数和运行目录：

```bash
../generative_beauty/.venv/bin/python -u train.py --resume --run-dir /绝对路径/原运行目录 \
  --epochs 5 --batch-size 512 --eval-batch-size 64 --val-users 20000 --beam-size 20
```

代码、数据签名或配置变更会拒绝恢复，需建立新实验。恢复点是上一个完整 epoch；中断 epoch 会重跑。

### 20 轮训练预算试验（2026-10-09）

首组 5 轮实验已完成：测试 Recall@10，TIGER 为 2.1197%，Decoder-only 为 2.3587%；两者验证最佳均为第 5 轮。增加训练预算后，用 `bash start_20epochs.sh` 启动独立实验，两种模型均从 seed=42 重新初始化，最多各 20 轮，不覆盖原运行目录。完整数据、固定商品编码、模型结构、batch、验证用户、beam 和优化器设置保持一致；无需重新下载或训练量化器。

学习率峰值仍为 0.001，warmup 仍为 1,000 updates，余弦衰减按 20 轮总步数安排，最终降至 0.0001。它不是原第 5 轮权重的继续训练，前 5 轮的学习率轨迹也不同于原 5 轮试验。旧任务已完成余弦周期，直接改 `--epochs` 恢复会被配置校验拒绝，也会改变下一步学习率。此轮保持严格恢复校验，不修改旧 checkpoint。

每轮仍在固定 20,000 用户上按 NDCG@10 选择最佳权重，连续 3 轮不提升早停；两者选模完成后才运行全量测试。延长预算依据验证曲线仍在上升，不用测试结果选具体轮次。脚本复用 `runs/pipeline.lock` 防止与总流程重复启动，阶段与最新目录仍从 `runs/pipeline_stage.txt`、`runs/latest_run.txt` 查看。按首组实测吞吐，均跑满约需 15 小时，可能因早停缩短。

### Decoder-only 低学习率恢复（2026-10-10）

20 轮对照中，TIGER 完成 20 轮并选中第 18 轮；Decoder-only 第 6 轮开始失稳，第 7 轮因非有限梯度范数退出。其第 5 轮 `best.pt` 验证 Recall@10=2.64%、NDCG@10=0.0156406，仍可复现。学习率较高维持时间变长是待验证的原因之一，不视为已确认根因。

恢复脚本位于 `decoder_recovery/`，不改原训练器和原实验。`bash decoder_recovery/start_train.sh` 会创建独立运行目录，只训练 Decoder-only：

- 从失败运行 `books-generative-20261009-153342-14533/decoder_only/best.pt` 的第 5 轮权重开始，最多再跑 15 轮，累计到第 20 轮。
- 原 best.pt 只有模型权重，因此 AdamW 重新初始化；这不是完整优化器状态的续训。原 epoch 6 的 last.pt 已退化，不用作恢复来源。
- 学习率峰值由 0.001 降到 0.0001；前 1,000 updates 从 0.00001 预热到峰值，然后余弦下降至 0.00001。
- 结构、固定商品编码、每轮全部 7,087,037 样本、batch=512、BF16、weight decay=0.01、clip=1、seed=42、beam=20 均保持。按累计 epoch 编排样本顺序。
- 固定同一批 20,000 验证用户，以 NDCG@10 选模，连续 3 轮不提升早停；初始第 5 轮权重也参与最佳选择。训练结束仅测试所选最佳权重，全量测试 603,668 用户。
- 每 200 batches 记录裁剪前梯度范数与本轮最大值，并检查参数/优化器有限性。梯度范数异常在裁剪修改梯度前抛错；保存 `numerical_failure.json` 的逐层信息和 `failure_snapshot.pt` 的状态、批次索引及随机状态，不跳过坏批次。
- 来源 checkpoint、数据、继承代码和恢复代码均有哈希；正式训练前复现第 5 轮验证成绩。完成 epoch 后保存 last.pt 的优化器、scheduler 和 Torch RNG，恢复时核对配置及代码。

4 项检查覆盖 warmup/余弦边界、优化器与学习率恢复一致性、异常梯度不被静默裁剪、有限梯度裁剪。真实 Books 权重的短流程先运行 1 个小 epoch 后退出，再 resume 完成下一轮，最佳权重恢复验证通过，测试集未使用。短流程目录为 `runs/decoder-recovery-smoke-20261010-a`，不是正式结果。

后台运行遵循同一个 `pipeline.lock`。日志通过 `runs/latest_run.txt` 定位，外层日志为 `runs/pipeline-decoder-recovery-20261010.log`。正式累计进度在 status.json 的 `cumulative_epoch`，本次新增轮次在 `recovery_epoch`。新的结果只能解释为低学习率和重置优化器后的恢复实验，不能与 TIGER 声称严格同学习率轨迹的结构对照。

新恢复实验自身若中断，可在锁保护下使用以下命令（保持原参数，替换实际新目录）：

```bash
flock -n runs/pipeline.lock ../generative_beauty/.venv/bin/python -u decoder_recovery/train_recovery.py \
  --parent-run runs/books-generative-20261009-153342-14533 --run-dir runs/实际恢复实验目录 --resume \
  --additional-epochs 15 --lr 0.0001 --warmup-steps 1000 --batch-size 512 --eval-batch-size 64 --patience 3 --seed 42
```

## 已完成的验证

- 两种模型约束 beam 搜索，与小商品库逐商品枚举的完整联合概率一致；包含同前三码商品、已看过滤、非法空槽和 GPT KV cache 重排。
- 训练/验证/测试的前缀位置隔离；历史截断后仍过滤全部已看商品。
- 原始 gzip 读取、时间排序、重复交互去重、短序列剔除、元数据匹配和留出位置经过小型完整输入验证。
- Padding 不改变目标 loss；主模型和量化器梯度有限且非零。
- 指标与手算命中/排名结果一致。
- 复用小份真实 Beauty 数据的 GPU 全流程通过，两个模型的最佳 checkpoint 恢复后验证结果一致，未评测测试集。
- 扩容前 400,000 个合成商品的容量检查通过，原始显存值见 `runs/capacity_check.json`；该检查不报告推荐准确率，也不代表真实服务延迟。扩容后的实际容量由 Books GPU 短流程再次验证。
- Books 下载与编码完成后，后台流程还会执行 Books 自身的真实 GPU 短流程，成功才启动正式训练。

文件：`download.py` 下载与校验；`prepare.py` 全量前缀和编码；`models.py` 两种生成器；`train.py` 训练/选模/测试；`test_pipeline.py` 正确性检查；`run_pipeline.sh` 总流程。
