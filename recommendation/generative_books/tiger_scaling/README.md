# Books TIGER 宽度实验

> 历史实验记录：下文结果、路径和运行状态来自此前实验；2026-10-20 归档，尚未与远端最终文件核对。数据、图表与模型产物不随 Git 分发。

在原 TIGER 20 轮实验基础上只扩大主特征宽度和前馈层，独立保存代码与结果，不改已完成模型。此目录的模型与训练器从 `generative_books/models.py`、`train.py` 派生。

| 配置 | 原 TIGER | 本轮 |
|---|---:|---:|
| d_model | 128 | 256 |
| d_ff | 1024 | 2048 |
| Encoder / Decoder 层数 | 4 / 4 | 4 / 4 |
| Attention heads / d_kv | 6 / 64 | 6 / 64 |
| 参数量 | 5,043,712 | 14,281,344 |
| 词表 | 4,563 | 4,563 |

Attention 内部投影宽度保持 6×64=384；T5 的主特征宽度与 head 数乘 d_kv 是独立参数。两处宽度扩大后总参数为原来的 2.8315 倍。此轮没有增加层数。

从 seed=42 随机初始化开始训练，不加载原 TIGER 权重。原商品语义编码和全部数据复用，不重新训练 RQ-VAE。每轮全量 7,087,037 样本，最近 20 件商品历史，batch=512，BF16，AdamW lr=0.001、weight_decay=0.01、clip=1、dropout=0.1。warmup 1,000 updates，余弦下降到 0.0001，最多 20 epochs，与原 TIGER 20 轮学习率轨迹一致。连续 3 次验证 NDCG@10 不改善早停。

固定同一批 20,000 验证用户及全量商品候选，beam=20，过滤全部历史商品；通过验证选模型，最后对全部 603,668 用户测试。本轮只跑 TIGER 和热门商品测试基线，不重跑 Decoder-only。单随机种子结果仅说明本次配置差异，不能宣称普遍架构结论。

训练器新增裁剪前梯度范数日志，在非有限梯度导致裁剪前抛错，并保存逐层统计及错误批次/随机状态。此保护不改变正常有限梯度的裁剪和更新。原恢复训练目录保持原样。

## 已验证

- 3 项测试：新旧模型配置只相差 d_model/d_ff；小商品库约束生成等于逐商品联合概率；padding、有限梯度和权重恢复。
- 真实完整商品库容量检查：训练 batch=512、评测 batch=64、beam=20，24 个训练 batch 中后 20 个测得约 4,422 样本/秒，峰值 allocated=4,979 MiB、reserved=5,280 MiB。仅代表短时容量与吞吐，不是推荐质量或线上延迟。
- 真实 Books 短流程 `runs/tiger-w256-smoke-20261010-a`：4,096 样本、64 验证用户，CHECKPOINT_RELOAD_OK 和 SMOKE_COMPLETE，测试集未用。

## 启动与恢复

在远端 `generative_books` 目录下运行：

```bash
bash tiger_scaling/start_train.sh
cat runs/latest_run.txt
cat runs/pipeline_stage.txt
```

启动脚本使用与其他 Books 任务相同的 `pipeline.lock`，新日志为 `runs/books-tiger-w256-20epochs-时间-进程号.log`。配置、代码快照和数据签名会存入各运行目录。正式阶段为 `tiger_w256_training`，结束为 `complete_tiger_w256`；必须同时检查 status.json、TRAINING_COMPLETE 和进程。

若中断，可从上一个完整 epoch 恢复，保持配置、数据和代码一致：

```bash
flock -n runs/pipeline.lock ../generative_beauty/.venv/bin/python -u tiger_scaling/train.py \
  --resume --run-dir runs/实际运行目录 --epochs 20 --lr 0.001 \
  --batch-size 512 --eval-batch-size 64 --val-users 20000 --beam-size 20 --patience 3 --seed 42
```

参照的原 TIGER 最佳模型位于 `runs/books-generative-20261009-153342-14533/tiger/best.pt`，其测试 Recall@10=0.033337861208478835、NDCG@10=0.020612178234259957。原模型在第 18 轮取得最佳验证。
