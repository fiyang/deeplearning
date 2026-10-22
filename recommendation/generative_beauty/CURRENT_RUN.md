# 2026-10-09 首轮正式实验

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

已通过 4 项协议/算法校验，以及真实 GPU 短流程和 checkpoint 恢复一致性校验。短流程没有评估测试集。

正式实验启动时间：2026-10-09 09:18:20（Asia/Shanghai）。

- 机器：`ssh deeplearning-dl-1`，本轮现场确认 RTX 4080 SUPER，32,760 MiB 显存。
- 后台 screen：`beauty-generative-20261009`。
- 运行目录：`/root/autodl-tmp/deeplearning-lab/recommendation/generative_beauty/runs/beauty-tiger-20261009-091820-3491`。
- 训练顺序：热门商品验证基线 → SASRec 风格模型 → TIGER 风格生成式模型 → 各模型最终测试。
- 每个模型最多 30 轮，5 次验证不改善则早停；详细配置见 [README.md](README.md)。

查看日志：

```bash
ssh deeplearning-dl-1
tail -f /root/autodl-tmp/deeplearning-lab/recommendation/generative_beauty/runs/beauty-tiger-20261009-091820-3491.log
```

查看当前阶段：

```bash
cat /root/autodl-tmp/deeplearning-lab/recommendation/generative_beauty/runs/beauty-tiger-20261009-091820-3491/status.json
```

该文件是启动记录，不是自动刷新的监控。后续状态以远端 `status.json`、日志、进程和 GPU 为准。正式完成应同时出现 `TRAINING_COMPLETE`、`status.json` 的 `stage=complete`、以及 `summary.json`。

完整验证集的固定热门商品基线：Recall@10=0.0163663，NDCG@10=0.00799684。模型的中途验证值不作为最终测试成绩。

准备数据已同步到本机的核对记录：[数据规模](dataset_info.json)、[量化器状态](quantizer_info.json)、[下载校验](download_manifest.json)。完整原始数据、模型和训练输出保存在远端数据盘。

## 2026-10-09 已现场核对完成

确认 `stage=complete`、`TRAINING_COMPLETE`、完整 `summary.json`，该训练进程已退出。最终测试 22,363 个用户：

| 模型 | 测试 Recall@10 | 测试 NDCG@10 |
|---|---:|---:|
| 热门推荐 | 1.1537% | 0.0054593 |
| SASRec 风格 | 4.9859% | 0.0272515 |
| TIGER 风格 | 5.7059% | 0.0299614 |

SASRec 选择 epoch 28，TIGER 选择 epoch 30；二者都训练了 30 epochs。前面讨论的 SASRec 6.72% 和 TIGER 7.16% 是验证结果，不能当作最终测试结果。单 seed，未形成统计显著性或工业部署结论。

本机保存的原始最终结果：[summary.json](results/beauty-tiger-20261009-091820-3491/summary.json)。新一轮大数据结构对照见 [Amazon Books](../generative_books/README.md)。
