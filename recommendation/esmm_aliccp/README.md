# Ali-CCP ESMM

> 历史实验记录：下文结果、路径和运行状态来自此前实验；2026-10-20 归档，尚未与远端最终文件核对。数据、图表与模型产物不随 Git 分发。

PyTorch 实验，使用 PaddleRec 官方脚本公开的预处理 CSV。训练与测试文件各 100,001 行，分别仅有 41 和 42 个转化正例；不是 8,400 万条原始全量数据。

数据来源：https://github.com/PaddlePaddle/PaddleRec/blob/master/datasets/ali-ccp/run.sh

- https://paddlerec.bj.bcebos.com/esmm/traindata_10w.csv
- https://paddlerec.bj.bcebos.com/esmm/testdata_10w.csv

机器：`ssh deeplearning-dl-1`。代码位于 `/root/autodl-tmp/deeplearning-lab/recommendation/esmm_aliccp`，数据位于相邻 `../data/aliccp-paddlerec`。

## 训练

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation/esmm_aliccp
bash start.sh
```

默认先训练独立 CTR/CVR 对照模型，再训练共享 Embedding 的 ESMM。每个模型最多 30 轮，验证集 CTCVR AUC 早停 patience=6。启动时打印完整日志路径。若需要调整参数：

```bash
bash start.sh --epochs 30 --patience 6 --batch-size 1024 --lr 0.001
```

当前实验路径记录在 `current_run.txt`：

```bash
run_dir=$(cat current_run.txt)
tail -f "${run_dir}.log"
/root/miniconda3/bin/python verify.py --run "$run_dir"
/root/miniconda3/bin/python report.py --run "$run_dir"
/root/miniconda3/bin/python score.py --checkpoint "$run_dir/esmm/best.pt" \
  --csv ../data/aliccp-paddlerec/testdata_10w.csv --row-index 0
```

模型：23 个字段，每字段 18 维，字段内完整 sum pooling，拼接 414 维；CTR/CVR 两塔都是 414→360→200→80→2。ESMM 输出 pCTCVR=pCTR×pCVR，以全曝光点击和转化 BCE 之和训练。

词表仅基于训练分区建立，缺失和未知值映射零；不使用样本 ID 或标签作为特征。原始 CSV 第四列已经是特征，不能跳过；重复字段值全部保留。原始特征权重和时间戳不在这些 CSV 中。

`summary.json` / `comparison.csv` 保存 AUC、AP、LogLoss、0.5 阈值准确率及全负例准确率；`EXPERIMENT.md` / `training_curves.png` 为报告和曲线。转化极稀疏，不能以约 99.96% 的准确率宣称模型有效。
