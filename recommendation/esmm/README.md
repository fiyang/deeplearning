# KuaiRand ESMM

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

在 AutoDL 上运行：

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation/esmm
bash start.sh
```

这会检查关键逻辑，运行独立 CTR/CVR 基线和 ESMM。每个模型最多 25 轮，验证集联合目标 AUC 连续 5 轮没有超过 0.0001 的提升则早停。最佳 checkpoint 始终保存实际最高验证 AUC。

需要已有的 PyTorch、NumPy；绘图还需要 Matplotlib。默认使用 CUDA；训练数据位于 `../data/kuairand/KuaiRand-Pure/data`。既有 MovieLens 代码和推理服务独立保留。

本实验的联合目标是 `is_click × is_like`。这是点击/有效播放与点赞的联合行为，不是购买或全部点赞。相关定义、完整特征、时间切分和限制记录在每次运行的 `protocol.json`、`feature_schema.json` 与 `data_manifest.json`。

训练输出路径会显示在启动日志中。用实际路径替换下面的 `RUN`：

```bash
python verify.py --run RUN
python report.py --run RUN
python score.py --checkpoint RUN/esmm/best.pt --user-id 0 --video-id 0 --date 20220422 --hourmin 1200 --tab 1
```

`score.py` 输出点击概率、给定点击的点赞概率、点击且点赞概率；需要原始用户与视频属性文件，并验证它们及特征词表与训练时的 hash 一致。
