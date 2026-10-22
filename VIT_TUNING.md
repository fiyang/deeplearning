# ViT 第二次实验：更长训练与 DeiT 风格配置

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

本实验按用户确认的方案训练一个 12 层 ViT，共 100 轮。原来的 6 层 `vit` 及其 checkpoint 继续可用；新模型名是 `vit_tiny12`。

## 实际配置

| 项目 | 设置 |
|---|---|
| 输入 | CIFAR-10，32×32；patch 4×4，共 64 个 patch，另加分类 token |
| 编码器 | 12 层，192 维，3 个注意力头，MLP 768 维 |
| 参数量 | 5,362,762，以运行生成的 config.json 为准 |
| 初始化 | 从随机权重开始；不使用预训练模型或教师模型 |
| 训练 | 100 轮，batch size 256，seed 42，无早停 |
| 优化器 | AdamW，betas=(0.9, 0.999)，eps=1e-8 |
| 学习率 | 按每次参数更新调度；前 5 轮从 1e-6 升至 2.5e-4，之后余弦下降至 1e-5 |
| 权重衰减 | 0.05；bias、LayerNorm、位置编码和分类 token 不衰减 |
| 图片增强 | RandomCrop(32, padding=4)、水平翻转、RandAugment(N=2, M=9，双线性插值) |
| 混合增强 | 每个 batch 随机选 Mixup(alpha=0.8) 或 CutMix(alpha=1.0)，各 50% |
| 标签平滑 | 0.1，在混合后的概率标签上应用 |
| Dropout | 0；没有另加 stochastic depth |
| 数值精度 | 训练使用 BF16 自动混合精度，损失计算/权重/优化器保持 FP32；验证、测试、推理使用 FP32 |

这是参考 [DeiT 论文](https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf)及[官方训练代码](https://github.com/facebookresearch/deit/blob/main/main.py)后适配 CIFAR-10 的教学实验，不是论文完整复现。图片尺寸、patch、增强实现、训练长度、初始化等都存在差异。
学习率按官方的批量缩放公式计算：`5e-4 × 256 / 512 = 2.5e-4`。使用机器已有的 PyTorch/torchvision，不新增训练依赖。

## 如何运行

### 后台训练 300 轮

启动脚本 `start_vit.sh` 默认从随机权重开始训练 300 轮，预热 5 轮，其余参数沿用本页配置：

```bash
cd /root/autodl-tmp/deeplearning-lab
bash start_vit.sh
```

脚本自动创建带时间戳和进程号的结果目录、日志以及独立 screen 会话，并打印查看进度的方法。断开 SSH 后继续运行，训练正常结束后写入同名 `.exit` 文件，`0` 表示成功。查看日志时按 `Ctrl+C` 只退出查看，不停止训练。

使用 `bash start_vit.sh --epochs 200` 可改变轮数。使用 `bash start_vit.sh --dry-run` 只检查启动配置，不创建结果文件或启动训练。
这只是新的启动入口，不提供断点续训，也不会自动改变已记录的 100 轮实验结果。

### 前台运行或指定输出目录

登录后，在远端执行：

```bash
cd /root/autodl-tmp/deeplearning-lab
bash run_vit.sh
```

脚本先检查本地数据，再创建新的时间戳结果目录，已有目录会拒绝覆盖。主要代码是 `train_vit.py`，模型在 `models.py`。

也可以指定输出目录和轮数：

```bash
bash run_vit.sh runs/my-vit-100 --epochs 100 --warmup-epochs 5
```

如要使用 FP32 训练，添加 `--precision fp32`。`best.pt` 和 `last.pt` 是推理权重，未保存完整优化器状态，不能作为无缝续训检查点。

## 怎么读结果

- 沿用原先的 45,000 张训练、5,000 张验证划分；10,000 张测试图片仅在训练结束、验证选定模型后评估一次。
- `metrics.csv` 的 `train_loss` 使用混合标签和标签平滑，`val_loss` 使用原始标签，二者不是完全相同的目标函数。
- `train_soft_target_accuracy` 表示预测类别分配到的混合标签概率，不能当成普通准确率。
- 每 10 轮在固定、未增强的 5,000 张训练图片上计算 `clean_train_accuracy`，可以与验证曲线一起观察拟合情况。
- 训练结束后，额外计算最佳 checkpoint 在全部 45,000 张未增强训练图片上的准确率。
- `best.pt` 由验证准确率选择；`last.pt` 保存第 100 轮。`source/` 保存运行时的代码。
- `inference_check.json` 和 `sample_features.npy` 验证独立加载与 192 维特征提取。

本次同时改变了网络深度、训练长度和训练方法；与旧实验的差值表示整体方案效果，无法单独归因于参数量。旧 ResNet 只训练了 16 轮，因此也不是相同训练预算的比较。

## 结果

本次运行目录：`runs/vit-tiny12-20260930-160733`。在远端实时查看进度：

```bash
cd /root/autodl-tmp/deeplearning-lab
tail -f runs/vit-tiny12-20260930-160733.log
```

训练在后台 screen 会话 `vit-tiny12-20260930-160733` 中执行，断开 SSH 后仍然运行。日志同名 `.exit` 文件为 `0` 且包含 `TRAINING_COMPLETE`，才代表训练和独立推理检查全部成功。

训练结束后可单独加载模型：

```bash
/root/miniconda3/bin/python infer_model.py \
  --checkpoint runs/vit-tiny12-20260930-160733/best.pt \
  --image runs/vit-tiny12-20260930-160733/sample_test_image.png
```

训练已完成，退出码为 0，独立加载、预测一致性与特征提取检查通过。

| 模型 | 参数量 | 训练轮数 | 最佳验证准确率 | 测试准确率 |
|---|---:|---:|---:|---:|
| 原 ViT，6 层 | 2,693,578 | 16 | 60.20% | 59.33% |
| 新 ViT，12 层 | 5,362,762 | 100 | **76.84%** | **76.67%** |
| 原 ResNet-18，参考 | 11,173,962 | 16 | 90.32% | 89.56% |

新 ViT 在 10,000 张测试图片中预测正确 7,667 张，比原 ViT 多 1,734 张，提升 **17.34 个百分点**；仍比原 ResNet-18 低 12.89 个百分点。
最佳 checkpoint 出现在第 100 轮。训练及每轮评估合计 1,123.52 秒；含最终评估和图表生成约 1,141.99 秒，即约 19 分钟，不含随后独立加载检查。
PyTorch 记录的峰值已分配显存为 1,481.25 MiB，该数值不是整个进程的显存占用。

最佳模型在完整、未增强的 45,000 张训练图片上准确率为 **80.53%**，与验证集相差约 3.69 个百分点。训练集也未达到很高准确率，因此没有证据表明“单纯增大参数量”就能解决剩余差距。更长训练、增强强度与模型结构的影响，需要后续分别控制变量验证。本轮只运行一个随机种子。

![新旧模型对比](runs/vit-tiny12-20260930-160733/comparison_to_baseline.png)

左图显示验证准确率随实际轮数变化；旧 ViT 的曲线在第 16 轮结束。右图是最终测试准确率，明确标出了不同训练预算。

![100 轮训练曲线](runs/vit-tiny12-20260930-160733/learning_curves.png)

中图的橙线是固定 5,000 张训练子集的无增强评估，不是训练时混合标签的分数；它与上文完整 45,000 张训练集准确率的口径不同。

原始指标见 [summary.json](runs/vit-tiny12-20260930-160733/summary.json)、[config.json](runs/vit-tiny12-20260930-160733/config.json)、[metrics.csv](runs/vit-tiny12-20260930-160733/metrics.csv)。
模型文件是 `runs/vit-tiny12-20260930-160733/best.pt`。完整源码快照在同目录的 `source/`。

重建对比图并检查结果文件：

```bash
/root/miniconda3/bin/python report_vit.py \
  --run runs/vit-tiny12-20260930-160733 \
  --baseline runs/model-comparison-20260930-153806
```
