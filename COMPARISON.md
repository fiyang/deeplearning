# 模型对比结果：ResNet 提高了准确率，ViT 已跑通特征提取

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

实验日期：2026 年 9 月 30 日（上海时间）。
实验目录：`runs/model-comparison-20260930-153806`。

**在统一训练 16 轮的设置下，ResNet-18 测试准确率为 89.56%，比原始 CNN 的 85.10% 高 4.46 个百分点。**
小型 ViT 也完成训练和特征导出，但本轮测试准确率为 59.33%。

## 同一训练条件下的实际结果

| 模型 | 参数量 | 最佳验证准确率 | 测试准确率 | 最佳轮次 | 训练与每轮验证耗时 |
|---|---:|---:|---:|---:|---:|
| 原始 Small CNN | 814,570 | 85.64% | 85.10% | 16 | 66.9 秒 |
| ResNet-18，适配 CIFAR 输入 | 11,173,962 | **90.32%** | **89.56%** | 14 | 127.0 秒 |
| Small ViT，教学实现 | 2,693,578 | 60.20% | 59.33% | 15 | 123.5 秒 |

耗时为每轮训练和验证之和，不包含数据准备、初始评估、最终测试、保存图表和下载结果。
三个模型都实际跑完 16 轮，没有早停；表中最佳轮次是验证集选出的 checkpoint。

![准确率与训练耗时对比](runs/model-comparison-20260930-153806/comparison.png)

原始 CNN 复跑得到的 85.10%，与用户手动运行 `manual-20260930-152210` 的 16 轮结果一致。
第一课的 81.26% 对应 8 轮，因此本轮不把它作为等轮数的比较基线。

## 对我们的实验意味着什么

**想先提高当前任务准确率，可以使用这次的 ResNet-18。** 它在相同 10,000 张测试图片上比 CNN 多识别正确 446 张。
例如，猫这一类从 70.2% 提高到 79.8%。代价是参数量更大，训练与每轮验证耗时约为原来的 1.9 倍。
ResNet 本身仍属于 CNN，改进主要是更深的网络和残差连接。

**Transformer 可以提取图像特征，换成 Transformer 并不保证在当前设置下提分。**
这次 ViT 的训练和验证曲线都持续改善，但 16 轮内达到的准确率低于 CNN 和 ResNet。
这是小型教学 ViT、随机初始化、固定共享训练方案的一次结果，不能作为所有 Transformer、预训练 ViT 或各自充分调参后模型的结论。

如果继续研究 ViT，值得单独比较预训练后的微调，或研究更合适的训练轮数、学习率和增强方式。
预训练引入外部数据和额外训练成本，应在报告中另列，不与本轮从零训练混为同一条件。
参见 [ViT 原论文](https://arxiv.org/abs/2010.11929) 对大规模预训练与迁移学习的说明。

## Transformer 的特征已经可以直接导出

这次 ViT 将 32×32 图片切成 64 个 4×4 小块，通过 6 层 Transformer，再得到一张图的 192 维表示。
特征提取器和分类器在训练时一起更新；推理时可单独取出分类头前的表示。

已验证的样例文件：

```text
runs/model-comparison-20260930-153806/vit/sample_features.npy
```

实际形状为 `(1, 192)`，表示 1 张图片、192 个特征数值。CNN 和 ResNet 也分别导出了 `(1, 256)` 和 `(1, 512)` 的样例。

在远端提取另一份特征：

```bash
cd /root/autodl-tmp/deeplearning-lab
/root/miniconda3/bin/python infer_model.py \
  --checkpoint runs/model-comparison-20260930-153806/vit/best.pt \
  --image runs/model-comparison-20260930-153806/vit/sample_test_image.png \
  --features-output /root/autodl-tmp/vit-my-image-features.npy
```

可把 `--image` 替换为自己的图片；再次运行需换一个尚不存在的输出文件名。

## 自己再跑一轮

例如单独重跑表现最好的 ResNet：

```bash
cd /root/autodl-tmp/deeplearning-lab
/root/miniconda3/bin/python -u compare_models.py \
  --models resnet18 \
  --epochs 16 --batch-size 256 --lr 0.001 \
  --weight-decay 0.0005 --seed 42 \
  --data-dir data \
  --output "runs/resnet-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
```

将 `--models resnet18` 改成 `--models vit` 可运行 Transformer；改成 `--models smallcnn resnet18 vit` 可重跑完整对比。
运行入口和结构说明见 [MODELS.md](MODELS.md)，模型源码见 [models.py](models.py)。

## 对比的范围与验证

- 同一台 GPU、同样的数据增强、AdamW 及余弦衰减、相同 batch size 256、学习率 0.001、权重衰减 0.0005、种子 42。
- 训练 / 验证 / 测试数量为 45,000 / 5,000 / 10,000；三个模型的训练、验证索引哈希一致，训练和验证之间没有交集。
- 所有模型从随机初始化开始。参数量和计算量不同，这不是等算力或等参数量比较，也不是各自最优超参数的比较。
- 验证准确率决定各自 checkpoint 和推荐模型；测试集仅用于最终报告，每个模型测试一次。
- 梯度、批量维度一致性、ViT 各层独立初始化、模型权重保存与恢复均通过检查。
- 单独进程重新加载各模型后，预测与评估样例一致；特征维度正确、数值有限。
- 混淆矩阵总数和正确数与结果报告一致；训练任务以退出码 0 完成。
- 模型权重、配置、指标、图表、特征和实际执行源码快照，均保存在本机和远端对应结果目录。

原始数据见 [comparison.json](runs/model-comparison-20260930-153806/comparison.json) 和 [comparison.csv](runs/model-comparison-20260930-153806/comparison.csv)。
