# 第二课：用 CNN、ResNet 和 Transformer 提取图像特征

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

所有实现继续使用现有 PyTorch 2.8 和 torchvision 0.23，无需添加依赖。
原始训练入口是 `train.py`；新增的 `compare_models.py` 在相同训练设置下比较多个模型。
本轮已完成，准确率、耗时和结论见 [实测对比报告](COMPARISON.md)。

## 三种模型

| 参数名 | 模型 | 参数量 | 分类前的图像特征 |
|---|---|---:|---:|
| `smallcnn` | 第一课的 6 层卷积 CNN | 814,570 | 256 维 |
| `resnet18` | 适配 CIFAR 小图的 ResNet-18 | 11,173,962 | 512 维 |
| `vit` | 教学用小型 Vision Transformer | 2,693,578 | 192 维 |

**ResNet 仍然属于卷积神经网络。** 它使用残差连接，允许特征跨过一些层，与后续结果相加，帮助更深的网络学习。
我们将 torchvision ResNet-18 的输入卷积改为 3×3、步长 1，移除最开始的最大池化，适配 32×32 输入。

**ViT 使用注意力机制学习不同图像区域之间的关系。** 本项目的处理过程是：

```text
32×32 RGB 图片
  → 切成 64 个 4×4 小块（每块有 48 个像素通道数值）
  → 线性投影成 64 个 192 维向量
  → 添加分类 token 和位置编码，共 65 个 token
  → 6 层 Transformer，每层 3 个注意力头
  → 取分类 token，得到整张图片的 192 维特征
  → 线性分类器，输出 10 个类别分数
```

这里的小型 ViT 是为教学和 CIFAR-10 编写的实现，不是官方 ViT-B/16 或 timm ViT-Tiny 的等同配置。
Transformer 各层独立初始化；训练过程中会更新特征提取器和分类器的全部参数。

## 本轮对比方法

用户手动实验已经把第一课 CNN 训练到 16 轮，因此本轮三个模型也统一训练 16 轮。

- 相同 45,000 / 5,000 / 10,000 的训练、验证、测试划分，种子 42。记录索引哈希，确保划分一致。
- 相同 batch size 256、AdamW、学习率 0.001、weight decay 0.0005、16 轮余弦衰减。
- 相同随机裁剪、水平翻转和归一化。
- 模型全部从随机初始化开始，不使用额外预训练数据。
- 不使用早停。各模型按验证准确率保存最佳 checkpoint，再对测试集做一次最终评估。
- 同一张 GPU 上依次运行，并记录训练耗时、参数量和显存使用。

这比较的是**一套固定训练方案下的表现**。模型参数量、内部 Dropout 和计算量不同；固定轮数不等于固定算力，也不代表分别调参后的最高准确率。
本轮只有一个随机种子，不能据此断言某类架构在所有任务中都优于其他架构。模型选择依据验证集，测试集仅用于最终报告。

## 自己运行

进入远端项目目录后，运行三个模型：

```bash
cd /root/autodl-tmp/deeplearning-lab
RUN_DIR="runs/compare-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
/root/miniconda3/bin/python -u compare_models.py \
  --models smallcnn resnet18 vit \
  --epochs 16 --batch-size 256 --lr 0.001 \
  --weight-decay 0.0005 --seed 42 \
  --data-dir data --output "$RUN_DIR"
```

只跑 Transformer 时，把 `--models smallcnn resnet18 vit` 改成 `--models vit`。
长任务可以在 `screen` 中运行。每次使用一个新结果目录；脚本拒绝覆盖既有目录。

结果目录内按模型名建立子目录。每个模型保存 `best.pt`、`config.json`、`summary.json`、`metrics.csv`、学习曲线、混淆矩阵、样例预测和样例特征。
共同目录保存 `comparison.csv`、`comparison.json`、`comparison.png` 和实际运行代码的 `source/` 快照。

## 用 Transformer 导出图片特征

例如刚运行完上面的对比，当前终端仍保留 `$RUN_DIR` 时：

```bash
/root/miniconda3/bin/python infer_model.py \
  --checkpoint "$RUN_DIR/vit/best.pt" \
  --image "$RUN_DIR/vit/sample_test_image.png" \
  --features-output "$RUN_DIR/vit/my_features.npy"
```

输出包含类别预测和 `feature_shape: [1, 192]`，特征向量保存在指定的 NumPy 文件中。
也可替换 `--image` 为自己的图片。图片会转为 RGB 并缩放到 32×32。
特征是模型学习到的数值表示，不是类别名称；可以作为图像检索或其他分类器的输入。这里的特征只经过 CIFAR-10 任务训练，其迁移效果需要另做验证。

`infer_model.py` 同样支持 ResNet 和 CNN checkpoint，会自动读取模型类型；原始第一课的 checkpoint 也可按 CNN 加载。

## 代码位置和资料

- `models.py`：模型结构，以及 `forward_features` / `extract_features`。
- `compare_models.py`：统一训练、验证、测试和比较。
- `infer_model.py`：单图预测与特征导出。
- `check_models.py`：梯度、特征形状、图片批量维度、Transformer 各层初始化和权重恢复检查。
- [torchvision ResNet-18 官方文档](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.resnet18.html)
- [PyTorch TransformerEncoder 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.TransformerEncoder.html)
- [Vision Transformer 原论文](https://arxiv.org/abs/2010.11929)：描述图像分块与 Transformer，也展示了大规模预训练后迁移学习的效果。
