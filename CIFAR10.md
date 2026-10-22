# 深度学习第一课：CIFAR-10 图像分类

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

用 AutoDL 公共盘已有的图片训练一个小型卷积神经网络，练习完整的训练、评估、保存和加载模型流程。

**第一轮已完成：独立测试集准确率 81.26%。** 先读 [第一轮实验结果与图表解读](RESULTS.md)，再按本文复现。

第二课新增了 ResNet 和 Vision Transformer，训练与特征提取方法见 [模型对比说明](MODELS.md)。
已完成三模型统一 16 轮对比：ResNet 测试准确率 89.56%，见 [完整对比结果](COMPARISON.md)。

第三课把 ViT 调整为 12 层，使用 100 轮、学习率预热与混合增强，测试准确率从 59.33% 提升至 **76.67%**，见 [ViT 调参实验与运行方法](VIT_TUNING.md)。

随后将同一 12 层 ViT 训练到 300 轮，测试准确率为 **90.56%**。宽度增加至 384 后，同样 300 轮达到 **90.88%**；192 维网络训练 600 轮达到 **93.68%**。配置、曲线和结果见 [ViT 扩大实验](VIT_SCALING.md)。

按验证集成绩选出的 192 维 / 600 轮最佳模型已部署为 HTTP 推理服务，支持分类和特征提取。调用方法与性能实测见 [推理服务](INFERENCE.md)。

## 我们用什么

- 机器：`ssh deeplearning-dl-1`，1 张 RTX 4080 SUPER，约 32 GB 显存。
- 远端项目：`/root/autodl-tmp/deeplearning-lab`。
- Python：`/root/miniconda3/bin/python`。
- 已有环境：Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、matplotlib 3.10.5。
- 原始数据：`/root/autodl-pub/cifar-10/cifar-10-python.tar.gz`。
- 数据集为 10 类、32×32 彩色图片：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。

## 第一课的学习顺序

1. **看数据**：一张图片对应一个正确类别，训练就是根据这些例子调整模型参数。
2. **看模型**：`train.py` 中的 `SmallCNN` 用卷积识别图像特征，最后输出 10 类的分数。
3. **看训练循环**：预测 → 计算误差 `loss` → `backward()` 求梯度 → `optimizer.step()` 更新参数。
4. **看曲线**：误差是否下降、验证准确率是否上升，以及两者是否开始分离。
5. **看预测**：在从未参与训练的图片上检查表现，同时观察预测错误。
6. **加载模型**：`predict.py` 使用保存的模型预测单张图片，验证模型可以被复用。

## 数据划分与本次实验

官方训练部分共 50,000 张。固定种子 42，每类留 500 张做验证，共 5,000 张；剩下 45,000 张用于训练。
官方 10,000 张测试图片只在训练结束、按验证准确率选择模型后做一次最终评估。
验证和测试不做随机增强。训练时使用随机裁剪和水平翻转。

模型从随机参数开始训练，不使用预训练权重。默认 8 轮、batch size 256、AdamW 初始学习率 0.001、余弦学习率衰减。
固定种子与数据划分便于复现，但不同软硬件下不保证逐位一致。单次实验成绩不等于模型在真实照片上的可靠性。
训练准确率包含随机增强和 Dropout，不能只用它与验证准确率的差值判断过拟合。

## 运行

在 Mac 终端登录后运行：

```bash
ssh deeplearning-dl-1
cd /root/autodl-tmp/deeplearning-lab
bash run.sh
```

每次默认创建新的结果目录，不覆盖既有实验。复制的压缩包先核对官方 MD5，再解压；不需要从公网下载。
如果希望退出 SSH 后继续训练，在远端使用 `screen`：

```bash
screen -S cifar10-lesson
cd /root/autodl-tmp/deeplearning-lab
mkdir -p runs
bash run.sh 2>&1 | tee runs/manual-training.log
```

按 `Ctrl+A`，再按 `D` 可离开会话；`screen -r cifar10-lesson` 返回。

## 结果文件

- `config.json`：实际版本、机器、超参数和数据量。
- `metrics.csv` / `learning_curves.png`：每轮训练、验证损失和准确率。
- `best.pt`：验证准确率最高的模型权重。
- `summary.json`：最终测试准确率、各类别准确率、时间和显存占用。
- `predictions.png`：测试集固定前 16 张图片的预测，包含错误样本。
- `confusion_matrix.png`：哪些真实类别容易被识别为其他类别。
- `inference_check.json`：独立加载模型后，对第一张测试图片的预测。
- `split_indices.json`：训练、验证图片的准确划分。

单张图片推理，把 `<结果目录>` 替换成实际目录：

```bash
/root/miniconda3/bin/python predict.py \
  --checkpoint <结果目录>/best.pt \
  --image <结果目录>/sample_test_image.png
```

模型的输出分数未进行置信度校准。处理自己的照片时会缩放到 32×32，而且仅能在这 10 类中选择。

## 下一轮可以研究的问题

每次只改变一个因素，再比较验证集结果：

- 多训练几轮，是否继续提高？
- 改变学习率，误差曲线有什么变化？
- 去掉数据增强，训练和验证的差距会怎样变化？

避免用测试集反复挑选参数；继续实验时仍用验证集比较。

## 来源

- [CIFAR-10/100 官方数据与校验值](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch 图像分类入门教程](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [torchvision CIFAR10 接口](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
- [AutoDL 公共数据清单](https://www.autodl.com/console/public-data)

本项目是独立编写的教学实现；使用当前机器已有的 PyTorch 2.8 环境。数据和结果保存在数据盘，重要成果另外同步到本机。
