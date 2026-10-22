# Ali-CCP 全量 ESMM 实验

> 历史实验记录：代码于 2026-10-22 与远端核对归档；下文成绩和运行状态仍来自此前实验，本次没有复训。数据、图表与模型产物不随 Git 分发。

使用原始 Ali-CCP 公开数据，保留 23 个字段的全部多值特征及原始浮点权重。PyTorch ESMM 共享 embedding，CTR/CVR 两个独立前馈塔，`pCTCVR = pCTR × pCVR`。训练损失是在全部有效曝光上计算的 CTR BCE + CTCVR BCE。

## 数据来源和验证

- 原始发布：https://tianchi.aliyun.com/dataset/408
- 镜像：https://aistudio.baidu.com/datasetdetail/281066
- 分享出处：https://github.com/datawhalechina/torch-rechub/issues/70#issuecomment-2230552128
- 公开仓库：`torch-rechub/e8L8YpeZ`，版本 `47a5a7c80abde9163c99d1d9598138e023b771f8`
- 页面许可标识：CC BY-NC-SA。

| 压缩包 | 精确字节数 | GB（十进制） |
|---|---:|---:|
| sample_train.tar.gz | 4,405,491,733 | 4.405 |
| sample_test.tar.gz | 5,029,299,086 | 5.029 |
| 合计 | 9,434,790,819 | 9.435 |

两个包已完整下载并匹配仓库 Git LFS SHA256，gzip 校验和解压已完成。四个原始 CSV 共 41,497,739,803 字节（41.50 GB / 38.65 GiB）。完整来源清单见 `source_manifest.json`。原始文件、压缩包和解析缓存均保留在服务器的 `/root/autodl-tmp/deeplearning-lab/recommendation/data/aliccp-full`。

原始训练文件共 42,300,135 条曝光、1,644,256 个点击，其中 230 条 click=0/conversion=1。实验排除这些不满足点击→转化顺序的记录，原文件与缓存不修改，保存排除行号。此标签过滤与 [AITM 原作者的公开预处理](https://github.com/xidongbo/AITM/blob/main/process_public_dataset.py)一致；本实验仍保留全部多值字段与权重。

| 分区 | 曝光 | 点击 | 转化 |
|---|---:|---:|---:|
| 训练 | 38,069,915 | 1,479,831 | 7,922 |
| 验证 | 4,229,990 | 164,425 | 880 |
| 测试 | 43,016,614 | 1,673,447 | 9,195 |

原始测试文件共 43,016,840 条，剔除 226 条 click=0/conversion=1。两份原始曝光文件合计 85,316,975 条，过滤后有效曝光 85,316,519 条，有效转化 17,997 条。

验证集是原始训练文件内按点击/转化标签分层抽取的 10%，seed=42。没有事件时间戳，不能解释为时间外推实验，同一用户可能出现在不同分区。测试文件保持独立，并应用同一标签过滤规则。仅从训练分区及其引用的公共特征建立词表，验证/测试未知特征映射到 0。没有负例下采样、正例过采样或类别加权。

## 网络和超参数

- 原始用户公共特征表：730,600 行，存储一次，按每条曝光的引用关联。
- 基线：23 字段，各 18 维；字段内对所有 embedding 按原始权重求和；拼接输入 414 维。
- 训练词表大小 7,901,977（含 padding/unknown 0），共享 embedding。
- CTR/CVR 双塔分别为 `414 → 360 → 200 → 80 → 2`，ReLU、dropout=0.1，两个输出的差为二分类 logit。
- 总参数 142,711,270。
- Adam，学习率 0.001，无 weight decay，batch=4096，梯度裁剪范数 5。
- 最多 10 轮，验证 CTCVR AUC 早停，patience=3，min_delta=0.0001。保留实际最高验证 AUC 的模型。
- 选定模型后才评估原始测试文件。报告 CTR AUC、点击子集 CVR AUC、全曝光 CTCVR AUC、AP、log loss 等。原始准确率不能单独说明效果。

64 维对照实验：用户 ID（101）和物品 ID（205）各 64 维，其余 21 个字段保持 18 维，总输入 `64 + 64 + 21 × 18 = 506`，两塔均为 `506 → 360 → 200 → 80 → 2`。ID 与其他特征使用两个紧凑的 embedding 表；用户与物品占不同的行，两塔共享这两个表。保留所有多值特征与原始权重。CPU/CUDA 测试覆盖加权聚合、空值、未知 ID、梯度、checkpoint 恢复及旧 18 维模型兼容性。

本实验不做哈希分桶。原始 ID 按字段区分后，用仅训练分区建立的词表映射为连续索引，没有碰撞；缺失或未见 ID 仍映射到零向量，由其余特征参与预测。64 维指每个 ID 的向量宽度，与原始 ID 的数值范围无关。哈希分桶可在固定内存预算下限制词表大小，但会让不同 ID 共享向量，且不能自动解决未见 ID 的冷启动。

`start_id64.sh` 固定引用下方 18 维基线，检查训练/验证行号及词表文件的 SHA256 完全一致，并核对非维度超参数一致。该对照保持学习率、批大小、随机种子及早停规则，只扩大这两个 ID 域；参数形状变化意味着初始随机权重不保证逐项相同。

## 运行和日志

服务器代码目录：`/root/autodl-tmp/deeplearning-lab/recommendation/esmm_aliccp_full`。

首次下载需要网页正常登录后取得两个临时 HTTPS 文件链接，放入服务器 `/tmp/aliccp-links.json`，权限 600。链接不写入代码、日志或实验快照。官方 SDK 0.3.9 的元数据访问成功，但大文件请求实测返回 403，所以当前采用网页提供的下载链接。下载支持断点续传，也可用 `download_ranges.py` 以四个连接分段传输，最终必须匹配公开 SHA256。

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation/esmm_aliccp_full
bash download_full.sh /tmp/aliccp-links.json
bash prepare_full.sh
/root/miniconda3/bin/python verify_raw.py --split train
/root/miniconda3/bin/python verify_raw.py --split test
bash start_full.sh
```

`prepare.cpp` 建立紧凑的加权稀疏表，避免将公共特征复制数千万次。解析时验证特征数量、字段名、引用、数值和偏移范围。`verify_raw.py` 独立抽查随机行和转化正例，对照原始 CSV 验证全部特征、权重、标签、关联及文件行数。

18 维基线：`esmm-aliccp-full-20261007-133218-3374`，已完成。第 5 轮触发早停，选中第 2 轮。查看日志：

```bash
tail -f /root/autodl-tmp/deeplearning-lab/recommendation/runs/esmm-aliccp-full-20261007-133218-3374.log
```

实验目录包含 `config.json`、`protocol.json`、数据来源、行号划分、词表、排除记录行号、训练历史和最佳模型。结束后生成 `summary.json`、`test_scores.npy`、`test_rows.npy`。后者将预测行映射回原始测试 CSV 的零基行号。

基线最终测试 CTR AUC=0.610822、点击子集 CVR AUC=0.664881、全曝光 CTCVR AUC=0.623247；第 2 轮验证 CTCVR AUC=0.689887。训练损失继续下降时验证 AUC 已回落，扩大 ID 向量是否有益需要实测。

64 维对照已于 2026-10-07 完成，运行名 `esmm-aliccp-id64-20261007-153915-7377`。同样第 5 轮早停、选中第 2 轮；训练/验证划分及词表 SHA256 与基线匹配。查看日志：

```bash
tail -f /root/autodl-tmp/deeplearning-lab/recommendation/runs/esmm-aliccp-id64-20261007-153915-7377.log
```

| 指标 | 全部 18 维 | 用户/物品 ID 64 维 |
|---|---:|---:|
| 验证 CTCVR AUC | 0.689887 | 0.693163 |
| 测试 CTR AUC | 0.610822 | 0.610467 |
| 测试点击子集 CVR AUC | 0.664881 | 0.665628 |
| 测试 CTCVR AUC | 0.623247 | 0.625594 |
| 测试 CTCVR AP | 0.000353787 | 0.000349451 |
| 测试 CTCVR log loss | 0.002078110 | 0.002078715 |
| 参数量 | 142,711,270 | 295,655,952 |
| 峰值 GPU 分配内存（GiB） | 14.24 | 16.77 |
| 总耗时（分钟） | 30.27 | 53.95 |

本次单种子对照中，扩大 ID 向量只带来较小的测试 CTCVR AUC 增幅（+0.002347），AP 和 log loss 未同步改善，尚不能证明稳定收益。64 维版本第 2–5 轮训练损失从 0.14731 降到 0.11764，验证 CTCVR AUC 却从 0.69316 回落到 0.65204，表现出过拟合迹象。测试集独立用户 ID 中有 150,150 / 295,787 个、独立物品 ID 中有 1,234,220 / 3,206,393 个未在实际训练分区出现；这是不同 ID 的数量比例，并非曝光占比。扩大向量无法改善这些 ID 当前的零向量表示。后续应先按见过/未见过用户和物品分组评估，再决定正则化或特征改进实验，避免单纯增加维度和轮数。

如需另开一次同配置实验，先确认 GPU 没有正在运行的训练，再在代码目录运行 `bash start_id64.sh`。每次自动创建独立目录。原 `start_full.sh` 保持 18 维基线配置。

## 学习率单因素对照

`start_id64_lr3e4.sh` 在 64 维 ESMM 基线上，把学习率从 **0.001 改为 0.0003**。网络、用户/物品 64 维、其余字段 18 维、batch=4096、dropout=0.1、Adam、无 weight decay、无学习率调度、梯度裁剪 5、seed=42、最多 10 轮及 CTCVR 验证 AUC 早停规则均保持一致。从头训练，不加载之前已训练的权重。

脚本使用 `--reference-run .../esmm-aliccp-id64-20261007-153915-7377 --comparison-change lr`。运行前要求非学习率超参数一致，且 `model.py`、`data.py`、`metrics.py` 与基线源代码快照 SHA256 一致；训练/验证行号和词表也必须一致。核对结果记录到 `reference_comparison.json`。`config.json`、启动日志及每轮 `history.csv` 均记录实际学习率。原维度对照的默认检查行为保持兼容。

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation/esmm_aliccp_full
bash start_id64_lr3e4.sh
```

只根据验证集选 checkpoint，再评估原始完整测试集一次。对照基线测试 CTR/CVR/CTCVR AUC 分别为 0.610467 / 0.665628 / 0.625594。这个新实验只检验降低固定学习率是否有帮助；不将学习率、batch 和正则化同时修改。

2026-10-07 已完成 `esmm-id64-lr3e-4-20261007-174558-11340`；训练/验证划分及词表一致，唯一允许变化的超参数为 `lr: 0.001 → 0.0003`。第 4 轮早停，按验证 CTCVR AUC 选中第 1 轮，总耗时 43.17 分钟。

| 指标 | 学习率 0.001 | 学习率 0.0003 |
|---|---:|---:|
| 验证 CTCVR AUC | 0.693163 | 0.704165 |
| 测试 CTR AUC | 0.610467 | 0.619729 |
| 测试点击子集 CVR AUC | 0.665628 | 0.691417 |
| 测试 CTCVR AUC | 0.625594 | 0.647462 |
| 测试 CTCVR AP | 0.000349451 | 0.000412990 |
| 测试 CTCVR log loss | 0.002078715 | 0.001999365 |

单种子结果显示降低学习率有益，仍需多种子验证稳定性。第 1–4 轮验证 CTCVR AUC 为 0.704165、0.700721、0.681085、0.673139，训练损失同时下降，说明更多训练轮数未改善泛化。

```bash
tail -f /root/autodl-tmp/deeplearning-lab/recommendation/runs/esmm-id64-lr3e-4-20261007-174558-11340.log
```

下一档固定学习率为 `0.0001`，启动脚本为 `start_id64_lr1e4.sh`。直接引用上面的 `0.0003` 实验，从相同随机种子重新初始化训练，继续核对非学习率超参数、模型/数据/指标代码、训练/验证行号及词表完全一致。保持最多 10 轮、patience=3、batch=4096；不加入 attention、学习率调度或新的正则化。根据验证指标选择 checkpoint，原始测试集只在选定 checkpoint 后评估一次；后续配置选择以验证集为依据，重复观察同一测试集会削弱其最终独立评估作用。

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation/esmm_aliccp_full
bash start_id64_lr1e4.sh
```

2026-10-07 已启动 `esmm-id64-lr1e-4-20261007-185351-13286`；已通过与 0.0003 实验的代码、划分和词表核对，唯一超参数变化为 `lr: 0.0003 → 0.0001`。结果待训练完成后比较。

```bash
tail -f /root/autodl-tmp/deeplearning-lab/recommendation/runs/esmm-id64-lr1e-4-20261007-185351-13286.log
```

之前的小样本实验保留在相邻 `esmm_aliccp` 目录，不能当作本次全量结果。
