# 深度学习实验室

集中保存我们在 AutoDL 上学习和实现的图像分类、推荐排序、生成式推荐与大模型强化学习代码。主要框架是 PyTorch。

## 从哪里开始

| 学习主题 | 代码入口 | 说明 |
|---|---|---|
| CIFAR-10：CNN、ResNet18、ViT | 根目录 `train.py`、`models.py`、`train_vit.py` | [视觉实验](CIFAR10.md)、[模型对比](MODELS.md)、[ViT 调参](VIT_TUNING.md)、[扩大训练](VIT_SCALING.md) |
| 图像分类推理与性能测试 | `inference_service.py`、`benchmark_inference.py` | [推理服务](INFERENCE.md) |
| MovieLens：ID Embedding＋MLP、侧信息、DeepFM 等对照 | `recommendation/` | [推荐入门](recommendation/README.md)、[特征](recommendation/FEATURES.md)、[AUC 实验](recommendation/AUC_EXPERIMENT.md) |
| 用户历史行为 Transformer | `recommendation/sequence_model.py` | [序列实验](recommendation/SEQUENCE_EXPERIMENT.md) |
| KuaiRand：ESMM 多目标学习 | `recommendation/esmm/` | [说明](recommendation/esmm/README.md)；点击与点赞联合目标，不是购买 CVR |
| Ali-CCP 小样本 ESMM | `recommendation/esmm_aliccp/` | [说明](recommendation/esmm_aliccp/README.md) |
| Ali-CCP 全量 ESMM、64 维 ID、学习率对照 | `recommendation/esmm_aliccp_full/` | [说明](recommendation/esmm_aliccp_full/README.md) |
| 特征域 Attention＋ESMM | `recommendation/esmm_aliccp_attention/` | [说明](recommendation/esmm_aliccp_attention/README.md) |
| AITM 作者方法与参数对照 | `recommendation/aitm_aliccp_official/` | [说明](recommendation/aitm_aliccp_official/README.md)，含固定版本参考源码和 MIT 许可证 |
| Amazon Beauty：RQ-VAE、TIGER、SASRec | `recommendation/generative_beauty/` | [说明](recommendation/generative_beauty/README.md) |
| Amazon Books：TIGER、Decoder-only、恢复与扩大网络 | `recommendation/generative_books/` | [说明](recommendation/generative_books/README.md) |
| Qwen2.5-7B：LoRA＋GRPO、规则奖励 | `rl/qwen7b_grpo/` | [强化学习实验](rl/qwen7b_grpo/README.md) |
| Qwen3-30B-A3B：单卡 BF16 MoE LoRA＋GRPO | `rl/qwen3_moe_grpo/` | [训练、评估与采样诊断](rl/qwen3_moe_grpo/README.md)，完整 1319 题：基座 91.58% → 第一轮 92.87% → 第二轮 93.33%；含结果表、轮次图和有效训练信号分析 |

[Encoder–Decoder 与 Decoder-only 推理交互图](docs/visualizations/encoder-decoder-inference.html)需下载后用浏览器打开。

## 当前归档状态

2026-10-22 已连接两台无卡模式 AutoDL，核对 168 个源码、配置和说明文件，合并远端差异，补齐 GRPO 预检查、Ali-CCP 数据核验和生成式推荐环境记录。文件来源、SHA-256 与保留差异见 [本次迁移记录](docs/migrations/2026-10-22/README.md)。2026-10-20 的原始恢复记录仍保留。

2026-10-23 补充第三台 A800 上已完成的 Qwen3 BF16 GRPO 两轮实验、全量评估和采样诊断。小型汇总保存在 [reports](rl/qwen3_moe_grpo/reports/)，原始题目、回答、权重与完整日志不进入 Git。本次代码归档没有启动新训练。

各实验文档中的成绩、机器、运行目录和“正在运行”等描述均是当时的历史记录。训练产物与服务状态没有随代码迁移。本次核对两台项目源码仍未发现独立 MMoE 实现，已有的是 ESMM、AITM 和特征域 Attention。后续验证与实验见 [TODO](TODO.md)，来源与检查边界见 [归档记录](docs/IMPORT.md)。

## 环境与数据

- 原实验以 Linux、Python 3.12、PyTorch 2.8.0＋CUDA 12.8 为基础。不同实验使用独立环境，GRPO 有自己的固定依赖。
- **数据集、模型权重、Python 依赖均在远端机器直接下载**；不通过本地电脑中转大文件。
- Git 仅保存代码、配置、文档和小型来源清单；忽略原始数据、checkpoint、日志、缓存、授权下载链接和机器账户信息。
- 保留原目录层级，避免破坏相对导入。远端可将本仓库克隆到 `/root/autodl-tmp/deeplearning-lab`，与历史脚本路径保持一致；已有同名目录时先核对，不覆盖。
- CIFAR-10 使用 AutoDL 公共盘；MovieLens 用 `recommendation/prepare_data.py`；Ali-CCP 全量下载流程见其文档，需要有效授权链接。Amazon 内容编码器与数据准备参照各实验说明。

根目录 [requirements.txt](requirements.txt)记录基础实验环境；生成式推荐额外依赖见 [requirements-generative.txt](requirements-generative.txt)。这些文件供远端安装使用，不会在本地检查时自动安装任何依赖。

[历史环境快照](docs/environments/README.md)保存了实际版本和内容编码器来源哈希；快照中的镜像构建路径不应当作可直接安装的依赖地址。

## 本地轻量检查

```bash
python3 scripts/check_local.py
```

检查 Python/JSON/Shell 语法、AITM 参考文件哈希，运行奖励解析、下载校验、训练排队控制及 Qwen3 实验与诊断测试；不下载数据，不启动训练。2026-10-22 已另在远端通过 61 项 CPU 模型单元测试与 5 种视觉结构检查，详见 [验证记录](docs/VALIDATION.md)。Qwen3 的 GPU 短流程及正式实验已完成；其他历史模型的 GPU 复跑仍列在 TODO。

## 复现规则

每次运行使用独立结果目录，记录实际配置、数据划分、依赖版本和随机种子。用验证集选模型，固定独立测试集做最终评估；历史分数不代表跨数据集、跨候选集或不同评估口径下的模型优劣。引用的第三方代码和数据遵循各自的许可证。
