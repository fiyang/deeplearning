# Qwen2.5-7B：LoRA＋GRPO

规则打分的数学题强化学习实验。底座为 `Qwen/Qwen2.5-7B-Instruct`，以 BF16 加载；这是 LoRA，不是 4-bit QLoRA。

## 数据到模型更新

原始 GSM8K 的 question → system/user prompt → 模型为同一道题采样 4 个回答 → 按最终答案打分 → GRPO 计算组内相对优势 → 反向传播更新 LoRA A、B。

标准数值答案放在 `solution` 字段，只供奖励函数使用，不输入模型。每个回答必须以唯一的 `<answer>NUMBER</answer>` 结束，数字正确且格式合法得 1，否则得 0。奖励不单独验证中间推导。

## 已有配置

| 项目 | 取值 |
|---|---|
| 训练／测试 | 固定 512／128 道，去重并检查无重叠 |
| 轮数 | 1，128 次优化器更新 |
| LoRA | rank=8、alpha=16、dropout=0 |
| 作用模块 | 每层 q/k/v/o/gate/up/down，共 28 层 |
| 可训练参数 | 20,185,088 |
| 学习率 | 5e-6，预热 5%，cosine |
| 每题生成数 | 4 |
| microbatch／梯度累积 | 1／16 |
| prompt／completion 上限 | 768／512 tokens |
| 采样 | temperature=0.9、top_p=0.95 |
| KL 系数 | beta=0.04 |
| 测试解码 | greedy，每题一个回答 |

实际配置以 [config.json](config.json) 和 [experiment.py](experiment.py) 为准。rank=32 只是后续候选，没有改动当前配置。

## 历史结果

2026-10-15 那一轮：训练前 105/128（82.03125%），训练后 115/128（89.84375%），增加 7.8125 个百分点；格式正确率从 91.40625% 升至 99.21875%。训练约 29 分 44 秒，最终评估约 127 秒。

这些是历史运行记录中的结果，本次归档没有重新运行；128 题、一个随机种子的观察不足以证明通用推理能力稳定提升。历史运行名：`qwen7b-grpo-20261015-094619`。

## 在远端运行

需要已安装 CUDA PyTorch 的 Linux GPU 机器，以及足够的模型缓存空间。**仅在远端下载数据、模型和依赖。**

```bash
cd /root/autodl-tmp/deeplearning-lab/rl/qwen7b_grpo
bash run.sh
```

脚本建立独立环境、安装固定依赖、检查奖励函数、准备 GSM8K、下载底座，先执行 smoke 再执行正式训练。默认不关机；需要结束后关机时显式运行 `AUTO_SHUTDOWN=1 bash run.sh`，这会在成功或失败退出时执行关机。

日志保存在 `runs/<运行名>.log`，运行目录包含 `config.json`、`parameters.json`、`data_manifest.json`、抽样后的数据、训练指标、评估和 `summary.json`。`adapter/` 保存 LoRA 权重；没有合并到底座。

```bash
tail -f "runs/$(cat runs/latest_attempt.txt).log"
```

仅运行奖励函数检查，不需要 GPU 或数据：

```bash
python3 -m unittest -v test_rewards.py
```

需要先恢复资源、核对远端版本再复跑，见根目录 [TODO](../../TODO.md)。
