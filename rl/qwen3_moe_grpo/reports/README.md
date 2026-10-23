# 已完成实验的小型汇总

这些 JSON 从本地保留的远端运行产物提取，只包含聚合指标和运行标识。原始题目、生成回答、模型权重、日志及云平台账户信息未发布。`source_manifest.json` 记录原文件相对位置与 SHA-256，用于与保留的完整产物核对；不能仅凭哈希重建原文件。

| 文件 | 内容 |
|---|---|
| `round1_full_eval.json` | 1319 道题的基座／第一轮对比、配对检验、原子集复现 |
| `round2_summary.json` | 第二轮完整测试、两次验证、训练步数、显存与耗时 |
| `round2_reward_groups.json` | 512 步日志重建的奖励组统计与假设，省略逐步明细 |
| `sampling_diagnostic.json` | 四组采样最终指标、有效数值混合组、定义与限制 |
| `source_manifest.json` | 上述汇总的原始来源哈希 |

路径中的运行日期是历史实验时间，不代表当前机器状态。`accuracy`、`*_rate` 为 0～1 比例，`accuracy_delta_pp` 与奖励组中的 `percent` 为百分比／百分点。`plot_epoch_accuracy.py` 直接读取前两份汇总即可重画图表。
