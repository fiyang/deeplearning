# Books 验证集召回诊断

两种诊断都使用已选定的模型权重、原固定 20,000 名验证用户和完整商品库，过滤全部历史物品，不更新参数，不用于测试集选模。

已完成的曲线与训练快照见 [2026-10-28 结果](RESULTS_20261028.md)。

当前诊断加载器支持原 5,043,712 参数 TIGER 及其 `tiger_refinement` 权重。1,428 万参数的 `tiger_growth` / `tiger_continuation` 使用不同模型宽度，需先适配加载器并重新验证，不能直接套用本目录命令或沿用小版模型的 Top-100 结果。

- `beam_sweep.py`：改变 beam=20/50/100，比较搜索范围对同一个 Recall@10 的影响。旧预测文件仅保留前 20 个物品，不能拿来计算 Recall@50/@100。
- `recall_curve.py`：冻结指定运行的最佳 checkpoint，先用 beam=20 精确复现已保存的验证指标，再用 beam=100 保存全部 100 个结果。从这一份列表的嵌套前缀计算 Recall@10/@50/@100，并统计命中名次区间。用于区分候选列表长度收益和 beam 搜索收益。

```bash
cd /root/autodl-tmp/deeplearning-lab/recommendation/generative_books
python_bin=../generative_beauty/.venv/bin/python
"$python_bin" -m unittest discover -s diagnostics -p test_recall_curve.py -v
"$python_bin" diagnostics/recall_curve.py \
  --model-run "$(cat runs/latest_tiger_refinement.txt)" \
  --run-dir runs/books-recall-curve-UNIQUE_NAME
```

诊断独立保存权重与源码副本，并校验数据签名、用户集合和原指标。训练器以原子替换方式写入最佳权重，因此即便训练继续推进，本次评估仍针对复制时固定的那个 checkpoint。输出 `summary.json`、`status.json` 和包含 `users/predicted/target` 的 `top100_validation.npz`。

只读评估可以与训练并行，但有独立诊断锁，并要求至少 5 GiB 空闲显存。2026-10-28 的运行使用外层 600 秒 timeout。并行时的耗时不能当作独占 GPU 推理基准。脚本不改变训练进程或机器开关机状态。

注意：beam search 是近似搜索。目标未出现在生成的 Top-100，只证明本次列表未命中，不证明其精确全库概率排名一定超过 100。Recall 曲线应与 NDCG、搜索宽度诊断和同数据强基线一起解释。
