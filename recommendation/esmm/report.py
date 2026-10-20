"""Generate an experiment report and validation curves from measured results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); args = p.parse_args()
    run = args.run
    s = json.loads((run/"summary.json").read_text()); cfg=json.loads((run/"experiment_config.json").read_text())
    model_cfg=json.loads((run/"esmm/config.json").read_text())
    rows=[]
    for kind, r in s["results"].items():
        for split, metrics in r["evaluation"].items():
            for task, m in metrics.items():
                rows.append(dict(model=kind, split=split, task=task, auc=m["auc"], ap=m["average_precision"],
                                 accuracy=m["accuracy_at_0_5"], all_negative_accuracy=m["all_negative_accuracy"],
                                 log_loss=m["log_loss"], count=m["count"], positives=m["positives"]))
    with (run/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig, axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for kind in s["results"]:
        h=list(csv.DictReader((run/kind/"history.csv").open()))
        epochs=[int(v["epoch"]) for v in h]
        for ax, key, title in zip(axes,["val_ctr_auc","val_cvr_auc","val_ctcvr_auc"],["CTR","Like given click","Click AND like"]):
            ax.plot(epochs,[float(v[key]) for v in h],marker="o",ms=3,label=kind)
            ax.set(title=title,xlabel="Epoch",ylabel="Validation AUC");ax.grid(alpha=.2)
    axes[0].legend();fig.savefig(run/"validation_curves.png",dpi=170);fig.savefig(run/"validation_curves.pdf");plt.close(fig)
    text=["# KuaiRand-Pure ESMM 实验", "", "本实验用快手短视频反馈复现 ESMM 的核心训练方式，不是阿里购买转化实验。",
          "", "## 数据与标签", "", "- CTR：`is_click`，在不同界面可能表示点击或有效播放。",
          "- CVR：在 `is_click=1` 样本中发生点赞的概率。",
          "- CTCVR：`is_click AND is_like`，不是所有点赞的概率。未点击但点赞的记录保留，联合标签为 0。",
          "- 两个常规日志文件拼接后，按 `time_ms` 以北京时间 2022-04-19、2022-04-22 午夜切分；原始 date 与 time_ms 在午夜附近偶有出入。",
          "- 随机曝光数据单独测试；无负采样、无正例过采样，不丢弃冷用户/冷视频。", "",
          "| 数据部分 | 样本数 | 点击数 | 点击且点赞数 |", "|---|---:|---:|---:|"]
    for split,c in s["counts"].items():text.append(f"| {split} | {c['rows']:,} | {c['clicks']:,} | {c['joint_click_likes']:,} |")
    text += ["", "## 网络与超参数", "", f"32 个单值类别字段加 1 个多标签视频类别字段；每个字段 Embedding 为 {cfg['embedding_dim']} 维，视频多标签做 sum pooling。拼接宽度 {model_cfg['input_dim']}。",
             "", "输入包含用户 ID、视频 ID、tab、小时、星期、18 个匿名用户特征，以及作者、类型、音乐、视频时长/尺寸分桶、视频年龄、标签。词表仅在训练集出现的值上建立；未知值为零向量。",
             "", "不输入当前反馈、播放时长、点赞/关注状态或全月聚合统计。匿名用户画像没有 as-of 时间，无法保证其严格历史可用性；视频元数据亦按发布快照使用。",
             "", f"ESMM：共享 Embedding → 两个独立的 {cfg['hidden']} ReLU MLP → 每个分支 2 个 logits → softmax。pCTCVR = pCTR × pCVR。总参数 {model_cfg['parameter_count']:,}。",
             "", "损失为全曝光上的 CTR BCE + 联合目标 BCE；没有额外 CVR BCE。乘积损失在 log 域计算以保持数值稳定。",
             "", "基线使用两份独立 Embedding：CTR 在全部曝光上训练，CVR 只在点击样本上训练。两者起始权重与 ESMM 对应模块相同，批次打乱种子相同；基线参数更多，不是严格参数量匹配的消融。",
             "", f"Adam lr={cfg['lr']}，betas=(0.9,0.999)，eps=1e-8，weight_decay={cfg['weight_decay']}；batch={cfg['batch_size']}，上限 {cfg['epochs']} 轮，patience={cfg['patience']}，min_delta={cfg['min_delta']}，dropout={cfg['dropout']}，seed={cfg['seed']}。",
             "", "每个模型按验证集 CTCVR AUC 保存 checkpoint；全部 checkpoint 固定后才评估常规测试集和随机曝光集。此处是单种子结果，不做显著性结论。", "",
             "## 实测结果", "", "| 模型 | 测试集 | 目标 | AUC | AP | 准确率（0.5） | 全预测负例准确率 | log loss |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:text.append(f"| {r['model']} | {r['split']} | {r['task']} | {r['auc']:.6f} | {r['ap']:.6f} | {r['accuracy']:.2%} | {r['all_negative_accuracy']:.2%} | {r['log_loss']:.6f} |")
    text += ["", "低正例率会使准确率非常高；模型效果应结合 AUC、AP、log loss 和全负例基线判断。", "", "![验证曲线](validation_curves.png)", "", "## 复现与文件", "",
             "在 recommendation/esmm 目录运行 `bash start.sh`。日志在 recommendation/runs/esmm-kuairand-<时间>-<进程号>.log。",
             "", "`esmm/best.pt` 是 ESMM 最佳 checkpoint，`independent/best.pt` 是基线。`feature_schema.json` 保存词表；`data_manifest.json` 保存原始文件 hash、切分口径和计数；`protocol.json` 保存评估约定；`verification.json` 保存独立复核。",
             "", f"总实验耗时 {s['elapsed_seconds']:.1f} 秒；峰值 GPU 张量显存 {s['peak_gpu_allocated_mib']:.1f} MiB。",
             "", "参考：[ESMM 论文](https://arxiv.org/pdf/1804.07931)、[KuaiRand 字段说明](https://kuairand.com/)、[原始数据](https://zenodo.org/records/10439422)。"]
    (run/"EXPERIMENT.md").write_text("\n".join(text)+"\n")
    print(run/"EXPERIMENT.md")


if __name__ == "__main__":
    main()
