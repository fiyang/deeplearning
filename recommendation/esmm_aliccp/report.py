"""Generate reproducible comparison figures and a short Chinese experiment report."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();root=args.run
    summary=json.loads((root/'summary.json').read_text());config=json.loads((root/'experiment_config.json').read_text())
    schema=json.loads((root/'feature_schema.json').read_text());results=summary['results']
    plt.style.use('seaborn-v0_8-whitegrid');fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for kind,color in [('independent','#64748b'),('esmm','#2563eb')]:
        with (root/kind/'history.csv').open() as f:history=list(csv.DictReader(f))
        epochs=[int(row['epoch']) for row in history]
        for ax,key,title in zip(axes,['train_loss','val_ctr_auc','val_ctcvr_auc'],['Training objective (different losses)','Validation CTR AUC','Validation CTCVR AUC (only 8 positives)']):
            ax.plot(epochs,[float(row[key]) for row in history],label=kind.upper(),color=color,marker='o',markersize=3)
            ax.set_title(title,fontsize=11);ax.set_xlabel('Epoch')
        axes[2].axvline(results[kind]['best_epoch'],color=color,linestyle=':',alpha=.5)
    axes[0].set_ylabel('Mean loss');axes[1].set_ylabel('AUC');axes[2].set_ylabel('AUC');axes[0].legend()
    fig.suptitle('Ali-CCP public preprocessed sample | train 80,001 / validation 20,000 / test 100,001',fontsize=13)
    fig.savefig(root/'training_curves.png',dpi=180);fig.savefig(root/'training_curves.pdf');plt.close(fig)
    comparison=[]
    for kind,result in results.items():
        for task,m in result['test'].items():
            comparison.append(dict(model=kind,task=task,auc=m['auc'],average_precision=m['average_precision'],
                log_loss=m['log_loss'],accuracy_at_0_5=m['accuracy_at_0_5'],all_negative_accuracy=m['all_negative_accuracy'],
                positives=m['positives'],count=m['count']))
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    lines=['# Ali-CCP ESMM 实验记录','',
        '使用 PaddleRec 官方脚本公开链接中的预处理样本，PyTorch 从头训练。文件体积约 1.49 GB，但训练和测试文件各只有 100,001 条，**不是原始 8,400 万条 Ali-CCP 全量数据**。',
        '', '数据来源：[PaddleRec 下载脚本](https://github.com/PaddlePaddle/PaddleRec/blob/master/datasets/ali-ccp/run.sh)。模型参考：[ESMM 论文](https://arxiv.org/abs/1804.07931)。','',
        '| 分区 | 曝光数 | 点击数 | 转化数 |','|---|---:|---:|---:|']
    for name,count in summary['counts'].items():lines.append(f"| {name} | {count['rows']:,} | {count['clicks']:,} | {count['conversions']:,} |")
    lines+=['','保留来源中的训练／测试文件分离；训练文件按 (click, conversion) 固定种子分层划分 80%／20%。文件没有时间戳，因此不是按时间验证。未负采样、未过采样、未丢弃任何行。各文件的样本 ID 独立编号，不作为输入。','',
        '## 模型与训练','',
        f"23 个特征字段 → 共享 Embedding({schema['vocabulary_size']:,} × {config['embedding_dim']}) → 字段内 sum pooling → 拼接 {schema['input_dim']} 维 → CTR/CVR 两个 MLP：{schema['input_dim']} → {' → '.join(map(str,config['hidden']))} → 2。两类 softmax 的正类概率分别为 pCTR 与 pCVR，pCTCVR = pCTR × pCVR。",
        '', '所有重复字段值完整保留，无长度截断；词表仅使用训练分区建立，未见特征统一映射到零向量。原文件不提供特征权重，所以使用等权求和。',
        '', 'ESMM 损失：全曝光 BCE(click, pCTR) + BCE(conversion, pCTCVR)。CVR 不直接施加单独标签损失。对照模型的 CTR 与 CVR 分别使用独立 Embedding，CVR 仅在点击样本上训练；两个模型初始化权重、样本顺序种子相同，但参数量及损失尺度不同。',
        '', f"Adam，学习率 {config['lr']}，batch size {config['batch_size']}，最多 {config['epochs']} 轮，早停 patience={config['patience']}，min_delta={config['min_delta']}，dropout={config['dropout']}，weight decay={config['weight_decay']}，梯度裁剪 5，seed={config['seed']}。按验证集 CTCVR AUC 保存最佳权重；两个模型权重均固定后才评估测试集。",
        '', '## 测试集结果','',
        '| 模型 | CTR AUC | CVR AUC（点击内） | CTCVR AUC（全曝光） | CTCVR AP | CTCVR LogLoss |','|---|---:|---:|---:|---:|---:|']
    for name,result in results.items():
        m=result['test'];lines.append(f"| {name} | {m['ctr']['auc']:.6f} | {m['cvr']['auc']:.6f} | {m['ctcvr']['auc']:.6f} | {m['ctcvr']['average_precision']:.6f} | {m['ctcvr']['log_loss']:.6f} |")
    for name,result in results.items():
        m=result['test']['ctcvr'];ci=m['auc_interval_approx_95'];lines+=['',f"{name}：训练 {result['completed_epochs']} 轮，最佳第 {result['best_epoch']} 轮；参数 {result['parameter_count']:,}；CTCVR 在 0.5 阈值下准确率 {m['accuracy_at_0_5']:.4%}，全部预测不转化也有 {m['all_negative_accuracy']:.4%}。CTCVR AUC 的近似 95% 区间 [{ci['lower']:.4f}, {ci['upper']:.4f}]（假设样本独立，未校正用户聚类）。"]
    esmm=results['esmm']['test']['ctcvr'];baseline=results['independent']['test']['ctcvr']
    lines+=['',f"本次 ESMM 相比独立模型的测试 CTCVR AUC 差值为 {esmm['auc']-baseline['auc']:+.6f}。单次结果仅作探索，不能据此宣称模型稳定优越。",
        '', '**主要限制：** 验证集仅 8 个、测试集仅 42 个转化正例，早停和 AUC 都很容易受少数样本影响；原发布样本的采样／预处理过程不能完全还原。本实验只跑一个随机种子，与论文全量结果及快手点赞任务不可直接比较。准确率必须与全负例基线一起看。',
        '', f"设备：{config['gpu']}；完整训练与评估 {summary['elapsed_seconds']:.1f} 秒；GPU 峰值分配 {summary['peak_gpu_allocated_mib']:.1f} MiB。",'',
        '![训练曲线](training_curves.png)','',
        '复现、验证和单条推理命令见源代码目录 README.md；本目录保存 source/、完整参数、原始文件 SHA256、划分索引、词表、权重、每轮指标和测试预测。验证结果见 verification.json。']
    (root/'EXPERIMENT.md').write_text('\n'.join(lines)+'\n')
    print(f'REPORT_COMPLETE {root}',flush=True)


if __name__=='__main__':main()
