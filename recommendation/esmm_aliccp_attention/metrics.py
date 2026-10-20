"""Binary ranking/calibration metrics, with explicit tie handling."""
import numpy as np


def binary_metrics(y, probability):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) == 0:
        return {"count": 0, "auc": None, "average_precision": None}
    assert y.shape == p.shape and np.isin(y, [0, 1]).all()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    n = len(y); positive = int(y.sum()); negative = n - positive
    order = np.argsort(p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    counts = np.diff(np.r_[starts, n])
    ranks = np.repeat(starts + (counts + 1) / 2, counts)
    auc = float((ranks[y[order] == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)) if positive and negative else None
    descending = order[::-1]
    ends = np.r_[np.flatnonzero(np.diff(p[descending])), n - 1]
    true_positive = np.cumsum(y[descending])[ends]
    ap = float(np.sum(np.diff(np.r_[0, true_positive]) / positive * true_positive / (ends + 1))) if positive else None
    pred = p >= .5
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = positive - tp; tn = negative - fp
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(count=n, positives=positive, positive_fraction=positive / n,
                auc=auc, average_precision=ap,
                log_loss=float(-(y * np.log(q) + (1 - y) * np.log1p(-q)).mean()),
                brier=float(np.square(p - y).mean()), mean_probability=float(p.mean()),
                accuracy_at_0_5=float((pred == y).mean()), all_negative_accuracy=negative / n,
                precision_at_0_5=tp / (tp + fp) if tp + fp else 0.,
                recall_at_0_5=tp / positive if positive else None,
                confusion_at_0_5=dict(tp=tp, fp=fp, fn=fn, tn=tn))


def funnel_metrics(labels, scores):
    click, conversion = labels.T
    return {"ctr": binary_metrics(click, scores[:, 0]),
            "cvr": binary_metrics(conversion[click == 1], scores[click == 1, 1]),
            "ctcvr": binary_metrics(conversion, scores[:, 2])}


def auc_interval(y, scores):
    """Approximate DeLong normal interval; limited when positives/clusters are few."""
    y=np.asarray(y);scores=np.asarray(scores,dtype=np.float64)
    positive=np.sort(scores[y==1]);negative=np.sort(scores[y==0])
    if min(len(positive),len(negative))<2:return None
    v10=(np.searchsorted(negative,positive,side='left')+np.searchsorted(negative,positive,side='right'))/(2*len(negative))
    v01=1-(np.searchsorted(positive,negative,side='left')+np.searchsorted(positive,negative,side='right'))/(2*len(positive))
    se=float(np.sqrt(v10.var(ddof=1)/len(positive)+v01.var(ddof=1)/len(negative)))
    center=float(v10.mean())
    return dict(lower=max(0.,center-1.96*se),upper=min(1.,center+1.96*se),standard_error=se,
                method='Approximate row-independent DeLong 95% interval; does not account for user clustering or data preprocessing.')
