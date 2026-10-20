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
    click, like = labels.T
    return {"ctr": binary_metrics(click, scores[:, 0]),
            "cvr_like_given_click": binary_metrics(like[click == 1], scores[click == 1, 1]),
            "ctcvr_click_and_like": binary_metrics(click * like, scores[:, 2])}
