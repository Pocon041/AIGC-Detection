from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve


@dataclass
class MetricResult:
    n: int
    positives: int
    negatives: int
    positive_rate: float
    ap_baseline: float
    auroc: float
    ap: float
    acc: float
    balanced_acc: float
    precision: float
    recall: float
    false_alarm: float
    f1: float
    tpr_at_fpr_1: float
    threshold_at_fpr_1: float
    tpr_at_fpr_5: float
    threshold_at_fpr_5: float
    best_threshold: float
    balanced_threshold: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "n": int(self.n),
            "positives": int(self.positives),
            "negatives": int(self.negatives),
            "positive_rate": float(self.positive_rate),
            "ap_baseline": float(self.ap_baseline),
            "auroc": float(self.auroc),
            "ap": float(self.ap),
            "acc": float(self.acc),
            "balanced_acc": float(self.balanced_acc),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "false_alarm": float(self.false_alarm),
            "f1": float(self.f1),
            "tpr_at_fpr_1": float(self.tpr_at_fpr_1),
            "threshold_at_fpr_1": float(self.threshold_at_fpr_1),
            "tpr_at_fpr_5": float(self.tpr_at_fpr_5),
            "threshold_at_fpr_5": float(self.threshold_at_fpr_5),
            "best_threshold": float(self.best_threshold),
            "balanced_threshold": float(self.balanced_threshold),
        }


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def threshold_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> tuple[float, float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return float("nan"), 0.0, float("nan")
    best = valid[np.argmax(tpr[valid])]
    return float(thresholds[best]), float(tpr[best]), float(fpr[best])


def best_acc_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(labels) == 0:
        return float("nan"), float("nan")
    thresholds = np.unique(scores)
    if len(thresholds) > 2000:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 2000))
    best_acc = -1.0
    best_thr = float(thresholds[0])
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int64)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = float(acc)
            best_thr = float(thr)
    return best_acc, best_thr


def operating_metrics(labels, suspicious_scores, threshold: float) -> Dict[str, float]:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(suspicious_scores).astype(np.float64)
    preds = (scores >= threshold).astype(np.int64)
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.logical_and(preds == 1, positives).sum())
    fp = int(np.logical_and(preds == 1, negatives).sum())
    tn = int(np.logical_and(preds == 0, negatives).sum())
    fn = int(np.logical_and(preds == 0, positives).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    specificity = tn / (fp + tn) if (fp + tn) > 0 else float("nan")
    balanced_acc = (recall + specificity) / 2.0 if not np.isnan(recall) and not np.isnan(specificity) else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(labels, preds)) if len(labels) else float("nan"),
        "balanced_acc": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "false_alarm": float(false_alarm),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(labels) == 0:
        return float("nan"), float("nan")
    thresholds = np.unique(scores)
    if len(thresholds) > 2000:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 2000))
    best_balanced = float("nan")
    best_thr = float("nan")
    for thr in thresholds:
        metrics = operating_metrics(labels, scores, float(thr))
        value = metrics["balanced_acc"]
        if not np.isfinite(value):
            continue
        if not np.isfinite(best_balanced) or value > best_balanced:
            best_balanced = float(value)
            best_thr = float(thr)
    return best_balanced, best_thr


def compute_binary_metrics(labels, suspicious_scores) -> MetricResult:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(suspicious_scores).astype(np.float64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    positive_rate = positives / len(labels) if len(labels) else float("nan")
    auroc = _safe_auc(labels, scores)
    ap = _safe_ap(labels, scores)
    acc, thr = best_acc_threshold(labels, scores)
    balanced_acc, balanced_thr = best_balanced_threshold(labels, scores)
    if np.isfinite(balanced_thr):
        best_balanced_ops = operating_metrics(labels, scores, balanced_thr)
    else:
        best_balanced_ops = {
            "precision": float("nan"),
            "recall": float("nan"),
            "false_alarm": float("nan"),
            "f1": float("nan"),
        }
    thr1, tpr1, _ = threshold_at_fpr(labels, scores, 0.01)
    thr5, tpr5, _ = threshold_at_fpr(labels, scores, 0.05)
    return MetricResult(
        n=len(labels),
        positives=positives,
        negatives=negatives,
        positive_rate=positive_rate,
        ap_baseline=positive_rate,
        auroc=auroc,
        ap=ap,
        acc=acc,
        balanced_acc=balanced_acc,
        precision=best_balanced_ops["precision"],
        recall=best_balanced_ops["recall"],
        false_alarm=best_balanced_ops["false_alarm"],
        f1=best_balanced_ops["f1"],
        tpr_at_fpr_1=tpr_at_fpr(labels, scores, 0.01),
        threshold_at_fpr_1=thr1,
        tpr_at_fpr_5=tpr5,
        threshold_at_fpr_5=thr5,
        best_threshold=thr,
        balanced_threshold=balanced_thr,
    )


def grouped_metrics(labels, scores, groups) -> Dict[str, Dict[str, float]]:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    groups = np.asarray(groups).astype(str)
    out = {}
    for group in sorted(set(groups.tolist())):
        idx = groups == group
        if idx.sum() == 0:
            continue
        out[group] = compute_binary_metrics(labels[idx], scores[idx]).to_dict()
    return out


def _macro_average(grouped: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "auroc",
        "ap",
        "acc",
        "balanced_acc",
        "tpr_at_fpr_1",
        "tpr_at_fpr_5",
        "precision",
        "recall",
        "false_alarm",
        "f1",
    ]
    macro = {}
    values = [v for k, v in grouped.items() if not k.startswith("_")]
    for key in keys:
        arr = np.asarray([v.get(key, float("nan")) for v in values], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        macro[key] = float(arr.mean()) if len(arr) else float("nan")
    macro["num_groups"] = int(len(values))
    return macro


def one_vs_real_grouped_metrics(labels, scores, groups) -> Dict[str, Dict[str, float]]:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    groups = np.asarray(groups).astype(str)
    real_idx = labels == 0
    out = {}
    for group in sorted(set(groups.tolist())):
        fake_idx = np.logical_and(groups == group, labels == 1)
        if fake_idx.sum() == 0:
            continue
        idx = np.logical_or(real_idx, fake_idx)
        item = compute_binary_metrics(labels[idx], scores[idx]).to_dict()
        item["real_count"] = int(real_idx.sum())
        item["fake_count"] = int(fake_idx.sum())
        out[group] = item
    if out:
        out["_macro"] = _macro_average(out)
    return out


def format_binary_metrics(name: str, metrics: Dict[str, float]) -> str:
    def pct(key: str) -> str:
        value = metrics.get(key, float("nan"))
        return "nan" if value is None or not np.isfinite(value) else f"{value * 100:.2f}%"

    def num(key: str) -> str:
        value = metrics.get(key, float("nan"))
        return "nan" if value is None or not np.isfinite(value) else f"{value:.4f}"

    return (
        f"{name}: n={int(metrics.get('n', 0))} "
        f"real={int(metrics.get('negatives', 0))} fake={int(metrics.get('positives', 0))} "
        f"fake_rate={pct('positive_rate')} | "
        f"AUROC={num('auroc')} AP={num('ap')} AP_base={num('ap_baseline')} | "
        f"TPR@FPR1={pct('tpr_at_fpr_1')} TPR@FPR5={pct('tpr_at_fpr_5')} | "
        f"bACC={pct('balanced_acc')} recall={pct('recall')} FA={pct('false_alarm')}"
    )


def format_grouped_metrics(title: str, grouped: Dict[str, Dict[str, float]], max_rows: int = 20) -> str:
    if not grouped:
        return f"{title}: no groups"
    rows = [f"{title}:"]
    macro = grouped.get("_macro")
    if macro:
        rows.append(
            "  _macro "
            f"AUROC={macro.get('auroc', float('nan')):.4f} "
            f"AP={macro.get('ap', float('nan')):.4f} "
            f"TPR@FPR1={macro.get('tpr_at_fpr_1', float('nan')) * 100:.2f}% "
            f"TPR@FPR5={macro.get('tpr_at_fpr_5', float('nan')) * 100:.2f}% "
            f"bACC={macro.get('balanced_acc', float('nan')) * 100:.2f}%"
        )
    shown = 0
    for group, metrics in grouped.items():
        if group.startswith("_"):
            continue
        rows.append(
            f"  {group} "
            f"real={metrics.get('real_count', metrics.get('negatives', 0))} "
            f"fake={metrics.get('fake_count', metrics.get('positives', 0))} "
            f"AUROC={metrics.get('auroc', float('nan')):.4f} "
            f"AP={metrics.get('ap', float('nan')):.4f} "
            f"TPR@FPR1={metrics.get('tpr_at_fpr_1', float('nan')) * 100:.2f}% "
            f"TPR@FPR5={metrics.get('tpr_at_fpr_5', float('nan')) * 100:.2f}% "
            f"bACC={metrics.get('balanced_acc', float('nan')) * 100:.2f}%"
        )
        shown += 1
        if shown >= max_rows:
            remaining = len([k for k in grouped if not k.startswith("_")]) - shown
            if remaining > 0:
                rows.append(f"  ... {remaining} more groups in json")
            break
    return "\n".join(rows)

