"""Classification metrics: precision, recall, F1, IoU and confusion matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.config import CLASS_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(CLASS_NAMES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # IoU = TP / (TP + FP + FN), the usual metric for point-cloud segmentation.
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)

    per_class = {
        CLASS_NAMES[label]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "iou": round(float(iou[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    return {
        "overall_accuracy": round(float(tp.sum() / cm.sum()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "mean_iou": round(float(iou.mean()), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def metrics_table(metrics: dict) -> str:
    """Markdown table, ready to paste in the docs."""
    rows = ["| Class | Precision | Recall | F1 | IoU | Support |", "|---|--:|--:|--:|--:|--:|"]
    for name, m in metrics["per_class"].items():
        rows.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['iou']:.3f} | {m['support']:,} |")
    rows.append(f"| **macro / mean** | | | **{metrics['macro_f1']:.3f}** | **{metrics['mean_iou']:.3f}** | |")
    return "\n".join(rows)


def plot_confusion_matrix(metrics: dict, path: str | Path, title: str = "Confusion matrix") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion_matrix"], dtype=float)
    cm_normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # row = true class
    names = list(metrics["per_class"])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            value = cm_normalized[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.5 else "black")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
