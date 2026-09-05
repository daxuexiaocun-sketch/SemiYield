"""Classification metrics with explicit undefined-value reasons."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report(y, probabilities, threshold=0.5):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(y):
        return {"status": "skipped", "reason": "No test samples"}
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    reasons = {}
    both = len(np.unique(y)) == 2
    result = {
        "status": "completed",
        "rows": len(y),
        "failure_rate": float(y.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pr_auc": float(average_precision_score(y, probabilities)) if both else None,
        "roc_auc": float(roc_auc_score(y, probabilities)) if both else None,
        "recall": float(recall_score(y, predicted)) if y.sum() else None,
        "precision": float(precision_score(y, predicted)) if predicted.sum() else None,
        "f1": float(f1_score(y, predicted)) if y.sum() + predicted.sum() else None,
        "mcc": float(matthews_corrcoef(y, predicted))
        if both and len(np.unique(predicted)) == 2
        else None,
    }
    for key, value in result.items():
        if value is None:
            reasons[key] = "Required observed or predicted class is absent"
    result["undefined_reasons"] = reasons
    return result
