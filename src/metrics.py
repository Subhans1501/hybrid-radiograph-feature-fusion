import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)


def compute_clinical_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred) * 100.0), 2),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0) * 100.0), 2),
        "recall_sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0) * 100.0), 2),
        "specificity": round(float(spec * 100.0), 2),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0) * 100.0), 2),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0), 2),
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": [round(float(x), 4) for x in fpr], "tpr": [round(float(x), 4) for x in tpr]},
    }
