"""Metric bundles that mirror the two graders exactly.

Task 2's `within_binders` / `within_non_binders` are the numbers the brief says it
looks at first, so they are computed everywhere the overall rho is.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

# Reference points printed by the two graders, for side-by-side reporting.
TASK1_REFERENCES = {
    "majority class (no model)": 0.500,
    "amino-acid composition only (1-mer)": 0.799,
    "1-NN 3-mer retrieval (no learning)": 0.931,
    "3-mer counts + logreg (starter.py)": 0.958,
    "logistic regression on 3-mer TF-IDF": 0.967,
}

TASK2_REFERENCES = {
    "3-mer TF-IDF ridge, raw target": 0.626,
    "3-mer TF-IDF ridge, percentile-rank target": 0.654,
    "oracle that knows only binder/non-binder": 0.835,
}


def classification_metrics(y_true, scores, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_hat = (scores >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat)),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
    }


def _safe_spearman(y, p) -> float:
    """Spearman that returns nan rather than raising on degenerate input."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(spearmanr(y, p).statistic)


def regression_metrics(y_true, predictions, binary_label) -> dict[str, float]:
    """Overall and within-class metrics, matching evaluate_regression.py.

    Callers must pass rows with the +/-500 clipped values already removed; the
    grader drops them and so do we, so that our validation number is comparable.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predictions, dtype=float)
    cls = np.asarray(binary_label)

    out = {
        "spearman": _safe_spearman(y, p),
        "pearson": float(pearsonr(y, p).statistic),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "within_binders": _safe_spearman(y[cls == 1], p[cls == 1]),
        "within_non_binders": _safe_spearman(y[cls == 0], p[cls == 0]),
        "n_binders": int((cls == 1).sum()),
        "n_non_binders": int((cls == 0).sum()),
    }
    # One number to rank candidate models by: the overall rho is mostly the class
    # gap, so we track the mean of the two within-class rhos alongside it. This is
    # a model-selection convenience, not a metric the graders compute.
    #
    # Both are nan for a prediction that is constant within each class (the class
    # oracle), where within-class rank correlation is genuinely undefined rather
    # than zero. np.nanmean would warn on an all-nan slice, so handle it directly.
    within = [out["within_binders"], out["within_non_binders"]]
    finite = [v for v in within if np.isfinite(v)]
    out["within_mean"] = float(np.mean(finite)) if finite else float("nan")
    return out


def format_regression(metrics: dict[str, float]) -> str:
    return (
        f"rho={metrics['spearman']:+.4f}  "
        f"within(binders)={metrics['within_binders']:+.4f}  "
        f"within(non)={metrics['within_non_binders']:+.4f}  "
        f"MAE={metrics['mae']:.3f}  R2={metrics['r2']:+.3f}"
    )
