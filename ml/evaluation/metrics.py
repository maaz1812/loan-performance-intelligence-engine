"""
Evaluation metrics.

Implements the metric set mandated by prd.md Section 4.2 and orchestrator.md
Section 3.5:

    ROC-AUC, PR-AUC, F1, recall at fixed precision, Brier score, macro-F1

plus calibration diagnostics, because a probability that is well-ranked but
badly calibrated is unusable for decisioning -- and `backend.md` Section 3.3
maps calibrated probabilities directly onto reviewer-facing risk tiers.

PR-AUC (average precision) is the headline ranking metric here rather than
ROC-AUC. Default and prepayment are rare events, and ROC-AUC is optimistic
under heavy class imbalance because the true-negative pool is enormous; PR-AUC
reflects performance on the positive class a reviewer actually cares about.
Both are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)


def recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray,
                        min_precision: float = 0.50) -> dict:
    """
    Highest recall achievable while holding precision at or above a floor.

    This is the metric a reviewer queue is actually tuned on: "if I accept a
    50%-precision alert list, what share of true events do I catch?"
    Returns the operating threshold too, so the result is directly actionable.
    """
    if len(np.unique(y_true)) < 2:
        return {"recall": float("nan"), "threshold": float("nan"),
                "precision": float("nan"), "min_precision": min_precision}
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thr) == len(prec) - 1
    ok = prec[:-1] >= min_precision
    if not ok.any():
        return {"recall": 0.0, "threshold": 1.0,
                "precision": float(prec.max()), "min_precision": min_precision}
    idx = int(np.argmax(rec[:-1] * ok))
    return {
        "recall": float(rec[idx]),
        "threshold": float(thr[idx]),
        "precision": float(prec[idx]),
        "min_precision": min_precision,
    }


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov separation -- the standard credit-risk discrimination
    statistic, reported alongside AUC because risk teams read it directly.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(y_prob)
    y = np.asarray(y_true)[order]
    pos = np.cumsum(y) / max(y.sum(), 1)
    neg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """Event rate in the top-k% of scores, divided by the base rate."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    base = float(np.mean(y_true))
    if base == 0:
        return float("nan")
    top = max(1, int(n * k))
    idx = np.argsort(-np.asarray(y_prob))[:top]
    return float(np.mean(np.asarray(y_true)[idx]) / base)


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    precision_floors: tuple[float, ...] = (0.30, 0.50, 0.70),
) -> dict:
    """
    Full metric suite for one binary target.

    If `threshold` is None the F1-optimal threshold is chosen from the
    precision-recall curve. Reporting metrics at a fixed 0.5 cut would be
    meaningless for a target with a 1% base rate -- almost nothing would ever be
    predicted positive.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    out: dict = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "base_rate": float(y_true.mean()) if len(y_true) else float("nan"),
    }
    if len(np.unique(y_true)) < 2:
        out["degenerate"] = True
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["log_loss"] = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)))
    out["ks"] = ks_statistic(y_true, y_prob)
    out["lift_top_10pct"] = lift_at_k(y_true, y_prob, 0.10)

    # Brier skill score against the base-rate-constant predictor: positive means
    # the model beats "always predict the base rate", which a raw Brier cannot
    # tell you when the base rate is tiny.
    base = out["base_rate"]
    ref = brier_score_loss(y_true, np.full_like(y_prob, base))
    out["brier_skill_score"] = float(1 - out["brier"] / ref) if ref > 0 else float("nan")

    if threshold is None:
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        f1s = np.divide(2 * prec[:-1] * rec[:-1], prec[:-1] + rec[:-1],
                        out=np.zeros_like(prec[:-1]), where=(prec[:-1] + rec[:-1]) > 0)
        threshold = float(thr[int(np.argmax(f1s))]) if len(thr) else 0.5
    out["threshold"] = float(threshold)

    y_hat = (y_prob >= threshold).astype(int)
    out["f1"] = float(f1_score(y_true, y_hat, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_hat, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_hat, zero_division=0))
    out["accuracy"] = float(accuracy_score(y_true, y_hat))
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    for floor in precision_floors:
        out[f"recall_at_precision_{int(floor * 100)}"] = recall_at_precision(
            y_true, y_prob, floor)
    return out


def multiclass_metrics(y_true, y_prob, labels: list[str]) -> dict:
    """Metrics for the next_state target (prd.md requires macro-F1)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_hat = np.asarray(labels)[np.argmax(y_prob, axis=1)]
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "macro_f1": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_hat, average="weighted", zero_division=0)),
        "labels": labels,
        "support": {l: int((y_true == l).sum()) for l in labels},
        "per_class_f1": {},
    }
    per = f1_score(y_true, y_hat, average=None, labels=labels, zero_division=0)
    out["per_class_f1"] = {l: float(v) for l, v in zip(labels, per)}
    try:
        present = [i for i, l in enumerate(labels) if (y_true == l).any()]
        if len(present) > 1:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(
                y_true, y_prob[:, present] / y_prob[:, present].sum(axis=1, keepdims=True),
                multi_class="ovr", average="macro",
                labels=[labels[i] for i in present]))
    except (ValueError, IndexError):
        pass
    out["confusion"] = confusion_matrix(y_true, y_hat, labels=labels).tolist()
    return out


def calibration_curve_data(y_true, y_prob, n_bins: int = 10,
                           strategy: str = "quantile") -> dict:
    """
    Reliability-curve points plus Expected Calibration Error.

    Quantile binning is the default: with a 1% base rate, uniform bins would put
    almost every observation in the first bin and report a meaningless curve.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(0, max(y_prob.max(), 1e-6), n_bins + 1)
    else:
        edges = np.linspace(0, 1, n_bins + 1)

    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, len(edges) - 2)
    rows, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        pred = float(y_prob[m].mean())
        obs = float(y_true[m].mean())
        w = float(m.mean())
        ece += w * abs(pred - obs)
        rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": pred,
                     "observed_rate": obs, "weight": w})
    return {"bins": rows, "ece": float(ece), "n_bins_used": len(rows),
            "strategy": strategy}


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.

    Used for the train-vs-test drift checks orchestrator.md Section 3.3 requires.
    Convention: <0.10 stable, 0.10-0.25 moderate shift, >0.25 significant shift.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    if len(e) < 10 or len(a) < 10:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(train: pd.DataFrame, other: pd.DataFrame,
                 features: list[str], n_bins: int = 10) -> pd.DataFrame:
    """Per-feature PSI between the training window and a later window."""
    rows = []
    for c in features:
        if c not in train.columns or c not in other.columns:
            continue
        v = psi(train[c].to_numpy(dtype="float64", na_value=np.nan),
                other[c].to_numpy(dtype="float64", na_value=np.nan), n_bins)
        rows.append({
            "feature": c,
            "psi": v,
            "severity": ("stable" if v < 0.10 else
                         "moderate" if v < 0.25 else "significant"),
            "train_mean": float(pd.to_numeric(train[c], errors="coerce").mean()),
            "other_mean": float(pd.to_numeric(other[c], errors="coerce").mean()),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


__all__ = [
    "binary_metrics", "multiclass_metrics", "calibration_curve_data",
    "recall_at_precision", "ks_statistic", "lift_at_k", "psi", "drift_report",
]
