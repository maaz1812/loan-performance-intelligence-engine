"""
Time-aware training pipeline.

Implements orchestrator.md Section 3.4 and implementation.md Phase 4:

  * baselines      LogisticRegression, RandomForest      (the required
                   baseline-vs-improved comparison)
  * advanced       XGBoost, LightGBM                     (decision.md ADR-3)
  * five targets   4 binary + next_state multiclass
  * imbalance      scale_pos_weight / class_weight
  * calibration    isotonic or Platt, fitted on VALIDATION, never on train
  * registry       every fit registered with full lineage

The split is NOT computed here. It is materialised once, deterministically, by
ml/data_pipeline/build_model_dataset.py, so every model trains on byte-identical
inputs and the leakage guarantee lives in exactly one testable place
(decision.md ADR-2).

Calibration discipline
----------------------
The calibrator is fitted on the VALIDATION window and evaluated on TEST. Fitting
it on training predictions would be near-useless -- a boosted ensemble is
massively overconfident in-sample, so the resulting map would correct a
distortion that does not exist out of sample. Brier score is reported pre- and
post-calibration so the effect is visible rather than asserted.
"""
from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import CFG, REFERENCE_DIR, REPORTS_DIR, SPLITS_DIR
from ..data_pipeline.feature_engineering import feature_columns
from ..evaluation.metrics import (
    binary_metrics, calibration_curve_data, multiclass_metrics,
)
from ..models.registry import ModelMetadata, next_version, register

warnings.filterwarnings("ignore", category=UserWarning)

SEED = CFG.model.random_seed

BINARY_TARGETS = tuple(CFG.model.targets)
MULTICLASS_TARGET = CFG.model.multiclass_target

# Model name in the registry, per orchestrator.md Section 5's naming.
REGISTRY_NAME = {
    "next_3m_delinquency_flag": "delinquency_3m_model",
    "next_6m_delinquency_flag": "delinquency_6m_model",
    "next_12m_default_flag": "default_model",
    "next_12m_prepayment_flag": "prepayment_model",
    "next_state": "next_state_model",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _split_path(name: str) -> Path:
    return SPLITS_DIR / f"{name}.parquet"


def load_split(
    name: str,
    target: str | None = None,
    columns: list[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Load one materialised split, keeping only rows whose label is COMPLETE.

    dataset_usage.md Section 8.3 requires that rows whose forward-label window
    extends past the observable data are excluded rather than trained on as if
    the outcome were known. The `<target>_complete` mask built during feature
    engineering is what enforces that here.

    `max_rows` subsamples with a fixed seed. This is a ROW-level sample inside an
    already loan-contained, time-bounded split, so it cannot introduce loan or
    temporal leakage -- it only bounds memory.
    """
    p = _split_path(name)
    if not p.exists():
        raise FileNotFoundError(f"split not built: {p}")

    cols = None
    if columns is not None:
        import pyarrow.parquet as pq
        available = set(pq.ParquetFile(p).schema_arrow.names)
        need = set(columns)
        if target:
            need |= {target, f"{target}_complete"}
        cols = sorted(need & available)

    df = pd.read_parquet(p, columns=cols) if cols else pd.read_parquet(p)

    if target and f"{target}_complete" in df.columns:
        before = len(df)
        df = df[df[f"{target}_complete"] == 1]
        df.attrs["dropped_incomplete_labels"] = before - len(df)

    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=SEED)

    return df.reset_index(drop=True)


def prepare_xy(df: pd.DataFrame, target: str, features: list[str]):
    """Feature matrix and label vector. NaNs are left in place for the tree
    models (both XGBoost and LightGBM handle them natively and learn a default
    direction); the linear baseline imputes inside its own pipeline."""
    X = df[features].astype("float32")
    y = df[target].astype("int8").to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_models(n_pos: int, n_neg: int, task: str = "binary") -> dict:
    """
    Build the model zoo for one target.

    `scale_pos_weight` = negatives/positives is the standard imbalance
    correction for gradient boosting on rare events (decision.md ADR-3 cites
    built-in class weighting as a reason for choosing GBMs). Default and
    prepayment base rates here are low single digits, so without it the trees
    would simply predict the majority class everywhere.
    """
    spw = max(1.0, n_neg / max(n_pos, 1))

    if task == "multiclass":
        import lightgbm as lgb
        return {
            "logistic_regression": Pipeline([
                ("impute", _MedianImputer()),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=300, n_jobs=CFG.model.n_jobs,
                                           class_weight="balanced",
                                           random_state=SEED)),
            ]),
            "lightgbm": lgb.LGBMClassifier(
                objective="multiclass", n_estimators=300, learning_rate=0.08,
                num_leaves=48, max_depth=-1, min_child_samples=100,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=1.0, class_weight="balanced",
                random_state=SEED, n_jobs=CFG.model.n_jobs, verbose=-1),
        }

    import lightgbm as lgb
    import xgboost as xgb
    return {
        # --- baselines (required comparison) ------------------------------
        "logistic_regression": Pipeline([
            ("impute", _MedianImputer()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, n_jobs=CFG.model.n_jobs,
                                       class_weight="balanced",
                                       random_state=SEED)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=50,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=CFG.model.n_jobs, random_state=SEED),
        # --- advanced -----------------------------------------------------
        "xgboost": xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.06, max_depth=6,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.5, gamma=0.1, scale_pos_weight=spw,
            tree_method="hist", eval_metric="aucpr",
            n_jobs=CFG.model.n_jobs, random_state=SEED),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.06, num_leaves=48,
            max_depth=-1, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.5,
            scale_pos_weight=spw, random_state=SEED,
            n_jobs=CFG.model.n_jobs, verbose=-1),
    }


class _MedianImputer:
    """Minimal median imputer; avoids a sklearn version-dependent import."""

    def fit(self, X, y=None):
        self.medians_ = np.nanmedian(np.asarray(X, dtype="float64"), axis=0)
        self.medians_ = np.nan_to_num(self.medians_, nan=0.0)
        return self

    def transform(self, X):
        A = np.asarray(X, dtype="float32").copy()
        idx = np.where(np.isnan(A))
        if len(idx[0]):
            A[idx] = np.take(self.medians_, idx[1]).astype("float32")
        return A

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kw):
        return self


def predict_proba(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_calibrator(y_valid: np.ndarray, p_valid: np.ndarray,
                   method: str | None = None):
    """
    Fit a post-hoc probability calibrator on the VALIDATION window.

    Isotonic is the default: it is non-parametric and handles the sigmoid-shaped
    miscalibration that `scale_pos_weight` induces (up-weighting positives
    inflates predicted probabilities roughly monotonically, which isotonic can
    undo exactly while Platt cannot).
    """
    method = method or CFG.model.calibration_method
    if len(np.unique(y_valid)) < 2:
        return None
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_valid, y_valid)
        return iso
    lr = LogisticRegression(max_iter=1000)
    lr.fit(p_valid.reshape(-1, 1), y_valid)
    return lr


def apply_calibrator(cal, p: np.ndarray) -> np.ndarray:
    if cal is None:
        return p
    if isinstance(cal, IsotonicRegression):
        return np.clip(cal.predict(p), 0.0, 1.0)
    return cal.predict_proba(p.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Training one target
# ---------------------------------------------------------------------------

def train_target(
    target: str,
    features: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    windows: dict | None = None,
) -> dict:
    """Train every model in the zoo for one binary target and register each."""
    Xtr, ytr = prepare_xy(train_df, target, features)
    Xva, yva = prepare_xy(valid_df, target, features)
    Xte, yte = prepare_xy(test_df, target, features)

    n_pos, n_neg = int(ytr.sum()), int((1 - ytr).sum())
    zoo = make_models(n_pos, n_neg, "binary")
    results: dict = {
        "target": target,
        "n_train": len(ytr), "n_valid": len(yva), "n_test": len(yte),
        "train_base_rate": float(ytr.mean()),
        "valid_base_rate": float(yva.mean()),
        "test_base_rate": float(yte.mean()),
        "models": {},
    }

    print(f"\n  target={target}  train={len(ytr):,} (pos {n_pos:,}, "
          f"{ytr.mean()*100:.3f}%)  valid={len(yva):,}  test={len(yte):,}",
          flush=True)

    if n_pos < 50 or len(np.unique(ytr)) < 2:
        results["skipped"] = f"insufficient positives in training window ({n_pos})"
        print(f"    SKIPPED: {results['skipped']}", flush=True)
        return results

    for algo, model in zoo.items():
        t0 = time.time()
        try:
            model.fit(Xtr, ytr)
        except Exception as exc:  # noqa: BLE001
            results["models"][algo] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"    [{algo}] FAILED: {exc}", flush=True)
            continue
        fit_s = time.time() - t0

        p_va_raw = predict_proba(model, Xva)
        p_te_raw = predict_proba(model, Xte)

        cal = fit_calibrator(yva, p_va_raw)
        p_te_cal = apply_calibrator(cal, p_te_raw)
        p_va_cal = apply_calibrator(cal, p_va_raw)

        m_va = binary_metrics(yva, p_va_cal)
        m_te = binary_metrics(yte, p_te_cal)
        m_te_raw = binary_metrics(yte, p_te_raw)

        entry = {
            "algorithm": algo,
            "fit_seconds": round(fit_s, 1),
            "validation": m_va,
            "test": m_te,
            "test_uncalibrated": {"brier": m_te_raw.get("brier"),
                                  "pr_auc": m_te_raw.get("pr_auc"),
                                  "roc_auc": m_te_raw.get("roc_auc")},
            "calibration": {
                "method": CFG.model.calibration_method,
                "fitted_on": "validation",
                "brier_before": m_te_raw.get("brier"),
                "brier_after": m_te.get("brier"),
                "curve": calibration_curve_data(yte, p_te_cal),
            },
        }

        if holdout_df is not None and len(holdout_df):
            Xho, yho = prepare_xy(holdout_df, target, features)
            entry["holdout_2022"] = binary_metrics(
                yho, apply_calibrator(cal, predict_proba(model, Xho)))
            del Xho

        md = ModelMetadata(
            model_name=REGISTRY_NAME[target],
            version=next_version(REGISTRY_NAME[target]),
            algorithm=algo,
            target=target,
            task="binary",
            features=features,
            n_train_rows=len(ytr),
            hyperparameters=_safe_params(model),
            metrics={"validation": _slim(m_va), "test": _slim(m_te)},
            calibration={k: v for k, v in entry["calibration"].items() if k != "curve"},
            class_balance={"n_pos": n_pos, "n_neg": n_neg,
                           "base_rate": float(ytr.mean())},
            train_window=(windows or {}).get("train", []),
            valid_window=(windows or {}).get("validation", []),
            test_window=(windows or {}).get("test", []),
            data_snapshot=(windows or {}).get("snapshot", ""),
        )
        register(model, md, calibrator=cal)
        entry["registry"] = {"model_name": md.model_name, "version": md.version}

        results["models"][algo] = entry
        print(f"    [{algo:20s}] PR-AUC {m_te.get('pr_auc', float('nan')):.4f}  "
              f"ROC-AUC {m_te.get('roc_auc', float('nan')):.4f}  "
              f"Brier {m_te.get('brier', float('nan')):.5f}"
              f" (raw {m_te_raw.get('brier', float('nan')):.5f})  "
              f"{fit_s:.0f}s  -> {md.version}", flush=True)

        del p_va_raw, p_te_raw, p_te_cal, p_va_cal
        gc.collect()

    # best-by-PR-AUC on test, used for the baseline-vs-improved comparison
    scored = {a: e["test"].get("pr_auc") for a, e in results["models"].items()
              if "test" in e and e["test"].get("pr_auc") is not None}
    if scored:
        best = max(scored, key=scored.get)
        results["best_model"] = best
        results["best_pr_auc"] = scored[best]
        baselines = {a: v for a, v in scored.items()
                     if a in ("logistic_regression", "random_forest")}
        if baselines:
            bb = max(baselines, key=baselines.get)
            results["baseline_best"] = bb
            results["baseline_pr_auc"] = baselines[bb]
            results["uplift_vs_baseline"] = scored[best] - baselines[bb]
            results["uplift_pct"] = (
                (scored[best] - baselines[bb]) / baselines[bb] * 100
                if baselines[bb] else float("nan"))

    del Xtr, Xva, Xte
    gc.collect()
    return results


def train_next_state(
    features: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    windows: dict | None = None,
) -> dict:
    """Multi-class next-state model (dataset_usage.md Section 7.4)."""
    tgt = MULTICLASS_TARGET
    labels = [s for s in CFG.model.loan_states
              if (train_df[tgt] == s).any()]

    Xtr = train_df[features].astype("float32")
    ytr = train_df[tgt].astype("string").to_numpy()
    Xva = valid_df[features].astype("float32")
    yva = valid_df[tgt].astype("string").to_numpy()
    Xte = test_df[features].astype("float32")
    yte = test_df[tgt].astype("string").to_numpy()

    keep = np.isin(ytr, labels)
    Xtr, ytr = Xtr[keep], ytr[keep]

    results = {"target": tgt, "labels": labels, "n_train": int(len(ytr)),
               "models": {}}
    print(f"\n  target={tgt}  train={len(ytr):,}  classes={labels}", flush=True)

    for algo, model in make_models(1, 1, "multiclass").items():
        t0 = time.time()
        try:
            model.fit(Xtr, ytr)
        except Exception as exc:  # noqa: BLE001
            results["models"][algo] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"    [{algo}] FAILED: {exc}", flush=True)
            continue
        classes = list(model.classes_) if hasattr(model, "classes_") else labels
        m_va = multiclass_metrics(yva, model.predict_proba(Xva), classes)
        m_te = multiclass_metrics(yte, model.predict_proba(Xte), classes)

        md = ModelMetadata(
            model_name=REGISTRY_NAME[tgt], version=next_version(REGISTRY_NAME[tgt]),
            algorithm=algo, target=tgt, task="multiclass", features=features,
            n_train_rows=int(len(ytr)), hyperparameters=_safe_params(model),
            metrics={"validation": {"macro_f1": m_va["macro_f1"],
                                    "accuracy": m_va["accuracy"]},
                     "test": {"macro_f1": m_te["macro_f1"],
                              "accuracy": m_te["accuracy"]}},
            train_window=(windows or {}).get("train", []),
            valid_window=(windows or {}).get("validation", []),
            test_window=(windows or {}).get("test", []),
        )
        register(model, md)
        results["models"][algo] = {
            "algorithm": algo, "fit_seconds": round(time.time() - t0, 1),
            "validation": m_va, "test": m_te,
            "registry": {"model_name": md.model_name, "version": md.version},
        }
        print(f"    [{algo:20s}] macro-F1 {m_te['macro_f1']:.4f}  "
              f"acc {m_te['accuracy']:.4f}  -> {md.version}", flush=True)

    scored = {a: e["test"]["macro_f1"] for a, e in results["models"].items()
              if "test" in e}
    if scored:
        results["best_model"] = max(scored, key=scored.get)
        results["best_macro_f1"] = scored[results["best_model"]]

    del Xtr, Xva, Xte
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_params(model) -> dict:
    try:
        p = model.get_params(deep=False)
        return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in p.items()}
    except Exception:  # noqa: BLE001
        return {}


_SLIM_KEYS = ("roc_auc", "pr_auc", "f1", "precision", "recall", "brier",
              "brier_skill_score", "ks", "lift_top_10pct", "base_rate", "n",
              "n_positive", "threshold", "recall_at_precision_50")


def _slim(m: dict) -> dict:
    return {k: m[k] for k in _SLIM_KEYS if k in m}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all(max_train_rows: int | None = None,
            targets: tuple[str, ...] | None = None) -> dict:
    """Train every model for every target and write the results bundle."""
    max_train_rows = max_train_rows or CFG.model.max_train_rows
    targets = targets or BINARY_TARGETS

    build = json.loads((REFERENCE_DIR / "feature_build_summary.json").read_text())
    windows = {
        "train": ["2018-01-01", CFG.split.train_end],
        "validation": [CFG.split.valid_start, CFG.split.valid_end],
        "test": build["test_window"],
        "snapshot": build.get("data_end", ""),
    }

    probe = pd.read_parquet(_split_path("train"))
    features = feature_columns(probe)
    del probe
    gc.collect()
    print(f"features: {len(features)}", flush=True)

    out: dict = {
        "features": features, "n_features": len(features),
        "windows": windows, "seed": SEED,
        "max_train_rows": max_train_rows,
        "config_version": CFG.config_version,
        "feature_set_version": CFG.feature_set_version,
        "targets": {},
    }

    for tgt in targets:
        tr = load_split("train", tgt, max_rows=max_train_rows)
        va = load_split("validation", tgt, max_rows=max_train_rows // 2)
        te = load_split("test", tgt, max_rows=max_train_rows // 2)
        ho = None
        if _split_path("holdout_2022").exists():
            ho = load_split("holdout_2022", tgt, max_rows=max_train_rows // 4)
        out["targets"][tgt] = train_target(tgt, features, tr, va, te, ho, windows)
        del tr, va, te, ho
        gc.collect()

    # next_state
    tr = load_split("train", MULTICLASS_TARGET, max_rows=max_train_rows)
    va = load_split("validation", MULTICLASS_TARGET, max_rows=max_train_rows // 2)
    te = load_split("test", MULTICLASS_TARGET, max_rows=max_train_rows // 2)
    out["targets"][MULTICLASS_TARGET] = train_next_state(features, tr, va, te, windows)
    del tr, va, te
    gc.collect()

    (REFERENCE_DIR / "training_results.json").write_text(
        json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    a = ap.parse_args()
    r = run_all(a.max_train_rows, tuple(a.targets) if a.targets else None)
    print("\n=== summary ===")
    for t, res in r["targets"].items():
        if "best_model" in res:
            k = "best_pr_auc" if "best_pr_auc" in res else "best_macro_f1"
            print(f"{t:32s} best={res['best_model']:20s} {k}={res[k]:.4f}"
                  + (f"  uplift_vs_baseline={res.get('uplift_vs_baseline', 0):+.4f}"
                     if "uplift_vs_baseline" in res else ""))
