"""
Versioned model registry -- the local stand-in for the MLflow Model Registry.

orchestrator.md Section 5 requires a registry with a
`staging -> production -> archived` lifecycle where "each registered model
version links back to the exact feature_set_version and training data snapshot
used." backend.md Section 2.6 requires a `ModelRegistry` whose models are
"never loaded per-request" but cached in-process and keyed by version.

No MLflow server is available in this environment, so this module provides the
same contract over the local filesystem:

  models/<model_name>/<version>/
      model.joblib          fitted estimator
      calibrator.joblib     fitted probability calibrator (optional)
      metadata.json         metrics, features, lineage, stage
  models/registry.json      index: name -> {stage -> version}

Because the interface matches backend.md's `ModelRegistry.get_model(name, stage)`
signature, swapping MLflow back in is a change inside this one file -- no call
site moves. That was the point of the deviation being recorded in
docs/documentation_analysis.md Section 6.2 rather than hidden.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ..config import CFG, MODELS_DIR

REGISTRY_INDEX = MODELS_DIR / "registry.json"

STAGING = "staging"
PRODUCTION = "production"
ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Everything needed to reproduce and audit a model version."""
    model_name: str
    version: str
    algorithm: str
    target: str
    task: str                       # "binary" | "multiclass"
    stage: str = STAGING

    feature_set_version: str = CFG.feature_set_version
    config_version: str = CFG.config_version
    schema_version: str = CFG.schema_version
    random_seed: int = CFG.model.random_seed

    features: list[str] = field(default_factory=list)
    n_features: int = 0
    n_train_rows: int = 0
    train_window: list[str] = field(default_factory=list)
    valid_window: list[str] = field(default_factory=list)
    test_window: list[str] = field(default_factory=list)

    hyperparameters: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    class_balance: dict = field(default_factory=dict)

    data_snapshot: str = ""         # ingest summary hash / id
    created_at: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.n_features = len(self.features)


def _load_index() -> dict:
    if REGISTRY_INDEX.exists():
        return json.loads(REGISTRY_INDEX.read_text())
    return {"models": {}}


def _save_index(idx: dict) -> None:
    REGISTRY_INDEX.write_text(json.dumps(idx, indent=2))


def next_version(model_name: str) -> str:
    idx = _load_index()
    versions = idx["models"].get(model_name, {}).get("versions", {})
    n = 1
    while f"v{n}" in versions:
        n += 1
    return f"v{n}"


def model_dir(model_name: str, version: str) -> Path:
    return MODELS_DIR / model_name / version


def register(
    model: Any,
    metadata: ModelMetadata,
    calibrator: Any | None = None,
    explainer: Any | None = None,
) -> ModelMetadata:
    """Persist a fitted model plus its lineage; lands in `staging`."""
    d = model_dir(metadata.model_name, metadata.version)
    d.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, d / "model.joblib", compress=3)
    if calibrator is not None:
        joblib.dump(calibrator, d / "calibrator.joblib", compress=3)
    if explainer is not None:
        joblib.dump(explainer, d / "explainer.joblib", compress=3)
    (d / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2, default=str))

    idx = _load_index()
    entry = idx["models"].setdefault(metadata.model_name, {"versions": {}, "stages": {}})
    entry["versions"][metadata.version] = {
        "stage": metadata.stage,
        "algorithm": metadata.algorithm,
        "created_at": metadata.created_at,
        "target": metadata.target,
        "key_metric": metadata.metrics.get("test", {}).get("pr_auc")
        or metadata.metrics.get("validation", {}).get("pr_auc"),
    }
    entry["stages"].setdefault(STAGING, metadata.version)
    _save_index(idx)
    return metadata


def promote(model_name: str, version: str, stage: str = PRODUCTION,
            require_uplift: bool = True, min_uplift: float = 0.0) -> dict:
    """
    Move a version to a lifecycle stage.

    orchestrator.md Section 3.5: "a model is only promoted if it meets a
    configured minimum uplift threshold" versus the incumbent. Section 6.4 makes
    promotion a HUMAN approval gate, so this function is never called
    automatically by the pipeline -- it is invoked explicitly by
    `pipelines/promote_models.py`, which records who approved it.
    """
    idx = _load_index()
    entry = idx["models"].get(model_name)
    if not entry or version not in entry["versions"]:
        raise KeyError(f"unknown model version {model_name}:{version}")

    incumbent = entry["stages"].get(PRODUCTION)
    if (stage == PRODUCTION and require_uplift and incumbent
            and incumbent != version):
        new_m = _key_metric(model_name, version)
        old_m = _key_metric(model_name, incumbent)
        if new_m is not None and old_m is not None and new_m < old_m + min_uplift:
            return {
                "promoted": False,
                "reason": (f"uplift gate failed: candidate PR-AUC {new_m:.4f} does "
                           f"not exceed incumbent {old_m:.4f} by {min_uplift}"),
                "candidate": version, "incumbent": incumbent,
            }
        entry["versions"][incumbent]["stage"] = ARCHIVED
        entry["stages"][ARCHIVED] = incumbent

    entry["versions"][version]["stage"] = stage
    entry["stages"][stage] = version
    _save_index(idx)

    md_path = model_dir(model_name, version) / "metadata.json"
    if md_path.exists():
        md = json.loads(md_path.read_text())
        md["stage"] = stage
        md_path.write_text(json.dumps(md, indent=2))

    return {"promoted": True, "model": model_name, "version": version,
            "stage": stage, "previous_production": incumbent}


def _key_metric(model_name: str, version: str) -> float | None:
    p = model_dir(model_name, version) / "metadata.json"
    if not p.exists():
        return None
    md = json.loads(p.read_text())
    m = md.get("metrics", {})
    for split in ("test", "validation"):
        v = m.get(split, {}).get("pr_auc")
        if v is not None:
            return float(v)
    return None


def resolve_version(model_name: str, version: str = PRODUCTION) -> str:
    """Map a stage alias ('production') or an explicit version to a version id."""
    idx = _load_index()
    entry = idx["models"].get(model_name)
    if not entry:
        raise KeyError(f"model '{model_name}' is not registered")
    if version in (STAGING, PRODUCTION, ARCHIVED):
        v = entry["stages"].get(version)
        if not v:
            # Fall back to the newest version so a fresh install is still usable
            # rather than hard-failing before any promotion has happened.
            v = sorted(entry["versions"], key=lambda s: int(s[1:]))[-1]
        return v
    if version not in entry["versions"]:
        raise KeyError(f"unknown version {model_name}:{version}")
    return version


class ModelRegistry:
    """
    In-process cached loader, matching backend.md Section 2.6.

    Models and their SHAP explainers are loaded once and cached by
    (name, version) so `/predict` never pays a deserialisation cost per request.
    """
    _cache: dict[str, dict] = {}

    @classmethod
    def get(cls, model_name: str, version: str = PRODUCTION) -> dict:
        v = resolve_version(model_name, version)
        key = f"{model_name}:{v}"
        if key in cls._cache:
            return cls._cache[key]

        d = model_dir(model_name, v)
        if not (d / "model.joblib").exists():
            raise FileNotFoundError(f"no artifact for {key}")
        bundle = {
            "model": joblib.load(d / "model.joblib"),
            "calibrator": (joblib.load(d / "calibrator.joblib")
                           if (d / "calibrator.joblib").exists() else None),
            "explainer": (joblib.load(d / "explainer.joblib")
                          if (d / "explainer.joblib").exists() else None),
            "metadata": json.loads((d / "metadata.json").read_text()),
            "version": v,
        }
        cls._cache[key] = bundle
        return bundle

    @classmethod
    def get_model(cls, model_name: str, version: str = PRODUCTION):
        return cls.get(model_name, version)["model"]

    @classmethod
    def metadata(cls, model_name: str, version: str = PRODUCTION) -> dict:
        return cls.get(model_name, version)["metadata"]

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()

    @classmethod
    def available(cls) -> dict:
        return _load_index()["models"]


def list_models() -> list[dict]:
    out = []
    for name, entry in _load_index()["models"].items():
        for v, meta in entry["versions"].items():
            out.append({"model_name": name, "version": v, **meta})
    return sorted(out, key=lambda r: (r["model_name"], r["version"]))


def reset_registry() -> None:
    """Wipe the registry -- used by tests and by a clean full retrain."""
    for p in MODELS_DIR.iterdir():
        if p.is_dir() and p.name not in {"classification", "survival", "anomaly",
                                         "explainers"}:
            shutil.rmtree(p, ignore_errors=True)
    if REGISTRY_INDEX.exists():
        REGISTRY_INDEX.unlink()
    ModelRegistry.clear()


__all__ = [
    "ModelMetadata", "ModelRegistry", "register", "promote", "resolve_version",
    "next_version", "list_models", "reset_registry", "model_dir",
    "STAGING", "PRODUCTION", "ARCHIVED",
]
