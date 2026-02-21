from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np

@dataclass
class ModelArtifacts:
    features: List[str]
    coef: np.ndarray
    intercept: float
    calibrator: Optional[Dict[str, Any]]  # isotonic x/y
    meta: Dict[str, Any]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.coef + self.intercept
        z = np.clip(z, -50, 50)
        p = 1/(1+np.exp(-z))
        if self.calibrator and "x" in self.calibrator:
            # isotonic mapping
            x = np.asarray(self.calibrator["x"], dtype=float)
            y = np.asarray(self.calibrator["y"], dtype=float)
            p = np.interp(p, x, y, left=y[0], right=y[-1])
        return np.clip(p, 0.0, 1.0)

def default_artifacts(features: List[str]) -> ModelArtifacts:
    # conservative defaults
    coef = np.zeros(len(features), dtype=float)
    intercept = -3.0
    return ModelArtifacts(features=features, coef=coef, intercept=intercept, calibrator=None, meta={"trained": False, "note":"Default fallback weights"})

def save_artifacts(art: ModelArtifacts, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": art.features,
        "coef": art.coef.tolist(),
        "intercept": float(art.intercept),
        "calibrator": art.calibrator,
        "meta": art.meta,
    }
    (out_dir/"model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_artifacts(out_dir: Path) -> Optional[ModelArtifacts]:
    p = out_dir/"model.json"
    if not p.exists():
        return None
    payload=json.loads(p.read_text(encoding="utf-8"))
    return ModelArtifacts(
        features=payload["features"],
        coef=np.asarray(payload["coef"], dtype=float),
        intercept=float(payload["intercept"]),
        calibrator=payload.get("calibrator"),
        meta=payload.get("meta", {})
    )
