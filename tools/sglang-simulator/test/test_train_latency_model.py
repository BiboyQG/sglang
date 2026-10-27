"""Minimal example: train on collected steps, then load the models for prediction."""

import json
import math
import random
from pathlib import Path

import joblib
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest
from sglang_simulator.time_predictor.ml import MLTimePredictor
from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)
from sglang_simulator.tools.train_ml_time_predictor import train as trainer


def test_train_and_predict(tmp_path):
    example = Path(trainer.__file__).with_name("example_steps.jsonl")
    output = tmp_path / "models"
    trainer.main(["--data-root", str(example), "--out-dir", str(output)])

    summary = json.loads((output / "summary.json").read_text())
    assert summary["train_rows"] > 0 and summary["test_rows"] > 0
    assert summary["train_rows"] + summary["test_rows"] == summary["final_fit_rows"]
    assert set(summary["metrics"]) == {"l2", "p50"}

    batch = ScheduleBatch(
        [
            ScheduleRequest(extend_length=4, past_kv_length=12),
            ScheduleRequest(extend_length=8, past_kv_length=0),
        ]
    )
    for loss in ("l2", "p50"):
        model = output / f"latency_model_hgbmono_{loss}.pkl"
        predictor = MLTimePredictor(None, None, None, str(model))
        latency = predictor.predict_infer_time(batch)
        assert math.isfinite(latency) and latency > 0


def test_walk_matches_sklearn_predict(tmp_path):
    """The predictor walks the trees itself; it must match sklearn's predict exactly."""
    rng = random.Random(0)

    def batch():
        if rng.random() < 0.5:  # decode: one new token per request
            return [(1, rng.randint(0, 65536)) for _ in range(rng.randint(1, 16))]
        return [
            (rng.randint(1, 8192), rng.randint(0, 65536))
            for _ in range(rng.randint(1, 4))
        ]

    def features(reqs):
        return extract_ml_features([e for e, _ in reqs], [p for _, p in reqs])

    # The example steps alone give single-leaf trees; these rows give the trainer real splits.
    rows = []
    for _ in range(2000):
        feats = features(batch())
        rows.append((None, feats, 0.003 + 1e-6 * feats[1] + 2e-9 * feats[4]))
    model = trainer.train_model(rows, "l2", 0)
    assert sum(len(t[0].nodes) for t in model._predictors) > len(model._predictors)
    bundle = tmp_path / "walk.pkl"
    joblib.dump({"model": model, "features": list(ML_FEATURE_NAMES)}, bundle)
    predictor = MLTimePredictor(None, None, None, str(bundle))

    for _ in range(500):
        reqs = batch()
        expected = float(model.predict([features(reqs)])[0])
        got = predictor.predict_infer_time(
            ScheduleBatch([ScheduleRequest(e, p) for e, p in reqs])
        )
        assert got == expected
