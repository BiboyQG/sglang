"""ML-trained per-iter latency predictor.

Loads a joblib pickle of a sklearn HistGradientBoostingRegressor and predicts forward
latency from batch composition features. Train one with `sglang-simulator-train`.

sim_config.json usage:
    "predictor": {
        "name": "ml",
        "database_path": "/path/to/latency_model.pkl"
    }
"""

import os

import joblib
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import InferTimePredictor, ScheduleBatch
from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)
from sglang_simulator.utils import get_logger
from sklearn.ensemble import HistGradientBoostingRegressor

logger = get_logger("sgl_simulator")


class MLTimePredictor(InferTimePredictor):
    """Predict forward latency using the shared 18-feature batch representation.

    The trees are walked from the model's node arrays instead of calling
    ``model.predict``: a simulation makes one prediction per scheduled batch, and
    ``predict`` on a one-row input spends milliseconds on input validation and on
    one predictor call per tree, each allocating an output array and entering an
    OpenMP region, while the walk itself takes microseconds.
    For the models the trainer produces (squared-error and quantile losses, whose
    link is the identity) the walk reproduces ``predict`` exactly: the same leaf
    values are summed in the same order. This ties the class to the node layout of
    the scikit-learn version the model was trained with.
    """

    # Feature order is shared by training and inference.
    FEATURE_NAMES = list(ML_FEATURE_NAMES)

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: str,
        latency_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, hw, config)
        database_path = os.path.expandvars(os.path.expanduser(database_path))
        if not database_path or not os.path.exists(database_path):
            raise FileNotFoundError(
                f"MLTimePredictor database_path not found: {database_path}. "
                "Train one with `sglang-simulator-train` first."
            )

        bundle = joblib.load(database_path)
        if (
            not isinstance(bundle, dict)
            or "model" not in bundle
            or "features" not in bundle
        ):
            raise ValueError(
                "MLTimePredictor requires a joblib bundle containing both "
                "'model' and ordered 'features' metadata"
            )
        model = bundle["model"]
        saved_features = list(bundle["features"])

        if saved_features != self.FEATURE_NAMES:
            raise ValueError(
                "MLTimePredictor feature contract mismatch: "
                f"saved={saved_features}, expected={self.FEATURE_NAMES}. "
                "Retrain or export the model with the exact 18-feature ABI."
            )
        if not isinstance(model, HistGradientBoostingRegressor):
            raise TypeError(
                "MLTimePredictor evaluates sklearn HistGradientBoostingRegressor "
                f"trees; the bundle holds {type(model).__name__}"
            )

        # Python lists index faster than numpy scalars in the per-node loop below.
        self._baseline = float(model._baseline_prediction[0, 0])
        self._trees = []
        for (tree,) in model._predictors:  # one tree per iteration for regression
            nodes = tree.nodes
            self._trees.append(
                (
                    nodes["is_leaf"].tolist(),
                    nodes["feature_idx"].tolist(),
                    nodes["num_threshold"].tolist(),
                    nodes["left"].tolist(),
                    nodes["right"].tolist(),
                    nodes["value"].tolist(),
                )
            )

        self._latency_scale = float(latency_scale)
        logger.info(
            "MLTimePredictor loaded from %s (model=%s, n_trees=%d, n_features=%d, latency_scale=%.4f)",
            database_path,
            type(model).__name__,
            len(self._trees),
            len(saved_features),
            self._latency_scale,
        )

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if batch.is_empty():
            return 0.0

        exts = [req.extend_length for req in batch.reqs]
        pasts = [req.past_kv_length for req in batch.reqs]

        feats = extract_ml_features(exts, pasts)

        # The features are finite, so the missing-value branch never applies.
        latency = self._baseline
        for is_leaf, feature_idx, threshold, left, right, value in self._trees:
            node = 0
            while not is_leaf[node]:
                if feats[feature_idx[node]] <= threshold[node]:
                    node = left[node]
                else:
                    node = right[node]
            latency += value[node]
        return latency * self._latency_scale
