#!/usr/bin/env python3
"""Rebuild the small ML and tokenizer assets used by examples and tests."""

from pathlib import Path

import joblib
from sglang_simulator.time_predictor.ml_features import ML_FEATURE_NAMES
from sglang_simulator.tools.train_ml_time_predictor import train as trainer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

ASSETS = Path(__file__).parent / "assets"


def build_ml_model() -> None:
    """The trainer's own model type on its example steps; illustrative, not calibrated."""
    example = Path(trainer.__file__).with_name("example_steps.jsonl")
    rows, _ = trainer.load_rows([example], {1}, 30.0)
    joblib.dump(
        {
            "model": trainer.train_model(rows, "l2", 0),
            "features": list(ML_FEATURE_NAMES),
        },
        ASSETS / "model.pkl",
    )


def build_tokenizer() -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "prefix": 1,
                "caching": 2,
                "latency": 3,
                "decode": 4,
                "token": 5,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
    ).save_pretrained(ASSETS / "tokenizer")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_ml_model()
    build_tokenizer()


if __name__ == "__main__":
    main()
