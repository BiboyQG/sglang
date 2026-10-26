import json
import os
import sys
from pathlib import Path

import pytest
from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig

ASSETS = Path(__file__).parent / "assets"
SGLANG_ROOT = Path(__file__).parents[3]
if str(SGLANG_ROOT) not in sys.path:
    sys.path.insert(0, str(SGLANG_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _write_sim_config(tmp_path: Path) -> Path:
    table_path = tmp_path / "replay.json"
    table_path.write_text(json.dumps({"[[1, 64]]": 0.001, "[[65, 0]]": 0.01}))
    config = {
        "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80},
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": {
            "name": "replay",
            "database_path": str(table_path),
            "miss_strategy": "knn",
            "miss_knn_k": 1,
        },
        "scheduler": {"tp_size": 1, "ep_size": 1, "dp_size": 1, "cpu_overhead_s": 0.001},
    }
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def _request(seed: int, **custom_params) -> GenericRequest:
    length = 64 + 16 * (seed % 5)
    return GenericRequest(
        token_ids=[1000 + seed] * length,
        input_length=length,
        output_length=3 + (seed % 3),
        custom_params=custom_params,
    )


def _dataset() -> SimpleDataset:
    # Session a: three turns, the later ones think_time_s after the previous turn finished.
    # Session b: one turn. Two open-loop requests mixed in; the trace times of a's later turns
    # are deliberately in the past, which a closed loop must ignore.
    return SimpleDataset(
        reqs=[
            _request(0, created_time=0.0, session_id="a", turn=0),
            _request(1, created_time=0.0, session_id="a", turn=1, think_time_s=0.05),
            _request(2, created_time=0.0, session_id="a", turn=2, think_time_s=0.02),
            _request(3, created_time=0.03, session_id="b", turn=0),
            _request(4, created_time=0.01),
            _request(5, created_time=0.02),
        ]
    )


def _runner(tmp_path: Path):
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(_write_sim_config(tmp_path))

    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs

    return SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=str(ASSETS / "qwen3-8b"),
            load_format="dummy",
            device="cpu",
            max_total_tokens=1024,
            max_running_requests=4,
            page_size=16,
            skip_tokenizer_init=True,
        )
    )


def _logical(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "rid"}


def test_later_turns_arrive_think_time_after_the_previous_turn_finishes(tmp_path):
    # One engine for everything: a second engine in the same process keeps the interpreter alive at exit.
    runner = _runner(tmp_path)
    config = BenchmarkConfig(ignore_request_timestamp=False)
    try:
        metrics = runner.benchmark(config, dataset=_dataset())
        first = [_logical(r) for r in runner.get_request_stats()]
        runner.flush_cache()
        runner.benchmark(config, dataset=_dataset())
        second = [_logical(r) for r in runner.get_request_stats()]
        # A turn without its predecessor is rejected before anything is sent.
        orphan = SimpleDataset(reqs=[_request(0, created_time=0.0, session_id="c", turn=1, think_time_s=0.1)])
        with pytest.raises(ValueError, match="no turn 0"):
            runner.benchmark(config, dataset=orphan)
    finally:
        runner.shutdown()

    assert metrics["completed"] == 6
    by_turn = {(r["session_id"], r["turn"]): r for r in first if r["session_id"] is not None}
    assert sorted(by_turn) == [("a", 0), ("a", 1), ("a", 2), ("b", 0)]
    a0, a1, a2, b0 = (by_turn[k] for k in sorted(by_turn))
    assert a0["created_time"] == 0.0
    assert a1["created_time"] == pytest.approx(a0["last_event_time"] + 0.05)
    assert a2["created_time"] == pytest.approx(a1["last_event_time"] + 0.02)
    assert b0["created_time"] == pytest.approx(0.03)
    # A held turn enters the waiting queue at the first scheduling pass after it arrives: not before,
    # and at most one pass later (the longest pass with this table is a prefill batch, 0.01 s, plus
    # cpu_overhead_s). queue_start is that moment; admission into a batch (queue_end) comes later.
    for turn in (a1, a2):
        assert 0 <= turn["queue_start"] - turn["created_time"] <= 0.011 + 1e-9
    open_loop = sorted(r["created_time"] for r in first if r["session_id"] is None)
    assert open_loop == pytest.approx([0.01, 0.02])
    assert first == second
