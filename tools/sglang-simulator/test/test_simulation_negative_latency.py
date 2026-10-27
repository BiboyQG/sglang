"""A predictor that returns a latency below zero must not move the OFFLINE clock backward."""

import json
import os
import sys
from pathlib import Path

from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig

ASSETS = Path(__file__).parent / "assets"
SGLANG_ROOT = Path(__file__).parents[3]
if str(SGLANG_ROOT) not in sys.path:
    sys.path.insert(0, str(SGLANG_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

CPU_OVERHEAD_S = 0.001


def _write_sim_config(tmp_path: Path) -> Path:
    # Decode batches "take" a negative time: the clock may only advance by cpu_overhead_s then.
    table_path = tmp_path / "replay.json"
    table_path.write_text(json.dumps({"[[1, 64]]": -0.004, "[[65, 0]]": 0.01}))
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
        "scheduler": {
            "tp_size": 1,
            "ep_size": 1,
            "dp_size": 1,
            "cpu_overhead_s": CPU_OVERHEAD_S,
        },
    }
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(json.dumps(config))
    return config_path


def _request(seed: int, **custom_params) -> GenericRequest:
    length = 64 + 16 * (seed % 5)
    return GenericRequest(
        token_ids=[1000 + seed] * length,
        input_length=length,
        output_length=8 + seed,
        custom_params=custom_params,
    )


def test_clock_never_runs_backward(tmp_path):
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(_write_sim_config(tmp_path))

    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs

    runner = SGLangBenchmarkRunner(
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
    # Two chained sessions and two open-loop requests; every decode pass predicts -4 ms.
    dataset = SimpleDataset(
        reqs=[
            _request(0, created_time=0.0, session_id="a", step=0),
            _request(
                1,
                created_time=0.0,
                session_id="a",
                step=1,
                wait_for=[{"session_id": "a", "step": 0, "gap_s": 0.01}],
            ),
            _request(2, created_time=0.0, session_id="b", step=0),
            _request(
                3,
                created_time=0.0,
                session_id="b",
                step=1,
                wait_for=[{"session_id": "b", "step": 0, "gap_s": 0.0}],
            ),
            _request(4, created_time=0.005),
            _request(5, created_time=0.006),
        ]
    )
    try:
        metrics = runner.benchmark(
            BenchmarkConfig(ignore_request_timestamp=False), dataset=dataset
        )
        records = runner.get_request_stats()
    finally:
        runner.shutdown()

    assert metrics["completed"] == 6
    for r in records:
        assert len(r["gen_token_latencies"]) == r["output_length"]
        assert all(
            latency >= CPU_OVERHEAD_S - 1e-12 for latency in r["gen_token_latencies"]
        )
        assert r["last_event_time"] >= r["created_time"] + r["gen_token_latencies"][0]
    by_key = {
        (r["session_id"], r["step"]): r for r in records if r["session_id"] is not None
    }
    assert (
        by_key[("a", 1)]["created_time"] >= by_key[("a", 0)]["last_event_time"] + 0.01
    )
    assert by_key[("b", 1)]["created_time"] >= by_key[("b", 0)]["last_event_time"]
