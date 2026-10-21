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


def _dataset() -> SimpleDataset:
    # Staggered arrival times, in tied pairs, so that batches form differently from pass to pass
    # and the arrival tie-break is exercised.
    return SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=[1000 + i] * (64 + 16 * (i % 5)),
                input_length=64 + 16 * (i % 5),
                output_length=4 + (i % 3),
                custom_params={"created_time": 0.01 * (i // 2)},
            )
            for i in range(12)
        ]
    )


def _logical(record: dict) -> dict:
    # Everything in request.jsonl except the engine-generated request id is simulated state.
    return {k: v for k, v in record.items() if k != "rid"}


def test_offline_runs_repeat_exactly_with_fixed_cpu_overhead(tmp_path):
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
    config = BenchmarkConfig(ignore_request_timestamp=False)
    try:
        first_metrics = runner.benchmark(config, dataset=_dataset())
        first = [_logical(r) for r in runner.get_request_stats()]
        runner.flush_cache()
        second_metrics = runner.benchmark(config, dataset=_dataset())
        second = [_logical(r) for r in runner.get_request_stats()]
    finally:
        runner.shutdown()

    assert first_metrics["completed"] == 12 == second_metrics["completed"]
    assert first == second
    assert first_metrics["mean_ttft_ms"] == second_metrics["mean_ttft_ms"]
    assert first_metrics["mean_tpot_ms"] == second_metrics["mean_tpot_ms"]
