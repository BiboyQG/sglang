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
        "scheduler": {
            "tp_size": 1,
            "ep_size": 1,
            "dp_size": 1,
            "cpu_overhead_s": 0.001,
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
        output_length=3 + (seed % 3),
        custom_params=custom_params,
    )


def _waits(*predecessors) -> list[dict]:
    return [{"session_id": s, "step": k, "gap_s": g} for s, k, g in predecessors]


def _dataset() -> SimpleDataset:
    # Session a: a chain of three steps, each released gap_s after the previous one finished.
    # Session p: a root step that spawns two sub-sessions (p/x with two steps, p/y with one); p's
    # next step waits for its own root and for the last step of each sub-session; p/y's gap is the
    # large one, so the latest (finish + gap) comes from a predecessor that finished earlier than
    # the one whose finish released the step. Session b: one root step. Two open-loop requests mixed in. The trace times of the waiting steps are
    # deliberately in the past, which a closed loop must ignore.
    return SimpleDataset(
        reqs=[
            _request(0, created_time=0.0, session_id="a", step=0, trigger="root"),
            _request(
                1,
                created_time=0.0,
                session_id="a",
                step=1,
                trigger="user",
                wait_for=_waits(("a", 0, 0.05)),
            ),
            _request(
                2,
                created_time=0.0,
                session_id="a",
                step=2,
                trigger="tool",
                wait_for=_waits(("a", 1, 0.02)),
            ),
            _request(3, created_time=0.03, session_id="b", step=0, trigger="root"),
            _request(4, created_time=0.01),
            _request(5, created_time=0.02),
            _request(
                6,
                created_time=0.04,
                session_id="p",
                step=0,
                trigger="root",
                after_finish={
                    "finish_reason": "tool_calls",
                    "tool_categories": ["subagent"],
                },
            ),
            _request(
                7,
                created_time=0.0,
                session_id="p/x",
                step=0,
                trigger="subagent",
                wait_for=_waits(("p", 0, 0.01)),
            ),
            _request(
                8,
                created_time=0.0,
                session_id="p/x",
                step=1,
                trigger="tool",
                wait_for=_waits(("p/x", 0, 0.0)),
            ),
            _request(
                9,
                created_time=0.0,
                session_id="p/y",
                step=0,
                trigger="subagent",
                wait_for=_waits(("p", 0, 0.03)),
            ),
            _request(
                10,
                created_time=0.0,
                session_id="p",
                step=1,
                trigger="tool",
                wait_for=_waits(("p", 0, 0.0), ("p/x", 1, 0.02), ("p/y", 0, 0.2)),
            ),
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


def test_waiting_steps_arrive_after_their_predecessors_finish(tmp_path):
    # One engine for everything: a second engine in the same process keeps the interpreter alive at exit.
    runner = _runner(tmp_path)
    config = BenchmarkConfig(ignore_request_timestamp=False)
    try:
        metrics = runner.benchmark(config, dataset=_dataset())
        first = [_logical(r) for r in runner.get_request_stats()]
        runner.flush_cache()
        runner.benchmark(config, dataset=_dataset())
        second = [_logical(r) for r in runner.get_request_stats()]
        # Rejected before anything is sent: a predecessor that is not in the dataset, a cycle,
        # a wait without a key.
        rejected = [
            (
                [
                    _request(
                        0,
                        created_time=0.0,
                        session_id="c",
                        step=1,
                        wait_for=_waits(("c", 0, 0.1)),
                    )
                ],
                "not in the dataset",
            ),
            (
                [
                    _request(
                        0,
                        created_time=0.0,
                        session_id="c",
                        step=0,
                        wait_for=_waits(("c", 1, 0.0)),
                    ),
                    _request(
                        1,
                        created_time=0.0,
                        session_id="c",
                        step=1,
                        wait_for=_waits(("c", 0, 0.0)),
                    ),
                ],
                "cycle",
            ),
            (
                [_request(0, created_time=0.0, wait_for=_waits(("c", 0, 0.0)))],
                "no session_id",
            ),
        ]
        for reqs, message in rejected:
            with pytest.raises(ValueError, match=message):
                runner.benchmark(config, dataset=SimpleDataset(reqs=reqs))
    finally:
        runner.shutdown()

    assert metrics["completed"] == 11
    by_key = {
        (r["session_id"], r["step"]): r for r in first if r["session_id"] is not None
    }
    assert sorted(by_key) == [
        ("a", 0),
        ("a", 1),
        ("a", 2),
        ("b", 0),
        ("p", 0),
        ("p", 1),
        ("p/x", 0),
        ("p/x", 1),
        ("p/y", 0),
    ]
    a0, a1, a2, b0, p0, p1, px0, px1, py0 = (by_key[k] for k in sorted(by_key))
    # A chain: each step arrives gap_s after the previous one finished, whatever its trace time.
    assert a0["created_time"] == 0.0
    assert a1["created_time"] == pytest.approx(a0["last_event_time"] + 0.05)
    assert a2["created_time"] == pytest.approx(a1["last_event_time"] + 0.02)
    assert b0["created_time"] == pytest.approx(0.03)
    # Fan-out and join: the sub-sessions start after p's root step; p's next step waits for its root
    # and for the last step of each sub-session, and arrives at the latest (finish + gap) of the three.
    assert px0["created_time"] == pytest.approx(p0["last_event_time"] + 0.01)
    assert px1["created_time"] == pytest.approx(px0["last_event_time"])
    assert py0["created_time"] == pytest.approx(p0["last_event_time"] + 0.03)
    assert p1["created_time"] == pytest.approx(
        max(
            p0["last_event_time"],
            px1["last_event_time"] + 0.02,
            py0["last_event_time"] + 0.2,
        )
    )
    assert p1["created_time"] == pytest.approx(py0["last_event_time"] + 0.2)
    assert py0["last_event_time"] < px1["last_event_time"]
    assert [r["trigger"] for r in (a0, a1, a2, p1, px0)] == [
        "root",
        "user",
        "tool",
        "tool",
        "subagent",
    ]
    # A waiting step enters the waiting queue at the first scheduling pass after it arrives: not before,
    # and at most one pass later (the longest pass with this table is a prefill batch, 0.01 s, plus
    # cpu_overhead_s). queue_start is that moment; admission into a batch (queue_end) comes later.
    for step in (a1, a2, px0, px1, py0, p1):
        assert 0 <= step["queue_start"] - step["created_time"] <= 0.011 + 1e-9
    open_loop = sorted(r["created_time"] for r in first if r["session_id"] is None)
    assert open_loop == pytest.approx([0.01, 0.02])
    assert first == second
