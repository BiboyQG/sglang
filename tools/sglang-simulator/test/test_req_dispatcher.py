"""ReqDispatcher without an engine: the hold and release of requests with predecessors."""

from sglang_simulator.simulation.sglang.scheduler import ReqDispatcher, StateManager
from sglang_simulator.simulation.types import SimulationMode


class _SamplingParams:
    def __init__(self, simulation: dict):
        self.custom_params = {"simulation": simulation}
        self.max_new_tokens = 3


class TokenizedGenerateReqInput:  # the dispatcher recognizes generate requests by class name
    def __init__(self, rid: str, **simulation):
        self.rid = rid
        self.sampling_params = _SamplingParams(simulation)
        self.input_ids = [1] * 64


def _waits(*predecessors) -> list[dict]:
    return [{"session_id": s, "step": k, "gap_s": g} for s, k, g in predecessors]


def _created_time(req) -> float:
    return req.sampling_params.custom_params["simulation"]["created_time"]


def test_requests_are_released_when_all_predecessors_finished():
    dispatcher = ReqDispatcher(SimulationMode.OFFLINE)
    dispatcher.mode = (
        SimulationMode.OFFLINE
    )  # the singleton may have been created in another mode
    dispatcher.reset()
    # The graph of test_simulation_closed_loop.py: a chain a0 -> a1 -> a2, a root p0 that spawns
    # p/x (two steps) and p/y (one step), p1 waiting for p0 and both sub-sessions' last steps.
    specs = {
        "a0": dict(session_id="a", step=0, created_time=0.0),
        "a1": dict(
            session_id="a", step=1, created_time=0.0, wait_for=_waits(("a", 0, 0.05))
        ),
        "a2": dict(
            session_id="a", step=2, created_time=0.0, wait_for=_waits(("a", 1, 0.02))
        ),
        "open": dict(created_time=0.01),
        "p0": dict(
            session_id="p",
            step=0,
            created_time=0.04,
            after_finish={"finish_reason": "tool_calls"},
        ),
        "px0": dict(
            session_id="p/x", step=0, created_time=0.0, wait_for=_waits(("p", 0, 0.01))
        ),
        "px1": dict(
            session_id="p/x", step=1, created_time=0.0, wait_for=_waits(("p/x", 0, 0.0))
        ),
        "py0": dict(
            session_id="p/y", step=0, created_time=0.0, wait_for=_waits(("p", 0, 0.01))
        ),
        "p1": dict(
            session_id="p",
            step=1,
            created_time=0.0,
            wait_for=_waits(("p", 0, 0.0), ("p/x", 1, 0.02), ("p/y", 0, 0.2)),
        ),
    }
    reqs = {
        rid: TokenizedGenerateReqInput(rid, total_request=len(specs), **spec)
        for rid, spec in specs.items()
    }
    dispatcher.add(list(reqs.values()))
    assert dispatcher.offline_recv_all_requests
    assert [req.rid for _, _, req in sorted(dispatcher.future_queue)] == [
        "a0",
        "open",
        "p0",
    ]
    assert sorted(dispatcher.waiting) == [
        ("a", 1),
        ("a", 2),
        ("p", 1),
        ("p/x", 0),
        ("p/x", 1),
        ("p/y", 0),
    ]
    # Replay control never reaches the scheduler: only the key and the clock stay on the request.
    for req in reqs.values():
        assert set(req.sampling_params.custom_params["simulation"]) <= {
            "total_request",
            "created_time",
            "session_id",
            "step",
        }

    StateManager.set_global_clock(0.05)
    assert [req.rid for req in dispatcher.dispatch()] == ["a0", "open", "p0"]

    dispatcher.on_finished(reqs["a0"], 0.1)
    assert _created_time(reqs["a1"]) == 0.1 + 0.05
    assert ("a", 1) not in dispatcher.waiting and ("a", 2) in dispatcher.waiting
    dispatcher.on_finished(reqs["open"], 0.11)  # no key: nothing to release

    dispatcher.on_finished(reqs["p0"], 0.2)
    assert not hasattr(reqs["p0"], "after_finish")  # attached at KV release, not at finish
    dispatcher.attach_after_finish(reqs["p0"])
    assert reqs["p0"].after_finish == {"finish_reason": "tool_calls"}
    assert (
        _created_time(reqs["px0"]) == 0.2 + 0.01
        and _created_time(reqs["py0"]) == 0.2 + 0.01
    )
    assert ("p", 1) in dispatcher.waiting
    # Requests released by the same finish are dispatched in the order they were added (ties between
    # different finishes follow the order the finishes were processed in their batch).
    StateManager.set_global_clock(0.3)
    assert [req.rid for req in dispatcher.dispatch()] == ["a1", "px0", "py0"]

    dispatcher.on_finished(reqs["py0"], 0.31)
    dispatcher.on_finished(reqs["px0"], 0.32)
    assert _created_time(reqs["px1"]) == 0.32 + 0.0
    assert ("p", 1) in dispatcher.waiting  # p/x's last step has not finished
    dispatcher.on_finished(reqs["px1"], 0.4)
    assert ("p", 1) not in dispatcher.waiting
    # The latest (finish + gap) wins, here a predecessor that finished before the one that released it.
    assert (
        _created_time(reqs["p1"])
        == max(0.2 + 0.0, 0.4 + 0.02, 0.31 + 0.2)
        == 0.31 + 0.2
    )

    dispatcher.on_finished(reqs["a1"], 0.5)
    assert _created_time(reqs["a2"]) == 0.5 + 0.02
    StateManager.set_global_clock(1.0)
    # Released requests are dispatched by created_time.
    assert [req.rid for req in dispatcher.dispatch()] == ["px1", "p1", "a2"]
    assert (
        not dispatcher.waiting
        and not dispatcher.dependents
        and not dispatcher.after_finish
    )


def test_after_finish_is_on_the_request_when_its_kv_cache_is_released():
    """The hook wraps the batch result processor's release_kv_cache: the request carries its
    after_finish before the original release runs."""
    import importlib

    from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook

    batch_result_processor = importlib.import_module(
        "sglang.srt.managers.scheduler_components.batch_result_processor"
    )
    seen = []
    original = batch_result_processor.release_kv_cache

    def record_release(req, tree_cache, *, checkpoint):
        seen.append((req.rid, getattr(req, "after_finish", None), checkpoint))

    batch_result_processor.release_kv_cache = record_release

    class Target:  # the scheduler class the hook patches; only the attributes the hook reads
        def __init__(self):
            pass

        recv_requests = _prefetch_kvcache = get_new_batch_prefill = run_batch = None
        process_batch_result = event_loop_normal = init_request_dispatcher = None
        event_loop_overlap = None

    try:
        C_SchedulerHook.hook(Target)
        dispatcher = C_SchedulerHook.REQ_DISPATCHER
        dispatcher.mode = SimulationMode.OFFLINE
        dispatcher.reset()
        req = TokenizedGenerateReqInput(
            "p0",
            total_request=1,
            session_id="p",
            step=0,
            created_time=0.0,
            after_finish={"finish_reason": "stop", "tool_categories": []},
        )
        dispatcher.add([req])
        assert not hasattr(req, "after_finish")
        batch_result_processor.release_kv_cache(req, None, checkpoint=True)
        assert seen == [("p0", {"finish_reason": "stop", "tool_categories": []}, True)]
        assert not dispatcher.after_finish
    finally:
        batch_result_processor.release_kv_cache = original


def test_scheduler_clock_is_the_logical_clock_in_offline_mode():
    """After the hook in OFFLINE mode, mem_cache.common.scheduler_clock reads the simulator's
    global clock, whichever way a caller imported it."""
    import importlib

    from sglang.srt.mem_cache.common import scheduler_clock
    from sglang_simulator.simulation.sglang import scheduler as sim_scheduler

    common = importlib.import_module("sglang.srt.mem_cache.common")
    batch_result_processor = importlib.import_module(
        "sglang.srt.managers.scheduler_components.batch_result_processor"
    )
    original_clock = common._clock
    original_mode = sim_scheduler.C_SchedulerHook.SIM_MODE
    original_release = batch_result_processor.release_kv_cache

    class Target:  # the scheduler class the hook patches; only the attributes the hook reads
        def __init__(self):
            pass

        recv_requests = _prefetch_kvcache = get_new_batch_prefill = run_batch = None
        process_batch_result = event_loop_normal = init_request_dispatcher = None

    try:
        sim_scheduler.C_SchedulerHook.SIM_MODE = SimulationMode.OFFLINE
        sim_scheduler.C_SchedulerHook.hook(Target)
        StateManager.set_global_clock(12.5)
        assert scheduler_clock() == common.scheduler_clock() == 12.5
        StateManager.set_global_clock(13.0)
        assert scheduler_clock() == 13.0
    finally:
        common._clock = original_clock
        sim_scheduler.C_SchedulerHook.SIM_MODE = original_mode
        batch_result_processor.release_kv_cache = original_release
