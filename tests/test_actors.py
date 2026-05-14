import importlib.util
from pathlib import Path

import ray


def load_chain_actor():
    actor_module = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("agent_actors_actors", actor_module)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ChainActor


ChainActor = load_chain_actor()


class DummyChain:
    def run(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


def test_chain_actor_run_dispatches_to_wrapped_chain():
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        actor = ChainActor.remote(DummyChain())

        result = ray.get(actor.run.remote("task", force=True))

        assert result == {"args": ("task",), "kwargs": {"force": True}}
    finally:
        if started_ray:
            ray.shutdown()
