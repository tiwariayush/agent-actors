import ray

from agent_actors import ChainActor


class DummyChain:
    def run(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


def test_chain_actor_run_dispatches_to_wrapped_chain():
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(ignore_reinit_error=True, include_dashboard=False, local_mode=True)
    try:
        actor = ChainActor.remote(DummyChain())

        result = ray.get(actor.run.remote("task", force=True))

        assert result == {"args": ("task",), "kwargs": {"force": True}}
    finally:
        if started_ray:
            ray.shutdown()
