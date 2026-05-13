import ray

from agent_actors.actors import ChainActor


class DummyChain:
    def run(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


def test_chain_actor_run_delegates_to_wrapped_chain():
    if not ray.is_initialized():
        ray.init(local_mode=True, include_dashboard=False, num_cpus=1)

    actor = ChainActor.remote(DummyChain())

    assert ray.get(actor.run.remote("task", option=True)) == {
        "args": ("task",),
        "kwargs": {"option": True},
    }
