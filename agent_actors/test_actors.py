from agent_actors.actors import ChainActor


class DummyChain:
    def __init__(self):
        self.received = None

    def run(self, *args, **kwargs):
        self.received = (args, kwargs)
        return "chain-result"


def test_chain_actor_run_delegates_to_chain():
    actor = ChainActor.__ray_metadata__.modified_class(DummyChain())

    result = actor.run("task", flag=True)

    assert result == "chain-result"
    assert actor.chain.received == (("task",), {"flag": True})
