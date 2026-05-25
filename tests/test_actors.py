import importlib.util
import sys
import types
import unittest
from pathlib import Path


class RayStub(types.ModuleType):
    def remote(self, *args, **kwargs):
        def decorate(cls):
            return cls

        return decorate


def load_actors_module():
    sys.modules["ray"] = RayStub("ray")
    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActorForwardingTest(unittest.TestCase):
    def test_chain_actor_run_forwards_to_wrapped_chain(self):
        actors = load_actors_module()

        class Chain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = actors.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", limit=3),
            (("task",), {"limit": 3}),
        )

    def test_agent_actor_run_forwards_to_wrapped_agent(self):
        actors = load_actors_module()

        class Agent:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = actors.AgentActor(Agent())

        self.assertEqual(
            actor.run("task", limit=3),
            (("task",), {"limit": 3}),
        )


if __name__ == "__main__":
    unittest.main()
