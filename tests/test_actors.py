import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_actors_module():
    ray_stub = types.ModuleType("ray")

    def remote(*_args, **_kwargs):
        def decorator(cls):
            return cls

        return decorator

    ray_stub.remote = remote
    sys.modules["ray"] = ray_stub

    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChainActorTest(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = load_actors_module()

        class DummyChain:
            def __init__(self):
                self.received = None

            def run(self, *args, **kwargs):
                self.received = (args, kwargs)
                return "chain result"

        chain = DummyChain()
        actor = actors.ChainActor(chain)

        self.assertEqual(
            actor.run("input", option=True),
            "chain result",
        )
        self.assertEqual(chain.received, (("input",), {"option": True}))


if __name__ == "__main__":
    unittest.main()
