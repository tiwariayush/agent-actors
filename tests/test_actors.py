import importlib.util
import sys
import unittest
from pathlib import Path


class RayStub:
    def remote(self, *args, **kwargs):
        def decorate(cls):
            return cls

        return decorate


def load_actors_module():
    sys.modules["ray"] = RayStub()
    actors_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", actors_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyChain:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "chain result"


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = load_actors_module()
        chain = DummyChain()

        result = actors.ChainActor(chain).run("task", option=True)

        self.assertEqual(result, "chain result")
        self.assertEqual(chain.calls, [(("task",), {"option": True})])


if __name__ == "__main__":
    unittest.main()
