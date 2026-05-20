import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_actors_module():
    def remote(**_remote_kwargs):
        def decorator(cls):
            return cls

        return decorator

    sys.modules["ray"] = types.SimpleNamespace(remote=remote)

    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeChain:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "chain result"


class ChainActorTest(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = load_actors_module()
        chain = FakeChain()

        result = actors.ChainActor(chain).run("task", option=True)

        self.assertEqual(result, "chain result")
        self.assertEqual(chain.calls, [(("task",), {"option": True})])


if __name__ == "__main__":
    unittest.main()
