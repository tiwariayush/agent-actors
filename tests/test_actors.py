import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_actors_module():
    ray_stub = types.SimpleNamespace(
        remote=lambda **_decorator_kwargs: lambda decorated_class: decorated_class
    )
    sys.modules["ray"] = ray_stub

    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingChain:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "chain result"


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = load_actors_module()
        chain = RecordingChain()
        actor = actors.ChainActor(chain)

        result = actor.run("input", option=True)

        self.assertEqual(result, "chain result")
        self.assertEqual(chain.calls, [(("input",), {"option": True})])


if __name__ == "__main__":
    unittest.main()
