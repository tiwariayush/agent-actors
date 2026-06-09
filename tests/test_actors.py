import importlib.util
import pathlib
import sys
import types
import unittest


class RayStub(types.SimpleNamespace):
    def remote(self, *args, **kwargs):
        def decorator(cls):
            return cls

        return decorator


def load_actors_module():
    sys.modules["ray"] = RayStub()
    module_path = pathlib.Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyChain:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append(("run", args, kwargs))
        return "chain-run-result"

    def custom(self, *args, **kwargs):
        self.calls.append(("custom", args, kwargs))
        return "chain-custom-result"


class ChainActorTests(unittest.TestCase):
    def setUp(self):
        self.actors = load_actors_module()

    def test_run_delegates_to_wrapped_chain_run(self):
        chain = DummyChain()
        actor = self.actors.ChainActor(chain)

        result = actor.run("task", force=True)

        self.assertEqual(result, "chain-run-result")
        self.assertEqual(chain.calls, [("run", ("task",), {"force": True})])

    def test_call_delegates_to_named_chain_method(self):
        chain = DummyChain()
        actor = self.actors.ChainActor(chain)

        result = actor.call("custom", "payload", attempt=2)

        self.assertEqual(result, "chain-custom-result")
        self.assertEqual(chain.calls, [("custom", ("payload",), {"attempt": 2})])


if __name__ == "__main__":
    unittest.main()
