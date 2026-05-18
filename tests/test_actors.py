import importlib.util
import sys
import types
import unittest
from pathlib import Path


class RayStub(types.SimpleNamespace):
    @staticmethod
    def remote(*decorator_args, **decorator_kwargs):
        def decorate(cls):
            return cls

        if len(decorator_args) == 1 and isinstance(decorator_args[0], type):
            return decorator_args[0]
        return decorate


def load_actors_module():
    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    previous_ray = sys.modules.get("ray")
    sys.modules["ray"] = RayStub()
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_ray is None:
            del sys.modules["ray"]
        else:
            sys.modules["ray"] = previous_ray
    return module


class ChainActorTest(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = load_actors_module()

        class FakeChain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain-result"

        chain = FakeChain()
        actor = actors.ChainActor(chain)

        self.assertEqual(actor.run("task", attempt=1), "chain-result")
        self.assertEqual(chain.calls, [(("task",), {"attempt": 1})])


if __name__ == "__main__":
    unittest.main()
