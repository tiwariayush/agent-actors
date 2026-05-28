import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_actors_module():
    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"

    previous_ray = sys.modules.get("ray")
    sys.modules["ray"] = types.SimpleNamespace(
        remote=lambda *args, **kwargs: (lambda cls: cls)
    )

    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_ray is None:
            del sys.modules["ray"]
        else:
            sys.modules["ray"] = previous_ray

    return module


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = load_actors_module()

        class RecordingChain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain result"

        chain = RecordingChain()
        actor = actors.ChainActor(chain)

        result = actor.run("task", retry=True)

        self.assertEqual(result, "chain result")
        self.assertEqual(chain.calls, [(("task",), {"retry": True})])


if __name__ == "__main__":
    unittest.main()
