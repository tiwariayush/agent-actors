import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_actors_module():
    module_name = "actors_under_test"
    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"

    ray_stub = types.SimpleNamespace(
        remote=lambda **_: (lambda cls: cls),
    )
    previous_ray = sys.modules.get("ray")
    sys.modules["ray"] = ray_stub

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_ray is None:
            sys.modules.pop("ray", None)
        else:
            sys.modules["ray"] = previous_ray


class StubChain:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "chain-result"


class ChainActorTest(unittest.TestCase):
    def test_run_dispatches_to_wrapped_chain(self):
        actors = load_actors_module()
        chain = StubChain()

        result = actors.ChainActor(chain).run("input", option=True)

        self.assertEqual(result, "chain-result")
        self.assertEqual(chain.calls, [(("input",), {"option": True})])


if __name__ == "__main__":
    unittest.main()
