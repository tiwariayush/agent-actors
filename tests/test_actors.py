import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_actors_module():
    ray = types.ModuleType("ray")

    def remote(**_options):
        return lambda cls: cls

    ray.remote = remote
    sys.modules["ray"] = ray

    actors_path = Path(__file__).parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", actors_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChainActorTest(unittest.TestCase):
    def test_run_delegates_to_chain(self):
        actors = _load_actors_module()

        class Chain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = actors.ChainActor(Chain())

        self.assertEqual(actor.run("input", mode="fast"), (("input",), {"mode": "fast"}))


if __name__ == "__main__":
    unittest.main()
