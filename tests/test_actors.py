import importlib.util
import sys
import unittest
from pathlib import Path

import ray


def load_actors_module():
    module_name = "_agent_actors_actors_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    actors_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location(module_name, actors_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RecordingChain:
    def run(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


class ChainActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=1)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_run_delegates_to_wrapped_chain_run(self):
        actors = load_actors_module()
        actor = actors.ChainActor.remote(RecordingChain())

        result = ray.get(actor.run.remote("task", attempt=1))

        self.assertEqual(result, {"args": ("task",), "kwargs": {"attempt": 1}})


if __name__ == "__main__":
    unittest.main()
