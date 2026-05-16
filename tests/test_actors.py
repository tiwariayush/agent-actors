import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def load_actors_module():
    class RayStub:
        @staticmethod
        def remote(**_kwargs):
            def decorate(cls):
                return cls

            return decorate

    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"ray": RayStub()}):
        spec.loader.exec_module(module)
    return module


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_chain_run(self):
        actors = load_actors_module()

        class RecordingChain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        result = actors.ChainActor(RecordingChain()).run("task", priority=3)

        self.assertEqual(result, {"args": ("task",), "kwargs": {"priority": 3}})

    def test_call_delegates_to_chain_methods(self):
        actors = load_actors_module()

        class RecordingChain:
            def custom(self, value):
                return value * 2

        self.assertEqual(actors.ChainActor(RecordingChain()).call("custom", 21), 42)


if __name__ == "__main__":
    unittest.main()
