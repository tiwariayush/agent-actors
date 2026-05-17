import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


class RayStub:
    def remote(self, **_remote_options):
        def decorate(actor_class):
            actor_class.remote = classmethod(
                lambda cls, *args, **kwargs: cls(*args, **kwargs)
            )
            return actor_class

        return decorate


def load_actors_module():
    module_name = "_agent_actors_actors_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    actors_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location(module_name, actors_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with mock.patch.dict(sys.modules, {"ray": RayStub()}):
        spec.loader.exec_module(module)
    return module


class RecordingChain:
    def run(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain_run(self):
        actors = load_actors_module()
        actor = actors.ChainActor.remote(RecordingChain())

        result = actor.run("task", attempt=1)

        self.assertEqual(result, {"args": ("task",), "kwargs": {"attempt": 1}})


if __name__ == "__main__":
    unittest.main()
