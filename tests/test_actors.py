import importlib.util
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


class _RayStub:
    def remote(self, *args, **kwargs):
        def decorate(actor_class):
            return actor_class

        if len(args) == 1 and not kwargs and isinstance(args[0], type):
            return args[0]
        return decorate


def load_actors_module():
    actors_path = Path(__file__).parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", actors_path)
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, {"ray": _RayStub()}):
        spec.loader.exec_module(module)

    return module


class ChainActorTests(TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = load_actors_module()

        class DummyChain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "ran"

        chain = DummyChain()
        actor = actors.ChainActor(chain)

        self.assertEqual(actor.run("task", flag=True), "ran")
        self.assertEqual(chain.calls, [(("task",), {"flag": True})])
