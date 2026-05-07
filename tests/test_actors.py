import importlib.util
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


class _RayStub:
    def remote(self, *args, **kwargs):
        def decorate(cls):
            return cls

        if len(args) == 1 and not kwargs and isinstance(args[0], type):
            return args[0]
        return decorate


def _load_actors_module():
    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, {"ray": _RayStub()}):
        spec.loader.exec_module(module)

    return module


class ChainActorTests(TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = _load_actors_module()

        class Chain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        chain_actor = actors.ChainActor(Chain())

        self.assertEqual(
            chain_actor.run("task", priority=1),
            {"args": ("task",), "kwargs": {"priority": 1}},
        )
