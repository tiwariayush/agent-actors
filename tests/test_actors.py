import importlib.util
import sys
import types
import unittest
from pathlib import Path


class RayStub(types.ModuleType):
    def remote(self, *args, **kwargs):
        if args and len(args) == 1 and isinstance(args[0], type) and not kwargs:
            return args[0]

        def decorator(cls):
            return cls

        return decorator


class ChainActorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault("ray", RayStub("ray"))
        actors_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
        spec = importlib.util.spec_from_file_location("actors_under_test", actors_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.ChainActor = module.ChainActor

    def test_run_delegates_to_wrapped_chain(self):
        class Chain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        actor = self.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", limit=3),
            {"args": ("task",), "kwargs": {"limit": 3}},
        )


if __name__ == "__main__":
    unittest.main()
