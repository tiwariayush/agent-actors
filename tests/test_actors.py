import importlib.util
import pathlib
import sys
import types
import unittest


def load_actors_module():
    ray = types.ModuleType("ray")
    ray.remote = lambda *args, **kwargs: (lambda cls: cls) if kwargs else args[0]
    sys.modules["ray"] = ray

    actors_path = pathlib.Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", actors_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChainActorTest(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = load_actors_module()

        class Chain:
            def run(self, *args, **kwargs):
                return "ran", args, kwargs

        result = actors.ChainActor(Chain()).run("task", urgent=True)

        self.assertEqual(result, ("ran", ("task",), {"urgent": True}))


if __name__ == "__main__":
    unittest.main()
