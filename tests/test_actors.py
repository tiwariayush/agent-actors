import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_actors_module():
    had_ray = "ray" in sys.modules
    original_ray = sys.modules.get("ray")
    ray = types.SimpleNamespace()

    def remote(*args, **kwargs):
        if args and isinstance(args[0], type):
            return args[0]

        def decorate(cls):
            return cls

        return decorate

    ray.remote = remote
    try:
        sys.modules["ray"] = ray
        module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
        spec = importlib.util.spec_from_file_location("agent_actors_actors", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if had_ray:
            sys.modules["ray"] = original_ray
        else:
            sys.modules.pop("ray", None)


class TestActors(unittest.TestCase):
    def test_chain_actor_run_delegates_to_chain(self):
        actors = load_actors_module()
        chain = RecordingRunnable("chain")

        result = actors.ChainActor(chain).run("task", mode="fast")

        self.assertEqual(result, ("chain", ("task",), {"mode": "fast"}))

    def test_agent_actor_run_delegates_to_agent(self):
        actors = load_actors_module()
        agent = RecordingRunnable("agent")

        result = actors.AgentActor(agent).run("task", mode="fast")

        self.assertEqual(result, ("agent", ("task",), {"mode": "fast"}))


class RecordingRunnable:
    def __init__(self, label):
        self.label = label

    def run(self, *args, **kwargs):
        return (self.label, args, kwargs)


if __name__ == "__main__":
    unittest.main()
