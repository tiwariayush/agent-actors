import importlib.util
import sys
import types
import unittest
from pathlib import Path


class RemoteMethod:
    def __init__(self, method):
        self._method = method

    def remote(self, *args, **kwargs):
        return self._method(*args, **kwargs)


class ActorHandle:
    def __init__(self, instance):
        self._instance = instance

    def __getattr__(self, name):
        attr = getattr(self._instance, name)
        if callable(attr):
            return RemoteMethod(attr)
        return attr


class RemoteClass:
    def __init__(self, cls):
        self._cls = cls

    def remote(self, *args, **kwargs):
        return ActorHandle(self._cls(*args, **kwargs))


def fake_remote(*decorator_args, **_decorator_kwargs):
    if decorator_args and len(decorator_args) == 1 and isinstance(decorator_args[0], type):
        return RemoteClass(decorator_args[0])

    def decorate(cls):
        return RemoteClass(cls)

    return decorate


def load_actors_module_with_fake_ray():
    module_name = "_agent_actors_actors_for_test"
    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    original_ray = sys.modules.get("ray")
    fake_ray = types.SimpleNamespace(remote=fake_remote)
    sys.modules["ray"] = fake_ray
    sys.modules.pop(module_name, None)

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_ray is None:
            sys.modules.pop("ray", None)
        else:
            sys.modules["ray"] = original_ray


class AgentActorTest(unittest.TestCase):
    def test_run_remote_delegates_to_wrapped_agent_run(self):
        actors = load_actors_module_with_fake_ray()
        agent = RecordingRunner()

        result = actors.AgentActor.remote(agent).run.remote("task", flag=True)

        self.assertEqual(result, {"args": ("task",), "kwargs": {"flag": True}})
        self.assertEqual(agent.calls, [(("task",), {"flag": True})])


class ChainActorTest(unittest.TestCase):
    def test_run_remote_delegates_to_wrapped_chain_run(self):
        actors = load_actors_module_with_fake_ray()
        chain = RecordingRunner()

        result = actors.ChainActor.remote(chain).run.remote("task", flag=True)

        self.assertEqual(result, {"args": ("task",), "kwargs": {"flag": True}})
        self.assertEqual(chain.calls, [(("task",), {"flag": True})])


class RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"args": args, "kwargs": kwargs}


if __name__ == "__main__":
    unittest.main()
