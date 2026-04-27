import importlib.util
import sys
import types
from pathlib import Path


class RayStub:
    @staticmethod
    def remote(**_kwargs):
        def decorator(cls):
            return cls

        return decorator


class Runnable:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "run-result"

    def named(self, value):
        return f"named-{value}"


def load_actors_module():
    module_name = "actors_under_test"
    module_path = Path(__file__).with_name("actors.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    previous_ray = sys.modules.get("ray")
    sys.modules["ray"] = types.SimpleNamespace(remote=RayStub.remote)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_ray is None:
            del sys.modules["ray"]
        else:
            sys.modules["ray"] = previous_ray

    return module


def test_chain_actor_run_delegates_to_wrapped_chain():
    actors = load_actors_module()
    chain = Runnable()

    result = actors.ChainActor(chain).run("task", option=True)

    assert result == "run-result"
    assert chain.calls == [(("task",), {"option": True})]


def test_chain_actor_call_delegates_to_named_chain_method():
    actors = load_actors_module()

    assert actors.ChainActor(Runnable()).call("named", "value") == "named-value"
