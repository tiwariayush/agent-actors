import importlib.util
import sys
from pathlib import Path


def load_actors_module(monkeypatch):
    class RayStub:
        @staticmethod
        def remote(**_kwargs):
            def decorate(cls):
                return cls

            return decorate

    monkeypatch.setitem(sys.modules, "ray", RayStub())
    module_path = Path(__file__).resolve().parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chain_actor_run_delegates_to_chain_run(monkeypatch):
    actors = load_actors_module(monkeypatch)

    class RecordingChain:
        def run(self, *args, **kwargs):
            return {"args": args, "kwargs": kwargs}

    result = actors.ChainActor(RecordingChain()).run("task", priority=3)

    assert result == {"args": ("task",), "kwargs": {"priority": 3}}


def test_chain_actor_call_delegates_to_chain_methods(monkeypatch):
    actors = load_actors_module(monkeypatch)

    class RecordingChain:
        def custom(self, value):
            return value * 2

    assert actors.ChainActor(RecordingChain()).call("custom", 21) == 42
