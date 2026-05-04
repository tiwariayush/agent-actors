import importlib.util
import sys
import types
from pathlib import Path


def load_actors_module(monkeypatch):
    fake_ray = types.SimpleNamespace(
        remote=lambda **_kwargs: lambda actor_class: actor_class
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    actors_path = Path(__file__).parents[1] / "agent_actors" / "actors.py"
    spec = importlib.util.spec_from_file_location("actors_under_test", actors_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chain_actor_run_delegates_to_wrapped_chain(monkeypatch):
    actors = load_actors_module(monkeypatch)

    class DummyChain:
        def __init__(self):
            self.calls = []

        def run(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "ran"

    chain = DummyChain()
    actor = actors.ChainActor(chain)

    assert actor.run("task", flag=True) == "ran"
    assert chain.calls == [(("task",), {"flag": True})]
