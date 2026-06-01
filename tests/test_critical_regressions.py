import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def patched_modules(modules):
    original = {name: sys.modules.get(name) for name in modules}
    missing = {name for name in modules if name not in sys.modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in modules:
            if name in missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original[name]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fake_ray_module():
    ray = types.ModuleType("ray")
    ray.ObjectRef = object

    def remote(*_args, **_kwargs):
        def decorate(cls):
            return cls

        return decorate

    ray.remote = remote
    return ray


def fake_pydantic_module():
    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, *args, **kwargs):
            super().__init__()
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    return pydantic


def fake_agent_dependencies():
    langchain = types.ModuleType("langchain")
    langchain.LLMChain = type("LLMChain", (), {})

    langchain_agents = types.ModuleType("langchain.agents")
    langchain_agents.Tool = type("Tool", (), {})

    chat_models_base = types.ModuleType("langchain.chat_models.base")
    chat_models_base.BaseChatModel = type("BaseChatModel", (), {})

    schema = types.ModuleType("langchain.schema")
    schema.BaseRetriever = type("BaseRetriever", (), {})
    schema.Document = type("Document", (), {})

    agent_actors = types.ModuleType("agent_actors")
    actors = types.ModuleType("agent_actors.actors")
    actors.AgentActor = type("AgentActor", (), {"remote": staticmethod(lambda *_args, **_kwargs: None)})

    chains = types.ModuleType("agent_actors.chains")
    chains_agent = types.ModuleType("agent_actors.chains.agent")
    for name in ("GenerateInsights", "MemoryStrength", "Synthesis", "WorkingMemory"):
        setattr(chains_agent, name, type(name, (), {"from_llm": classmethod(lambda cls, **_kwargs: cls())}))

    return {
        "ray": fake_ray_module(),
        "pydantic": fake_pydantic_module(),
        "langchain": langchain,
        "langchain.agents": langchain_agents,
        "langchain.chat_models.base": chat_models_base,
        "langchain.schema": schema,
        "agent_actors": agent_actors,
        "agent_actors.actors": actors,
        "agent_actors.chains": chains,
        "agent_actors.chains.agent": chains_agent,
    }


class CriticalRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        with patched_modules({"ray": fake_ray_module()}):
            actors = load_module("actors_under_test", "agent_actors/actors.py")

        class FakeChain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        actor = actors.ChainActor(FakeChain())

        self.assertEqual(
            actor.run("input", option=True),
            {"args": ("input",), "kwargs": {"option": True}},
        )

    def test_memory_strength_uses_matched_digits(self):
        with patched_modules(fake_agent_dependencies()):
            agent_module = load_module("agent_under_test", "agent_actors/agent.py")

        agent = object.__new__(agent_module.Agent)

        class FakeMemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **_kwargs):
                return self.response

        agent.memory_strength = FakeMemoryStrength("Relevance: 10")
        self.assertEqual(agent._predict_memory_strength("important memory"), 1.0)

        agent.memory_strength = FakeMemoryStrength("10")
        self.assertEqual(agent._predict_memory_strength("important memory"), 1.0)


if __name__ == "__main__":
    unittest.main()
