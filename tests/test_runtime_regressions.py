import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def patched_modules(modules):
    previous = {name: sys.modules.get(name) for name in modules}
    missing = {name for name, value in previous.items() if value is None}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in modules:
            if name in missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ray_stub():
    ray = types.ModuleType("ray")
    ray.ObjectRef = object

    def remote(*decorator_args, **_decorator_kwargs):
        if len(decorator_args) == 1 and isinstance(decorator_args[0], type):
            return decorator_args[0]

        def decorate(cls):
            return cls

        return decorate

    ray.remote = remote
    return ray


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        with patched_modules({"ray": ray_stub()}):
            actors = load_module("actors_under_test", "agent_actors/actors.py")

        class Chain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        actor = actors.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", priority=1),
            {"args": ("task",), "kwargs": {"priority": 1}},
        )

    def test_memory_strength_parses_prefixed_score(self):
        agent_module = self._load_agent_module()
        agent = object.__new__(agent_module.Agent)
        agent.memory_strength = MemoryStrengthResponse("Relevance: 8")

        self.assertEqual(agent._predict_memory_strength("memory"), 0.8)

    def test_memory_strength_preserves_ten_out_of_ten_score(self):
        agent_module = self._load_agent_module()
        agent = object.__new__(agent_module.Agent)
        agent.memory_strength = MemoryStrengthResponse("10")

        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

    def _load_agent_module(self):
        return load_agent_module()


class MemoryStrengthResponse:
    def __init__(self, response):
        self.response = response

    def run(self, memory_content):
        return self.response


def load_agent_module():
    langchain = types.ModuleType("langchain")
    langchain.LLMChain = object

    agents = types.ModuleType("langchain.agents")
    agents.Tool = object

    chat_models = types.ModuleType("langchain.chat_models")
    chat_models_base = types.ModuleType("langchain.chat_models.base")
    chat_models_base.BaseChatModel = object

    schema = types.ModuleType("langchain.schema")
    schema.BaseRetriever = object
    schema.Document = object

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = object

    def field(default=None, default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.Field = field

    package = types.ModuleType("agent_actors")
    package.__path__ = []

    actors = types.ModuleType("agent_actors.actors")
    actors.AgentActor = object

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = []

    agent_chains = types.ModuleType("agent_actors.chains.agent")
    agent_chains.GenerateInsights = object
    agent_chains.MemoryStrength = object
    agent_chains.Synthesis = object
    agent_chains.WorkingMemory = object

    modules = {
        "ray": ray_stub(),
        "langchain": langchain,
        "langchain.agents": agents,
        "langchain.chat_models": chat_models,
        "langchain.chat_models.base": chat_models_base,
        "langchain.schema": schema,
        "pydantic": pydantic,
        "agent_actors": package,
        "agent_actors.actors": actors,
        "agent_actors.chains": chains_package,
        "agent_actors.chains.agent": agent_chains,
    }

    with patched_modules(modules):
        return load_module("agent_under_test", "agent_actors/agent.py")


if __name__ == "__main__":
    unittest.main()
