import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_dependency_stubs():
    for module_name in [
        "agent_actors",
        "agent_actors.actors",
        "agent_actors.agent",
        "agent_actors.chains",
        "agent_actors.chains.agent",
        "langchain",
        "langchain.agents",
        "langchain.chat_models",
        "langchain.chat_models.base",
        "langchain.schema",
        "pydantic",
        "ray",
    ]:
        sys.modules.pop(module_name, None)

    ray = types.ModuleType("ray")

    def remote(*decorator_args, **decorator_kwargs):
        def decorate(cls):
            cls.remote = classmethod(lambda remote_cls, *args, **kwargs: remote_cls(*args, **kwargs))
            return cls

        if decorator_args and len(decorator_args) == 1 and isinstance(decorator_args[0], type):
            return decorate(decorator_args[0])
        return decorate

    ray.remote = remote
    ray.ObjectRef = object
    sys.modules["ray"] = ray

    langchain = types.ModuleType("langchain")
    langchain.LLMChain = type("LLMChain", (), {})
    sys.modules["langchain"] = langchain

    langchain_agents = types.ModuleType("langchain.agents")
    langchain_agents.Tool = type("Tool", (), {})
    sys.modules["langchain.agents"] = langchain_agents

    langchain_chat_models = types.ModuleType("langchain.chat_models")
    sys.modules["langchain.chat_models"] = langchain_chat_models

    langchain_chat_models_base = types.ModuleType("langchain.chat_models.base")
    langchain_chat_models_base.BaseChatModel = type("BaseChatModel", (), {})
    sys.modules["langchain.chat_models.base"] = langchain_chat_models_base

    langchain_schema = types.ModuleType("langchain.schema")
    langchain_schema.BaseRetriever = type("BaseRetriever", (), {})
    langchain_schema.Document = type("Document", (), {})
    sys.modules["langchain.schema"] = langchain_schema

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules["pydantic"] = pydantic

    agent_actors = types.ModuleType("agent_actors")
    agent_actors.__path__ = [str(REPO_ROOT / "agent_actors")]
    sys.modules["agent_actors"] = agent_actors

    chains = types.ModuleType("agent_actors.chains")
    chains.__path__ = [str(REPO_ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains

    chain_agent = types.ModuleType("agent_actors.chains.agent")

    class DummyChain:
        @classmethod
        def from_llm(cls, **kwargs):
            return cls()

    chain_agent.WorkingMemory = DummyChain
    chain_agent.Synthesis = DummyChain
    chain_agent.GenerateInsights = DummyChain
    chain_agent.MemoryStrength = DummyChain
    sys.modules["agent_actors.chains.agent"] = chain_agent


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime_modules():
    _install_dependency_stubs()
    actors = _load_module("agent_actors.actors", "agent_actors/actors.py")
    agent = _load_module("agent_actors.agent", "agent_actors/agent.py")
    return actors, agent


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        actors, _ = _load_runtime_modules()

        class Chain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain result"

        chain = Chain()

        self.assertEqual(
            actors.ChainActor(chain).run("task", force=True),
            "chain result",
        )
        self.assertEqual(chain.calls, [(("task",), {"force": True})])

    def test_memory_strength_uses_matched_number_from_labelled_response(self):
        _, agent = _load_runtime_modules()

        class MemoryStrength:
            def run(self, **kwargs):
                return "Relevance: 8"

        instance = agent.Agent.__new__(agent.Agent)
        instance.memory_strength = MemoryStrength()

        self.assertEqual(instance._predict_memory_strength("important memory"), 0.8)

    def test_memory_strength_preserves_two_digit_scores(self):
        _, agent = _load_runtime_modules()

        class MemoryStrength:
            def run(self, **kwargs):
                return "10"

        instance = agent.Agent.__new__(agent.Agent)
        instance.memory_strength = MemoryStrength()

        self.assertEqual(instance._predict_memory_strength("critical memory"), 1.0)


if __name__ == "__main__":
    unittest.main()
