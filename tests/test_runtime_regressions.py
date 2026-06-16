import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_dependency_stubs():
    ray = types.ModuleType("ray")
    ray.ObjectRef = object

    def remote(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return lambda decorated: decorated

    ray.remote = remote
    sys.modules["ray"] = ray

    langchain = types.ModuleType("langchain")

    class LLMChain:
        pass

    langchain.LLMChain = LLMChain
    sys.modules["langchain"] = langchain

    agents = types.ModuleType("langchain.agents")

    class Tool:
        pass

    agents.Tool = Tool
    sys.modules["langchain.agents"] = agents

    chat_models = types.ModuleType("langchain.chat_models")
    chat_models_base = types.ModuleType("langchain.chat_models.base")

    class BaseChatModel:
        pass

    chat_models_base.BaseChatModel = BaseChatModel
    sys.modules["langchain.chat_models"] = chat_models
    sys.modules["langchain.chat_models.base"] = chat_models_base

    schema = types.ModuleType("langchain.schema")

    class BaseRetriever:
        pass

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    schema.BaseRetriever = BaseRetriever
    schema.Document = Document
    sys.modules["langchain.schema"] = schema

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules["pydantic"] = pydantic

    package = types.ModuleType("agent_actors")
    package.__path__ = [str(ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = [str(ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains_package

    chains_agent = types.ModuleType("agent_actors.chains.agent")

    class ChainFactory:
        @classmethod
        def from_llm(cls, **kwargs):
            return cls()

    chains_agent.GenerateInsights = ChainFactory
    chains_agent.MemoryStrength = ChainFactory
    chains_agent.Synthesis = ChainFactory
    chains_agent.WorkingMemory = ChainFactory
    sys.modules["agent_actors.chains.agent"] = chains_agent


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        _install_dependency_stubs()
        actors = _load_module("agent_actors.actors", "agent_actors/actors.py")

        class Chain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = actors.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", force=True),
            (("task",), {"force": True}),
        )


class MemoryStrengthTests(unittest.TestCase):
    def test_prefixed_score_uses_captured_digits(self):
        _install_dependency_stubs()
        _load_module("agent_actors.actors", "agent_actors/actors.py")
        agent_module = _load_module("agent_actors.agent", "agent_actors/agent.py")

        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **kwargs):
                return self.response

        agent = object.__new__(agent_module.Agent)
        agent.memory_strength = MemoryStrength("Relevance: 8")

        self.assertEqual(agent._predict_memory_strength("memory"), 0.8)

    def test_two_digit_score_is_not_truncated_to_first_digit(self):
        _install_dependency_stubs()
        _load_module("agent_actors.actors", "agent_actors/actors.py")
        agent_module = _load_module("agent_actors.agent", "agent_actors/agent.py")

        class MemoryStrength:
            def run(self, **kwargs):
                return "10"

        agent = object.__new__(agent_module.Agent)
        agent.memory_strength = MemoryStrength()

        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)


if __name__ == "__main__":
    unittest.main()
