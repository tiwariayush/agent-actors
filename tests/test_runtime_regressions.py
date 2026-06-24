import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_stubs():
    ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        def decorate(obj):
            return obj

        return decorate

    ray.remote = remote
    ray.ObjectRef = object
    sys.modules["ray"] = ray

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        pass

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules["pydantic"] = pydantic

    langchain = types.ModuleType("langchain")
    langchain.LLMChain = type("LLMChain", (), {})
    sys.modules["langchain"] = langchain

    langchain_agents = types.ModuleType("langchain.agents")
    langchain_agents.Tool = type("Tool", (), {})
    sys.modules["langchain.agents"] = langchain_agents

    langchain_chat_models = types.ModuleType("langchain.chat_models")
    sys.modules["langchain.chat_models"] = langchain_chat_models
    langchain_chat_base = types.ModuleType("langchain.chat_models.base")
    langchain_chat_base.BaseChatModel = type("BaseChatModel", (), {})
    sys.modules["langchain.chat_models.base"] = langchain_chat_base

    langchain_schema = types.ModuleType("langchain.schema")
    langchain_schema.BaseRetriever = type("BaseRetriever", (), {})
    langchain_schema.Document = type("Document", (), {})
    sys.modules["langchain.schema"] = langchain_schema

    package = types.ModuleType("agent_actors")
    package.__path__ = [str(ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = [str(ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains_package

    chains_agent = types.ModuleType("agent_actors.chains.agent")

    class ChainStub:
        @classmethod
        def from_llm(cls, **kwargs):
            return cls()

    chains_agent.GenerateInsights = ChainStub
    chains_agent.MemoryStrength = ChainStub
    chains_agent.Synthesis = ChainStub
    chains_agent.WorkingMemory = ChainStub
    sys.modules["agent_actors.chains.agent"] = chains_agent


class RuntimeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_stubs()
        cls.actors_module = _load_module("agent_actors.actors", "agent_actors/actors.py")
        cls.agent_module = _load_module("agent_actors.agent", "agent_actors/agent.py")

    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        class Chain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        actor = self.actors_module.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", priority="high"),
            {"args": ("task",), "kwargs": {"priority": "high"}},
        )

    def test_predict_memory_strength_uses_matched_digits(self):
        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, memory_content):
                return self.response

        agent = self.agent_module.Agent.__new__(self.agent_module.Agent)

        agent.memory_strength = MemoryStrength("Relevance: 8")
        self.assertEqual(agent._predict_memory_strength("important memory"), 0.8)

        agent.memory_strength = MemoryStrength("10")
        self.assertEqual(agent._predict_memory_strength("critical memory"), 1.0)


if __name__ == "__main__":
    unittest.main()
