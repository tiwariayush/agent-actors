import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_dependency_stubs():
    package = types.ModuleType("agent_actors")
    package.__path__ = [str(ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package

    ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(cls):
            return cls

        return decorator

    ray.remote = remote
    ray.ObjectRef = object
    sys.modules["ray"] = ray

    langchain = types.ModuleType("langchain")
    langchain.LLMChain = type("LLMChain", (), {})
    sys.modules["langchain"] = langchain

    agents = types.ModuleType("langchain.agents")
    agents.Tool = type("Tool", (), {})
    sys.modules["langchain.agents"] = agents

    chat_models = types.ModuleType("langchain.chat_models")
    sys.modules["langchain.chat_models"] = chat_models

    chat_models_base = types.ModuleType("langchain.chat_models.base")
    chat_models_base.BaseChatModel = type("BaseChatModel", (), {})
    sys.modules["langchain.chat_models.base"] = chat_models_base

    schema = types.ModuleType("langchain.schema")
    schema.BaseRetriever = type("BaseRetriever", (), {})
    schema.Document = type("Document", (), {})
    sys.modules["langchain.schema"] = schema

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = lambda default=None, **kwargs: default
    sys.modules["pydantic"] = pydantic

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


class RuntimeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_dependency_stubs()
        cls.actors_module = _load_module("agent_actors.actors", "agent_actors/actors.py")
        cls.agent_module = _load_module("agent_actors.agent", "agent_actors/agent.py")

    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        class Chain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = self.actors_module.ChainActor(Chain())

        self.assertEqual(
            actor.run("prompt", temperature=0),
            (("prompt",), {"temperature": 0}),
        )

    def test_memory_strength_uses_matched_numeric_group(self):
        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **kwargs):
                return self.response

        agent = self.agent_module.Agent.__new__(self.agent_module.Agent)
        agent.memory_strength = MemoryStrength("Relevance: 8")
        self.assertEqual(agent._predict_memory_strength("memory"), 0.8)

        agent.memory_strength = MemoryStrength("10")
        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

        agent.memory_strength = MemoryStrength("not scored")
        self.assertEqual(agent._predict_memory_strength("memory"), 0.0)


if __name__ == "__main__":
    unittest.main()
