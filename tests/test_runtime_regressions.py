import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_ray_stub():
    ray = types.ModuleType("ray")
    ray.ObjectRef = object

    def remote(*_args, **_kwargs):
        def decorate(cls):
            return cls

        return decorate

    ray.remote = remote
    sys.modules["ray"] = ray


def install_agent_dependency_stubs(actors_module):
    langchain = types.ModuleType("langchain")
    langchain.LLMChain = type("LLMChain", (), {})
    sys.modules["langchain"] = langchain

    agents = types.ModuleType("langchain.agents")
    agents.Tool = type("Tool", (), {})
    sys.modules["langchain.agents"] = agents

    chat_models = types.ModuleType("langchain.chat_models")
    chat_models_base = types.ModuleType("langchain.chat_models.base")
    chat_models_base.BaseChatModel = type("BaseChatModel", (), {})
    sys.modules["langchain.chat_models"] = chat_models
    sys.modules["langchain.chat_models.base"] = chat_models_base

    schema = types.ModuleType("langchain.schema")
    schema.BaseRetriever = type("BaseRetriever", (), {})
    schema.Document = type("Document", (), {})
    sys.modules["langchain.schema"] = schema

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules["pydantic"] = pydantic

    package = types.ModuleType("agent_actors")
    package.__path__ = []
    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = []
    chains_agent = types.ModuleType("agent_actors.chains.agent")

    class ChainFactory:
        @classmethod
        def from_llm(cls, **_kwargs):
            return cls()

    chains_agent.GenerateInsights = ChainFactory
    chains_agent.MemoryStrength = ChainFactory
    chains_agent.Synthesis = ChainFactory
    chains_agent.WorkingMemory = ChainFactory

    sys.modules["agent_actors"] = package
    sys.modules["agent_actors.actors"] = actors_module
    sys.modules["agent_actors.chains"] = chains_package
    sys.modules["agent_actors.chains.agent"] = chains_agent


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_chain(self):
        install_ray_stub()
        actors = load_module("agent_actors.actors", ROOT / "agent_actors" / "actors.py")

        class Chain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain-result"

        chain = Chain()
        actor = actors.ChainActor(chain)

        self.assertEqual(actor.run("task", flag=True), "chain-result")
        self.assertEqual(chain.calls, [(("task",), {"flag": True})])

    def test_memory_strength_parser_uses_matched_digits(self):
        install_ray_stub()
        actors = load_module("agent_actors.actors", ROOT / "agent_actors" / "actors.py")
        install_agent_dependency_stubs(actors)
        agent_module = load_module(
            "agent_actors.agent", ROOT / "agent_actors" / "agent.py"
        )

        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **_kwargs):
                return self.response

        agent = object.__new__(agent_module.Agent)
        agent.memory_strength = MemoryStrength("Relevance: 8")
        self.assertEqual(agent_module.Agent._predict_memory_strength(agent, "memory"), 0.8)

        agent.memory_strength = MemoryStrength("10")
        self.assertEqual(agent_module.Agent._predict_memory_strength(agent, "memory"), 1.0)

        agent.memory_strength = MemoryStrength("none")
        self.assertEqual(agent_module.Agent._predict_memory_strength(agent, "memory"), 0.0)


if __name__ == "__main__":
    unittest.main()
