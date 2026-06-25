import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_ray_stub():
    ray = types.ModuleType("ray")
    ray.ObjectRef = object

    def remote(*args, **kwargs):
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(cls):
            return cls

        return decorator

    ray.remote = remote
    sys.modules["ray"] = ray


def _install_agent_dependency_stubs():
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

    chat_models_base = types.ModuleType("langchain.chat_models.base")

    class BaseChatModel:
        pass

    chat_models_base.BaseChatModel = BaseChatModel
    sys.modules["langchain.chat_models.base"] = chat_models_base

    schema = types.ModuleType("langchain.schema")

    class BaseRetriever:
        pass

    class Document:
        pass

    schema.BaseRetriever = BaseRetriever
    schema.Document = Document
    sys.modules["langchain.schema"] = schema

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

    agent_chains = types.ModuleType("agent_actors.chains.agent")

    class _ChainFactory:
        @classmethod
        def from_llm(cls, **kwargs):
            return cls()

    agent_chains.GenerateInsights = _ChainFactory
    agent_chains.MemoryStrength = _ChainFactory
    agent_chains.Synthesis = _ChainFactory
    agent_chains.WorkingMemory = _ChainFactory
    sys.modules["agent_actors.chains.agent"] = agent_chains

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = [str(REPO_ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains_package


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ray_stub()

        package = types.ModuleType("agent_actors")
        package.__path__ = [str(REPO_ROOT / "agent_actors")]
        sys.modules["agent_actors"] = package

        cls.actors_module = _load_module("agent_actors.actors", "agent_actors/actors.py")
        _install_agent_dependency_stubs()
        cls.agent_module = _load_module("agent_actors.agent", "agent_actors/agent.py")

    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        class Chain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = self.actors_module.ChainActor(Chain())

        self.assertEqual(actor.run("task", retries=2), (("task",), {"retries": 2}))

    def test_memory_strength_accepts_prefixed_scores(self):
        agent = types.SimpleNamespace(
            memory_strength=types.SimpleNamespace(
                run=lambda memory_content: "Relevance: 8"
            )
        )

        self.assertEqual(self.agent_module.Agent._predict_memory_strength(agent, "memory"), 0.8)

    def test_memory_strength_preserves_ten_score(self):
        agent = types.SimpleNamespace(
            memory_strength=types.SimpleNamespace(run=lambda memory_content: "10")
        )

        self.assertEqual(self.agent_module.Agent._predict_memory_strength(agent, "memory"), 1.0)


if __name__ == "__main__":
    unittest.main()
