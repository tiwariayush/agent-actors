import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_dependency_stubs():
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

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

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


def _install_agent_actors_package_stubs():
    package = types.ModuleType("agent_actors")
    package.__path__ = [str(ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = [str(ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains_package

    chains_agent = types.ModuleType("agent_actors.chains.agent")

    class StubChain:
        @classmethod
        def from_llm(cls, **kwargs):
            return cls()

    chains_agent.GenerateInsights = StubChain
    chains_agent.MemoryStrength = StubChain
    chains_agent.Synthesis = StubChain
    chains_agent.WorkingMemory = StubChain
    sys.modules["agent_actors.chains.agent"] = chains_agent


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime_modules():
    _install_dependency_stubs()
    _install_agent_actors_package_stubs()
    actors = _load_module("agent_actors.actors", "agent_actors/actors.py")
    agent = _load_module("agent_actors.agent", "agent_actors/agent.py")
    return actors, agent


class RuntimeRegressionTest(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        actors, _ = load_runtime_modules()

        class Chain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        actor = actors.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", priority="high"),
            {"args": ("task",), "kwargs": {"priority": "high"}},
        )

    def test_memory_strength_accepts_labelled_scores(self):
        _, agent = load_runtime_modules()

        scorer = types.SimpleNamespace(run=lambda memory_content: "Relevance: 8")
        subject = types.SimpleNamespace(memory_strength=scorer)

        self.assertEqual(agent.Agent._predict_memory_strength(subject, "memory"), 0.8)

    def test_memory_strength_preserves_two_digit_scores(self):
        _, agent = load_runtime_modules()

        scorer = types.SimpleNamespace(run=lambda memory_content: "10")
        subject = types.SimpleNamespace(memory_strength=scorer)

        self.assertEqual(agent.Agent._predict_memory_strength(subject, "memory"), 1.0)


if __name__ == "__main__":
    unittest.main()
