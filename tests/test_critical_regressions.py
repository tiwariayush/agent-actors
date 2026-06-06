import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubModules:
    def __init__(self, modules):
        self.modules = modules
        self.originals = {}

    def __enter__(self):
        for name, module in self.modules.items():
            self.originals[name] = sys.modules.get(name)
            sys.modules[name] = module

    def __exit__(self, exc_type, exc, tb):
        for name in self.modules:
            original = self.originals[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _stub_ray():
    ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        def decorate(cls):
            return cls

        return decorate

    ray.remote = remote
    ray.ObjectRef = object
    return ray


class ChainActorTest(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        with StubModules({"ray": _stub_ray()}):
            actors = _load_module("actors_under_test", ROOT / "agent_actors" / "actors.py")

        class Chain:
            def run(self, value, *, suffix):
                return f"{value}:{suffix}"

        self.assertEqual(actors.ChainActor(Chain()).run("work", suffix="done"), "work:done")


def _agent_import_stubs():
    langchain = types.ModuleType("langchain")

    class LLMChain:
        pass

    langchain.LLMChain = LLMChain

    agents = types.ModuleType("langchain.agents")

    class Tool:
        pass

    agents.Tool = Tool

    chat_models_base = types.ModuleType("langchain.chat_models.base")

    class BaseChatModel:
        pass

    chat_models_base.BaseChatModel = BaseChatModel

    schema = types.ModuleType("langchain.schema")

    class BaseRetriever:
        pass

    class Document:
        pass

    schema.BaseRetriever = BaseRetriever
    schema.Document = Document

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        pass

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field

    actors = types.ModuleType("agent_actors.actors")

    class AgentActor:
        @staticmethod
        def remote(*args, **kwargs):
            return None

    actors.AgentActor = AgentActor

    chains_agent = types.ModuleType("agent_actors.chains.agent")
    for name in ("GenerateInsights", "MemoryStrength", "Synthesis", "WorkingMemory"):
        setattr(chains_agent, name, type(name, (), {}))

    agent_actors = types.ModuleType("agent_actors")
    agent_actors.__path__ = []
    chains = types.ModuleType("agent_actors.chains")
    chains.__path__ = []

    return {
        "ray": _stub_ray(),
        "langchain": langchain,
        "langchain.agents": agents,
        "langchain.chat_models.base": chat_models_base,
        "langchain.schema": schema,
        "pydantic": pydantic,
        "agent_actors": agent_actors,
        "agent_actors.actors": actors,
        "agent_actors.chains": chains,
        "agent_actors.chains.agent": chains_agent,
    }


class MemoryStrengthParsingTest(unittest.TestCase):
    def _predict_strength(self, response):
        with StubModules(_agent_import_stubs()):
            agent_module = _load_module(
                "agent_under_test", ROOT / "agent_actors" / "agent.py"
            )

        class MemoryStrength:
            def run(self, **kwargs):
                return response

        agent = types.SimpleNamespace(memory_strength=MemoryStrength())
        return agent_module.Agent._predict_memory_strength(agent, "memory")

    def test_prefixed_score_uses_matched_digits(self):
        self.assertEqual(self._predict_strength("Relevance: 8"), 0.8)

    def test_ten_scores_as_full_strength(self):
        self.assertEqual(self._predict_strength("10"), 1.0)

    def test_missing_score_defaults_to_zero(self):
        self.assertEqual(self._predict_strength("not relevant"), 0.0)


if __name__ == "__main__":
    unittest.main()
