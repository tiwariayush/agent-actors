import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def patched_modules(replacements):
    missing = object()
    originals = {name: sys.modules.get(name, missing) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ray_stub():
    module = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and len(args) == 1 and isinstance(args[0], type):
            return args[0]

        def decorate(cls):
            return cls

        return decorate

    module.remote = remote
    module.ObjectRef = object
    return module


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        with patched_modules({"ray": ray_stub()}):
            actors = load_module("_actors_under_test", "agent_actors/actors.py")

        class Chain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain result"

        chain = Chain()
        actor = actors.ChainActor(chain)

        self.assertEqual(actor.run("task", option=True), "chain result")
        self.assertEqual(chain.calls, [(("task",), {"option": True})])


class MemoryStrengthTests(unittest.TestCase):
    def test_prefixed_memory_strength_score_uses_matched_digits(self):
        agent = self._agent_with_memory_strength_output("Relevance: 8")

        self.assertEqual(agent._predict_memory_strength("important memory"), 0.8)

    def test_ten_memory_strength_score_is_not_truncated(self):
        agent = self._agent_with_memory_strength_output("10")

        self.assertEqual(agent._predict_memory_strength("important memory"), 1.0)

    def _agent_with_memory_strength_output(self, score):
        agent_module = self._load_agent_module()
        agent = object.__new__(agent_module.Agent)
        agent.memory_strength = FakeMemoryStrength(score)
        return agent

    def _load_agent_module(self):
        class BaseModel:
            pass

        def Field(default=None, default_factory=None, **kwargs):
            if default_factory is not None:
                return default_factory()
            return default

        class LLMChain:
            pass

        class Tool:
            pass

        class BaseChatModel:
            pass

        class BaseRetriever:
            pass

        class Document:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = BaseModel
        pydantic.Field = Field

        langchain = types.ModuleType("langchain")
        langchain.LLMChain = LLMChain

        agents = types.ModuleType("langchain.agents")
        agents.Tool = Tool

        chat_models = types.ModuleType("langchain.chat_models")
        chat_models_base = types.ModuleType("langchain.chat_models.base")
        chat_models_base.BaseChatModel = BaseChatModel

        schema = types.ModuleType("langchain.schema")
        schema.BaseRetriever = BaseRetriever
        schema.Document = Document

        agent_actors = types.ModuleType("agent_actors")
        agent_actors.__path__ = [str(ROOT / "agent_actors")]

        actors = types.ModuleType("agent_actors.actors")
        actors.AgentActor = type("AgentActor", (), {})

        chains = types.ModuleType("agent_actors.chains")
        chains.__path__ = [str(ROOT / "agent_actors" / "chains")]

        chains_agent = types.ModuleType("agent_actors.chains.agent")
        for class_name in (
            "GenerateInsights",
            "MemoryStrength",
            "Synthesis",
            "WorkingMemory",
        ):
            setattr(chains_agent, class_name, type(class_name, (), {}))

        replacements = {
            "agent_actors": agent_actors,
            "agent_actors.actors": actors,
            "agent_actors.chains": chains,
            "agent_actors.chains.agent": chains_agent,
            "langchain": langchain,
            "langchain.agents": agents,
            "langchain.chat_models": chat_models,
            "langchain.chat_models.base": chat_models_base,
            "langchain.schema": schema,
            "pydantic": pydantic,
            "ray": ray_stub(),
        }

        with patched_modules(replacements):
            return load_module("_agent_under_test", "agent_actors/agent.py")


class FakeMemoryStrength:
    def __init__(self, score):
        self.score = score

    def run(self, **kwargs):
        return self.score


if __name__ == "__main__":
    unittest.main()
