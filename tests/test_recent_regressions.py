import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DependencyStubs:
    def __enter__(self):
        self.previous_modules = sys.modules.copy()
        self._install_ray_stub()
        self._install_langchain_stubs()
        self._install_pydantic_stub()
        self._install_agent_actor_stub()
        self._install_agent_chain_stubs()

    def __exit__(self, exc_type, exc, tb):
        sys.modules.clear()
        sys.modules.update(self.previous_modules)

    def _install_ray_stub(self):
        ray = types.ModuleType("ray")

        def remote(*args, **kwargs):
            if args and len(args) == 1 and isinstance(args[0], type):
                return args[0]

            def decorate(cls):
                return cls

            return decorate

        ray.remote = remote
        ray.ObjectRef = object
        sys.modules["ray"] = ray

    def _install_langchain_stubs(self):
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

    def _install_pydantic_stub(self):
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

    def _install_agent_actor_stub(self):
        package = types.ModuleType("agent_actors")
        package.__path__ = []
        actors = types.ModuleType("agent_actors.actors")

        class AgentActor:
            @classmethod
            def remote(cls, *args, **kwargs):
                return cls(*args, **kwargs)

        actors.AgentActor = AgentActor
        sys.modules["agent_actors"] = package
        sys.modules["agent_actors.actors"] = actors

    def _install_agent_chain_stubs(self):
        chains = types.ModuleType("agent_actors.chains")
        agent_chains = types.ModuleType("agent_actors.chains.agent")

        class StubChain:
            @classmethod
            def from_llm(cls, **kwargs):
                return cls()

        agent_chains.GenerateInsights = StubChain
        agent_chains.MemoryStrength = StubChain
        agent_chains.Synthesis = StubChain
        agent_chains.WorkingMemory = StubChain
        sys.modules["agent_actors.chains"] = chains
        sys.modules["agent_actors.chains.agent"] = agent_chains


class ChainActorTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        with DependencyStubs():
            actors = load_module("actors_under_test", "agent_actors/actors.py")

            class Chain:
                def run(self, *args, **kwargs):
                    return args, kwargs

            result = actors.ChainActor(Chain()).run("input", option=True)

        self.assertEqual(result, (("input",), {"option": True}))


class MemoryStrengthTests(unittest.TestCase):
    def test_prefixed_strength_score_uses_matched_number(self):
        with DependencyStubs():
            agent_module = load_module("agent_under_test", "agent_actors/agent.py")
            agent = object.__new__(agent_module.Agent)

            class MemoryStrength:
                def run(self, **kwargs):
                    return "Score: 8"

            agent.memory_strength = MemoryStrength()
            score = agent._predict_memory_strength("important memory")

        self.assertEqual(score, 0.8)

    def test_two_digit_strength_score_is_not_truncated(self):
        with DependencyStubs():
            agent_module = load_module(
                "agent_under_test_two_digit", "agent_actors/agent.py"
            )
            agent = object.__new__(agent_module.Agent)

            class MemoryStrength:
                def run(self, **kwargs):
                    return "10"

            agent.memory_strength = MemoryStrength()
            score = agent._predict_memory_strength("critical memory")

        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
