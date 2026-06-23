import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_ray_stub():
    ray = types.ModuleType("ray")

    def remote(*decorator_args, **_decorator_kwargs):
        if decorator_args and len(decorator_args) == 1 and isinstance(
            decorator_args[0], type
        ):
            return decorator_args[0]

        def decorate(cls):
            return cls

        return decorate

    ray.remote = remote
    ray.ObjectRef = object
    sys.modules["ray"] = ray


def _install_agent_dependencies():
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
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    schema.BaseRetriever = BaseRetriever
    schema.Document = Document
    sys.modules["langchain.schema"] = schema

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        pass

    def Field(default=None, default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules["pydantic"] = pydantic

    package = types.ModuleType("agent_actors")
    package.__path__ = [str(REPO_ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = [str(REPO_ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains_package

    chains_agent = types.ModuleType("agent_actors.chains.agent")

    class ChainFactory:
        @classmethod
        def from_llm(cls, **_kwargs):
            return cls()

    chains_agent.GenerateInsights = ChainFactory
    chains_agent.MemoryStrength = ChainFactory
    chains_agent.Synthesis = ChainFactory
    chains_agent.WorkingMemory = ChainFactory
    sys.modules["agent_actors.chains.agent"] = chains_agent


class RuntimeRegressionTests(unittest.TestCase):
    def setUp(self):
        self._previous_modules = sys.modules.copy()
        _install_ray_stub()

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self._previous_modules)

    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        actors = _load_module("agent_actors.actors", "agent_actors/actors.py")

        class FakeChain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain-result"

        chain = FakeChain()
        actor = actors.ChainActor(chain)

        self.assertEqual(
            actor.run("input", mode="test"),
            "chain-result",
        )
        self.assertEqual(chain.calls, [(("input",), {"mode": "test"})])

    def test_memory_strength_uses_captured_number(self):
        _install_agent_dependencies()
        _load_module("agent_actors.actors", "agent_actors/actors.py")
        agent_module = _load_module("agent_actors.agent", "agent_actors/agent.py")

        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **_kwargs):
                return self.response

        subject = types.SimpleNamespace(memory_strength=MemoryStrength("Relevance: 8"))
        self.assertEqual(agent_module.Agent._predict_memory_strength(subject, "x"), 0.8)

        subject.memory_strength = MemoryStrength("10")
        self.assertEqual(agent_module.Agent._predict_memory_strength(subject, "x"), 1.0)


if __name__ == "__main__":
    unittest.main()
