import importlib.util
import pathlib
import sys
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_ray_stub():
    ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and len(args) == 1 and isinstance(args[0], type) and not kwargs:
            return args[0]
        return lambda cls: cls

    ray.remote = remote
    ray.ObjectRef = object
    sys.modules["ray"] = ray


def clear_agent_actor_modules():
    for module_name in list(sys.modules):
        if module_name.startswith("agent_actors") or module_name in {
            "ray",
            "langchain",
            "langchain.agents",
            "langchain.chat_models",
            "langchain.chat_models.base",
            "langchain.schema",
            "pydantic",
        }:
            del sys.modules[module_name]


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        clear_agent_actor_modules()
        install_ray_stub()
        actors = load_module("agent_actors.actors", "agent_actors/actors.py")

        class Chain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain result"

        chain = Chain()

        self.assertEqual(actors.ChainActor(chain).run("task", context="memory"), "chain result")
        self.assertEqual(chain.calls, [(("task",), {"context": "memory"})])


class MemoryStrengthTests(unittest.TestCase):
    def setUp(self):
        clear_agent_actor_modules()
        install_ray_stub()

        langchain = types.ModuleType("langchain")

        class LLMChain:
            pass

        class PromptTemplate:
            @classmethod
            def from_template(cls, *args, **kwargs):
                return cls()

        langchain.LLMChain = LLMChain
        langchain.PromptTemplate = PromptTemplate
        sys.modules["langchain"] = langchain

        agents = types.ModuleType("langchain.agents")

        class Tool:
            pass

        agents.Tool = Tool
        sys.modules["langchain.agents"] = agents

        sys.modules["langchain.chat_models"] = types.ModuleType("langchain.chat_models")
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
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def Field(default=None, default_factory=None, **kwargs):
            return default_factory() if default_factory else default

        pydantic.BaseModel = BaseModel
        pydantic.Field = Field
        sys.modules["pydantic"] = pydantic

        agent_actors = types.ModuleType("agent_actors")
        agent_actors.__path__ = [str(REPO_ROOT / "agent_actors")]
        sys.modules["agent_actors"] = agent_actors
        chains = types.ModuleType("agent_actors.chains")
        chains.__path__ = [str(REPO_ROOT / "agent_actors" / "chains")]
        sys.modules["agent_actors.chains"] = chains
        load_module("agent_actors.actors", "agent_actors/actors.py")
        load_module("agent_actors.chains.agent", "agent_actors/chains/agent.py")
        self.agent_module = load_module("agent_actors.agent", "agent_actors/agent.py")

    def predict_strength(self, llm_output):
        class MemoryStrength:
            def run(self, memory_content):
                return llm_output

        agent = self.agent_module.Agent.__new__(self.agent_module.Agent)
        agent.memory_strength = MemoryStrength()
        return agent._predict_memory_strength("memory")

    def test_prefixed_score_uses_first_number(self):
        self.assertEqual(self.predict_strength("Relevance: 8"), 0.8)

    def test_two_digit_score_is_not_truncated(self):
        self.assertEqual(self.predict_strength("10"), 1.0)

    def test_missing_score_defaults_to_zero(self):
        self.assertEqual(self.predict_strength("not relevant"), 0.0)


if __name__ == "__main__":
    unittest.main()
