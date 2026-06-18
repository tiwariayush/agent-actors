import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_ray_stub():
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


def install_agent_dependency_stubs(actors_module):
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

    package = types.ModuleType("agent_actors")
    package.__path__ = [str(ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package
    sys.modules["agent_actors.actors"] = actors_module

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = [str(ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains_package

    chains_agent = types.ModuleType("agent_actors.chains.agent")
    for class_name in (
        "GenerateInsights",
        "MemoryStrength",
        "Synthesis",
        "WorkingMemory",
    ):
        chain_class = type(
            class_name,
            (),
            {"from_llm": classmethod(lambda cls, **kwargs: cls())},
        )
        setattr(chains_agent, class_name, chain_class)
    sys.modules["agent_actors.chains.agent"] = chains_agent


class RuntimeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_ray_stub()
        cls.actors_module = load_module("agent_actors.actors", "agent_actors/actors.py")
        install_agent_dependency_stubs(cls.actors_module)
        cls.agent_module = load_module("agent_actors.agent", "agent_actors/agent.py")

    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        class DummyChain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        actor = self.actors_module.ChainActor(DummyChain())

        self.assertEqual(
            actor.run("task", retries=2),
            {"args": ("task",), "kwargs": {"retries": 2}},
        )

    def test_memory_strength_uses_matched_digits(self):
        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **kwargs):
                return self.response

        agent = types.SimpleNamespace(memory_strength=MemoryStrength("Relevance: 8"))

        self.assertEqual(
            self.agent_module.Agent._predict_memory_strength(agent, "important memory"),
            0.8,
        )

    def test_memory_strength_handles_ten_as_full_strength(self):
        class MemoryStrength:
            def run(self, **kwargs):
                return "10"

        agent = types.SimpleNamespace(memory_strength=MemoryStrength())

        self.assertEqual(
            self.agent_module.Agent._predict_memory_strength(agent, "critical memory"),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
