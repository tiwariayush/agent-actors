import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def patched_modules(replacements):
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ray_stub():
    ray = types.ModuleType("ray")
    ray.ObjectRef = object

    def remote(*args, **kwargs):
        if args and isinstance(args[0], type):
            return args[0]

        def decorator(cls):
            return cls

        return decorator

    ray.remote = remote
    return ray


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        with patched_modules({"ray": ray_stub()}):
            actors = load_module(
                "tests._actors_under_test",
                REPO_ROOT / "agent_actors" / "actors.py",
            )

        class Chain:
            def run(self, *args, **kwargs):
                return args, kwargs

        self.assertEqual(
            actors.ChainActor(Chain()).run("task", verbose=True),
            (("task",), {"verbose": True}),
        )


class MemoryStrengthTests(unittest.TestCase):
    def test_predict_memory_strength_uses_first_number_in_response(self):
        agent_module = load_agent_module()

        for response, expected in (
            ("8", 0.8),
            ("Relevance: 8", 0.8),
            ("10", 1.0),
            ("Score: 10/10", 1.0),
        ):
            with self.subTest(response=response):
                agent = types.SimpleNamespace(
                    memory_strength=types.SimpleNamespace(
                        run=lambda memory_content, response=response: response
                    )
                )

                self.assertEqual(
                    agent_module.Agent._predict_memory_strength(agent, "memory"),
                    expected,
                )

    def test_predict_memory_strength_returns_zero_without_number(self):
        agent_module = load_agent_module()
        agent = types.SimpleNamespace(
            memory_strength=types.SimpleNamespace(
                run=lambda memory_content: "not relevant"
            )
        )

        self.assertEqual(
            agent_module.Agent._predict_memory_strength(agent, "memory"),
            0.0,
        )


def load_agent_module():
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

    langchain = types.ModuleType("langchain")
    langchain.LLMChain = object
    langchain_agents = types.ModuleType("langchain.agents")
    langchain_agents.Tool = object
    chat_models_base = types.ModuleType("langchain.chat_models.base")
    chat_models_base.BaseChatModel = object
    schema = types.ModuleType("langchain.schema")

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    schema.BaseRetriever = object
    schema.Document = Document

    agent_actors = types.ModuleType("agent_actors")
    agent_actors.__path__ = []
    actors = types.ModuleType("agent_actors.actors")

    class AgentActor:
        @classmethod
        def remote(cls, agent):
            return agent

    actors.AgentActor = AgentActor

    chains = types.ModuleType("agent_actors.chains")
    chains_agent = types.ModuleType("agent_actors.chains.agent")

    class ChainFactory:
        @classmethod
        def from_llm(cls, **kwargs):
            return object()

    chains_agent.GenerateInsights = ChainFactory
    chains_agent.MemoryStrength = ChainFactory
    chains_agent.Synthesis = ChainFactory
    chains_agent.WorkingMemory = ChainFactory

    replacements = {
        "ray": ray_stub(),
        "pydantic": pydantic,
        "langchain": langchain,
        "langchain.agents": langchain_agents,
        "langchain.chat_models.base": chat_models_base,
        "langchain.schema": schema,
        "agent_actors": agent_actors,
        "agent_actors.actors": actors,
        "agent_actors.chains": chains,
        "agent_actors.chains.agent": chains_agent,
    }

    module_name = "tests._agent_under_test"
    with patched_modules(replacements):
        return load_module(
            module_name,
            REPO_ROOT / "agent_actors" / "agent.py",
        )


if __name__ == "__main__":
    unittest.main()
