import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_ray_stub():
    ray = types.ModuleType("ray")
    ray.ObjectRef = object

    def remote(*args, **kwargs):
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(cls):
            return cls

        return decorator

    ray.remote = remote
    return ray


@contextmanager
def stubbed_modules(replacements):
    original = {}
    missing = object()
    for name, module in replacements.items():
        original[name] = sys.modules.get(name, missing)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, module in original.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        with stubbed_modules({"ray": build_ray_stub()}):
            actors = load_module(
                "actors_under_test", ROOT / "agent_actors" / "actors.py"
            )

        class Chain:
            def run(self, *args, **kwargs):
                return ("chain-run", args, kwargs)

        actor = actors.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", force=True),
            ("chain-run", ("task",), {"force": True}),
        )


class MemoryStrengthTests(unittest.TestCase):
    def load_agent_module(self):
        langchain = types.ModuleType("langchain")
        langchain.LLMChain = type("LLMChain", (), {})

        langchain_agents = types.ModuleType("langchain.agents")
        langchain_agents.Tool = type("Tool", (), {})

        langchain_chat_base = types.ModuleType("langchain.chat_models.base")
        langchain_chat_base.BaseChatModel = type("BaseChatModel", (), {})

        langchain_schema = types.ModuleType("langchain.schema")
        langchain_schema.BaseRetriever = type("BaseRetriever", (), {})
        langchain_schema.Document = type("Document", (), {})

        pydantic = types.ModuleType("pydantic")

        class BaseModel:
            pass

        def Field(default=None, default_factory=None, **kwargs):
            if default_factory is not None:
                return default_factory()
            return default

        pydantic.BaseModel = BaseModel
        pydantic.Field = Field

        agent_actors = types.ModuleType("agent_actors")
        agent_actors.__path__ = []

        actors = types.ModuleType("agent_actors.actors")
        actors.AgentActor = types.SimpleNamespace(remote=lambda agent: agent)

        chains = types.ModuleType("agent_actors.chains")
        chains.__path__ = []

        chains_agent = types.ModuleType("agent_actors.chains.agent")
        for name in (
            "GenerateInsights",
            "MemoryStrength",
            "Synthesis",
            "WorkingMemory",
        ):
            setattr(chains_agent, name, type(name, (), {}))

        replacements = {
            "ray": build_ray_stub(),
            "langchain": langchain,
            "langchain.agents": langchain_agents,
            "langchain.chat_models.base": langchain_chat_base,
            "langchain.schema": langchain_schema,
            "pydantic": pydantic,
            "agent_actors": agent_actors,
            "agent_actors.actors": actors,
            "agent_actors.chains": chains,
            "agent_actors.chains.agent": chains_agent,
        }

        with stubbed_modules(replacements):
            return load_module(
                "agent_under_test", ROOT / "agent_actors" / "agent.py"
            )

    def test_strength_parser_accepts_prefixed_scores(self):
        agent_module = self.load_agent_module()
        agent = agent_module.Agent.__new__(agent_module.Agent)
        agent.memory_strength = types.SimpleNamespace(
            run=lambda memory_content: "Relevance: 10"
        )

        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

    def test_strength_parser_uses_full_matched_number(self):
        agent_module = self.load_agent_module()
        agent = agent_module.Agent.__new__(agent_module.Agent)
        agent.memory_strength = types.SimpleNamespace(run=lambda memory_content: "10")

        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)


if __name__ == "__main__":
    unittest.main()
