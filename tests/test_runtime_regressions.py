import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_dependency_stubs():
    def remote(*decorator_args, **decorator_kwargs):
        if decorator_args and len(decorator_args) == 1 and callable(decorator_args[0]):
            return decorator_args[0]

        def decorate(cls):
            return cls

        return decorate

    ray = types.ModuleType("ray")
    ray.remote = remote
    ray.ObjectRef = object
    sys.modules["ray"] = ray

    langchain = types.ModuleType("langchain")
    langchain.LLMChain = type("LLMChain", (), {})
    sys.modules["langchain"] = langchain

    langchain_agents = types.ModuleType("langchain.agents")
    langchain_agents.Tool = type("Tool", (), {})
    sys.modules["langchain.agents"] = langchain_agents

    langchain_chat_models = types.ModuleType("langchain.chat_models")
    sys.modules["langchain.chat_models"] = langchain_chat_models

    langchain_chat_models_base = types.ModuleType("langchain.chat_models.base")
    langchain_chat_models_base.BaseChatModel = type("BaseChatModel", (), {})
    sys.modules["langchain.chat_models.base"] = langchain_chat_models_base

    langchain_schema = types.ModuleType("langchain.schema")
    langchain_schema.BaseRetriever = type("BaseRetriever", (), {})
    langchain_schema.Document = type("Document", (), {})
    sys.modules["langchain.schema"] = langchain_schema

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

    agent_actors = types.ModuleType("agent_actors")
    agent_actors.__path__ = [str(REPO_ROOT / "agent_actors")]
    sys.modules["agent_actors"] = agent_actors

    chains = types.ModuleType("agent_actors.chains")
    chains.__path__ = [str(REPO_ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains

    chains_agent = types.ModuleType("agent_actors.chains.agent")
    for name in ("GenerateInsights", "MemoryStrength", "Synthesis", "WorkingMemory"):
        setattr(chains_agent, name, type(name, (), {}))
    sys.modules["agent_actors.chains.agent"] = chains_agent


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_install_dependency_stubs()
actors_module = _load_module("agent_actors.actors", "agent_actors/actors.py")
agent_module = _load_module("agent_actors.agent", "agent_actors/agent.py")


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        class Chain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain result"

        chain = Chain()
        actor = actors_module.ChainActor(chain)

        self.assertEqual(
            actor.run("task", context="memory"),
            "chain result",
        )
        self.assertEqual(chain.calls, [(("task",), {"context": "memory"})])

    def test_predict_memory_strength_parses_prefixed_multi_digit_scores(self):
        class MemoryStrength:
            def run(self, memory_content):
                self.memory_content = memory_content
                return "Relevance: 10"

        agent = types.SimpleNamespace(memory_strength=MemoryStrength())

        self.assertEqual(
            agent_module.Agent._predict_memory_strength(agent, "important memory"),
            1.0,
        )
        self.assertEqual(agent.memory_strength.memory_content, "important memory")


if __name__ == "__main__":
    unittest.main()
