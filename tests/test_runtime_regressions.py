import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _install_external_stubs():
    ray = types.ModuleType("ray")

    def remote(*decorator_args, **decorator_kwargs):
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

    langchain = types.ModuleType("langchain")

    class LLMChain:
        pass

    langchain.LLMChain = LLMChain
    sys.modules["langchain"] = langchain

    agents = types.ModuleType("langchain.agents")
    agents.Tool = object
    sys.modules["langchain.agents"] = agents

    chat_models = types.ModuleType("langchain.chat_models")
    chat_models_base = types.ModuleType("langchain.chat_models.base")
    chat_models_base.BaseChatModel = object
    sys.modules["langchain.chat_models"] = chat_models
    sys.modules["langchain.chat_models.base"] = chat_models_base

    schema = types.ModuleType("langchain.schema")
    schema.BaseRetriever = object
    schema.Document = object
    sys.modules["langchain.schema"] = schema

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules["pydantic"] = pydantic


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_actor_module():
    _install_external_stubs()
    return _load_module("agent_actors.actors", ROOT / "agent_actors" / "actors.py")


def _load_agent_module():
    _install_external_stubs()

    package = types.ModuleType("agent_actors")
    package.__path__ = [str(ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package
    _load_actor_module()

    chains = types.ModuleType("agent_actors.chains")
    chains.__path__ = [str(ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains

    chains_agent = types.ModuleType("agent_actors.chains.agent")
    chains_agent.GenerateInsights = object
    chains_agent.MemoryStrength = object
    chains_agent.Synthesis = object
    chains_agent.WorkingMemory = object
    sys.modules["agent_actors.chains.agent"] = chains_agent

    return _load_module("agent_actors.agent", ROOT / "agent_actors" / "agent.py")


class ChainActorTests(unittest.TestCase):
    def test_run_delegates_to_wrapped_chain(self):
        actors = _load_actor_module()

        class Chain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        actor = actors.ChainActor(Chain())

        self.assertEqual(
            actor.run("prompt", temperature=0),
            {"args": ("prompt",), "kwargs": {"temperature": 0}},
        )


class MemoryStrengthTests(unittest.TestCase):
    def _agent_with_score(self, score):
        agent_module = _load_agent_module()
        agent = agent_module.Agent.__new__(agent_module.Agent)

        class MemoryStrength:
            def run(self, **kwargs):
                return score

        agent.memory_strength = MemoryStrength()
        return agent

    def test_prefixed_memory_strength_score_is_parsed(self):
        agent = self._agent_with_score("Relevance: 8")

        self.assertEqual(agent._predict_memory_strength("memory"), 0.8)

    def test_two_digit_memory_strength_score_is_not_truncated(self):
        agent = self._agent_with_score("10")

        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

    def test_unparseable_memory_strength_score_defaults_to_zero(self):
        agent = self._agent_with_score("not relevant")

        self.assertEqual(agent._predict_memory_strength("memory"), 0.0)


if __name__ == "__main__":
    unittest.main()
