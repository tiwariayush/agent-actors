import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def fake_ray_module():
    ray = types.ModuleType("ray")
    ray.ObjectRef = object

    def remote(*decorator_args, **decorator_kwargs):
        if decorator_args and len(decorator_args) == 1 and not decorator_kwargs:
            return decorator_args[0]

        def decorate(obj):
            return obj

        return decorate

    ray.remote = remote
    return ray


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        with patch.dict(sys.modules, {"ray": fake_ray_module()}):
            actors = load_module(
                "agent_actors_actors_under_test", "agent_actors/actors.py"
            )

        class FakeChain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        actor = actors.ChainActor(FakeChain())

        self.assertEqual(
            actor.run("task", option=True),
            {"args": ("task",), "kwargs": {"option": True}},
        )

    def test_memory_strength_uses_matched_numeric_score(self):
        module_stubs = self._agent_module_stubs()
        with patch.dict(sys.modules, module_stubs):
            agent_module = load_module(
                "agent_actors.agent_under_test", "agent_actors/agent.py"
            )

        agent = object.__new__(agent_module.Agent)

        for raw_score, expected in (
            ("10", 1.0),
            ("Relevance: 10", 1.0),
            ("2", 0.2),
            ("no score", 0.0),
        ):
            with self.subTest(raw_score=raw_score):
                agent.memory_strength = FakeMemoryStrength(raw_score)
                self.assertEqual(
                    agent._predict_memory_strength("important memory"),
                    expected,
                )

    def _agent_module_stubs(self):
        class DummyBaseModel:
            def __init__(self, *args, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class DummyChain:
            @classmethod
            def from_llm(cls, **kwargs):
                return cls()

        def field(default=None, default_factory=None, **kwargs):
            return default_factory() if default_factory else default

        langchain = types.ModuleType("langchain")
        langchain.LLMChain = DummyChain

        agents = types.ModuleType("langchain.agents")
        agents.Tool = object

        chat_models = types.ModuleType("langchain.chat_models")
        chat_models_base = types.ModuleType("langchain.chat_models.base")
        chat_models_base.BaseChatModel = object

        schema = types.ModuleType("langchain.schema")
        schema.BaseRetriever = object
        schema.Document = object

        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = DummyBaseModel
        pydantic.Field = field

        agent_actors = types.ModuleType("agent_actors")
        agent_actors.__path__ = []

        actors = types.ModuleType("agent_actors.actors")
        actors.AgentActor = DummyAgentActor

        chains = types.ModuleType("agent_actors.chains")
        chains.__path__ = []

        chains_agent = types.ModuleType("agent_actors.chains.agent")
        chains_agent.GenerateInsights = DummyChain
        chains_agent.MemoryStrength = DummyChain
        chains_agent.Synthesis = DummyChain
        chains_agent.WorkingMemory = DummyChain

        return {
            "ray": fake_ray_module(),
            "langchain": langchain,
            "langchain.agents": agents,
            "langchain.chat_models": chat_models,
            "langchain.chat_models.base": chat_models_base,
            "langchain.schema": schema,
            "pydantic": pydantic,
            "agent_actors": agent_actors,
            "agent_actors.actors": actors,
            "agent_actors.chains": chains,
            "agent_actors.chains.agent": chains_agent,
        }


class DummyAgentActor:
    @classmethod
    def remote(cls, *args, **kwargs):
        return cls()


class FakeMemoryStrength:
    def __init__(self, response: str):
        self.response = response

    def run(self, **kwargs):
        return self.response


if __name__ == "__main__":
    unittest.main()
