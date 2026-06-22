import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def patched_modules(modules):
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ray_stub():
    def remote(*args, **kwargs):
        def decorator(cls):
            return cls

        return decorator

    return types.SimpleNamespace(ObjectRef=object, remote=remote)


class RuntimeRegressionTest(unittest.TestCase):
    def test_chain_actor_run_delegates_to_chain(self):
        with patched_modules({"ray": ray_stub()}):
            actors = load_module("actors_under_test", "agent_actors/actors.py")

        class Chain:
            def run(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

        actor = actors.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", context="memory"),
            {"args": ("task",), "kwargs": {"context": "memory"}},
        )

    def test_memory_strength_uses_matched_score_digits(self):
        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = object
        pydantic.Field = lambda default=None, **kwargs: (
            kwargs["default_factory"]() if "default_factory" in kwargs else default
        )

        langchain = types.ModuleType("langchain")
        langchain.LLMChain = object

        langchain_agents = types.ModuleType("langchain.agents")
        langchain_agents.Tool = object

        chat_models = types.ModuleType("langchain.chat_models")
        chat_models_base = types.ModuleType("langchain.chat_models.base")
        chat_models_base.BaseChatModel = object

        schema = types.ModuleType("langchain.schema")
        schema.BaseRetriever = object
        schema.Document = object

        agent_actors = types.ModuleType("agent_actors")
        agent_actors.__path__ = [str(ROOT / "agent_actors")]

        actors = types.ModuleType("agent_actors.actors")

        class AgentActor:
            @classmethod
            def remote(cls, agent):
                return None

        actors.AgentActor = AgentActor

        chains = types.ModuleType("agent_actors.chains")
        chains.__path__ = [str(ROOT / "agent_actors/chains")]
        chains_agent = types.ModuleType("agent_actors.chains.agent")
        chains_agent.GenerateInsights = object
        chains_agent.MemoryStrength = object
        chains_agent.Synthesis = object
        chains_agent.WorkingMemory = object

        modules = {
            "ray": ray_stub(),
            "pydantic": pydantic,
            "langchain": langchain,
            "langchain.agents": langchain_agents,
            "langchain.chat_models": chat_models,
            "langchain.chat_models.base": chat_models_base,
            "langchain.schema": schema,
            "agent_actors": agent_actors,
            "agent_actors.actors": actors,
            "agent_actors.chains": chains,
            "agent_actors.chains.agent": chains_agent,
        }

        with patched_modules(modules):
            agent_module = load_module("agent_under_test", "agent_actors/agent.py")

        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **kwargs):
                return self.response

        agent = types.SimpleNamespace(memory_strength=MemoryStrength("Relevance: 8"))
        self.assertEqual(agent_module.Agent._predict_memory_strength(agent, "memory"), 0.8)

        agent.memory_strength = MemoryStrength("10")
        self.assertEqual(agent_module.Agent._predict_memory_strength(agent, "memory"), 1.0)

        agent.memory_strength = MemoryStrength("no score")
        self.assertEqual(agent_module.Agent._predict_memory_strength(agent, "memory"), 0.0)


if __name__ == "__main__":
    unittest.main()
