import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class RegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        ray = types.ModuleType("ray")
        ray.remote = lambda **_kwargs: lambda cls: cls

        previous_ray = sys.modules.get("ray")
        sys.modules["ray"] = ray
        try:
            actors = load_module("actors_under_test", ROOT / "agent_actors" / "actors.py")
        finally:
            if previous_ray is None:
                sys.modules.pop("ray", None)
            else:
                sys.modules["ray"] = previous_ray

        class Chain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = actors.ChainActor(Chain())

        self.assertEqual(
            actor.run("task", priority="high"),
            (("task",), {"priority": "high"}),
        )

    def test_memory_strength_uses_first_number_in_model_response(self):
        self._install_agent_import_stubs()
        agent_module = load_module(
            "agent_under_test", ROOT / "agent_actors" / "agent.py"
        )

        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **_kwargs):
                return self.response

        agent = types.SimpleNamespace(memory_strength=MemoryStrength("Relevance: 8"))
        self.assertEqual(agent_module.Agent._predict_memory_strength(agent, "memory"), 0.8)

        agent.memory_strength = MemoryStrength("10")
        self.assertEqual(agent_module.Agent._predict_memory_strength(agent, "memory"), 1.0)

    def _install_agent_import_stubs(self):
        ray = types.ModuleType("ray")
        ray.ObjectRef = object

        langchain = types.ModuleType("langchain")
        langchain.LLMChain = type("LLMChain", (), {})

        langchain_agents = types.ModuleType("langchain.agents")
        langchain_agents.Tool = type("Tool", (), {})

        langchain_chat_models = types.ModuleType("langchain.chat_models")
        langchain_chat_models_base = types.ModuleType("langchain.chat_models.base")
        langchain_chat_models_base.BaseChatModel = type("BaseChatModel", (), {})

        langchain_schema = types.ModuleType("langchain.schema")
        langchain_schema.BaseRetriever = type("BaseRetriever", (), {})
        langchain_schema.Document = type("Document", (), {})

        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = type("BaseModel", (), {})
        pydantic.Field = self._field_stub

        actors = types.ModuleType("agent_actors.actors")
        actors.AgentActor = type("AgentActor", (), {"remote": classmethod(lambda cls, agent: agent)})

        chains_agent = types.ModuleType("agent_actors.chains.agent")
        for name in ("GenerateInsights", "MemoryStrength", "Synthesis", "WorkingMemory"):
            setattr(chains_agent, name, type(name, (), {}))

        sys.modules.update(
            {
                "ray": ray,
                "langchain": langchain,
                "langchain.agents": langchain_agents,
                "langchain.chat_models": langchain_chat_models,
                "langchain.chat_models.base": langchain_chat_models_base,
                "langchain.schema": langchain_schema,
                "pydantic": pydantic,
                "agent_actors.actors": actors,
                "agent_actors.chains.agent": chains_agent,
            }
        )

    @staticmethod
    def _field_stub(default=None, default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default


if __name__ == "__main__":
    unittest.main()
