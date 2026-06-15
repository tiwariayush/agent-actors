import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_ray_stub():
    def remote(*decorator_args, **decorator_kwargs):
        if decorator_args and len(decorator_args) == 1 and isinstance(decorator_args[0], type):
            return decorator_args[0]

        def decorate(cls):
            return cls

        return decorate

    sys.modules["ray"] = types.SimpleNamespace(remote=remote, ObjectRef=object)


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        _install_ray_stub()
        actors = _load_module("agent_actors.actors", "agent_actors/actors.py")

        class DummyChain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain-result"

        chain = DummyChain()
        actor = actors.ChainActor(chain)

        self.assertEqual(actor.run("input", mode="fast"), "chain-result")
        self.assertEqual(chain.calls, [(("input",), {"mode": "fast"})])

    def test_memory_strength_uses_parsed_numeric_score(self):
        self._install_agent_dependencies()
        agent_module = _load_module("agent_actors.agent", "agent_actors/agent.py")
        agent = object.__new__(agent_module.Agent)

        for raw_score, expected in [
            ("Relevance: 8", 0.8),
            ("10", 1.0),
            ("Score = 10 out of 10", 1.0),
        ]:
            with self.subTest(raw_score=raw_score):
                agent.memory_strength = types.SimpleNamespace(
                    run=lambda **kwargs: raw_score
                )

                self.assertEqual(agent._predict_memory_strength("memory"), expected)

    def _install_agent_dependencies(self):
        _install_ray_stub()

        package = types.ModuleType("agent_actors")
        package.__path__ = [str(ROOT / "agent_actors")]
        sys.modules["agent_actors"] = package
        _load_module("agent_actors.actors", "agent_actors/actors.py")

        chains_package = types.ModuleType("agent_actors.chains")
        chains_package.__path__ = [str(ROOT / "agent_actors" / "chains")]
        sys.modules["agent_actors.chains"] = chains_package

        chains_agent = types.ModuleType("agent_actors.chains.agent")
        for class_name in [
            "GenerateInsights",
            "MemoryStrength",
            "Synthesis",
            "WorkingMemory",
        ]:
            setattr(chains_agent, class_name, type(class_name, (), {}))
        sys.modules["agent_actors.chains.agent"] = chains_agent

        langchain = types.ModuleType("langchain")
        langchain.LLMChain = type("LLMChain", (), {})
        sys.modules["langchain"] = langchain

        langchain_agents = types.ModuleType("langchain.agents")
        langchain_agents.Tool = type("Tool", (), {})
        sys.modules["langchain.agents"] = langchain_agents

        chat_models_base = types.ModuleType("langchain.chat_models.base")
        chat_models_base.BaseChatModel = type("BaseChatModel", (), {})
        sys.modules["langchain.chat_models.base"] = chat_models_base
        sys.modules["langchain.chat_models"] = types.ModuleType("langchain.chat_models")

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


if __name__ == "__main__":
    unittest.main()
