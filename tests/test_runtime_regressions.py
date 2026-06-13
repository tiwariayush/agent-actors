import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_dependency_stubs():
    ray = types.ModuleType("ray")

    def remote(**_options):
        return lambda cls: cls

    ray.remote = remote
    ray.ObjectRef = object
    sys.modules["ray"] = ray

    langchain = types.ModuleType("langchain")
    langchain.LLMChain = object
    sys.modules["langchain"] = langchain

    agents = types.ModuleType("langchain.agents")
    agents.Tool = object
    sys.modules["langchain.agents"] = agents

    chat_models_base = types.ModuleType("langchain.chat_models.base")
    chat_models_base.BaseChatModel = object
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

    def Field(default=None, default_factory=None, **_kwargs):
        return default_factory() if default_factory is not None else default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules["pydantic"] = pydantic

    package = types.ModuleType("agent_actors")
    package.__path__ = [str(REPO_ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = [str(REPO_ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains_package

    chains_agent = types.ModuleType("agent_actors.chains.agent")
    for class_name in ("GenerateInsights", "MemoryStrength", "Synthesis", "WorkingMemory"):
        setattr(chains_agent, class_name, type(class_name, (), {}))
    sys.modules["agent_actors.chains.agent"] = chains_agent


class RuntimeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_dependency_stubs()
        cls.actors = _load_module(
            "agent_actors.actors", REPO_ROOT / "agent_actors" / "actors.py"
        )
        cls.agent_module = _load_module(
            "agent_actors.agent", REPO_ROOT / "agent_actors" / "agent.py"
        )

    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        class Chain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = self.actors.ChainActor(Chain())

        self.assertEqual(actor.run("task", option=True), (("task",), {"option": True}))

    def test_memory_strength_uses_matched_number_from_prefixed_response(self):
        agent = self.agent_module.Agent.__new__(self.agent_module.Agent)
        agent.memory_strength = _MemoryStrength("Relevance: 8")

        self.assertEqual(agent._predict_memory_strength("important memory"), 0.8)

    def test_memory_strength_handles_ten_as_full_strength(self):
        agent = self.agent_module.Agent.__new__(self.agent_module.Agent)
        agent.memory_strength = _MemoryStrength("10")

        self.assertEqual(agent._predict_memory_strength("critical memory"), 1.0)

    def test_memory_strength_returns_zero_when_no_score_is_present(self):
        agent = self.agent_module.Agent.__new__(self.agent_module.Agent)
        agent.memory_strength = _MemoryStrength("not relevant")

        self.assertEqual(agent._predict_memory_strength("unclear memory"), 0.0)


class _MemoryStrength:
    def __init__(self, response):
        self.response = response

    def run(self, **_kwargs):
        return self.response


if __name__ == "__main__":
    unittest.main()
