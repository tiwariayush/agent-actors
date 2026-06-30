import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def stubbed_modules(**modules):
    previous = {name: sys.modules.get(name) for name in modules}
    missing = {name for name, value in previous.items() if value is None}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in missing:
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module


def ray_stub():
    ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and len(args) == 1 and isinstance(args[0], type) and not kwargs:
            return args[0]

        def decorate(cls):
            return cls

        return decorate

    ray.remote = remote
    ray.ObjectRef = object
    return ray


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        with stubbed_modules(ray=ray_stub()):
            actors = load_module("actors_under_test", "agent_actors/actors.py")

        class FakeChain:
            def __init__(self):
                self.calls = []

            def run(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "chain result"

        chain = FakeChain()
        actor = actors.ChainActor(chain)

        self.assertEqual(actor.run("task", retries=2), "chain result")
        self.assertEqual(chain.calls, [(("task",), {"retries": 2})])

    def test_predict_memory_strength_accepts_labelled_scores(self):
        agent_module = load_agent_module()
        agent = object.__new__(agent_module.Agent)
        agent.memory_strength = FixedStrength("Relevance: 8")

        self.assertEqual(agent._predict_memory_strength("memory"), 0.8)

    def test_predict_memory_strength_accepts_ten(self):
        agent_module = load_agent_module()
        agent = object.__new__(agent_module.Agent)
        agent.memory_strength = FixedStrength("10")

        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

    def test_predict_memory_strength_defaults_unparseable_scores_to_zero(self):
        agent_module = load_agent_module()
        agent = object.__new__(agent_module.Agent)
        agent.memory_strength = FixedStrength("not relevant")

        self.assertEqual(agent._predict_memory_strength("memory"), 0.0)


class FixedStrength:
    def __init__(self, response):
        self.response = response

    def run(self, **kwargs):
        return self.response


def load_agent_module():
    langchain = types.ModuleType("langchain")
    langchain.LLMChain = type("LLMChain", (), {})

    agents = types.ModuleType("langchain.agents")
    agents.Tool = type("Tool", (), {})

    chat_models = types.ModuleType("langchain.chat_models")
    chat_models_base = types.ModuleType("langchain.chat_models.base")
    chat_models_base.BaseChatModel = type("BaseChatModel", (), {})

    schema = types.ModuleType("langchain.schema")
    schema.BaseRetriever = type("BaseRetriever", (), {})
    schema.Document = type("Document", (), {})

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = field

    package = types.ModuleType("agent_actors")
    package.__path__ = []

    actors = types.ModuleType("agent_actors.actors")
    actors.AgentActor = type("AgentActor", (), {"remote": staticmethod(lambda agent: agent)})

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = []
    chains_agent = types.ModuleType("agent_actors.chains.agent")
    for name in ("GenerateInsights", "MemoryStrength", "Synthesis", "WorkingMemory"):
        setattr(chains_agent, name, type(name, (), {}))

    with stubbed_modules(
        ray=ray_stub(),
        langchain=langchain,
        **{
            "langchain.agents": agents,
            "langchain.chat_models": chat_models,
            "langchain.chat_models.base": chat_models_base,
            "langchain.schema": schema,
            "pydantic": pydantic,
            "agent_actors": package,
            "agent_actors.actors": actors,
            "agent_actors.chains": chains_package,
            "agent_actors.chains.agent": chains_agent,
        },
    ):
        return load_module("agent_under_test", "agent_actors/agent.py")


if __name__ == "__main__":
    unittest.main()
