import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def install_dependency_stubs():
    ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(obj):
            return obj

        return decorator

    ray.remote = remote
    ray.ObjectRef = object
    ray.get = lambda refs: refs
    ray.wait = lambda refs, num_returns=1: (refs[:num_returns], refs[num_returns:])
    sys.modules["ray"] = ray

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

    langchain = types.ModuleType("langchain")

    class LLMChain:
        pass

    langchain.LLMChain = LLMChain
    sys.modules["langchain"] = langchain

    agents = types.ModuleType("langchain.agents")

    class Tool:
        pass

    agents.Tool = Tool
    sys.modules["langchain.agents"] = agents

    chat_models = types.ModuleType("langchain.chat_models")
    chat_models_base = types.ModuleType("langchain.chat_models.base")

    class BaseChatModel:
        pass

    chat_models_base.BaseChatModel = BaseChatModel
    sys.modules["langchain.chat_models"] = chat_models
    sys.modules["langchain.chat_models.base"] = chat_models_base

    schema = types.ModuleType("langchain.schema")

    class AgentAction:
        def __init__(self, tool, tool_input, log):
            self.tool = tool
            self.tool_input = tool_input
            self.log = log

    class AgentFinish:
        def __init__(self, return_values, log):
            self.return_values = return_values
            self.log = log

    class BaseRetriever:
        pass

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    schema.AgentAction = AgentAction
    schema.AgentFinish = AgentFinish
    schema.BaseRetriever = BaseRetriever
    schema.Document = Document
    sys.modules["langchain.schema"] = schema

    package = types.ModuleType("agent_actors")
    package.__path__ = [str(ROOT / "agent_actors")]
    sys.modules["agent_actors"] = package

    chains_package = types.ModuleType("agent_actors.chains")
    chains_package.__path__ = [str(ROOT / "agent_actors" / "chains")]
    sys.modules["agent_actors.chains"] = chains_package

    chains_agent = types.ModuleType("agent_actors.chains.agent")

    class ChainFactory:
        @classmethod
        def from_llm(cls, **kwargs):
            return cls()

    chains_agent.GenerateInsights = ChainFactory
    chains_agent.MemoryStrength = ChainFactory
    chains_agent.Synthesis = ChainFactory
    chains_agent.WorkingMemory = ChainFactory
    sys.modules["agent_actors.chains.agent"] = chains_agent

    chains_parent = types.ModuleType("agent_actors.chains.parent")
    chains_parent.Adjust = ChainFactory
    chains_parent.Plan = ChainFactory
    sys.modules["agent_actors.chains.parent"] = chains_parent

    child = types.ModuleType("agent_actors.child")

    class ChildAgent:
        pass

    child.ChildAgent = ChildAgent
    sys.modules["agent_actors.child"] = child


def load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


install_dependency_stubs()
actors_module = load_module("agent_actors.actors", "agent_actors/actors.py")
agent_module = load_module("agent_actors.agent", "agent_actors/agent.py")
parent_module = load_module("agent_actors.parent", "agent_actors/parent.py")


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        class FakeChain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = actors_module.ChainActor(FakeChain())

        self.assertEqual(actor.run("task", option=True), (("task",), {"option": True}))

    def test_memory_strength_uses_matched_numeric_group(self):
        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, memory_content):
                return self.response

        agent = agent_module.Agent.__new__(agent_module.Agent)

        agent.memory_strength = MemoryStrength("Relevance: 8")
        self.assertEqual(agent._predict_memory_strength("memory"), 0.8)

        agent.memory_strength = MemoryStrength("Score: 10")
        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

        agent.memory_strength = MemoryStrength("not relevant")
        self.assertEqual(agent._predict_memory_strength("memory"), 0.0)

    def test_parent_formats_agent_finish_working_memory(self):
        AgentFinish = sys.modules["langchain.schema"].AgentFinish

        captured = {}

        class Plan:
            def __call__(self, inputs):
                captured["context"] = inputs["context"]
                return {"json": []}

        class Adjust:
            def run(self, **kwargs):
                captured["results"] = kwargs["results"]
                return {"confidence": 8, "result": "complete"}

        parent = parent_module.ParentAgent.__new__(parent_module.ParentAgent)
        parent.status = "idle"
        parent.task = ""
        parent.verbose = False
        parent.children = {}
        parent.reflect_every = 10
        parent.plan = Plan()
        parent.adjust = Adjust()
        parent.get_context = lambda: "base context"
        parent.pause_to_reflect = lambda: None

        result = parent_module.ParentAgent.run(
            parent,
            task="top-level task",
            working_memory=[AgentFinish({"result": "dependency complete"}, "done")],
        )

        self.assertEqual(captured["context"], "base context\ndependency complete")
        self.assertEqual(captured["results"], "")
        self.assertIsInstance(result, AgentFinish)


if __name__ == "__main__":
    unittest.main()
