import sys
import types
import unittest
from unittest.mock import patch


def install_dependency_stubs():
    ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(cls):
            return cls

        return decorator

    def get(refs):
        return refs

    def wait(refs, num_returns=1):
        return refs[:num_returns], refs[num_returns:]

    ray.remote = remote
    ray.get = get
    ray.wait = wait
    ray.ObjectRef = object
    sys.modules["ray"] = ray

    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, *args, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        if default is ...:
            return None
        return default

    pydantic.BaseModel = BaseModel
    pydantic.Field = Field
    sys.modules["pydantic"] = pydantic

    langchain = types.ModuleType("langchain")

    class LLMChain:
        @classmethod
        def from_llm(cls, **kwargs):
            return cls()

    class MRKLChain:
        @classmethod
        def from_agent_and_tools(cls, *args, **kwargs):
            return cls()

    class PromptTemplate:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_template(cls, *args, **kwargs):
            return cls()

    langchain.LLMChain = LLMChain
    langchain.MRKLChain = MRKLChain
    langchain.PromptTemplate = PromptTemplate
    sys.modules["langchain"] = langchain

    callbacks = types.ModuleType("langchain.callbacks")

    class StdOutCallbackHandler:
        def on_chain_end(self, outputs, **kwargs):
            pass

        def on_chain_start(self, serialized, inputs, **kwargs):
            pass

    class CallbackManager:
        def __init__(self, handlers=None):
            self.handlers = list(handlers or [])

    callbacks.CallbackManager = CallbackManager
    callbacks.StdOutCallbackHandler = StdOutCallbackHandler
    sys.modules["langchain.callbacks"] = callbacks

    agents = types.ModuleType("langchain.agents")

    class Tool:
        pass

    class ZeroShotAgent:
        @classmethod
        def from_llm_and_tools(cls, *args, **kwargs):
            return cls()

    agents.Tool = Tool
    agents.ZeroShotAgent = ZeroShotAgent
    sys.modules["langchain.agents"] = agents

    chat_models_base = types.ModuleType("langchain.chat_models.base")

    class BaseChatModel:
        pass

    chat_models_base.BaseChatModel = BaseChatModel
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
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    schema.AgentAction = AgentAction
    schema.AgentFinish = AgentFinish
    schema.BaseRetriever = BaseRetriever
    schema.Document = Document
    sys.modules["langchain.schema"] = schema


install_dependency_stubs()

from langchain.schema import AgentAction, AgentFinish

from agent_actors.agent import Agent
from agent_actors.actors import ChainActor
from agent_actors.parent import ParentAgent
from agent_actors.results import format_agent_result


class RuntimeRegressionTests(unittest.TestCase):
    def test_chain_actor_run_delegates_to_chain(self):
        class FakeChain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = ChainActor(FakeChain())

        self.assertEqual(
            actor.run("task", answer=True),
            (("task",), {"answer": True}),
        )

    def test_memory_strength_parses_labelled_and_two_digit_scores(self):
        agent = object.__new__(Agent)

        class FakeStrengthChain:
            def __init__(self, score):
                self.score = score

            def run(self, **kwargs):
                return self.score

        agent.memory_strength = FakeStrengthChain("Relevance: 8")
        self.assertEqual(agent._predict_memory_strength("memory"), 0.8)

        agent.memory_strength = FakeStrengthChain("10")
        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

    def test_agent_results_are_prompt_safe_text(self):
        self.assertEqual(
            format_agent_result(AgentFinish({"result": "final answer"}, "success")),
            "final answer",
        )
        self.assertEqual(
            format_agent_result(AgentFinish({"output": "tool output"}, "success")),
            "tool output",
        )
        self.assertEqual(
            format_agent_result(AgentAction("Search", "cats", "Need to search")),
            "Need to search",
        )

    def test_parent_run_normalizes_nested_parent_results(self):
        parent = object.__new__(ParentAgent)
        parent.status = "idle"
        parent.task = ""
        parent.verbose = False
        parent.reflect_every = 10

        class FakePlan:
            def __call__(self, inputs):
                self.inputs = inputs
                return {
                    "json": [
                        {
                            "child_id": 42,
                            "task_id": 0,
                            "task": "delegate",
                            "dependencies": [],
                        }
                    ]
                }

        class FakeAdjust:
            def run(self, **kwargs):
                self.inputs = kwargs
                return {
                    "confidence": 8,
                    "result": "complete",
                    "speak": "complete",
                }

        class FakeRun:
            def remote(self, **kwargs):
                return AgentFinish({"result": "nested result"}, "success")

        class FakeActor:
            run = FakeRun()

        class FakeChild:
            actor = FakeActor()

            def get_context(self):
                return "child context"

        parent.plan = FakePlan()
        parent.adjust = FakeAdjust()
        parent.children = {42: FakeChild()}
        parent.get_context = lambda: "parent context"
        parent.pause_to_reflect = lambda: []

        result = parent.run("top-level task")

        self.assertIsInstance(result, AgentFinish)
        self.assertIn("ID: 42", parent.plan.inputs["child_summary"])
        self.assertEqual(parent.adjust.inputs["results"], "nested result")

    def test_parent_creates_planned_child_under_requested_id(self):
        parent = object.__new__(ParentAgent)
        parent.status = "idle"
        parent.task = ""
        parent.verbose = False
        parent.reflect_every = 10
        parent.children = {}
        parent.llm = object()
        parent.tools = ["search"]
        parent.long_term_memory = object()

        class FakePlan:
            def __call__(self, inputs):
                return {
                    "json": [
                        {
                            "child_id": 42,
                            "task_id": 0,
                            "task": "delegate",
                            "dependencies": [],
                        }
                    ]
                }

        class FakeAdjust:
            def run(self, **kwargs):
                return {"confidence": 8, "result": "complete"}

        class FakeRun:
            def remote(self, **kwargs):
                return "child result"

        class FakeActor:
            run = FakeRun()

        class FakeChild:
            actor = FakeActor()

        child_kwargs = {}

        def create_child(**kwargs):
            child_kwargs.update(kwargs)
            return FakeChild()

        parent.plan = FakePlan()
        parent.adjust = FakeAdjust()
        parent.get_context = lambda: "parent context"
        parent.pause_to_reflect = lambda: []

        with patch("agent_actors.parent.ChildAgent", side_effect=create_child):
            result = parent.run("top-level task")

        self.assertIsInstance(result, AgentFinish)
        self.assertIn(42, parent.children)
        self.assertIs(child_kwargs["llm"], parent.llm)
        self.assertEqual(child_kwargs["tools"], parent.tools)
        self.assertIs(child_kwargs["long_term_memory"], parent.long_term_memory)


if __name__ == "__main__":
    unittest.main()
