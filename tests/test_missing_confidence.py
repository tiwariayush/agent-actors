import sys
import types
import unittest


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

    # Stub Pydantic so object.__new__(ParentAgent) can set attributes in tests.
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

from agent_actors.parent import ParentAgent


def _parent_with_adjust(adjustment):
    parent = object.__new__(ParentAgent)
    parent.status = "idle"
    parent.task = ""
    parent.verbose = False
    parent.reflect_every = 10
    parent.children = {}

    class FakePlan:
        def __call__(self, inputs):
            return {
                "json": [
                    {
                        "child_id": 0,
                        "task_id": 0,
                        "task": "do work",
                        "dependencies": [],
                    }
                ]
            }

    class FakeAdjust:
        def run(self, **kwargs):
            return adjustment

    class FakeRun:
        def remote(self, **kwargs):
            return "child result"

    class FakeActor:
        run = FakeRun()

    class FakeChild:
        actor = FakeActor()

        def get_context(self):
            return "child context"

    parent.plan = FakePlan()
    parent.adjust = FakeAdjust()
    parent.children = {0: FakeChild()}
    parent.get_context = lambda: "parent context"
    parent.pause_to_reflect = lambda: []
    return parent


class MissingConfidenceTests(unittest.TestCase):
    def test_missing_confidence_asks_human_instead_of_keyerror(self):
        """Adjust sometimes returns speak/result without confidence."""
        parent = _parent_with_adjust(
            {"speak": "Need another pass", "result": "partial findings"}
        )

        result = parent.run("top-level task")

        self.assertIsInstance(result, AgentAction)
        self.assertEqual(result.tool, "Human")
        self.assertEqual(parent.status, "idle")

    def test_null_confidence_asks_human_instead_of_typeerror(self):
        parent = _parent_with_adjust(
            {"confidence": None, "speak": "unsure", "result": "maybe"}
        )

        result = parent.run("top-level task")

        self.assertIsInstance(result, AgentAction)
        self.assertEqual(result.tool, "Human")

    def test_numeric_confidence_still_finishes(self):
        parent = _parent_with_adjust(
            {"confidence": 9, "speak": "done", "result": "ship it"}
        )

        result = parent.run("top-level task")

        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(result.return_values["result"], "ship it")

    def test_string_confidence_still_finishes(self):
        """Same call site as draft PR #70; keep coercion so missing-key fix is complete."""
        parent = _parent_with_adjust(
            {"confidence": "8", "speak": "done", "result": "all good"}
        )

        result = parent.run("top-level task")

        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(result.return_values["result"], "all good")

    def test_empty_adjustment_object_asks_human(self):
        parent = _parent_with_adjust({})

        result = parent.run("top-level task")

        self.assertIsInstance(result, AgentAction)
        self.assertEqual(result.tool, "Human")


if __name__ == "__main__":
    unittest.main()
