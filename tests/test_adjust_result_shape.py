import sys
import types
import unittest
from unittest.mock import MagicMock


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

from agent_actors.parent import ParentAgent, normalize_adjustment


class NormalizeAdjustmentTests(unittest.TestCase):
    def test_object_passthrough(self):
        raw = {"confidence": 9, "speak": "done", "result": "ok"}
        self.assertEqual(normalize_adjustment(raw), raw)

    def test_bare_confidence_number_becomes_object(self):
        """Adjust prompt asks for confidence as a number before showing schema."""
        self.assertEqual(
            normalize_adjustment(9),
            {"confidence": 9, "speak": "", "result": "9"},
        )
        self.assertEqual(
            normalize_adjustment(7.5),
            {"confidence": 7.5, "speak": "", "result": "7.5"},
        )

    def test_one_element_array_unwraps_to_object(self):
        """Models often wrap a single Adjust object in a one-element array."""
        raw = [{"confidence": 8, "speak": "ready", "result": "ship it"}]
        self.assertEqual(normalize_adjustment(raw), raw[0])

    def test_one_element_array_of_number_unwraps(self):
        self.assertEqual(
            normalize_adjustment([9]),
            {"confidence": 9, "speak": "", "result": "9"},
        )

    def test_multi_element_array_rejected(self):
        with self.assertRaises(TypeError):
            normalize_adjustment(
                [
                    {"confidence": 8, "speak": "a", "result": "a"},
                    {"confidence": 9, "speak": "b", "result": "b"},
                ]
            )

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            normalize_adjustment(True)

    def test_null_rejected(self):
        with self.assertRaises(TypeError):
            normalize_adjustment(None)


class ParentAdjustShapeTests(unittest.TestCase):
    def _make_parent(self, adjustment):
        parent = object.__new__(ParentAgent)
        parent.status = "idle"
        parent.task = ""
        parent.verbose = False
        parent.reflect_every = 10
        parent.children = {}
        parent.get_context = lambda: "context"
        parent.pause_to_reflect = MagicMock()
        parent.plan = MagicMock(
            return_value={
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "Research AGI",
                        "dependencies": [],
                    }
                ]
            }
        )

        child = MagicMock()
        child.get_context = MagicMock(return_value="child context")
        child.actor = MagicMock()
        child.actor.run = MagicMock()
        child.actor.run.remote = MagicMock(return_value="child-result-ref")
        parent.children = {0: child}

        parent.adjust = MagicMock()
        parent.adjust.run = MagicMock(return_value=adjustment)
        return parent

    def test_parent_accepts_bare_confidence_number(self):
        parent = self._make_parent(9)
        result = parent.run("Ship the report")
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(result.return_values["confidence"], 9)
        self.assertEqual(parent.status, "idle")

    def test_parent_accepts_one_element_adjust_array(self):
        parent = self._make_parent(
            [{"confidence": 9, "speak": "done", "result": "final"}]
        )
        result = parent.run("Ship the report")
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(result.return_values["result"], "final")
        self.assertEqual(parent.status, "idle")

    def test_parent_low_bare_confidence_asks_human(self):
        parent = self._make_parent(4)
        result = parent.run("Ship the report")
        self.assertIsInstance(result, AgentAction)
        self.assertEqual(result.tool, "Human")


if __name__ == "__main__":
    unittest.main()
