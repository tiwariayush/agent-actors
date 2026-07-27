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

from langchain.schema import AgentFinish

from agent_actors.parent import ParentAgent


class DuplicateTaskIdTests(unittest.TestCase):
    def _make_parent(self, plan_json, child_remote):
        parent = object.__new__(ParentAgent)
        parent.status = "idle"
        parent.task = ""
        parent.verbose = False
        parent.reflect_every = 10
        parent.get_context = lambda: "parent context"
        parent.pause_to_reflect = lambda: []

        class FakePlan:
            def __call__(self, inputs):
                return {"json": plan_json}

        class FakeAdjust:
            def run(self, **kwargs):
                return {
                    "confidence": 9,
                    "result": "all good",
                    "speak": "done",
                }

        class FakeRun:
            def remote(self, **kwargs):
                return child_remote(**kwargs)

        class FakeActor:
            run = FakeRun()

        class FakeChild:
            actor = FakeActor()

            def get_context(self):
                return "child context"

        parent.plan = FakePlan()
        parent.adjust = FakeAdjust()
        parent.children = {0: FakeChild()}
        return parent

    def test_duplicate_task_ids_raise_before_any_remote_submit(self):
        """Plan LLMs sometimes repeat child_id.task_id; that must not orphan work."""
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs)
            return f"result-{len(remote_calls)}"

        parent = self._make_parent(
            plan_json=[
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": "Research topic",
                    "dependencies": [],
                },
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": "Write summary",
                    "dependencies": [],
                },
            ],
            child_remote=child_remote,
        )

        with self.assertRaises(ValueError) as ctx:
            parent.run("top-level task")

        self.assertIn("0.0", str(ctx.exception))
        self.assertIn("duplicate task ids", str(ctx.exception).lower())
        self.assertEqual(remote_calls, [])
        self.assertEqual(parent.status, "idle")

    def test_unique_task_ids_still_dispatch(self):
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs["task"])
            return f"result for {kwargs['task']}"

        parent = self._make_parent(
            plan_json=[
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": "Research topic",
                    "dependencies": [],
                },
                {
                    "child_id": 0,
                    "task_id": 1,
                    "task": "Write summary",
                    "dependencies": [],
                },
            ],
            child_remote=child_remote,
        )

        result = parent.run("top-level task")

        self.assertEqual(remote_calls, ["Research topic", "Write summary"])
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(result.return_values["result"], "all good")
        self.assertEqual(parent.status, "idle")


if __name__ == "__main__":
    unittest.main()
