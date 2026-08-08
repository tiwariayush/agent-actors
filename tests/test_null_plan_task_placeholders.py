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

from agent_actors.parent import (
    ParentAgent,
    drop_null_plan_task_placeholders,
)


class DropNullPlanTaskPlaceholdersTests(unittest.TestCase):
    def test_null_document_becomes_empty_list(self):
        """Models return JSON null when they decide no sub-tasks are needed."""
        self.assertEqual(drop_null_plan_task_placeholders(None), [])

    def test_null_item_in_plan_array_dropped(self):
        """Prompt-shaped array with a null placeholder means no task."""
        self.assertEqual(drop_null_plan_task_placeholders([None]), [])

    def test_empty_object_placeholder_dropped(self):
        self.assertEqual(drop_null_plan_task_placeholders([{}]), [])

    def test_all_fields_null_object_dropped(self):
        self.assertEqual(
            drop_null_plan_task_placeholders(
                [{"task_id": None, "child_id": None, "task": None}]
            ),
            [],
        )

    def test_mixed_real_task_and_null_placeholder(self):
        task = {
            "child_id": 0,
            "task_id": 0,
            "task": "Research AGI",
            "dependencies": [],
        }
        self.assertEqual(
            drop_null_plan_task_placeholders([None, task, {}]),
            [task],
        )

    def test_populated_plan_preserved(self):
        tasks = [
            {
                "child_id": 0,
                "task_id": 0,
                "task": "Research topic",
                "dependencies": [],
            }
        ]
        self.assertEqual(drop_null_plan_task_placeholders(tasks), tasks)

    def test_non_list_plan_passed_through(self):
        """Single-object / wrapper shapes remain draft PRs #73 / #74."""
        task = {
            "child_id": 0,
            "task_id": 0,
            "task": "Research AGI",
            "dependencies": [],
        }
        self.assertIs(drop_null_plan_task_placeholders(task), task)


class NullPlanTaskPlaceholderParentTests(unittest.TestCase):
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

    def test_null_plan_document_reaches_adjust_without_typeerror(self):
        """Before the fix, ``for t in None`` raised TypeError during planning."""
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs["task"])
            return f"result for {kwargs['task']}"

        parent = self._make_parent(plan_json=None, child_remote=child_remote)
        result = parent.run("top-level task")

        self.assertEqual(remote_calls, [])
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(parent.status, "idle")

    def test_null_placeholder_in_plan_array_dispatches_real_tasks(self):
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs["task"])
            return f"result for {kwargs['task']}"

        parent = self._make_parent(
            plan_json=[
                None,
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": "Research AGI",
                    "dependencies": [],
                },
                {},
            ],
            child_remote=child_remote,
        )

        result = parent.run("top-level task")

        self.assertEqual(remote_calls, ["Research AGI"])
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(parent.status, "idle")


if __name__ == "__main__":
    unittest.main()
