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

from agent_actors.parent import ParentAgent, coerce_string_plan


class CoerceStringPlanTests(unittest.TestCase):
    def test_string_steps_become_tasks_for_child_zero(self):
        raw = ["Look up Sergey Brin's age", "  Multiply that age by 12  "]
        self.assertEqual(
            coerce_string_plan(raw, {0: object(), 1: object()}),
            [
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": "Look up Sergey Brin's age",
                    "dependencies": [],
                },
                {
                    "child_id": 0,
                    "task_id": 1,
                    "task": "Multiply that age by 12",
                    "dependencies": [],
                },
            ],
        )

    def test_string_steps_use_lowest_existing_child_when_zero_is_absent(self):
        coerced = coerce_string_plan(["Draft the report"], {42: object(), 2: object()})
        self.assertEqual(coerced[0]["child_id"], 2)
        self.assertEqual(coerced[0]["task_id"], 0)
        self.assertEqual(coerced[0]["task"], "Draft the report")

    def test_object_plans_are_unchanged(self):
        tasks = [
            {
                "child_id": 0,
                "task_id": 0,
                "task": "Research topic",
                "dependencies": [],
            }
        ]
        self.assertIs(coerce_string_plan(tasks, {0: object()}), tasks)
        single = {"child_id": 0, "task_id": 0, "task": "Research topic"}
        self.assertIs(coerce_string_plan(single, {0: object()}), single)

    def test_mixed_and_blank_lists_are_unchanged(self):
        mixed = ["Research topic", {"task": "Write summary"}]
        blank = ["   "]
        self.assertIs(coerce_string_plan(mixed, {0: object()}), mixed)
        self.assertIs(coerce_string_plan(blank, {0: object()}), blank)
        self.assertEqual(coerce_string_plan([], {0: object()}), [])


class StringPlanParentTests(unittest.TestCase):
    def _make_parent(self, plan_json, child_ids):
        parent = object.__new__(ParentAgent)
        parent.status = "idle"
        parent.task = ""
        parent.verbose = False
        parent.reflect_every = 10
        parent.get_context = lambda: "parent context"
        parent.pause_to_reflect = lambda: []
        remote_calls = []

        class FakePlan:
            def __call__(self, inputs):
                return {"json": plan_json}

        class FakeAdjust:
            def run(self, **kwargs):
                self.inputs = kwargs
                return {"confidence": 9, "result": "done", "speak": "done"}

        class FakeRun:
            def remote(self, **kwargs):
                remote_calls.append(kwargs)
                return f"result for {kwargs['task']}"

        class FakeActor:
            run = FakeRun()

        class FakeChild:
            actor = FakeActor()

            def get_context(self):
                return "child context"

        parent.plan = FakePlan()
        parent.adjust = FakeAdjust()
        parent.children = {child_id: FakeChild() for child_id in child_ids}
        return parent, remote_calls

    def test_string_plan_dispatches_instead_of_typeerror(self):
        parent, remote_calls = self._make_parent(
            ["Look up Sergey Brin's age", "Multiply that age by 12"],
            [0],
        )

        result = parent.run("What is Sergey Brin's age multiplied by 12?")

        self.assertEqual(
            [call["task"] for call in remote_calls],
            ["Look up Sergey Brin's age", "Multiply that age by 12"],
        )
        self.assertTrue(all(call["working_memory"] == [] for call in remote_calls))
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(result.return_values["confidence"], 9)
        self.assertEqual(parent.status, "idle")

    def test_string_plan_runs_on_existing_child_when_zero_is_absent(self):
        parent, remote_calls = self._make_parent(["Draft the report"], [42, 2])

        parent.run("Write the report")

        self.assertEqual(len(remote_calls), 1)
        self.assertEqual(remote_calls[0]["task"], "Draft the report")
        self.assertEqual(set(parent.children), {42, 2})


if __name__ == "__main__":
    unittest.main()
