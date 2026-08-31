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

from agent_actors.parent import ParentAgent, normalize_plan_tasks


TASK_A = {
    "child_id": 0,
    "task_id": 0,
    "task": "Research topic",
    "dependencies": [],
}
TASK_B = {
    "child_id": 1,
    "task_id": 0,
    "task": "Write summary",
    "dependencies": [],
}
TASK_C = {
    "child_id": 0,
    "task_id": 1,
    "task": "Cite sources",
    "dependencies": [],
}


class NormalizeNumericKeyedPlanTests(unittest.TestCase):
    def test_list_plan_unchanged(self):
        tasks = [TASK_A]
        self.assertIs(normalize_plan_tasks(tasks), tasks)

    def test_index_keyed_task_objects(self):
        """LLMs often emit {"0": {task}, "1": {task}} instead of an array."""
        raw = {"1": TASK_B, "0": TASK_A}
        self.assertEqual(normalize_plan_tasks(raw), [TASK_A, TASK_B])

    def test_child_id_keyed_task_lists(self):
        """Plan prompt assigns work per team member id, so models group by worker."""
        raw = {"0": [TASK_A, TASK_C], "1": [TASK_B]}
        self.assertEqual(normalize_plan_tasks(raw), [TASK_A, TASK_C, TASK_B])

    def test_mixed_object_and_list_values(self):
        raw = {"0": TASK_A, "1": [TASK_B]}
        self.assertEqual(normalize_plan_tasks(raw), [TASK_A, TASK_B])

    def test_integer_keys(self):
        raw = {0: TASK_A, 42: TASK_B}
        self.assertEqual(normalize_plan_tasks(raw), [TASK_A, TASK_B])

    def test_single_task_object_left_unchanged(self):
        """Bare task objects stay with draft PR #73."""
        self.assertIs(normalize_plan_tasks(TASK_A), TASK_A)

    def test_tasks_wrapper_left_unchanged(self):
        """{"tasks": [...]} wrappers stay with draft PR #74."""
        wrapped = {"tasks": [TASK_A]}
        self.assertIs(normalize_plan_tasks(wrapped), wrapped)

    def test_empty_object_left_unchanged(self):
        raw = {}
        self.assertIs(normalize_plan_tasks(raw), raw)

    def test_non_task_numeric_values_left_unchanged(self):
        raw = {"0": "Research topic", "1": "Write summary"}
        self.assertIs(normalize_plan_tasks(raw), raw)


class NumericKeyedPlanParentTests(unittest.TestCase):
    def _make_parent(self, plan_json, child_remote, children=None):
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
        if children is None:
            parent.children = {0: FakeChild(), 1: FakeChild()}
        else:
            parent.children = children
        return parent

    def test_index_keyed_object_dispatches_instead_of_typeerror(self):
        """Before the fix, iterating {"0": {task}} yielded the key '0' and crashed."""
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs["task"])
            return f"result for {kwargs['task']}"

        parent = self._make_parent(
            plan_json={"0": TASK_A, "1": TASK_B},
            child_remote=child_remote,
        )

        result = parent.run("top-level task")

        self.assertEqual(remote_calls, ["Research topic", "Write summary"])
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(parent.status, "idle")

    def test_worker_keyed_lists_dispatch(self):
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs["task"])
            return f"result for {kwargs['task']}"

        parent = self._make_parent(
            plan_json={"0": [TASK_A, TASK_C], "1": [TASK_B]},
            child_remote=child_remote,
        )

        parent.run("top-level task")
        self.assertEqual(
            remote_calls, ["Research topic", "Cite sources", "Write summary"]
        )

    def test_array_plan_still_dispatches(self):
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs["task"])
            return f"result for {kwargs['task']}"

        parent = self._make_parent(
            plan_json=[TASK_A, TASK_B],
            child_remote=child_remote,
        )

        parent.run("top-level task")
        self.assertEqual(remote_calls, ["Research topic", "Write summary"])


if __name__ == "__main__":
    unittest.main()
