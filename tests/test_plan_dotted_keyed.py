import json
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
    _parse_dotted_citation_key,
    normalize_plan_tasks,
)


TASK_A = {
    "child_id": 0,
    "task_id": 0,
    "task": "Look up Sergey Brin's age",
    "dependencies": [],
}
TASK_B = {
    "child_id": 0,
    "task_id": 1,
    "task": "Multiply that age by 12",
    "dependencies": [],
}
TASK_C = {
    "child_id": 1,
    "task_id": 0,
    "task": "Write the report",
    "dependencies": [],
}


class ParseDottedCitationKeyTests(unittest.TestCase):
    def test_plain_dotted_key(self):
        self.assertEqual(_parse_dotted_citation_key("0.0"), (0, 0))
        self.assertEqual(_parse_dotted_citation_key("1.2"), (1, 2))
        self.assertEqual(_parse_dotted_citation_key("42.7"), (42, 7))

    def test_bracketed_citation_key(self):
        self.assertEqual(_parse_dotted_citation_key("[0.0]"), (0, 0))
        self.assertEqual(_parse_dotted_citation_key("[1.2]"), (1, 2))

    def test_all_digit_index_keys_rejected(self):
        """All-digit keys stay with draft PR #92."""
        self.assertIsNone(_parse_dotted_citation_key("0"))
        self.assertIsNone(_parse_dotted_citation_key("42"))

    def test_unbalanced_brackets_rejected(self):
        self.assertIsNone(_parse_dotted_citation_key("[0.0"))
        self.assertIsNone(_parse_dotted_citation_key("0.0]"))

    def test_non_strings_rejected(self):
        self.assertIsNone(_parse_dotted_citation_key(0.0))
        self.assertIsNone(_parse_dotted_citation_key(0))


class NormalizeDottedKeyedPlanTests(unittest.TestCase):
    def test_list_plan_unchanged(self):
        tasks = [TASK_A]
        self.assertIs(normalize_plan_tasks(tasks), tasks)

    def test_dotted_keys_without_ids_are_filled_from_the_key(self):
        """Models key by [worker #.task #] and omit child_id/task_id on the value."""
        raw = {
            "0.0": {"task": TASK_A["task"], "dependencies": []},
            "0.1": {"task": TASK_B["task"], "dependencies": []},
        }
        self.assertEqual(
            normalize_plan_tasks(raw),
            [
                {
                    "task": TASK_A["task"],
                    "dependencies": [],
                    "child_id": 0,
                    "task_id": 0,
                },
                {
                    "task": TASK_B["task"],
                    "dependencies": [],
                    "child_id": 0,
                    "task_id": 1,
                },
            ],
        )

    def test_bracketed_keys_are_filled_from_the_key(self):
        raw = {
            "[0.0]": {"task": TASK_A["task"], "dependencies": []},
            "[1.0]": {"task": TASK_C["task"], "dependencies": []},
        }
        records = normalize_plan_tasks(raw)
        self.assertEqual(records[0]["child_id"], 0)
        self.assertEqual(records[0]["task_id"], 0)
        self.assertEqual(records[1]["child_id"], 1)
        self.assertEqual(records[1]["task_id"], 0)

    def test_json_loads_string_keys_from_citation_floats(self):
        """JSON object keys are strings; float-like 0.0 serializes as '0.0'."""
        raw = json.loads(
            '{"0.0": {"task": "Research AGI", "dependencies": []},'
            ' "1.0": {"task": "Write the report", "dependencies": []}}'
        )
        records = normalize_plan_tasks(raw)
        self.assertEqual(records[0]["child_id"], 0)
        self.assertEqual(records[0]["task_id"], 0)
        self.assertEqual(records[1]["child_id"], 1)
        self.assertEqual(records[1]["task_id"], 0)

    def test_does_not_overwrite_explicit_ids(self):
        raw = {
            "9.9": {
                "child_id": 0,
                "task_id": 1,
                "task": "Keep assigned ids",
                "dependencies": [],
            }
        }
        records = normalize_plan_tasks(raw)
        self.assertEqual(records[0]["child_id"], 0)
        self.assertEqual(records[0]["task_id"], 1)

    def test_all_digit_keyed_object_left_unchanged(self):
        """{"0": {task}} stays with draft PR #92."""
        raw = {"0": TASK_A, "1": TASK_C}
        self.assertIs(normalize_plan_tasks(raw), raw)

    def test_single_task_object_left_unchanged(self):
        """Bare task objects stay with draft PR #73."""
        self.assertIs(normalize_plan_tasks(TASK_A), TASK_A)

    def test_tasks_wrapper_left_unchanged(self):
        """{"tasks": [...]} wrappers stay with draft PR #74."""
        wrapped = {"tasks": [TASK_A]}
        self.assertIs(normalize_plan_tasks(wrapped), wrapped)

    def test_id_field_on_array_tasks_left_unchanged(self):
        """Array items with an `id` field stay with draft PRs #93/#94."""
        raw = [{"id": "0.0", "task": "Research AGI", "dependencies": []}]
        self.assertIs(normalize_plan_tasks(raw), raw)

    def test_empty_object_left_unchanged(self):
        raw = {}
        self.assertIs(normalize_plan_tasks(raw), raw)

    def test_non_task_dotted_values_left_unchanged(self):
        raw = {"0.0": "Research topic", "1.0": "Write summary"}
        self.assertIs(normalize_plan_tasks(raw), raw)


class DottedKeyedPlanParentTests(unittest.TestCase):
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

    def test_dotted_keyed_object_dispatches_instead_of_typeerror(self):
        """Before the fix, iterating {"0.0": {task}} yielded the key '0.0' and crashed."""
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs["task"])
            return f"result for {kwargs['task']}"

        parent = self._make_parent(
            plan_json={
                "0.0": {"task": TASK_A["task"], "dependencies": []},
                "1.0": {"task": TASK_C["task"], "dependencies": []},
            },
            child_remote=child_remote,
        )

        result = parent.run("top-level task")

        self.assertEqual(
            remote_calls,
            [TASK_A["task"], TASK_C["task"]],
        )
        self.assertIsInstance(result, AgentFinish)
        self.assertEqual(parent.status, "idle")

    def test_bracketed_keys_dispatch(self):
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs["task"])
            return f"result for {kwargs['task']}"

        parent = self._make_parent(
            plan_json={
                "[0.0]": {"task": TASK_A["task"], "dependencies": []},
                "[0.1]": {"task": TASK_B["task"], "dependencies": []},
            },
            child_remote=child_remote,
        )

        parent.run("top-level task")
        self.assertEqual(remote_calls, [TASK_A["task"], TASK_B["task"]])

    def test_array_plan_still_dispatches(self):
        remote_calls = []

        def child_remote(**kwargs):
            remote_calls.append(kwargs["task"])
            return f"result for {kwargs['task']}"

        parent = self._make_parent(
            plan_json=[TASK_A, TASK_C],
            child_remote=child_remote,
        )

        parent.run("top-level task")
        self.assertEqual(remote_calls, [TASK_A["task"], TASK_C["task"]])


if __name__ == "__main__":
    unittest.main()
