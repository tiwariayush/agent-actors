import sys
import types
import unittest


def install_dependency_stubs():
    ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        def decorator(cls):
            return cls

        if args and callable(args[0]) and not kwargs:
            return decorator(args[0])
        return decorator

    def get(refs):
        return [f"result:{ref}" for ref in refs]

    def wait(refs, num_returns=1):
        return list(refs[:num_returns]), list(refs[num_returns:])

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
        def __init__(self, *args, **kwargs):
            pass

    class MRKLChain:
        pass

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
        pass

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


class RecordingActor:
    def __init__(self):
        self.calls = []

    @property
    def run(self):
        actor = self

        class Remote:
            def remote(self, **kwargs):
                actor.calls.append(kwargs)
                return f"ref-{len(actor.calls)}"

        return Remote()


def arm(agent, name):
    agent.name = name
    agent.traits = ["focused"]
    agent.task = ""
    agent.status = "idle"
    agent.verbose = False
    agent.working_memory_state = "ready"
    agent.memories_since_last_reflection = 0
    agent.reflect_every = 10
    agent.cumulative_strength = 0.0
    agent.reflect_strength_trigger = float("inf")
    agent.children = {}
    agent.pause_to_reflect = lambda: None
    agent.actor = RecordingActor()


class NestedChildRosterTests(unittest.TestCase):
    def _parent(self, name, plan):
        parent = ParentAgent.__new__(ParentAgent)
        arm(parent, name)
        parent.plan = plan
        parent.adjust = types.SimpleNamespace(
            run=lambda **kwargs: {"confidence": 9, "result": "ok", "speak": "done"}
        )
        return parent

    def test_roster_is_visible_to_a_parent_actor_started_without_children(self):
        """Children attached after actor startup must still be the team that plans and runs."""
        seen = {}
        worker = types.SimpleNamespace(
            get_context=lambda: "AMIR",
            actor=RecordingActor(),
            children={},
        )

        def plan(inputs):
            seen["summary"] = inputs["child_summary"]
            return {
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "draft the report",
                        "dependencies": [],
                    }
                ]
            }

        # Simulates the Ray actor copy captured before luke.children = {0: amir}.
        luke = self._parent("Luke", plan)
        preserved = types.SimpleNamespace(get_context=lambda: "KEPT", actor=RecordingActor(), children={})
        luke.children = {7: preserved}

        result = luke.run(
            task="Write the report",
            child_roster={0: worker},
        )

        self.assertIsInstance(result, AgentFinish)
        self.assertIn("ID: 0", seen["summary"])
        self.assertIn("AMIR", seen["summary"])
        self.assertIn("ID: 7", seen["summary"])
        self.assertEqual(luke.children[0], worker)
        self.assertEqual(luke.children[7], preserved)
        self.assertEqual(worker.actor.calls[0]["task"], "draft the report")
        self.assertNotIn("child_roster", worker.actor.calls[0])
        self.assertEqual(preserved.actor.calls, [])

    def test_nested_parent_dispatch_forwards_its_current_children(self):
        grandchild = types.SimpleNamespace(name="Jiang")
        nested = ParentAgent.__new__(ParentAgent)
        arm(nested, "Luke")
        nested.children = {0: grandchild}

        def plan(inputs):
            return {
                "json": [
                    {
                        "task_id": 0,
                        "child_id": 0,
                        "task": "coordinate the team",
                        "dependencies": [],
                    }
                ]
            }

        cyrus = self._parent("Cyrus", plan)
        cyrus.children = {0: nested}

        cyrus.run(task="Write the report")

        self.assertEqual(len(nested.actor.calls), 1)
        forwarded = nested.actor.calls[0]["child_roster"]
        self.assertEqual(forwarded[0], grandchild)
        self.assertIsNot(forwarded, nested.children)


if __name__ == "__main__":
    unittest.main()
