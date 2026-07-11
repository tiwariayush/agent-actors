import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "agent_actors"


class RuntimeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._install_dependency_stubs()
        cls.results_module = cls._load_module(
            "agent_actors.results", PACKAGE_ROOT / "results.py"
        )
        cls.actors_module = cls._load_module(
            "agent_actors.actors", PACKAGE_ROOT / "actors.py"
        )
        cls.agent_module = cls._load_module(
            "agent_actors.agent", PACKAGE_ROOT / "agent.py"
        )
        cls.child_module = cls._load_module(
            "agent_actors.child", PACKAGE_ROOT / "child.py"
        )
        cls.parent_module = cls._load_module(
            "agent_actors.parent", PACKAGE_ROOT / "parent.py"
        )
        cls.AgentFinish = sys.modules["langchain.schema"].AgentFinish

    @classmethod
    def _load_module(cls, name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _install_dependency_stubs(cls):
        package = types.ModuleType("agent_actors")
        package.__path__ = [str(PACKAGE_ROOT)]
        sys.modules["agent_actors"] = package

        ray = types.ModuleType("ray")

        def remote(*args, **kwargs):
            if args and isinstance(args[0], type):
                return args[0]
            return lambda cls: cls

        ray.remote = remote
        ray.get = lambda refs: refs
        ray.wait = lambda refs, num_returns=1: (
            refs[:num_returns],
            refs[num_returns:],
        )
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
        chat_base = types.ModuleType("langchain.chat_models.base")

        class BaseChatModel:
            pass

        chat_base.BaseChatModel = BaseChatModel
        sys.modules["langchain.chat_models"] = chat_models
        sys.modules["langchain.chat_models.base"] = chat_base

        schema = types.ModuleType("langchain.schema")

        class BaseRetriever:
            pass

        class Document:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        class AgentFinish:
            def __init__(self, return_values, log):
                self.return_values = return_values
                self.log = log

        class AgentAction:
            def __init__(self, tool, tool_input, log):
                self.tool = tool
                self.tool_input = tool_input
                self.log = log

        schema.BaseRetriever = BaseRetriever
        schema.Document = Document
        schema.AgentFinish = AgentFinish
        schema.AgentAction = AgentAction
        sys.modules["langchain.schema"] = schema

        chains_package = types.ModuleType("agent_actors.chains")
        chains_package.__path__ = []
        sys.modules["agent_actors.chains"] = chains_package

        class StubChain:
            @classmethod
            def from_llm(cls, **kwargs):
                return cls()

        chains_agent = types.ModuleType("agent_actors.chains.agent")
        chains_agent.GenerateInsights = StubChain
        chains_agent.MemoryStrength = StubChain
        chains_agent.Synthesis = StubChain
        chains_agent.WorkingMemory = StubChain
        sys.modules["agent_actors.chains.agent"] = chains_agent

        chains_child = types.ModuleType("agent_actors.chains.child")
        chains_child.Check = StubChain
        chains_child.Do = StubChain
        sys.modules["agent_actors.chains.child"] = chains_child

        chains_parent = types.ModuleType("agent_actors.chains.parent")
        chains_parent.Adjust = StubChain
        chains_parent.Plan = StubChain
        sys.modules["agent_actors.chains.parent"] = chains_parent

        models = types.ModuleType("agent_actors.models")

        class TaskRef:
            def __init__(self, child_id, task_id):
                self.child_id = child_id
                self.task_id = task_id

            @property
            def id(self):
                return f"{self.child_id}.{self.task_id}"

        class TaskRecord(TaskRef):
            def __init__(self, child_id, task_id, task, dependencies=None):
                super().__init__(child_id=child_id, task_id=task_id)
                self.task = task
                self.dependencies = dependencies or []

        models.TaskRef = TaskRef
        models.TaskRecord = TaskRecord
        sys.modules["agent_actors.models"] = models

    def test_chain_actor_run_delegates_to_wrapped_chain(self):
        class Chain:
            def run(self, *args, **kwargs):
                return args, kwargs

        actor = self.actors_module.ChainActor(Chain())

        self.assertEqual(actor.run("task", option=True), (("task",), {"option": True}))

    def test_memory_strength_uses_matched_score(self):
        agent = object.__new__(self.agent_module.Agent)

        class MemoryStrength:
            def __init__(self, response):
                self.response = response

            def run(self, **kwargs):
                return self.response

        agent.memory_strength = MemoryStrength("Relevance: 8")
        self.assertEqual(agent._predict_memory_strength("memory"), 0.8)

        agent.memory_strength = MemoryStrength("10")
        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

        agent.memory_strength = MemoryStrength("Score: 15")
        self.assertEqual(agent._predict_memory_strength("memory"), 1.0)

    def test_child_formats_structured_dependency_results(self):
        child = object.__new__(self.child_module.ChildAgent)
        child.max_iterations = 1
        child.get_context = lambda: "base context"
        child.add_memory = lambda memory: None
        captured = {}

        class Do:
            def __call__(self, inputs, return_only_outputs=False):
                captured["context"] = inputs["context"]
                return {"intermediate_steps": [], "output": "complete answer"}

        class Check:
            def run(self, context, learning):
                return "Complete"

        child.do = Do()
        child.check = Check()

        result = child.run(
            "child task",
            working_memory=[self.AgentFinish({"result": "dependency done"}, "log")],
        )

        self.assertIn("dependency done", captured["context"])
        self.assertIn("complete answer", result)

    def test_parent_formats_nested_agent_results(self):
        parent = self._parent_with_plan(
            [
                {
                    "child_id": 0,
                    "task_id": 0,
                    "task": "subtask",
                    "dependencies": [],
                }
            ]
        )
        captured = parent._captured

        class RemoteRun:
            def remote(self, **kwargs):
                return self.AgentFinish({"result": "child complete"}, "child log")

        RemoteRun.AgentFinish = self.AgentFinish

        class Actor:
            run = RemoteRun()

        class Child:
            actor = Actor()

            def get_context(self):
                return "child context"

        parent.children = {0: Child()}

        result = parent.run(
            "parent task",
            working_memory=[self.AgentFinish({"result": "upstream complete"}, "log")],
        )

        self.assertIsInstance(result, self.AgentFinish)
        self.assertIn("upstream complete", captured["plan_context"])
        self.assertEqual(captured["adjust_results"], "child complete")

    def test_parent_creates_missing_child_at_requested_id(self):
        parent = self._parent_with_plan(
            [
                {
                    "child_id": 42,
                    "task_id": 0,
                    "task": "new child task",
                    "dependencies": [],
                }
            ]
        )
        captured = parent._captured
        created = {}
        original_child_agent = self.parent_module.ChildAgent

        class RemoteRun:
            def remote(self, **kwargs):
                captured["created_child_run"] = kwargs
                return "created child result"

        class Actor:
            run = RemoteRun()

        class FakeChildAgent:
            actor = Actor()

            def __init__(self, **kwargs):
                created["kwargs"] = kwargs

            def get_context(self):
                return "created child context"

        self.parent_module.ChildAgent = FakeChildAgent
        try:
            result = parent.run("parent task")
        finally:
            self.parent_module.ChildAgent = original_child_agent

        self.assertIsInstance(result, self.AgentFinish)
        self.assertIn(42, parent.children)
        self.assertEqual(parent.children[42].__class__, FakeChildAgent)
        self.assertEqual(created["kwargs"]["long_term_memory"], parent.long_term_memory)
        self.assertEqual(created["kwargs"]["tools"], parent.tools)
        self.assertEqual(captured["created_child_run"]["task"], "new child task")
        self.assertEqual(captured["adjust_results"], "created child result")

    def _parent_with_plan(self, planned_tasks):
        parent = object.__new__(self.parent_module.ParentAgent)
        parent.verbose = False
        parent.reflect_every = 1
        parent.llm = object()
        parent.long_term_memory = object()
        parent.tools = []
        parent.children = {}
        parent.get_context = lambda: "parent context"
        parent.pause_to_reflect = lambda: None
        parent._captured = {}

        class Plan:
            def __call__(self, inputs):
                parent._captured["plan_context"] = inputs["context"]
                return {"json": planned_tasks}

        class Adjust:
            def run(self, context, task, results):
                parent._captured["adjust_results"] = results
                return {"confidence": 8, "result": "final answer"}

        parent.plan = Plan()
        parent.adjust = Adjust()
        return parent


if __name__ == "__main__":
    unittest.main()
